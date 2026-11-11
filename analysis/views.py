import base64
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from io import BytesIO, StringIO

import matplotlib
import pandas as pd
import seaborn as sns
from django.contrib import messages
from django.core.files import File
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from matplotlib import font_manager
from matplotlib.figure import Figure
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .models import ColumnData, MLResult, UploadedFile

# 日本語Excelで保存したCSVにも対応するため、UTF-8(BOM付きも可) → Shift-JIS(cp932) の順に試す
CSV_ENCODINGS = ('utf-8-sig', 'cp932')

# Kaggle のデータセットIDは「ユーザー名/データセット名」の形式
KAGGLE_DATASET_PATTERN = re.compile(r'^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$')
KAGGLE_TIMEOUT_SECONDS = 300

# 学習用とテスト用(2割)に分けても評価できるだけの行数
MIN_ROWS_FOR_REGRESSION = 10

# 相関ヒートマップで数値を書き込む列数の上限（多すぎると読めないため）
MAX_ANNOTATED_COLUMNS = 15

JAPANESE_FONT_CANDIDATES = [
    'Hiragino Sans', 'Hiragino Kaku Gothic ProN',  # macOS
    'Yu Gothic', 'Meiryo',  # Windows
    'Noto Sans CJK JP', 'IPAexGothic', 'IPAGothic',  # Linux
]

MISSING_FILE_MESSAGE = (
    'このファイルはCSV本体が保存されていません（以前のバージョンで登録したデータです）。'
    'お手数ですが、もう一度アップロードしてください'
)


class CsvReadError(Exception):
    """CSVを読み込めなかったときのエラー（画面に表示するメッセージを持つ）"""


class AnalysisError(Exception):
    """分析の入力が不正なときのエラー（画面に表示するメッセージを持つ）"""


# ---------------------------------------------------------------------------
# 共通処理
# ---------------------------------------------------------------------------

def read_csv_any(source):
    """パスまたはファイルオブジェクトからCSVを読み込む。文字コードは自動で判別する"""
    if isinstance(source, (str, os.PathLike)):
        with open(source, 'rb') as f:
            raw = f.read()
    else:
        source.seek(0)
        raw = source.read()

    for encoding in CSV_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            return pd.read_csv(StringIO(text))
        except pd.errors.EmptyDataError:
            raise CsvReadError('CSVファイルが空です')
        except pd.errors.ParserError:
            raise CsvReadError('CSVの形式が正しくありません（列数がそろっているか確認してください）')
    raise CsvReadError('文字コードを判別できませんでした。UTF-8 か Shift-JIS で保存してください')


def to_float(value):
    """DBに保存できる値に変換する（NaN は None にする）"""
    if value is None or pd.isna(value):
        return None
    return float(value)


@transaction.atomic
def register_csv(name, df, csv_file):
    """CSV本体とファイル情報を保存し、数値列ごとの基本統計量を1列＝1行で保存する"""
    uploaded_file = UploadedFile.objects.create(
        name=name,
        csv_file=csv_file,
        row_count=len(df),
        column_count=len(df.columns),
    )
    column_stats = []
    for col in df.select_dtypes(include='number').columns:
        series = df[col]
        column_stats.append(ColumnData(
            file=uploaded_file,
            column_name=str(col),
            mean=to_float(series.mean()),
            median=to_float(series.median()),
            std=to_float(series.std()),
            min_value=to_float(series.min()),
            max_value=to_float(series.max()),
            null_count=int(series.isnull().sum()),
        ))
    ColumnData.objects.bulk_create(column_stats)
    return uploaded_file


def load_dataframe(uploaded_file):
    """保存してあるCSV本体を読み込む。本体が無い（古いデータ）場合は AnalysisError"""
    if not uploaded_file.csv_file:
        raise AnalysisError(MISSING_FILE_MESSAGE)
    try:
        return read_csv_any(uploaded_file.csv_file.path)
    except FileNotFoundError:
        raise AnalysisError(MISSING_FILE_MESSAGE)
    except CsvReadError as e:
        raise AnalysisError(str(e))


@lru_cache(maxsize=1)
def font_families():
    """インストールされている日本語フォントを優先し、無い文字は DejaVu Sans で表示する"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in JAPANESE_FONT_CANDIDATES if name in installed] + ['DejaVu Sans']


def render_heatmap(corr):
    """相関行列をヒートマップにして base64 のPNG文字列で返す"""
    # pyplot を使わず Figure を直接作る（Webサーバーの別スレッドでも安全に描画できる）
    with matplotlib.rc_context({'font.family': font_families()}):
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        sns.heatmap(
            corr,
            annot=len(corr.columns) <= MAX_ANNOTATED_COLUMNS,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title('相関係数ヒートマップ', fontsize=14, pad=20)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def run_regression(uploaded_file, target, features, strategy):
    """線形回帰で学習・評価し、結果を保存して返す"""
    if strategy not in dict(MLResult.MISSING_STRATEGIES):
        raise AnalysisError('欠損値の扱いを選んでください')
    if not features:
        raise AnalysisError('説明変数を1つ以上選んでください')
    if target in features:
        raise AnalysisError('目的変数と同じ列は説明変数に選べません')

    df = load_dataframe(uploaded_file)
    numeric_columns = set(df.select_dtypes(include='number').columns)
    if target not in numeric_columns:
        raise AnalysisError('目的変数には数値の列を選んでください')
    not_numeric = [f for f in features if f not in numeric_columns]
    if not_numeric:
        raise AnalysisError(f'数値ではない列が含まれています：{", ".join(not_numeric)}')
    all_missing = [f for f in features if df[f].isna().all()]
    if all_missing:
        raise AnalysisError(f'値がすべて欠損している列は使えません：{", ".join(all_missing)}')

    # 目的変数は補完すると正解を作ってしまうので、欠損している行は常に除く
    data = df[[target] + features].dropna(subset=[target])
    if strategy == 'drop':
        data = data.dropna()
    if len(data) < MIN_ROWS_FOR_REGRESSION:
        raise AnalysisError(
            f'分析に使える行が{len(data)}行しかありません（{MIN_ROWS_FOR_REGRESSION}行以上必要です）。'
            '欠損値を補完する設定にするか、説明変数を見直してください'
        )

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if strategy == 'drop':
        model = LinearRegression()
    else:
        # 補完に使う平均値・中央値は学習データだけから計算する（テストデータの情報を学習に混ぜない）
        model = make_pipeline(SimpleImputer(strategy=strategy), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))

    MLResult.objects.create(
        file=uploaded_file,
        target_column=target,
        feature_columns=','.join(features),
        missing_strategy=strategy,
        r2_score=r2,
        mse=mse,
    )
    return {
        'r2': round(r2, 4),
        'mse': round(mse, 4),
        'target': target,
        'features': features,
        'strategy_label': dict(MLResult.MISSING_STRATEGIES)[strategy],
        'train_rows': len(X_train),
        'test_rows': len(X_test),
    }


def find_csv_files(directory):
    """フォルダ以下（サブフォルダを含む）のCSVファイルのパスを返す"""
    csv_paths = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith('.csv'):
                csv_paths.append(os.path.join(root, name))
    return sorted(csv_paths)


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------

def upload(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')
        if csv_file is None or not csv_file.name.lower().endswith('.csv'):
            messages.error(request, 'CSVファイルを選択してください')
            return redirect('upload')
        try:
            df = read_csv_any(csv_file)
        except CsvReadError as e:
            messages.error(request, str(e))
            return redirect('upload')
        uploaded_file = register_csv(csv_file.name, df, csv_file)
        return redirect('dashboard', pk=uploaded_file.pk)

    files = UploadedFile.objects.order_by('-uploaded_at')
    return render(request, 'analysis/upload.html', {'files': files})


def dashboard(request, pk):
    uploaded_file = get_object_or_404(UploadedFile, pk=pk)
    columns = list(uploaded_file.columns.order_by('pk'))
    # グラフ用のデータは json_script でJSに渡す（列名の記号や欠損値でJSが壊れないように）
    chart_data = [
        {
            'id': f'chart_{i}',
            'label': col.column_name,
            'data': [to_float(col.min_value), to_float(col.mean),
                     to_float(col.median), to_float(col.max_value)],
        }
        for i, col in enumerate(columns, start=1)
    ]
    return render(request, 'analysis/dashboard.html', {
        'file': uploaded_file,
        'columns': columns,
        'chart_data': chart_data,
    })


def kaggle_import(request):
    if request.method == 'POST':
        dataset = request.POST.get('dataset', '').strip()
        if not KAGGLE_DATASET_PATTERN.match(dataset):
            messages.error(request, 'データセットIDは「ユーザー名/データセット名」の形式で入力してください')
            return redirect('kaggle_import')

        # 前回取得したCSVが混ざらないよう、取得のたびに新しい一時フォルダを使う
        with tempfile.TemporaryDirectory() as download_dir:
            try:
                subprocess.run(
                    ['kaggle', 'datasets', 'download', '-d', dataset, '-p', download_dir, '--unzip'],
                    check=True,
                    capture_output=True,
                    timeout=KAGGLE_TIMEOUT_SECONDS,
                )
            except FileNotFoundError:
                messages.error(request, 'kaggle コマンドが見つかりません。requirements.txt のパッケージをインストールしてください')
                return redirect('kaggle_import')
            except subprocess.TimeoutExpired:
                messages.error(request, 'データセットの取得がタイムアウトしました。時間をおいて再度お試しください')
                return redirect('kaggle_import')
            except subprocess.CalledProcessError:
                messages.error(request, 'データセットの取得に失敗しました。データセットIDと .env の Kaggle 認証情報を確認してください')
                return redirect('kaggle_import')

            registered = []
            for path in find_csv_files(download_dir):
                name = os.path.basename(path)
                try:
                    df = read_csv_any(path)
                except CsvReadError as e:
                    messages.warning(request, f'{name} は読み込めませんでした：{e}')
                    continue
                with open(path, 'rb') as f:
                    registered.append(register_csv(name, df, File(f, name=name)))

        if not registered:
            messages.error(request, 'CSVファイルが見つかりませんでした')
            return redirect('kaggle_import')
        if len(registered) == 1:
            return redirect('dashboard', pk=registered[0].pk)
        messages.success(request, f'{len(registered)}個のCSVを取り込みました。分析したいファイルを履歴から選んでください')
        return redirect('upload')

    return render(request, 'analysis/kaggle_import.html')


def ml_predict(request, pk):
    uploaded_file = get_object_or_404(UploadedFile, pk=pk)
    context = {
        'file': uploaded_file,
        'columns': uploaded_file.columns.order_by('pk'),
        'strategies': MLResult.MISSING_STRATEGIES,
        'selected_target': None,
        'selected_features': [],
        'selected_strategy': 'drop',
        'result': None,
    }

    if request.method == 'POST':
        target = request.POST.get('target', '')
        features = list(dict.fromkeys(request.POST.getlist('features')))  # 重複を除く
        strategy = request.POST.get('missing_strategy', 'drop')
        context.update(selected_target=target, selected_features=features, selected_strategy=strategy)
        try:
            context['result'] = run_regression(uploaded_file, target, features, strategy)
        except AnalysisError as e:
            messages.error(request, str(e))

    return render(request, 'analysis/ml_predict.html', context)


def correlation(request, pk):
    uploaded_file = get_object_or_404(UploadedFile, pk=pk)
    img_base64 = None
    try:
        df = load_dataframe(uploaded_file)
    except AnalysisError as e:
        messages.error(request, str(e))
    else:
        # 値がすべて欠損している列は相関を計算できないので除く
        numeric_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
        if len(numeric_df.columns) >= 2:
            img_base64 = render_heatmap(numeric_df.corr())

    return render(request, 'analysis/correlation.html', {
        'file': uploaded_file,
        'img': img_base64,
    })
