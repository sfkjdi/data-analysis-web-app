import pandas as pd
import os
import subprocess
from django.shortcuts import render, redirect
from .models import UploadedFile, ColumnData, MLResult
from django.contrib import messages


def upload(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'CSVファイルを選択してください')
            return redirect('upload')
        df = pd.read_csv(csv_file)
        uploaded_file = UploadedFile.objects.create(
            name=csv_file.name,
            row_count=len(df),
            column_count=len(df.columns)
        )
        for col in df.select_dtypes(include='number').columns:
            ColumnData.objects.create(
                file=uploaded_file,
                column_name=col,
                mean=df[col].mean(),
                median=df[col].median(),
                std=df[col].std(),
                min_value=df[col].min(),
                max_value=df[col].max(),
                null_count=int(df[col].isnull().sum())
            )
        return redirect('dashboard', pk=uploaded_file.pk)
    files = UploadedFile.objects.all().order_by('-uploaded_at')
    return render(request, 'analysis/upload.html', {'files': files})


def dashboard(request, pk):
    uploaded_file = UploadedFile.objects.get(pk=pk)
    columns = uploaded_file.columns.all()
    return render(request, 'analysis/dashboard.html', {
        'file': uploaded_file,
        'columns': columns
    })


def kaggle_import(request):
    if request.method == 'POST':
        dataset = request.POST.get('dataset')
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
        download_path = os.path.join('kaggle_data')
        os.makedirs(download_path, exist_ok=True)
        try:
            subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset, '-p', download_path, '--unzip'],
                check=True
            )
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            if not csv_files:
                messages.error(request, 'CSVファイルが見つかりませんでした')
                return redirect('kaggle_import')
            for csv_file_name in csv_files:
                csv_path = os.path.join(download_path, csv_file_name)
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='shift-jis')
                uploaded_file = UploadedFile.objects.create(
                    name=csv_file_name,
                    row_count=len(df),
                    column_count=len(df.columns)
                )
                for col in df.select_dtypes(include='number').columns:
                    ColumnData.objects.create(
                        file=uploaded_file,
                        column_name=col,
                        mean=df[col].mean(),
                        median=df[col].median(),
                        std=df[col].std(),
                        min_value=df[col].min(),
                        max_value=df[col].max(),
                        null_count=int(df[col].isnull().sum())
                    )
            return redirect('dashboard', pk=uploaded_file.pk)
        except subprocess.CalledProcessError:
            messages.error(request, 'データセットの取得に失敗しました。データセット名を確認してください')
            return redirect('kaggle_import')
    return render(request, 'analysis/kaggle_import.html')

def ml_predict(request, pk):
    uploaded_file = UploadedFile.objects.get(pk=pk)
    columns = uploaded_file.columns.all()
    result = None

    if request.method == 'POST':
        target = request.POST.get('target')
        features = request.POST.getlist('features')

        import glob
        csv_files = glob.glob(f'kaggle_data/*.csv') + glob.glob(f'*.csv')
        df = None
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if target in df.columns:
                    break
            except:
                continue

        if df is not None and target in df.columns:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error

            valid_features = [f for f in features if f in df.columns]
            df_clean = df[[target] + valid_features].dropna()

            X = df_clean[valid_features]
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            MLResult.objects.create(
                file=uploaded_file,
                target_column=target,
                feature_columns=','.join(valid_features),
                r2_score=r2,
                mse=mse
            )

            result = {
                'r2': round(r2, 4),
                'mse': round(mse, 4),
                'target': target,
                'features': valid_features
            }

    return render(request, 'analysis/ml_predict.html', {
        'file': uploaded_file,
        'columns': columns,
        'result': result
    })


def ml_predict(request, pk):
    uploaded_file = UploadedFile.objects.get(pk=pk)
    columns = uploaded_file.columns.all()
    result = None

    if request.method == 'POST':
        target = request.POST.get('target')
        features = request.POST.getlist('features')

        import glob
        csv_files = glob.glob(f'kaggle_data/*.csv') + glob.glob(f'*.csv')
        df = None
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if target in df.columns:
                    break
            except:
                continue

        if df is not None and target in df.columns:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error

            valid_features = [f for f in features if f in df.columns]
            df_clean = df[[target] + valid_features].dropna()

            X = df_clean[valid_features]
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            MLResult.objects.create(
                file=uploaded_file,
                target_column=target,
                feature_columns=','.join(valid_features),
                r2_score=r2,
                mse=mse
            )

            result = {
                'r2': round(r2, 4),
                'mse': round(mse, 4),
                'target': target,
                'features': valid_features
            }

    return render(request, 'analysis/ml_predict.html', {
        'file': uploaded_file,
        'columns': columns,
        'result': result
    })
def ml_predict(request, pk):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import glob

    uploaded_file = UploadedFile.objects.get(pk=pk)
    columns = uploaded_file.columns.all()
    result = None

    if request.method == 'POST':
        target = request.POST.get('target')
        features = request.POST.getlist('features')

        csv_files = glob.glob('kaggle_data/*.csv') + glob.glob('*.csv')
        df = None
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if target in df.columns:
                    break
            except:
                continue

        if df is not None and target in df.columns:
            valid_features = [f for f in features if f in df.columns]
            df_clean = df[[target] + valid_features].dropna()
            X = df_clean[valid_features]
            y = df_clean[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            MLResult.objects.create(
                file=uploaded_file,
                target_column=target,
                feature_columns=','.join(valid_features),
                r2_score=r2,
                mse=mse
            )
            result = {
                'r2': round(r2, 4),
                'mse': round(mse, 4),
                'target': target,
                'features': valid_features
            }

    return render(request, 'analysis/ml_predict.html', {
        'file': uploaded_file,
        'columns': columns,
        'result': result
    })
def correlation(request, pk):
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    uploaded_file = UploadedFile.objects.get(pk=pk)

    csv_files = glob.glob('kaggle_data/*.csv') + glob.glob('*.csv')
    df = None
    for f in csv_files:
        try:
            temp = pd.read_csv(f)
            if len(temp.columns) > 1:
                df = temp
                break
        except:
            continue

    img_base64 = None
    if df is not None:
        numeric_df = df.select_dtypes(include='number')
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                ax=ax,
                linewidths=0.5
            )
            ax.set_title('相関係数ヒートマップ', fontsize=14, pad=20)
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

    return render(request, 'analysis/correlation.html', {
        'file': uploaded_file,
        'img': img_base64
    })