# Data Analysis Web App

Django と PostgreSQL を使ったデータ分析 Web アプリケーションです。

## 概要

CSV ファイルのアップロードまたは Kaggle API を使ってデータを取得し、基本統計量の算出・グラフ表示・相関分析・機械学習による予測を行うことができます。

## 機能

- **CSV アップロード**：手元の CSV ファイルをアップロードしてデータ分析（UTF-8 / Shift-JIS に対応）
- **Kaggle API 連携**：Kaggle からデータセットを直接取得
- **基本統計量の表示**：平均・中央値・標準偏差・最小値・最大値・欠損値数
- **グラフ表示**：Chart.js で統計量を棒グラフ・折れ線グラフなどに切り替えて表示
- **相関分析**：数値列どうしの相関係数を seaborn のヒートマップで表示
- **機械学習**：目的変数と説明変数を選んで線形回帰を行い、R² と MSE で評価

## 使用技術

| カテゴリ | 技術 |
|---|---|
| バックエンド | Python / Django |
| データベース | PostgreSQL |
| フロントエンド | HTML / CSS / JavaScript / Bootstrap / Chart.js |
| データ分析 | pandas / matplotlib / seaborn / scikit-learn |
| API 連携 | Kaggle API |
| その他 | python-dotenv / psycopg2 |

## セットアップ

### 前提条件

- Python 3.9〜3.12（requirements.txt のバージョンが 3.13 に未対応のため）
- PostgreSQL
- Kaggle アカウント（Kaggle API 使用時）

### 手順

**1. リポジトリをクローン**
```bash
git clone https://github.com/sfkjdi/data-analysis-web-app.git
cd data-analysis-web-app
```

**2. 仮想環境を作成・有効化**
```bash
python -m venv venv
source venv/bin/activate
```

**3. パッケージをインストール**
```bash
pip install -r requirements.txt
```

**4. データベースを作成**
```bash
createdb data_analysis
```

**5. 環境変数を設定**

`.env.example` をコピーして `.env` を作成し、値を設定してください。
```bash
cp .env.example .env
```

`SECRET_KEY` は次のコマンドで生成できます。
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

| 変数 | 内容 |
|---|---|
| `SECRET_KEY` | Django のシークレットキー |
| `DEBUG` | 開発時は `True` |
| `ALLOWED_HOSTS` | カンマ区切りのホスト名（既定：`localhost,127.0.0.1`） |
| `DB_NAME` / `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` | PostgreSQL の接続情報 |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle API の認証情報（Kaggle 取得を使う場合） |

**6. マイグレーションを実行**
```bash
python manage.py migrate
```

**7. 開発サーバーを起動**
```bash
python manage.py runserver
```

ブラウザで `http://127.0.0.1:8000` にアクセスしてください。

### テスト

```bash
python manage.py test analysis
```

## 工夫した点

- 列ごとの統計量を 1 列＝1 行の形で縦に持たせる設計にし、列数の違う CSV でも同じテーブルで扱えるようにしました
- アップロードした CSV 本体を保存し、回帰分析・相関分析は選んだファイルのデータで行うようにしました
- 回帰分析では、説明変数の欠損値を「行を削除」「平均値で補完」「中央値で補完」から選べるようにしました。補完に使う値は学習データだけから計算し、テストデータの情報が学習に混ざらないようにしています
- CSV と Kaggle API の両方に対応し、データソースを柔軟に選択できる設計にしました
- Chart.js・seaborn でグラフを描画し、分析結果を直感的に確認できる UI を実装しました
