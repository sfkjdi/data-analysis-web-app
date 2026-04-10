# Data Analysis Web App

Django と PostgreSQL を使ったデータ分析 Web アプリケーションです。

## 概要

CSV ファイルのアップロードまたは Kaggle API を使ってデータを取得し、基本統計量の算出・グラフ表示・機械学習による予測を行うことができます。

## 機能

- **CSV アップロード**：手元の CSV ファイルをアップロードしてデータ分析
- **Kaggle API 連携**：Kaggle からデータセットを直接取得
- **基本統計量の表示**：平均・中央値・標準偏差・最小値・最大値・欠損値数
- **グラフ表示**：Chart.js・matplotlib・seaborn を使ったデータの可視化
- **機械学習**：回帰モデルによる予測

## 使用技術

| カテゴリ | 技術 |
|---|---|
| バックエンド | Python / Django |
| データベース | PostgreSQL |
| フロントエンド | HTML / CSS / JavaScript / Chart.js |
| データ分析 | pandas / numpy / matplotlib / seaborn / scikit-learn / scipy |
| API 連携 | Kaggle API |
| その他 | python-dotenv / psycopg2 |

## セットアップ

### 前提条件

- Python 3.x
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

**4. 環境変数を設定**

`.env` ファイルを作成し、以下を設定してください。
```
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

**5. マイグレーションを実行**
```bash
python manage.py migrate
```

**6. 開発サーバーを起動**
```bash
python manage.py runserver
```

ブラウザで `http://127.0.0.1:8000` にアクセスしてください。

## 工夫した点

- CSV と Kaggle API の両方に対応し、データソースを柔軟に選択できる設計にしました
- pandas / numpy を使って欠損値を含むデータも適切に処理できるようにしました
- Chart.js・matplotlib・seaborn でグラフを描画し、分析結果を直感的に確認できる UI を実装しました
- scikit-learn の回帰モデルを組み込み、データから予測まで一貫して行えるようにしました
