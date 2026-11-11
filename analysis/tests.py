import json
import os
import re
import shutil
import subprocess
import tempfile
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import ColumnData, MLResult, UploadedFile

TEMP_MEDIA_ROOT = tempfile.mkdtemp()


def make_csv(name, text, encoding='utf-8'):
    return SimpleUploadedFile(name, text.encode(encoding), content_type='text/csv')


def linear_csv(rows=20):
    """y = 2x + 1 がぴったり成り立つデータ（R² が 1 になる）"""
    lines = ['x,y,z']
    for i in range(rows):
        lines.append(f'{i},{2 * i + 1},{(i * 7) % 5}')
    return '\n'.join(lines)


def random_like_csv(rows=20):
    """x と y に関係がないデータ"""
    ys = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0, 5, 9, 1, 7, 3, 8, 2, 6, 4, 0]
    lines = ['x,y,z']
    for i in range(rows):
        lines.append(f'{i},{ys[i % len(ys)]},{i % 3}')
    return '\n'.join(lines)


@override_settings(MEDIA_ROOT=TEMP_MEDIA_ROOT)
class AnalysisTestBase(TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEMP_MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()

    def upload(self, name, text, encoding='utf-8'):
        response = self.client.post(reverse('upload'), {'csv_file': make_csv(name, text, encoding)})
        return response, UploadedFile.objects.order_by('-pk').first()


class UploadTests(AnalysisTestBase):
    def test_saves_csv_and_column_stats(self):
        response, uploaded = self.upload('data.csv', linear_csv())
        self.assertRedirects(response, reverse('dashboard', args=[uploaded.pk]))
        self.assertTrue(uploaded.csv_file)
        self.assertEqual(uploaded.row_count, 20)
        self.assertEqual(uploaded.column_count, 3)
        x = ColumnData.objects.get(file=uploaded, column_name='x')
        self.assertEqual(x.min_value, 0)
        self.assertEqual(x.max_value, 19)

    def test_reads_shift_jis(self):
        response, uploaded = self.upload('売上.csv', '店舗,売上\n東京,100\n大阪,200\n', encoding='cp932')
        self.assertRedirects(response, reverse('dashboard', args=[uploaded.pk]))
        self.assertTrue(ColumnData.objects.filter(file=uploaded, column_name='売上').exists())

    def test_rejects_non_csv(self):
        response = self.client.post(reverse('upload'), {'csv_file': make_csv('data.txt', 'a,b\n1,2')}, follow=True)
        self.assertContains(response, 'CSVファイルを選択してください')
        self.assertFalse(UploadedFile.objects.exists())

    def test_empty_csv_shows_error(self):
        response = self.client.post(reverse('upload'), {'csv_file': make_csv('empty.csv', '')}, follow=True)
        self.assertContains(response, 'CSVファイルが空です')
        self.assertFalse(UploadedFile.objects.exists())


class DashboardTests(AnalysisTestBase):
    def test_missing_file_returns_404(self):
        response = self.client.get(reverse('dashboard', args=[9999]))
        self.assertEqual(response.status_code, 404)

    def test_all_missing_column_is_sent_as_null(self):
        _, uploaded = self.upload('missing.csv', "a,b\n1,\n2,\n3,\n")
        response = self.client.get(reverse('dashboard', args=[uploaded.pk]))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        match = re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', html)
        chart_data = json.loads(match.group(1))
        column_b = next(item for item in chart_data if item['label'] == 'b')
        self.assertEqual(column_b['data'], [None, None, None, None])
        self.assertNotContains(response, 'nan')


class RegressionTests(AnalysisTestBase):
    def predict(self, uploaded, target, features, strategy='drop'):
        return self.client.post(reverse('ml_predict', args=[uploaded.pk]), {
            'target': target,
            'features': features,
            'missing_strategy': strategy,
        })

    def test_uses_data_of_selected_file(self):
        _, linear = self.upload('linear.csv', linear_csv())
        _, unrelated = self.upload('unrelated.csv', random_like_csv())

        response = self.predict(linear, 'y', ['x'])
        self.assertAlmostEqual(response.context['result']['r2'], 1.0)

        response = self.predict(unrelated, 'y', ['x'])
        self.assertLess(response.context['result']['r2'], 0.9)
        self.assertEqual(MLResult.objects.count(), 2)

    def test_rejects_target_in_features(self):
        _, uploaded = self.upload('linear.csv', linear_csv())
        response = self.predict(uploaded, 'y', ['x', 'y'])
        self.assertIsNone(response.context['result'])
        self.assertContains(response, '目的変数と同じ列は説明変数に選べません')

    def test_requires_features(self):
        _, uploaded = self.upload('linear.csv', linear_csv())
        response = self.predict(uploaded, 'y', [])
        self.assertIsNone(response.context['result'])
        self.assertContains(response, '説明変数を1つ以上選んでください')

    def test_mean_imputation_keeps_rows(self):
        lines = ['x,y']
        for i in range(12):
            x = '' if i % 2 == 0 else str(i)  # 半分の行で x が欠損
            lines.append(f'{x},{i}')
        _, uploaded = self.upload('half_missing.csv', '\n'.join(lines))

        response = self.predict(uploaded, 'y', ['x'], strategy='drop')
        self.assertIsNone(response.context['result'])
        self.assertContains(response, '分析に使える行が6行しかありません')

        response = self.predict(uploaded, 'y', ['x'], strategy='mean')
        result = response.context['result']
        self.assertIsNotNone(result)
        self.assertEqual(result['train_rows'] + result['test_rows'], 12)
        self.assertEqual(MLResult.objects.get().missing_strategy, 'mean')

    def test_old_record_without_csv_shows_message(self):
        uploaded = UploadedFile.objects.create(name='old.csv', row_count=1, column_count=1)
        response = self.predict(uploaded, 'y', ['x'])
        self.assertContains(response, 'もう一度アップロードしてください')


class CorrelationTests(AnalysisTestBase):
    def test_renders_heatmap(self):
        _, uploaded = self.upload('linear.csv', linear_csv())
        response = self.client.get(reverse('correlation', args=[uploaded.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['img'])

    def test_needs_two_numeric_columns(self):
        _, uploaded = self.upload('one.csv', 'name,value\na,1\nb,2\n')
        response = self.client.get(reverse('correlation', args=[uploaded.pk]))
        self.assertIsNone(response.context['img'])


class KaggleImportTests(AnalysisTestBase):
    @staticmethod
    def fake_download(files):
        """kaggle コマンドの代わりに、-p で指定されたフォルダへCSVを書き出す"""
        def run(args, **kwargs):
            download_dir = args[args.index('-p') + 1]
            for name, text in files.items():
                with open(os.path.join(download_dir, name), 'w', encoding='utf-8') as f:
                    f.write(text)
            return subprocess.CompletedProcess(args, 0)
        return run

    @mock.patch('analysis.views.subprocess.run')
    def test_invalid_dataset_id_is_rejected(self, run):
        response = self.client.post(reverse('kaggle_import'), {'dataset': '--help'}, follow=True)
        run.assert_not_called()
        self.assertContains(response, '「ユーザー名/データセット名」の形式')

    def test_registers_only_downloaded_csv(self):
        with mock.patch('analysis.views.subprocess.run', side_effect=self.fake_download({'first.csv': linear_csv()})):
            response = self.client.post(reverse('kaggle_import'), {'dataset': 'owner/first'})
        first = UploadedFile.objects.get()
        self.assertRedirects(response, reverse('dashboard', args=[first.pk]))

        with mock.patch('analysis.views.subprocess.run', side_effect=self.fake_download({'second.csv': linear_csv()})):
            self.client.post(reverse('kaggle_import'), {'dataset': 'owner/second'})
        # 前回取得した first.csv が再登録されていないこと
        self.assertEqual(sorted(UploadedFile.objects.values_list('name', flat=True)), ['first.csv', 'second.csv'])

    def test_multiple_csv_goes_to_history(self):
        files = {'train.csv': linear_csv(), 'test.csv': linear_csv()}
        with mock.patch('analysis.views.subprocess.run', side_effect=self.fake_download(files)):
            response = self.client.post(reverse('kaggle_import'), {'dataset': 'owner/two-files'})
        self.assertRedirects(response, reverse('upload'))
        self.assertEqual(UploadedFile.objects.count(), 2)

    def test_download_failure_shows_error(self):
        error = subprocess.CalledProcessError(1, ['kaggle'])
        with mock.patch('analysis.views.subprocess.run', side_effect=error):
            response = self.client.post(reverse('kaggle_import'), {'dataset': 'owner/missing'}, follow=True)
        self.assertContains(response, 'データセットの取得に失敗しました')
        self.assertFalse(UploadedFile.objects.exists())
