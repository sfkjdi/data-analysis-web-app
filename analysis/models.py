from django.db import models

class UploadedFile(models.Model):
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    row_count = models.IntegerField(default=0)
    column_count = models.IntegerField(default=0)

    def __str__(self):
        return self.name


class ColumnData(models.Model):
    file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='columns')
    column_name = models.CharField(max_length=255)
    mean = models.FloatField(null=True, blank=True)
    median = models.FloatField(null=True, blank=True)
    std = models.FloatField(null=True, blank=True)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    null_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.file.name} - {self.column_name}"
class MLResult(models.Model):
    file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='ml_results')
    target_column = models.CharField(max_length=255)
    feature_columns = models.TextField()
    r2_score = models.FloatField()
    mse = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file.name} - {self.target_column}"