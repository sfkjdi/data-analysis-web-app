from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('dashboard/<int:pk>/', views.dashboard, name='dashboard'),
    path('kaggle/', views.kaggle_import, name='kaggle_import'),
    path('ml/<int:pk>/', views.ml_predict, name='ml_predict'),
    path('correlation/<int:pk>/', views.correlation, name='correlation'),
]
