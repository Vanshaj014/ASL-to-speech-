from django.urls import path
from . import views

urlpatterns = [
    path("vocabulary/", views.vocabulary, name="vocabulary"),
    path("status/", views.model_status, name="model-status"),
    path("log-prediction/", views.log_prediction, name="log-prediction"),
    path("sessions/", views.SessionListCreateView.as_view(), name="sessions"),
    path("sessions/<int:pk>/", views.SessionDetailView.as_view(), name="session-detail"),
    path("tts/", views.tts_trigger, name="tts-trigger"),
]
