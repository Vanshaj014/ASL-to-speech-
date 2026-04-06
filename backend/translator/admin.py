from django.contrib import admin
from .models import TranslationSession, PredictionLog


@admin.register(TranslationSession)
class TranslationSessionAdmin(admin.ModelAdmin):
    list_display = ["id", "started_at", "ended_at", "total_predictions", "final_sentence"]
    list_filter = ["started_at"]
    search_fields = ["final_sentence"]


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ["id", "sign", "confidence", "mode", "predicted_at", "session"]
    list_filter = ["mode", "predicted_at"]
    search_fields = ["sign"]
