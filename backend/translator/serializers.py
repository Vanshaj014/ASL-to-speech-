from rest_framework import serializers
from .models import TranslationSession, PredictionLog


class PredictionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionLog
        fields = ["id", "sign", "confidence", "mode", "predicted_at"]


class TranslationSessionSerializer(serializers.ModelSerializer):
    predictions = PredictionLogSerializer(many=True, read_only=True)
    duration_seconds = serializers.ReadOnlyField()

    class Meta:
        model = TranslationSession
        fields = ["id", "started_at", "ended_at", "total_predictions",
                  "final_sentence", "duration_seconds", "predictions"]
