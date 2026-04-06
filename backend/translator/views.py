from django.http import JsonResponse
from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.http import require_http_methods

from .models import TranslationSession, PredictionLog
from .serializers import TranslationSessionSerializer, PredictionLogSerializer
from .ml.predictor import predictor


@api_view(["GET"])
def vocabulary(request):
    """Return all supported signs grouped by model type."""
    vocab = predictor.get_vocabulary()
    return Response({
        "static_signs": vocab["static"],
        "dynamic_signs": vocab["dynamic"],
        "total": len(vocab["static"]) + len(vocab["dynamic"])
    })


@api_view(["GET"])
def model_status(request):
    """Return model health status."""
    return Response({
        "static_model": predictor.static_model is not None,
        "dynamic_model": predictor.dynamic_model is not None,
        "ready": predictor.is_ready(),
        "static_classes": len(predictor.static_label_map) if predictor.static_label_map else 0,
        "dynamic_classes": len(predictor.dynamic_label_map) if predictor.dynamic_label_map else 0,
    })


@api_view(["POST"])
def log_prediction(request):
    """Log a confirmed prediction to the database."""
    serializer = PredictionLogSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SessionListCreateView(generics.ListCreateAPIView):
    queryset = TranslationSession.objects.all()[:50]
    serializer_class = TranslationSessionSerializer


class SessionDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = TranslationSession.objects.all()
    serializer_class = TranslationSessionSerializer


@api_view(["POST"])
def tts_trigger(request):
    """Text-to-speech endpoint: returns audio as base64 or triggers local TTS."""
    from .ml.tts_engine import speak_text
    text = request.data.get("text", "")
    if not text:
        return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)

    success = speak_text(text)
    return Response({"success": success, "text": text})
