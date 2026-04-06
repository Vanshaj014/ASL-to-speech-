from django.db import models
from django.utils import timezone


class TranslationSession(models.Model):
    """Represents one translation session (browser tab/connection)."""
    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)
    total_predictions = models.IntegerField(default=0)
    final_sentence = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"Session {self.pk} @ {self.started_at.strftime('%Y-%m-%d %H:%M')}"

    @property
    def duration_seconds(self):
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class PredictionLog(models.Model):
    """Log of individual sign predictions within a session."""
    MODE_CHOICES = [("static", "Static (A-Z)"), ("dynamic", "Dynamic (Words)")]

    session = models.ForeignKey(
        TranslationSession, on_delete=models.CASCADE,
        related_name="predictions", null=True, blank=True
    )
    sign = models.CharField(max_length=50)
    confidence = models.FloatField()
    mode = models.CharField(max_length=10, choices=MODE_CHOICES, default="static")
    predicted_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-predicted_at"]

    def __str__(self):
        return f"'{self.sign}' ({self.confidence:.2%}) [{self.mode}]"
