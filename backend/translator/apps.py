from django.apps import AppConfig


class TranslatorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "translator"

    def ready(self):
        """
        Load ML models in a background thread so they don't block
        Django/ASGI startup. TF + MediaPipe take 30-60s to import.
        The predictor singleton is thread-safe — the first WS request
        that arrives before loading completes will find models=None
        and return a graceful 'not ready' response.
        """
        import threading
        import logging
        logger = logging.getLogger(__name__)

        def _load():
            try:
                from translator.ml.predictor import predictor  # noqa: F401
                logger.info("[APP] ML models loaded successfully.")
            except Exception as e:
                logger.warning(f"[APP] Could not load ML models: {e}")

        t = threading.Thread(target=_load, daemon=True, name="ml-model-loader")
        t.start()

