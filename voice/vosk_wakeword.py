"""Vosk wake-word detection stub.

Listens on the default microphone for a configurable wake phrase using the
``vosk`` speech-recognition library and ``pyaudio`` for audio capture.  If
either library is not installed the module still imports cleanly and
``listen()`` returns ``True`` immediately with a warning so the rest of the
pipeline can be exercised without hardware.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_VOSK_AVAILABLE = False
_PYAUDIO_AVAILABLE = False
try:
    import vosk as _vosk  # type: ignore
    _VOSK_AVAILABLE = True
except ImportError:
    logger.debug("vosk library not installed — VoskWakeWord will skip detection")

try:
    import pyaudio as _pyaudio  # type: ignore
    _PYAUDIO_AVAILABLE = True
except ImportError:
    logger.debug("pyaudio library not installed — VoskWakeWord will skip detection")

_SAMPLE_RATE = 16000
_CHUNK = 4000


class VoskWakeWord:
    """Wake-word detector using Vosk offline speech recognition.

    Parameters
    ----------
    model_path:
        Path to the Vosk model directory.  When *None* Vosk's default
        small English model is expected at ``vosk-model-small-en-us``.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or "vosk-model-small-en-us"
        self._model: Optional[object] = None

    def _load_model(self) -> Optional[object]:
        if self._model is not None:
            return self._model
        try:
            self._model = _vosk.Model(self._model_path)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error("VoskWakeWord: failed to load model from %s: %s", self._model_path, exc)
            return None
        return self._model

    def listen(self, phrase: str = "hey jarvis") -> bool:
        """Block until *phrase* is detected in the microphone stream.

        Parameters
        ----------
        phrase:
            The wake phrase to listen for (case-insensitive).

        Returns
        -------
        bool
            ``True`` when the phrase is detected.  Returns ``True``
            immediately (with a warning) if the required libraries are not
            installed.
        """
        if not _VOSK_AVAILABLE or not _PYAUDIO_AVAILABLE:
            logger.warning(
                "VoskWakeWord: vosk/pyaudio not installed — returning True immediately"
            )
            return True

        model = self._load_model()
        if model is None:
            logger.warning("VoskWakeWord: model not available — returning True immediately")
            return True

        recogniser = _vosk.KaldiRecognizer(model, _SAMPLE_RATE)  # type: ignore[attr-defined]
        phrase_lower = phrase.lower()

        pa = _pyaudio.PyAudio()  # type: ignore[attr-defined]
        stream = pa.open(
            format=_pyaudio.paInt16,  # type: ignore[attr-defined]
            channels=1,
            rate=_SAMPLE_RATE,
            input=True,
            frames_per_buffer=_CHUNK,
        )
        logger.info("VoskWakeWord: listening for '%s' …", phrase)
        try:
            while True:
                data = stream.read(_CHUNK, exception_on_overflow=False)
                if recogniser.AcceptWaveform(data):
                    result = json.loads(recogniser.Result())
                    text = result.get("text", "").lower()
                    if phrase_lower in text:
                        logger.info("VoskWakeWord: detected wake phrase!")
                        return True
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
