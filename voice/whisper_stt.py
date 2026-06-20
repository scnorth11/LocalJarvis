"""Whisper speech-to-text stub.

Uses ``openai-whisper`` when available.  If the library is not installed the
module still imports cleanly and transcription returns a placeholder string so
the rest of the pipeline can be exercised without audio hardware.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_WHISPER_AVAILABLE = False
try:
    import whisper as _whisper  # type: ignore
    _WHISPER_AVAILABLE = True
except ImportError:
    logger.debug("whisper library not installed — WhisperSTT will return placeholders")


class WhisperSTT:
    """Thin wrapper around ``openai-whisper`` for audio transcription.

    Parameters
    ----------
    model_name:
        Whisper model size to load (e.g. ``"tiny"``, ``"base"``, ``"small"``).
        Defaults to ``"tiny"`` for speed on local hardware.
    """

    def __init__(self, model_name: str = "tiny") -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return
        if not _WHISPER_AVAILABLE:
            return
        logger.info("WhisperSTT: loading model '%s'", self._model_name)
        self._model = _whisper.load_model(self._model_name)  # type: ignore[attr-defined]

    def transcribe(self, audio_path: str) -> str:
        """Transcribe *audio_path* to text.

        Parameters
        ----------
        audio_path:
            Path to the audio file (WAV, MP3, etc.).

        Returns
        -------
        str
            Transcribed text, or a placeholder string if Whisper is not
            available or the file does not exist.
        """
        if not _WHISPER_AVAILABLE:
            logger.warning(
                "WhisperSTT: whisper library not installed — returning placeholder"
            )
            return "[whisper not available]"

        if not os.path.isfile(audio_path):
            logger.warning("WhisperSTT: audio file not found: %s", audio_path)
            return "[audio file not found]"

        self._load_model()
        try:
            result = self._model.transcribe(audio_path)  # type: ignore[union-attr]
            text = result.get("text", "").strip()
            logger.debug("WhisperSTT: transcribed %d chars", len(text))
            return text
        except Exception as exc:
            logger.error("WhisperSTT: transcription failed: %s", exc)
            return "[transcription failed]"

    def record(self, duration_seconds: float = 5.0, sample_rate: int = 16_000) -> str:
        """Record audio from the default microphone and return the path to a WAV file.

        Parameters
        ----------
        duration_seconds:
            How long to record in seconds.
        sample_rate:
            PCM sample rate (Hz).

        Returns
        -------
        str
            Absolute path to a temporary WAV file, or ``""`` if pyaudio is
            not available.
        """
        try:
            import pyaudio as _pa  # type: ignore
        except ImportError:
            logger.warning("WhisperSTT.record: pyaudio not installed — cannot record audio")
            return ""

        import tempfile
        import wave

        chunk = 1024
        fmt = _pa.paInt16
        channels = 1

        pa = _pa.PyAudio()
        sample_width = pa.get_sample_size(fmt)
        frames = []

        logger.info("WhisperSTT: recording %.1fs at %dHz …", duration_seconds, sample_rate)
        try:
            stream = pa.open(
                format=fmt,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            try:
                num_chunks = int(sample_rate / chunk * duration_seconds)
                for _ in range(num_chunks):
                    frames.append(stream.read(chunk, exception_on_overflow=False))
            finally:
                stream.stop_stream()
                stream.close()
        except Exception as exc:
            logger.error("WhisperSTT.record: capture error: %s", exc)
            return ""
        finally:
            pa.terminate()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))
        except Exception as exc:
            logger.error("WhisperSTT.record: WAV write error: %s", exc)
            return ""

        logger.debug("WhisperSTT: audio saved to %s", tmp.name)
        return tmp.name
