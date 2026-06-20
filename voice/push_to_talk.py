"""Push-to-talk voice input.

Records audio from the default microphone while a configurable key is held
down.  Uses ``pynput`` for keyboard event detection and ``pyaudio`` for
audio capture.  If either library is not installed the module still imports
cleanly and ``record_blocking()`` returns an empty string with a warning so
the rest of the pipeline can be exercised without hardware.
"""

import logging
import tempfile
import threading
import wave
from typing import Optional

logger = logging.getLogger(__name__)

_PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard as _keyboard  # type: ignore
    _PYNPUT_AVAILABLE = True
except ImportError:
    logger.debug("pynput not installed — PushToTalk will return empty paths")


class PushToTalk:
    """Record audio from the microphone while a key is held down.

    Parameters
    ----------
    key:
        Keyboard key name to use as the push-to-talk trigger.  Accepts
        pynput ``Key`` attribute names for special keys (e.g. ``"f12"``,
        ``"shift"``, ``"ctrl"``) or a single printable character.
        Defaults to ``"f12"``.
    max_duration:
        Maximum recording duration in seconds.  Recording stops sooner if
        the key is released before this limit.  Defaults to 30 seconds.
    sample_rate:
        PCM sample rate for audio capture (Hz).  Defaults to 16 000 Hz.
    """

    def __init__(
        self,
        key: str = "f12",
        max_duration: float = 30.0,
        sample_rate: int = 16_000,
    ) -> None:
        self._key_name = key
        self._max_duration = max_duration
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_blocking(self) -> str:
        """Wait for the PTT key press, record while held, return WAV path.

        Blocks until the configured key is pressed, then records audio
        until the key is released (or *max_duration* seconds pass).

        Returns
        -------
        str
            Absolute path to a temporary WAV file, or ``""`` if the
            required libraries are not installed or an error occurs.
        """
        if not _PYNPUT_AVAILABLE:
            logger.warning("PushToTalk: pynput not installed — returning empty path")
            return ""
        try:
            import pyaudio as _pa  # type: ignore
        except ImportError:
            logger.warning("PushToTalk: pyaudio not installed — returning empty path")
            return ""

        ptt_key = self._resolve_key()
        pressed_event = threading.Event()
        released_event = threading.Event()

        def on_press(k: object) -> None:
            if k == ptt_key and not pressed_event.is_set():
                logger.debug("PushToTalk: key pressed — starting capture")
                pressed_event.set()

        def on_release(k: object) -> Optional[bool]:
            if k == ptt_key:
                logger.debug("PushToTalk: key released — stopping capture")
                released_event.set()
                return False  # stop the listener
            return None

        logger.info("PushToTalk: hold %s to speak …", self._key_name.upper())
        with _keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            pressed_event.wait()
            wav_path = self._capture(_pa, released_event)
            listener.join(timeout=0)

        return wav_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self) -> object:
        """Resolve the key name string to a pynput Key or KeyCode object."""
        # Try special keys first (Key.f12, Key.shift, etc.)
        key_attr = getattr(_keyboard.Key, self._key_name.lower(), None)
        if key_attr is not None:
            return key_attr
        # Fall back to a single printable character.
        return _keyboard.KeyCode.from_char(self._key_name)

    def _capture(self, _pa: object, stop_event: threading.Event) -> str:
        """Capture PCM frames until *stop_event* fires or max_duration elapses."""
        chunk = 1024
        fmt = _pa.paInt16  # type: ignore[attr-defined]
        channels = 1

        pa = _pa.PyAudio()  # type: ignore[attr-defined]
        sample_width = pa.get_sample_size(fmt)
        frames = []

        try:
            stream = pa.open(
                format=fmt,
                channels=channels,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            total_chunks = int(self._sample_rate / chunk * self._max_duration)
            try:
                for _ in range(total_chunks):
                    if stop_event.is_set():
                        break
                    frames.append(stream.read(chunk, exception_on_overflow=False))
            finally:
                stream.stop_stream()
                stream.close()
        except Exception as exc:
            logger.error("PushToTalk: capture error: %s", exc)
            return ""
        finally:
            pa.terminate()

        if not frames:
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(self._sample_rate)
                wf.writeframes(b"".join(frames))
        except Exception as exc:
            logger.error("PushToTalk: WAV write error: %s", exc)
            return ""

        logger.debug("PushToTalk: recorded to %s", tmp.name)
        return tmp.name
