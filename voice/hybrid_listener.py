"""Hybrid voice listener.

Supports three voice trigger modes:

* ``WAKE_WORD`` — blocks until Vosk detects the wake phrase, then records a
  fixed-duration clip via :meth:`WhisperSTT.record`.
* ``PTT`` — delegates entirely to :class:`PushToTalk`; records while the
  configured key is held.
* ``HYBRID`` — races both detectors in background threads; whichever fires
  first wins and the other thread is abandoned (daemon).

Returns a ``(trigger_type, audio_path)`` tuple so callers know how the
session was activated.
"""

import logging
import threading
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class VoiceMode(str, Enum):
    """Voice activation mode."""
    WAKE_WORD = "wake_word"
    PTT = "ptt"
    HYBRID = "hybrid"


class HybridListener:
    """Race wake-word detection against push-to-talk.

    Parameters
    ----------
    wake_phrase:
        Phrase Vosk listens for in wake-word / hybrid mode.
    vosk_model_path:
        Path to the Vosk model directory.
    ptt_key:
        Keyboard key name for push-to-talk mode.
    record_seconds:
        Duration to record (seconds) after the wake-word fires.
    sample_rate:
        Audio sample rate shared by all recording paths (Hz).
    """

    def __init__(
        self,
        wake_phrase: str = "jarvis",
        vosk_model_path: str = "vosk-model-small-en-us",
        ptt_key: str = "f12",
        record_seconds: float = 5.0,
        sample_rate: int = 16_000,
    ) -> None:
        self._wake_phrase = wake_phrase
        self._vosk_model_path = vosk_model_path
        self._ptt_key = ptt_key
        self._record_seconds = record_seconds
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, mode: VoiceMode = VoiceMode.HYBRID) -> Tuple[str, str]:
        """Block until a voice trigger fires.

        Parameters
        ----------
        mode:
            Which activation method(s) to enable.

        Returns
        -------
        tuple[str, str]
            ``(trigger_type, audio_path)`` where *trigger_type* is one of
            ``"wake_word"``, ``"ptt"``, or ``"hybrid"`` and *audio_path* is
            the path to a WAV file (or ``""`` on failure).
        """
        if mode == VoiceMode.WAKE_WORD:
            return self._listen_wake_word()
        if mode == VoiceMode.PTT:
            return self._listen_ptt()
        return self._listen_hybrid()

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _listen_wake_word(self) -> Tuple[str, str]:
        from voice.vosk_wakeword import VoskWakeWord
        from voice.whisper_stt import WhisperSTT

        detector = VoskWakeWord(model_path=self._vosk_model_path)
        detector.listen(phrase=self._wake_phrase)
        logger.info("HybridListener: wake word detected — recording …")
        path = WhisperSTT().record(
            duration_seconds=self._record_seconds,
            sample_rate=self._sample_rate,
        )
        return ("wake_word", path)

    def _listen_ptt(self) -> Tuple[str, str]:
        from voice.push_to_talk import PushToTalk

        ptt = PushToTalk(
            key=self._ptt_key,
            max_duration=30.0,
            sample_rate=self._sample_rate,
        )
        path = ptt.record_blocking()
        return ("ptt", path)

    def _listen_hybrid(self) -> Tuple[str, str]:
        """Race wake-word and PTT — first to produce an audio path wins."""
        # result_holder is written at most once; done_event prevents double-writes.
        result_holder: list = []
        done_event = threading.Event()
        empty_count = [0]
        empty_lock = threading.Lock()

        def _on_empty() -> None:
            """Called by a thread that produced no path (deps missing)."""
            with empty_lock:
                empty_count[0] += 1
                if empty_count[0] >= 2:
                    done_event.set()

        def _run_wake_word() -> None:
            trigger, path = self._listen_wake_word()
            if path and not done_event.is_set():
                result_holder.append((trigger, path))
                done_event.set()
            elif not path:
                _on_empty()

        def _run_ptt() -> None:
            trigger, path = self._listen_ptt()
            if path and not done_event.is_set():
                result_holder.append((trigger, path))
                done_event.set()
            elif not path:
                _on_empty()

        t_ww = threading.Thread(target=_run_wake_word, daemon=True, name="wakeword")
        t_ptt = threading.Thread(target=_run_ptt, daemon=True, name="ptt")
        t_ww.start()
        t_ptt.start()

        done_event.wait()

        if result_holder:
            return result_holder[0]
        # Both threads returned empty — dependencies not installed.
        return ("hybrid", "")
