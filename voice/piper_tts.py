"""Piper TTS stub.

Delegates synthesis to the ``piper`` command-line binary via a subprocess
call.  If piper is not installed the module still imports cleanly and
``speak()`` no-ops with a warning so the rest of the pipeline can run
without audio hardware.
"""

import logging
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class PiperTTS:
    """Text-to-speech via the Piper binary.

    Parameters
    ----------
    voice:
        Piper voice model name (e.g. ``"en_US-lessac-medium"``).  When
        *None* the binary's default voice is used.
    piper_bin:
        Path to the ``piper`` executable.  Defaults to ``"piper"`` (resolved
        from ``PATH``).
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        piper_bin: str = "piper",
    ) -> None:
        self._voice = voice
        self._piper_bin = piper_bin

    def _is_available(self) -> bool:
        return shutil.which(self._piper_bin) is not None

    def speak(self, text: str) -> None:
        """Synthesise *text* and play it through the default audio device.

        Uses ``piper`` for synthesis and ``aplay`` (ALSA) for playback.  If
        either binary is absent the call is silently skipped with a warning.

        Parameters
        ----------
        text:
            The text to synthesise and play.
        """
        if not text.strip():
            return

        if not self._is_available():
            logger.warning(
                "PiperTTS: '%s' binary not found — skipping speech synthesis",
                self._piper_bin,
            )
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as wav_file:
            wav_path = wav_file.name

        cmd_piper = [self._piper_bin, "--output_file", wav_path]
        if self._voice:
            cmd_piper += ["--model", self._voice]

        try:
            subprocess.run(
                cmd_piper,
                input=text.encode(),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("PiperTTS: synthesis failed: %s", exc.stderr.decode())
            return
        except Exception as exc:
            logger.error("PiperTTS: unexpected error during synthesis: %s", exc)
            return

        # Playback — try aplay (Linux ALSA), fall back silently.
        if shutil.which("aplay"):
            try:
                subprocess.run(["aplay", "-q", wav_path], check=True, capture_output=True)
            except Exception as exc:
                logger.warning("PiperTTS: aplay failed: %s", exc)
        else:
            logger.warning("PiperTTS: aplay not found — WAV written to %s but not played", wav_path)
