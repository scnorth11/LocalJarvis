"""Voice command workflow.

Full pipeline:  listen_and_record → transcribe → run_pipeline → speak → format_output.

``listen_and_record`` uses :class:`~voice.hybrid_listener.HybridListener` which
races wake-word detection (Vosk) against push-to-talk (pynput + pyaudio) in
background threads.  Both blocking calls are offloaded to a thread-pool
executor so they do not starve the asyncio event loop.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from workflows.engine import BaseWorkflow, WorkflowStep

logger = logging.getLogger(__name__)


class VoiceCommandWorkflow(BaseWorkflow):
    """Single voice-command turn: listen → transcribe → respond → speak."""

    name = "voice_command"

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        config:
            A ``VoiceConfig`` instance (or *None* to use defaults).  When
            provided the workflow reads ``piper_voice``, ``vosk_model_path``,
            ``wake_phrase``, ``whisper_model``, ``voice_mode``, and
            ``ptt_key`` from it.
        """
        self._config = config

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get(self, attr: str, default: Any) -> Any:
        if self._config is not None and hasattr(self._config, attr):
            val = getattr(self._config, attr)
            if val:
                return val
        return default

    # ------------------------------------------------------------------
    # Step list
    # ------------------------------------------------------------------

    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                name="listen_and_record",
                description="Race wake-word and push-to-talk; return audio path",
                action=self._listen_and_record,
            ),
            WorkflowStep(
                name="transcribe",
                description="Transcribe recorded audio to text via Whisper",
                action=self._transcribe,
            ),
            WorkflowStep(
                name="run_pipeline",
                description="Route transcribed text through the agent pipeline",
                action=self._run_pipeline,
            ),
            WorkflowStep(
                name="speak",
                description="Synthesise and play the response via Piper TTS",
                action=self._speak,
                required=False,
            ),
            WorkflowStep(
                name="format_output",
                description="Store final output in context",
                action=self._format_output,
            ),
        ]

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _listen_and_record(self, context: Dict[str, Any]) -> str:
        """Block on wake-word / PTT in a thread; store audio_path in context."""
        from voice.hybrid_listener import HybridListener, VoiceMode

        wake_phrase = self._get("wake_phrase", "jarvis")
        vosk_model_path = self._get("vosk_model_path", "vosk-model-small-en-us")
        ptt_key = self._get("ptt_key", "f12")
        voice_mode_str = self._get("voice_mode", "hybrid")

        try:
            mode = VoiceMode(voice_mode_str)
        except ValueError:
            logger.warning(
                "VoiceCommandWorkflow: unknown voice_mode %r — defaulting to hybrid",
                voice_mode_str,
            )
            mode = VoiceMode.HYBRID

        listener = HybridListener(
            wake_phrase=wake_phrase,
            vosk_model_path=vosk_model_path,
            ptt_key=ptt_key,
        )

        loop = asyncio.get_running_loop()
        trigger_type, audio_path = await loop.run_in_executor(
            None, lambda: listener.listen(mode)
        )

        context["audio_path"] = audio_path
        context["trigger_type"] = trigger_type
        logger.info(
            "VoiceCommandWorkflow: activated via %s — audio at %s",
            trigger_type,
            audio_path or "(none)",
        )
        return audio_path

    async def _transcribe(self, context: Dict[str, Any]) -> str:
        from voice.whisper_stt import WhisperSTT

        audio_path = context.get("audio_path", "")
        if not audio_path:
            raise RuntimeError("listen_and_record produced no audio path — cannot transcribe.")

        whisper_model = self._get("whisper_model", "base")
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: WhisperSTT(model_name=whisper_model).transcribe(audio_path),
        )
        logger.info("VoiceCommandWorkflow: transcribed: %r", text)
        return text

    @staticmethod
    async def _run_pipeline(context: Dict[str, Any]) -> str:
        text = context.get("transcribe", "").strip()
        if not text or text.startswith("["):
            raise RuntimeError(f"Transcription unavailable or failed: {text!r}")
        pipeline = context["_pipeline"]
        return await pipeline(text)

    async def _speak(self, context: Dict[str, Any]) -> str:
        from voice.piper_tts import PiperTTS

        response = context.get("run_pipeline", "").strip()
        if response:
            piper_voice = self._get("piper_voice", "en_GB-alan-medium")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: PiperTTS(voice=piper_voice).speak(response),
                )
            except Exception as exc:
                logger.warning("VoiceCommandWorkflow: TTS failed: %s", exc)
        return response

    @staticmethod
    async def _format_output(context: Dict[str, Any]) -> str:
        output = context.get("run_pipeline", "").strip()
        if not output:
            output = "No response generated."
        context["output"] = output
        return output

