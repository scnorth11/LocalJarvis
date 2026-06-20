from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ModelsConfig:
    default: str
    tiers: Dict[str, str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelsConfig":
        if not isinstance(d, dict):
            raise ValueError("models must be a mapping")
        default = d.get("default")
        tiers = d.get("tiers")
        if not isinstance(default, str):
            raise ValueError("models.default must be a string")
        if not isinstance(tiers, dict):
            raise ValueError("models.tiers must be a mapping of str->str")
        for k, v in tiers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("models.tiers keys and values must be strings")
        return cls(default=default, tiers=dict(tiers))


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    cache_dir: str
    log_dir: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PathsConfig":
        if not isinstance(d, dict):
            raise ValueError("paths must be a mapping")
        data_dir = d.get("data_dir")
        cache_dir = d.get("cache_dir")
        log_dir = d.get("log_dir")
        if not all(isinstance(x, str) for x in (data_dir, cache_dir, log_dir)):
            raise ValueError("paths entries must be strings")
        return cls(data_dir=data_dir, cache_dir=cache_dir, log_dir=log_dir)


@dataclass(frozen=True)
class SecurityConfig:
    allowed_tools: Dict[str, List[str]]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SecurityConfig":
        if not isinstance(d, dict):
            raise ValueError("security must be a mapping")
        allowed = d.get("allowed_tools")
        if not isinstance(allowed, dict):
            raise ValueError("security.allowed_tools must be a mapping")
        cleaned: Dict[str, List[str]] = {}
        for agent, tools in allowed.items():
            if not isinstance(agent, str) or not isinstance(tools, list):
                raise ValueError("security.allowed_tools must map agent names to lists of tool names")
            for t in tools:
                if not isinstance(t, str):
                    raise ValueError("tool names must be strings")
            cleaned[agent] = list(tools)
        return cls(allowed_tools=cleaned)


@dataclass(frozen=True)
class VoiceConfig:
    default_voice: str
    tts_engine: str
    stt_engine: str
    piper_voice: str = "en_GB-alan-medium"
    vosk_model_path: str = "vosk-model-small-en-us"
    wake_phrase: str = "jarvis"
    whisper_model: str = "base"
    voice_mode: str = "hybrid"
    ptt_key: str = "f12"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VoiceConfig":
        if not isinstance(d, dict):
            raise ValueError("voice must be a mapping")
        default_voice = d.get("default_voice")
        tts_engine = d.get("tts_engine")
        stt_engine = d.get("stt_engine")
        if not all(isinstance(x, str) for x in (default_voice, tts_engine, stt_engine)):
            raise ValueError("voice entries must be strings")
        return cls(
            default_voice=default_voice,
            tts_engine=tts_engine,
            stt_engine=stt_engine,
            piper_voice=str(d.get("piper_voice", "en_GB-alan-medium")),
            vosk_model_path=str(d.get("vosk_model_path", "vosk-model-small-en-us")),
            wake_phrase=str(d.get("wake_phrase", "jarvis")),
            whisper_model=str(d.get("whisper_model", "base")),
            voice_mode=str(d.get("voice_mode", "hybrid")),
            ptt_key=str(d.get("ptt_key", "f12")),
        )


# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model_names: Dict[str, str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OllamaConfig":
        if not isinstance(d, dict):
            return cls(base_url="http://localhost:11434", model_names={})
        model_names = d.get("model_names", {})
        if not isinstance(model_names, dict):
            raise ValueError("ollama.model_names must be a mapping")
        return cls(
            base_url=str(d.get("base_url", "http://localhost:11434")),
            model_names={str(k): str(v) for k, v in model_names.items()},
        )


# ---------------------------------------------------------------------------
# Tool-specific config blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileConfig:
    allowed_paths: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileConfig":
        if not isinstance(d, dict):
            return cls(allowed_paths=[])
        paths = d.get("allowed_paths", [])
        if not isinstance(paths, list):
            raise ValueError("tools.file.allowed_paths must be a list")
        return cls(allowed_paths=[str(p) for p in paths])


@dataclass(frozen=True)
class CalendarConfig:
    db_path: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalendarConfig":
        if not isinstance(d, dict):
            return cls(db_path="data/calendar.db")
        return cls(db_path=str(d.get("db_path", "data/calendar.db")))


@dataclass(frozen=True)
class ResearchConfig:
    max_results: int
    source_guidance: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchConfig":
        if not isinstance(d, dict):
            return cls(max_results=10, source_guidance="")
        return cls(
            max_results=int(d.get("max_results", 10)),
            source_guidance=str(d.get("source_guidance", "")),
        )


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    client_secret: str
    refresh_token: str
    redirect_uri: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpotifyConfig":
        if not isinstance(d, dict):
            return cls(client_id="", client_secret="", refresh_token="", redirect_uri="http://localhost:8888/callback")
        return cls(
            client_id=str(d.get("client_id", "")),
            client_secret=str(d.get("client_secret", "")),
            refresh_token=str(d.get("refresh_token", "")),
            redirect_uri=str(d.get("redirect_uri", "http://localhost:8888/callback")),
        )


@dataclass(frozen=True)
class ToolsConfig:
    file: FileConfig
    calendar: CalendarConfig
    research: ResearchConfig
    spotify: SpotifyConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolsConfig":
        if not isinstance(d, dict):
            d = {}
        return cls(
            file=FileConfig.from_dict(d.get("file", {})),
            calendar=CalendarConfig.from_dict(d.get("calendar", {})),
            research=ResearchConfig.from_dict(d.get("research", {})),
            spotify=SpotifyConfig.from_dict(d.get("spotify", {})),
        )


# ---------------------------------------------------------------------------
# Top-level AppConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    models: ModelsConfig
    paths: PathsConfig
    security: SecurityConfig
    voice: VoiceConfig
    timeouts: Dict[str, Any]
    tools: ToolsConfig
    ollama: OllamaConfig = field(default_factory=lambda: OllamaConfig.from_dict({}))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        if not isinstance(d, dict):
            raise ValueError("config must be a mapping")
        models = ModelsConfig.from_dict(d.get("models", {}))
        paths = PathsConfig.from_dict(d.get("paths", {}))
        security = SecurityConfig.from_dict(d.get("security", {}))
        voice = VoiceConfig.from_dict(d.get("voice", {}))
        tools = ToolsConfig.from_dict(d.get("tools", {}))
        ollama = OllamaConfig.from_dict(d.get("ollama", {}))
        timeouts = d.get("timeouts", {})
        if not isinstance(timeouts, dict):
            raise ValueError("timeouts must be a mapping")
        return cls(
            models=models,
            paths=paths,
            security=security,
            voice=voice,
            timeouts=dict(timeouts),
            tools=tools,
            ollama=ollama,
        )
