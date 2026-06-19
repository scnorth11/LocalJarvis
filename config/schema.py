from dataclasses import dataclass
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VoiceConfig":
        if not isinstance(d, dict):
            raise ValueError("voice must be a mapping")
        default_voice = d.get("default_voice")
        tts_engine = d.get("tts_engine")
        stt_engine = d.get("stt_engine")
        if not all(isinstance(x, str) for x in (default_voice, tts_engine, stt_engine)):
            raise ValueError("voice entries must be strings")
        return cls(default_voice=default_voice, tts_engine=tts_engine, stt_engine=stt_engine)


@dataclass(frozen=True)
class AppConfig:
    models: ModelsConfig
    paths: PathsConfig
    security: SecurityConfig
    voice: VoiceConfig
    timeouts: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        if not isinstance(d, dict):
            raise ValueError("config must be a mapping")
        models = ModelsConfig.from_dict(d.get("models", {}))
        paths = PathsConfig.from_dict(d.get("paths", {}))
        security = SecurityConfig.from_dict(d.get("security", {}))
        voice = VoiceConfig.from_dict(d.get("voice", {}))
        timeouts = d.get("timeouts", {})
        if not isinstance(timeouts, dict):
            raise ValueError("timeouts must be a mapping")
        return cls(models=models, paths=paths, security=security, voice=voice, timeouts=dict(timeouts))
