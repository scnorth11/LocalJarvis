from dataclasses import dataclass
from typing import List

from .schema import AppConfig, ModelsConfig, PathsConfig, VoiceConfig


@dataclass(frozen=True)
class RestrictedConfig:
    models: ModelsConfig
    paths: PathsConfig
    voice: VoiceConfig
    allowed_tools: List[str]


class Config:
    def __init__(self, app_config: AppConfig):
        self._app = app_config
        self.models = app_config.models
        self.paths = app_config.paths
        self.voice = app_config.voice
        self.security = app_config.security
        self.timeouts = app_config.timeouts

    def for_agent(self, agent_name: str) -> RestrictedConfig:
        allowed = self._app.security.allowed_tools.get(agent_name, [])
        return RestrictedConfig(models=self.models, paths=self.paths, voice=self.voice, allowed_tools=list(allowed))
