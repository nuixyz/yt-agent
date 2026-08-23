from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model tag used for summarization.",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )

    camoufox_profile_dir: Path = Field(
        default=Path("./.camoufox-profile"),
        description="Persistent browser profile dir (cookies, localStorage, session).",
    )
    camoufox_headless: bool = Field(
        default=True,
        description="Run Camoufox headless after the first interactive login.",
    )

    default_playlist_url: str | None = Field(
        default=None,
        description="Playlist used by the background scheduler when no URL is supplied.",
    )

    max_videos_per_run: int = Field(
        default=5,
        description="Cap on how many playlist videos are processed in a single run.",
    )

    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory where generated Markdown digests are written.",
    )

    scheduler_enabled: bool = Field(
        default=False,
        description="If true, runs a daily background job in addition to /trigger.",
    )
    scheduler_cron_hour: int = Field(
        default=7,
        description="Hour (24h, local time) the scheduled job runs at.",
    )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camoufox_profile_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
