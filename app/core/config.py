from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "bookly-ai"
    debug: bool = False

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "nova"

    # Same secret as the Go backend: we validate its HS256 access tokens locally.
    jwt_secret: str = ""

    bookly_api_url: str = "http://127.0.0.1:8080"
    redis_url: str = "redis://127.0.0.1:6379/1"

    conversation_ttl_seconds: int = 1800
    request_timeout_seconds: float = 20.0
    max_tool_rounds: int = 6
