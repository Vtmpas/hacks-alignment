from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TELEGRAM_BOT_API_TOKEN: SecretStr

    class Config:
        """Pydantic Settings config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
