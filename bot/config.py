from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class uses Pydantic to parse and validate environment variables,
    and it specifically handles secrets like API tokens.

    Attributes:
        TELEGRAM_BOT_API_TOKEN (SecretStr): The API token for the Telegram bot, which is stored as a secret.

    Configuration:
        - Loads environment variables from a file named `.env`.
        - Uses UTF-8 encoding for the environment file.
    """

    TELEGRAM_BOT_API_TOKEN: SecretStr

    class Config:
        """
        Configuration for Pydantic Settings.

        Attributes:
            env_file (str): The path to the environment file.
            env_file_encoding (str): The encoding used for reading the environment file.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
