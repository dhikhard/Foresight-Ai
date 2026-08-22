import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    if settings.groq_api_key:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
    return settings
