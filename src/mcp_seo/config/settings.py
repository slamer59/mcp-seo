"""
Configuration settings for FastMCP SEO server.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # DataForSEO API credentials
    dataforseo_login: Optional[str] = None
    dataforseo_password: Optional[str] = None
    
    # MCP Server settings
    server_name: str = "mcp-seo"
    server_version: str = "1.0.0"
    
    # API rate limiting and timeout settings
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    
    # Default SEO analysis settings
    default_location_code: int = 2840  # USA
    default_language_code: str = "en"
    default_device: str = "desktop"
    
    # OnPage analysis settings
    onpage_max_crawl_pages: int = 100
    onpage_crawl_delay: int = 1
    
    # Keyword analysis settings
    keyword_limit: int = 100
    competitor_limit: int = 50
    
    # Task management
    task_completion_timeout: int = 300  # 5 minutes
    task_check_interval: int = 10  # 10 seconds
    
    # Logging settings
    log_level: str = "INFO"
    enable_request_logging: bool = False
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Location and language mappings for common use cases
LOCATION_MAPPINGS = {
    "usa": 2840,
    "uk": 2826,
    "canada": 2124,
    "australia": 2036,
    "germany": 2276,
    "france": 2250,
    "spain": 2724,
    "italy": 2380,
    "japan": 2392,
    "south_korea": 2410,
    "brazil": 2076,
    "india": 2356,
    "china": 2156,
    "russia": 2643
}

LANGUAGE_MAPPINGS = {
    "english": "en",
    "spanish": "es", 
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "russian": "ru",
    "dutch": "nl",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi"
}

DEVICE_MAPPINGS = {
    "desktop": "desktop",
    "mobile": "mobile",
    "tablet": "tablet"
}

def get_location_code(location: str) -> int:
    """Get location code from string identifier."""
    location_lower = location.lower().replace(" ", "_")
    return LOCATION_MAPPINGS.get(location_lower, 2840)  # Default to USA

def get_language_code(language: str) -> str:
    """Get language code from string identifier."""
    language_lower = language.lower()
    return LANGUAGE_MAPPINGS.get(language_lower, "en")  # Default to English