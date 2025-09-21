"""
Configuration management for MCP SEO.

This module handles application configuration, settings, and environment-specific
configurations.
"""

from .settings import get_settings, get_location_code, get_language_code

__all__ = [
    "get_settings",
    "get_location_code",
    "get_language_code"
]