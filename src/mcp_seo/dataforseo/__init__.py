"""
DataForSEO API client and infrastructure.

This module provides the client implementation for interacting with the
DataForSEO API for SEO data collection and analysis.
"""

from .client import DataForSEOClient, ApiException

__all__ = [
    "DataForSEOClient",
    "ApiException"
]