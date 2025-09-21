"""
MCP Data4SEO - FastMCP server for comprehensive SEO analysis using DataForSEO API

This package provides comprehensive SEO analysis capabilities including:
- Content analysis and markdown parsing
- SERP competitor analysis and keyword research
- PageRank analysis and internal link optimization
- Rich console reporting and progress tracking
"""

__version__ = "1.0.0"
__author__ = "Thomas PEDOT"

# Core server functionality
from mcp_seo.server import mcp

# Analysis modules
from mcp_seo.analysis import SERPCompetitorAnalyzer, SEORecommendationEngine

# Content analysis modules
from mcp_seo.content import (
    MarkdownParser,
    BlogAnalyzer,
    LinkOptimizer,
    LinkOpportunity,
    ClusterOpportunity
)

# Utility modules
from mcp_seo.utils import SEOReporter, ProgressTracker, create_seo_reporter

# Configuration and client
from mcp_seo.dataforseo.client import DataForSEOClient
from mcp_seo.config.settings import get_settings

__all__ = [
    # Core functionality
    "mcp",
    "main",

    # Analysis
    "SERPCompetitorAnalyzer",
    "SEORecommendationEngine",

    # Content analysis
    "MarkdownParser",
    "BlogAnalyzer",
    "LinkOptimizer",
    "LinkOpportunity",
    "ClusterOpportunity",

    # Utilities
    "SEOReporter",
    "ProgressTracker",
    "create_seo_reporter",

    # Configuration
    "DataForSEOClient",
    "get_settings"
]

def main():
    """Main entry point for the MCP Data4SEO server."""
    mcp.run()
