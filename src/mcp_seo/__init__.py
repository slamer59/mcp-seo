"""
MCP Data4SEO - FastMCP server for comprehensive SEO analysis using DataForSEO API
"""

__version__ = "1.0.0"
__author__ = "Thomas PEDOT"

from mcp_seo.server import mcp

def main():
    """Main entry point for the MCP Data4SEO server."""
    mcp.run()
