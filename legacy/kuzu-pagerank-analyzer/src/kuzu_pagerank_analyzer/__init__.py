"""
Kùzu PageRank SEO Analyzer for GitAlchemy Blog Content
=====================================================

A comprehensive SEO analysis tool that uses Kùzu graph database to analyze
internal linking structure and calculate PageRank scores for blog posts.

Author: GitAlchemy Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "GitAlchemy Team"
__description__ = "Comprehensive SEO PageRank analysis tool using Kùzu graph database for GitAlchemy blog content"

from .main import main

__all__ = ["main"]
