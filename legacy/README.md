# Legacy SEO Scripts

This directory contains the original SEO analysis scripts that were migrated from the GitAlchemy project.

## ⚠️ Legacy Status

These scripts are preserved for reference but are **no longer actively maintained**. The functionality has been modernized and integrated into the main MCP SEO project.

## 📁 Contents

### `kuzu-pagerank-analyzer/`
Original blog SEO analysis tools using Kuzu graph database:
- `seo_analyzer.py` - DataForSEO API client with GitAlchemy-specific analysis
- `run_seo_analysis.py` - Kuzu PageRank analyzer for blog posts
- `src/kuzu_pagerank_analyzer/main.py` - Core analysis components

## 🔄 Migration Status

✅ **Migrated Components:**
- DataForSEO API patterns → `mcp_seo/analysis/competitor_analyzer.py`
- SEO recommendation engine → `mcp_seo/analysis/recommendation_engine.py`
- Rich console reporting → `mcp_seo/utils/rich_reporter.py`
- Kuzu blog analysis → `mcp_seo/content/blog_analyzer.py`
- Markdown parsing → `mcp_seo/content/markdown_parser.py`
- Link optimization → `mcp_seo/content/link_optimizer.py`

## 🚀 Modern Replacements

Instead of using these legacy scripts, use the modern MCP SEO tools:

```python
# Legacy: seo_analyzer.py
# Modern: Use MCP SEO tools directly

from mcp_seo import SEORecommendationEngine, SERPCompetitorAnalyzer, SEOReporter

# Legacy: run_seo_analysis.py
# Modern: Use content analysis tools

from mcp_seo import BlogAnalyzer, MarkdownParser, LinkOptimizer
```

## 📚 Historical Reference

These scripts were developed for the GitAlchemy project and contained:
- GitAlchemy-specific keywords and competitor data
- Hardcoded domain analysis (gitalchemy.app)
- CLI-based interface with rich console output
- Custom blog content analysis for internal linking

The modern MCP SEO implementation provides the same functionality but with:
- Generic, configurable analysis
- MCP protocol integration
- Enhanced error handling and modularity
- Better type safety and documentation