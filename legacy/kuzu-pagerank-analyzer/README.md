# Kuzu PageRank SEO Analyzer

Comprehensive SEO analysis tool using Kuzu graph database for GitAlchemy blog content analysis.

## Overview

This tool analyzes your blog's internal linking structure using Kuzu graph database and provides actionable SEO recommendations to improve search rankings. It parses local markdown files, builds a graph database, and calculates PageRank scores to identify content opportunities.

## Features

- = **Markdown Parsing**: Direct analysis of local markdown files with frontmatter extraction
- =Ê **Kuzu Graph Database**: High-performance graph analysis with native algorithms
- <¯ **PageRank Calculation**: Uses Kuzu's native algo extension for optimized PageRank computation
- =È **Comprehensive Metrics**: PageRank, betweenness centrality, authority scores, and more
- <Û **Pillar Page Identification**: Automatically identifies content hubs and authority pages
- = **Link Opportunity Analysis**: Finds missing internal links based on content relevance
- =Ë **Actionable Recommendations**: Specific SEO improvement suggestions
- =¾ **Multiple Export Formats**: JSON and CSV export for further analysis

## Installation

```bash
# Navigate to the project directory
cd scripts/kuzu-pagerank-analyzer

# Install the package with uv
uv pip install -e .

# Or run directly with uv
uv run kuzu-pagerank-analyzer --help
```

## Usage

### Basic Analysis

```bash
# Analyze blog content with default settings
uv run kuzu-pagerank-analyzer

# Specify custom blog directory
uv run kuzu-pagerank-analyzer --blog-dir ../../src/content/blog

# Custom output directory
uv run kuzu-pagerank-analyzer --output-dir ./seo-results

# Export specific format
uv run kuzu-pagerank-analyzer --export-format json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--blog-dir` | Directory containing markdown blog posts | `../../src/content/blog` |
| `--output-dir` | Directory to save analysis results | `output` |
| `--export-format` | Export format: `json`, `csv`, or `both` | `both` |
| `--db-path` | Kuzu database path | `blog_seo_analysis` |

## Dependencies

- **kuzu**: Graph database engine
- **polars**: High-performance data manipulation  
- **networkx**: Graph algorithms (fallback)
- **rich**: Beautiful terminal output
- **click**: Command-line interface
- **python-frontmatter**: Markdown frontmatter parsing
- **beautifulsoup4**: HTML content cleaning

## License

MIT License - see LICENSE file for details.