#!/bin/bash

# GitAlchemy SEO PageRank Analyzer Runner
# =====================================

set -e  # Exit on error

echo "🚀 GitAlchemy SEO PageRank Analyzer"
echo "==================================="
echo

# Check if we're in the right directory
if [ ! -f "run_seo_analysis.py" ]; then
    echo "❌ Error: Please run this script from the kuzu-pagerank-analyzer directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run:"
    echo "   uv add kuzu polars pyarrow networkx numpy scipy rich click python-frontmatter beautifulsoup4"
    exit 1
fi

# Activate virtual environment and run analysis
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📊 Running SEO analysis..."
echo

# Run with all arguments passed to this script
python run_seo_analysis.py "$@"

echo
echo "✅ Analysis complete!"
echo "📁 Check the 'analysis-results' directory for output files"
echo "📖 See USAGE.md for guidance on interpreting results"