# GitAlchemy SEO PageRank Analyzer - Quick Usage Guide

## 🚀 Quick Start

```bash
# Navigate to the analyzer directory
cd scripts/kuzu-pagerank-analyzer

# Run the analysis (recommended way)
source .venv/bin/activate && python run_seo_analysis.py
```

## 📊 What You Get

The tool analyzes your blog content and provides:

### 1. **Pillar Pages Identification**
- Pages with highest PageRank and authority scores
- These should be your main content hubs
- Focus internal linking strategy around these pages

### 2. **Internal Linking Opportunities** 
- Specific page-to-page linking suggestions
- Based on keyword relevance and PageRank potential
- Prioritized by opportunity score

### 3. **Content Clusters**
- Related content grouped by shared keywords
- Helps identify topical authority opportunities
- Suggests which pages to interlink

### 4. **SEO Recommendations**
- High/Medium priority actionable items
- Specific improvements for better search rankings
- Link equity distribution strategies

## 📁 Output Files

### `analysis-results/seo_analysis.json`
Complete analysis with all metrics and recommendations

### `analysis-results/pillar_pages.csv` 
Top content hubs ranked by authority - use for content strategy

### `analysis-results/link_opportunities.csv`
Specific internal linking suggestions - use for content optimization

## 🔍 Key Metrics Explained

**PageRank Score**: Authority based on internal link structure (0.0-1.0)
**Authority Score**: Content hub potential based on incoming links
**Link Density**: Overall internal linking interconnectedness
**Opportunity Score**: Priority ranking for new internal links

## 📈 Current GitAlchemy Analysis Results

Based on the latest analysis of 56 blog posts:

- **Link Density**: 0.0149 (Low - needs improvement)
- **Average Links Per Page**: 0.82 (Very low - major opportunity)
- **Top Pillar Pages**: Basic Operations, What is GitLab?, Search Features
- **Link Opportunities**: 20 specific suggestions identified

### 🎯 Immediate Action Items

1. **Add internal links to underperforming pages** (High Priority)
   - Many pages have 0 incoming links
   - Target: 2-3 internal links per page minimum

2. **Strengthen pillar pages** (High Priority) 
   - Link more content TO your top authority pages
   - Make "Basic Operations" and "What is GitLab?" main hubs

3. **Create topic clusters** (Medium Priority)
   - Group GitLab feature content together
   - Link tutorial content in sequences
   - Cross-link comparison articles

## 💡 How to Use Results

### For Content Strategy:
- Use pillar pages as templates for new high-value content
- Focus keyword optimization on highest PageRank pages
- Plan content calendar around top authority topics

### For Technical SEO:
- Implement suggested internal links in content updates
- Update navigation to highlight pillar pages
- Create topic silos based on content clusters

### For Performance Tracking:
- Re-run analysis monthly to track improvements
- Monitor PageRank changes after implementing links
- Compare link density improvements over time

## 🔄 Running Custom Analysis

```bash
# Different blog directory
python run_seo_analysis.py --blog-dir /path/to/your/blog

# Custom output location  
python run_seo_analysis.py --output-dir ./custom-results

# JSON only export
python run_seo_analysis.py --export-format json

# Full help
python run_seo_analysis.py --help
```

## 🛠️ Troubleshooting

**"No blog posts found"**: Check that `../../src/content/blog` contains `.md` files

**"Database error"**: Delete existing database files and re-run

**"Import errors"**: Make sure you're running in the activated virtual environment:
```bash
source .venv/bin/activate
```

## 🎯 Integration with Existing SEO Tools

The JSON output can be imported into:
- Google Search Console (for comparing with actual search performance)
- SEO reporting tools
- Content management systems
- Analytics dashboards

Example Python integration:
```python
import json
with open('analysis-results/seo_analysis.json', 'r') as f:
    seo_data = json.load(f)
    
pillar_pages = seo_data['analysis']['pillar_pages']
recommendations = seo_data['analysis']['recommendations']
```