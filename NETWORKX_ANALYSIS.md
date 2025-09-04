# NetworkX Graph Analysis for SEO

This document explains the comprehensive NetworkX-based graph analysis suite for SEO optimization in the MCP Data4SEO server.

## Overview

The NetworkX analysis suite provides advanced graph algorithms to analyze website link structures beyond basic PageRank, offering multiple ways to organize and analyze nodes for strategic SEO decision-making.

## Available Tools

### 1. `analyze_centrality` - Authority and Hub Identification

**Purpose**: Identify authoritative pages and navigation hubs using multiple centrality measures.

**Parameters**:
- `domain`: Domain to analyze (requires existing link graph)
- `metrics`: List of centrality metrics to calculate (default: betweenness, closeness, eigenvector)
- `top_k`: Number of top pages to return per metric (default: 20)

**Centrality Measures**:
- **Betweenness Centrality**: Identifies bridge pages that connect different site sections
- **Closeness Centrality**: Finds pages that are easily accessible from other pages
- **Eigenvector Centrality**: Detects authoritative pages linked by other authoritative pages
- **Degree Centrality**: Basic link popularity (in-degree and out-degree)
- **Katz Centrality**: Alternative authority measure for directed graphs

**SEO Applications**:
- Identify pages with natural authority for strategic internal linking
- Find bridge pages that can connect different site sections
- Discover underutilized high-authority pages
- Optimize navigation structure around natural hubs

**Example Response**:
```json
{
  "domain": "https://example.com/",
  "analysis_type": "centrality",
  "metrics_analyzed": ["betweenness", "eigenvector"],
  "top_pages_by_metric": {
    "betweenness": [
      {
        "url": "https://example.com/about",
        "title": "About Us",
        "betweenness_centrality": 0.0234
      }
    ]
  },
  "seo_insights": [
    "🌉 Top bridge page: 'About Us' - Use this page to connect different site sections"
  ],
  "recommendations": [
    {
      "priority": "high",
      "type": "navigation_optimization",
      "action": "Leverage top bridge page for cross-linking"
    }
  ]
}
```

### 2. `detect_communities` - Content Clustering Analysis

**Purpose**: Identify content communities and topic clusters for content strategy optimization.

**Parameters**:
- `domain`: Domain to analyze
- `algorithm`: Community detection algorithm ("louvain" or "greedy_modularity")
- `min_community_size`: Minimum community size to report (default: 3)

**Algorithms**:
- **Louvain**: High-quality communities with optimal modularity
- **Greedy Modularity**: Interpretable clusters based on modularity maximization

**SEO Applications**:
- Identify content silos and topic clusters
- Find isolated content that needs better integration
- Optimize internal linking within topic areas
- Develop content hub strategies

**Key Metrics**:
- **Modularity Score**: Quality of community structure (>0.3 = good clustering)
- **Community Authority**: Total PageRank within each community
- **Internal vs External Links**: Link patterns within and between communities

### 3. `analyze_site_structure` - Architecture Analysis

**Purpose**: Analyze structural properties for site architecture optimization.

**Parameters**:
- `domain`: Domain to analyze

**Structural Analysis**:
- **K-core Decomposition**: Identifies the structural hierarchy and core strength
- **Clustering Coefficient**: Measures local connectivity and topic coherence
- **Critical Pages**: Articulation points essential for site connectivity
- **Graph Density**: Overall link interconnectedness

**SEO Applications**:
- Assess site architecture strength
- Identify critical pages that must be maintained
- Find opportunities for better content interconnection
- Optimize topic cluster formation

### 4. `analyze_navigation_paths` - UX Path Optimization

**Purpose**: Analyze navigation efficiency and path structure for UX optimization.

**Parameters**:
- `domain`: Domain to analyze

**Path Metrics**:
- **Average Path Length**: Mean clicks between pages
- **Site Diameter**: Maximum navigation distance
- **Connectivity Ratio**: Percentage of well-connected pages
- **Hard-to-reach Pages**: Valuable content with poor accessibility

**SEO Applications**:
- Identify navigation bottlenecks
- Improve content discoverability
- Optimize user journey paths
- Reduce clicks-to-content for important pages

### 5. `find_connector_pages` - Bridge Opportunity Identification

**Purpose**: Find pages that connect different site sections and identify linking opportunities.

**Parameters**:
- `domain`: Domain to analyze
- `min_betweenness`: Minimum betweenness centrality threshold (default: 0.0)

**Analysis Features**:
- **Natural Bridges**: Pages with high betweenness centrality
- **Underutilized Authorities**: High PageRank pages with low connector function
- **Cross-section Analysis**: How different site areas connect
- **Bridge Opportunities**: Strategic internal linking recommendations

**SEO Applications**:
- Improve internal link structure
- Connect isolated site sections
- Leverage high-authority pages as navigation hubs
- Strategic link equity distribution

### 6. `comprehensive_graph_analysis` - Complete Analysis Suite

**Purpose**: Run all NetworkX analyses in one comprehensive report.

**Parameters**:
- `domain`: Domain to analyze

**Features**:
- Combines all analysis types
- Unified insights and recommendations
- Prioritized action items
- Overall site health assessment

**Health Assessment**:
- **Excellent** (80%+): Strong structure, good navigation, clear topics
- **Good** (60-80%): Solid foundation with improvement opportunities
- **Fair** (40-60%): Some structural issues, needs optimization
- **Poor** (<40%): Significant structural problems requiring attention

## Prerequisites

Before using NetworkX analysis tools, you must first:

1. **Build Link Graph**: Use `build_link_graph` or `analyze_pagerank` to create the internal link database
2. **Have Sufficient Data**: Minimum 10 pages with internal links for meaningful analysis
3. **Connected Structure**: Site should have some internal linking (not all orphaned pages)

## Workflow Integration

### Typical Analysis Workflow:

1. **Initial Setup**:
   ```
   build_link_graph -> analyze_pagerank
   ```

2. **Comprehensive Analysis**:
   ```
   comprehensive_graph_analysis
   ```

3. **Focused Analysis** (choose based on needs):
   ```
   analyze_centrality (for authority identification)
   detect_communities (for content strategy)
   analyze_navigation_paths (for UX optimization)
   find_connector_pages (for internal linking)
   analyze_site_structure (for architecture review)
   ```

### Use Cases by SEO Goal:

**Content Strategy**:
- `detect_communities` → Identify topic clusters
- `analyze_centrality` → Find content authorities
- `find_connector_pages` → Connect content areas

**Technical SEO**:
- `analyze_site_structure` → Architecture assessment
- `analyze_navigation_paths` → Crawlability optimization
- `comprehensive_graph_analysis` → Full technical audit

**Link Building Strategy**:
- `analyze_centrality` → Identify link targets
- `find_connector_pages` → Internal linking opportunities
- `analyze_site_structure` → Critical page identification

## Performance Considerations

### Scalability:
- **Small sites** (< 100 pages): All algorithms run quickly
- **Medium sites** (100-1000 pages): Some algorithms use sampling for efficiency
- **Large sites** (1000+ pages): Automatic optimization and progress reporting

### Memory Usage:
- Graph construction: ~1MB per 1000 pages
- Centrality calculation: Additional ~500KB per 1000 pages
- Community detection: Minimal additional memory

## Interpretation Guide

### Centrality Scores:
- **Betweenness > 0.1**: Significant bridge function
- **Eigenvector > 0.05**: Strong authority signal
- **Closeness > 0.5**: High accessibility

### Community Quality:
- **Modularity > 0.3**: Good community structure
- **Modularity 0.1-0.3**: Moderate clustering
- **Modularity < 0.1**: Weak community structure

### Path Efficiency:
- **Average path < 3**: Excellent navigation
- **Average path 3-4**: Good navigation
- **Average path > 4**: Navigation issues

### Structural Health:
- **Max k-core ≥ 3**: Strong structural foundation
- **Clustering ≥ 0.3**: Good local connectivity
- **Density 0.01-0.05**: Optimal link density

## Integration with Other Tools

The NetworkX analysis suite works seamlessly with existing MCP SEO tools:

- **PageRank Analysis**: Provides baseline authority scores
- **OnPage Analysis**: Technical foundation for graph analysis
- **Competitor Analysis**: Compare graph structures
- **Keyword Analysis**: Map keywords to authority pages

## Error Handling

Common errors and solutions:

- **"No link graph data found"**: Run `build_link_graph` first
- **"Graph too small for analysis"**: Ensure minimum 10 connected pages
- **"Algorithm failed"**: May indicate disconnected graph components
- **"Insufficient connectivity"**: Add more internal links

## Advanced Features

### Custom Analysis:
- Filter by path patterns
- Focus on specific content areas  
- Authority threshold customization
- Community size filtering

### Export Capabilities:
- JSON format for further analysis
- Integration with visualization tools
- CSV export for spreadsheet analysis
- API response caching

## Best Practices

1. **Regular Analysis**: Run monthly for dynamic sites, quarterly for static sites
2. **Incremental Approach**: Start with comprehensive analysis, then focus on specific areas
3. **Action Prioritization**: Follow recommendation priority levels (high → medium → low)
4. **Progress Tracking**: Re-run analyses after implementing changes
5. **Integration**: Combine with other SEO tools for complete optimization strategy

## Support and Troubleshooting

For issues with NetworkX analysis:

1. Check prerequisites (link graph exists, sufficient data)
2. Verify graph connectivity (not all orphaned pages)
3. Review error messages for specific guidance
4. Consider reducing analysis scope for large sites
5. Check memory availability for intensive calculations