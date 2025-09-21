#!/usr/bin/env python3
"""
Content Analysis Example
========================

Example script demonstrating how to use the extracted blog analysis components
from the legacy GitAlchemy Kuzu PageRank analyzer.

This example shows how to:
1. Parse markdown blog content
2. Analyze content using graph metrics
3. Optimize internal linking structure
4. Generate SEO recommendations

Usage:
    python content_analysis_example.py --content-dir /path/to/blog/posts

Author: MCP SEO Content Analysis Example
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Import the extracted components
from mcp_seo.content import MarkdownParser, BlogAnalyzer, LinkOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_metrics(posts_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Create mock graph metrics for demonstration purposes.
    In a real implementation, these would come from Kuzu or NetworkX analysis.
    """
    mock_metrics = {}

    for i, slug in enumerate(posts_data.keys()):
        # Simulate some variation in metrics
        base_score = 1.0 / len(posts_data)  # Base PageRank
        variation = (i % 5) * 0.02  # Add some variation

        mock_metrics[slug] = {
            'pagerank': base_score + variation,
            'betweenness_centrality': variation * 2,
            'in_degree': (i % 3) + 1,
            'out_degree': (i % 4) + 1,
            'closeness_centrality': 0.5 + variation,
            'hub_score': variation * 1.5,
            'authority_score': base_score + (variation * 0.5),
            'clustering_coefficient': 0.3 + variation,
            'katz_centrality': base_score * 1.2,
            'total_degree': ((i % 3) + 1) + ((i % 4) + 1)
        }

    return mock_metrics


def analyze_blog_content(content_dir: Path) -> Dict[str, Any]:
    """
    Complete blog content analysis workflow.

    Args:
        content_dir: Directory containing markdown blog posts

    Returns:
        Comprehensive analysis results
    """
    logger.info(f"Starting blog content analysis for: {content_dir}")

    # Step 1: Parse markdown content
    logger.info("Parsing markdown content...")
    parser = MarkdownParser(
        content_dir=content_dir,
        # Configure for different link patterns if needed
        link_patterns=[
            r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]',  # WikiLinks
            r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)',   # Markdown links to .md
            r'\[([^\]]+)\]\(([^)]+)\)',            # General markdown links
        ]
    )

    posts_data = parser.parse_all_posts(recursive=True, filter_published=True)

    if not posts_data:
        logger.error("No blog posts found to analyze")
        return {}

    logger.info(f"Parsed {len(posts_data)} blog posts")

    # Step 2: Create mock graph metrics (replace with real graph analysis)
    logger.info("Generating graph metrics...")
    metrics = create_mock_metrics(posts_data)

    # Step 3: Analyze content with BlogAnalyzer
    logger.info("Performing comprehensive blog analysis...")
    analyzer = BlogAnalyzer(posts_data=posts_data, metrics=metrics)
    analysis = analyzer.generate_comprehensive_analysis()

    # Step 4: Optimize internal linking
    logger.info("Optimizing internal linking structure...")
    optimizer = LinkOptimizer(posts_data=posts_data, metrics=metrics)

    # Get link opportunities
    link_opportunities = optimizer.identify_link_opportunities(
        max_opportunities=20,
        min_relevance_score=0.3
    )

    # Get cluster opportunities
    cluster_opportunities = optimizer.identify_cluster_opportunities(
        min_cluster_size=2,
        min_cluster_strength=0.05
    )

    # Analyze link equity flow
    flow_analysis = optimizer.analyze_link_equity_flow()

    # Generate implementation plan
    implementation_plan = optimizer.generate_implementation_plan(
        link_opportunities, cluster_opportunities
    )

    # Step 5: Compile comprehensive results
    results = {
        'content_statistics': parser.get_content_statistics(),
        'blog_analysis': analysis,
        'link_opportunities': [
            {
                'source': opp.source_title,
                'target': opp.target_title,
                'score': opp.opportunity_score,
                'priority': opp.priority,
                'keywords': opp.shared_keywords,
                'suggestions': opp.context_suggestions
            }
            for opp in link_opportunities
        ],
        'cluster_opportunities': [
            {
                'keyword': cluster.cluster_keyword,
                'pages_count': len(cluster.cluster_pages),
                'pillar_page': cluster.pillar_page['title'] if cluster.pillar_page else None,
                'optimization_potential': cluster.optimization_potential,
                'missing_connections': len(cluster.missing_connections)
            }
            for cluster in cluster_opportunities
        ],
        'flow_analysis': flow_analysis,
        'implementation_plan': implementation_plan
    }

    return results


def print_analysis_summary(results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("BLOG CONTENT SEO ANALYSIS SUMMARY")
    print("="*60)

    # Content Statistics
    stats = results.get('content_statistics', {})
    print(f"\n📊 CONTENT STATISTICS:")
    print(f"   Total Posts: {stats.get('total_posts', 0)}")
    print(f"   Total Words: {stats.get('total_words', 0):,}")
    print(f"   Total Internal Links: {stats.get('total_internal_links', 0)}")
    print(f"   Average Links per Post: {stats.get('avg_links_per_post', 0):.2f}")
    print(f"   Link Density: {stats.get('link_density', 0):.4f}")

    # Blog Analysis Summary
    blog_analysis = results.get('blog_analysis', {})
    summary = blog_analysis.get('summary', {})
    print(f"\n🔍 SEO ANALYSIS:")
    print(f"   Average Word Count: {summary.get('avg_word_count', 0):.0f}")
    print(f"   Average Readability: {summary.get('avg_readability_score', 0):.2f}")

    # Pillar Pages
    pillar_pages = blog_analysis.get('pillar_pages', [])
    print(f"\n🏛️  TOP PILLAR PAGES ({len(pillar_pages)} found):")
    for i, page in enumerate(pillar_pages[:5], 1):
        print(f"   {i}. {page['title']} (PageRank: {page['pagerank']:.4f})")

    # Link Opportunities
    link_opps = results.get('link_opportunities', [])
    print(f"\n🔗 LINK OPPORTUNITIES ({len(link_opps)} found):")
    for i, opp in enumerate(link_opps[:5], 1):
        print(f"   {i}. {opp['source']} → {opp['target']}")
        print(f"      Score: {opp['score']:.2f}, Priority: {opp['priority']}")
        print(f"      Keywords: {', '.join(opp['keywords'][:3])}")

    # Cluster Opportunities
    cluster_opps = results.get('cluster_opportunities', [])
    print(f"\n📚 CONTENT CLUSTERS ({len(cluster_opps)} found):")
    for i, cluster in enumerate(cluster_opps[:5], 1):
        print(f"   {i}. '{cluster['keyword']}' cluster ({cluster['pages_count']} pages)")
        print(f"      Pillar: {cluster['pillar_page'] or 'None identified'}")
        print(f"      Potential: {cluster['optimization_potential']:.2f}")

    # Flow Analysis
    flow = results.get('flow_analysis', {})
    print(f"\n🌊 LINK EQUITY FLOW:")
    print(f"   Flow Health Score: {flow.get('flow_health_score', 0):.2f}")
    print(f"   Bottlenecks Found: {len(flow.get('bottlenecks', []))}")
    print(f"   Authority Sinks: {len(flow.get('authority_sinks', []))}")

    # Implementation Plan
    plan = results.get('implementation_plan', {})
    phases = plan.get('implementation_phases', {})
    print(f"\n📋 IMPLEMENTATION PLAN:")
    for phase_name, phase_data in phases.items():
        phase_num = phase_name.split('_')[1]
        link_count = len(phase_data.get('link_opportunities', []))
        cluster_count = len(phase_data.get('cluster_work', []))
        print(f"   Phase {phase_num}: {link_count} links, {cluster_count} clusters")
        print(f"              Effort: {phase_data.get('estimated_effort', 'Unknown')}")

    # ROI
    roi = plan.get('expected_roi', {})
    print(f"\n💰 EXPECTED ROI:")
    print(f"   Total Value: {roi.get('total_expected_value', 0):.2f}")
    print(f"   Estimated Hours: {roi.get('total_estimated_effort_hours', 0):.1f}")
    print(f"   Value per Hour: {roi.get('value_per_hour', 0):.2f}")

    print("\n" + "="*60)


def main():
    """Main entry point for the content analysis example."""
    parser = argparse.ArgumentParser(
        description="Analyze blog content for SEO optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--content-dir',
        type=Path,
        required=True,
        help='Directory containing markdown blog posts'
    )

    parser.add_argument(
        '--output-file',
        type=Path,
        help='Optional JSON file to save detailed results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate content directory
    if not args.content_dir.exists():
        logger.error(f"Content directory not found: {args.content_dir}")
        return 1

    try:
        # Run analysis
        results = analyze_blog_content(args.content_dir)

        if not results:
            logger.error("Analysis failed - no results generated")
            return 1

        # Print summary
        print_analysis_summary(results)

        # Save detailed results if requested
        if args.output_file:
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Detailed results saved to: {args.output_file}")

        logger.info("Content analysis completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())