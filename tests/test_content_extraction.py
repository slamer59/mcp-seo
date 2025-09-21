#!/usr/bin/env python3
"""
Test the extracted content analysis components
=============================================

Basic tests to verify that the extracted components from GitAlchemy
Kuzu PageRank analyzer work correctly in the MCP SEO environment.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from mcp_seo.content import MarkdownParser, BlogAnalyzer, LinkOptimizer


@pytest.fixture
def sample_blog_posts():
    """Create sample blog posts for testing."""
    posts = {
        "post1.md": """---
title: "Getting Started with Python"
keywords: ["python", "programming", "tutorial"]
author: "Test Author"
published: true
date: "2024-01-01"
---

# Getting Started with Python

Python is a great programming language for beginners. In this tutorial, we'll cover the basics.

## Why Python?

Python is easy to learn and has great [[data-science]] capabilities.

## Next Steps

Check out our [[advanced-python]] guide for more information.
""",
        "post2.md": """---
title: "Data Science with Python"
keywords: ["python", "data-science", "pandas"]
author: "Test Author"
published: true
date: "2024-01-02"
---

# Data Science with Python

Python is excellent for data science work. Let's explore pandas and visualization.

## Pandas Basics

Pandas is the most important library for data manipulation in Python.

## Visualization

Use matplotlib and seaborn for creating charts and graphs.
""",
        "post3.md": """---
title: "Advanced Python Techniques"
keywords: ["python", "advanced", "programming"]
author: "Test Author"
published: true
date: "2024-01-03"
---

# Advanced Python Techniques

Once you've mastered the basics from our [[post1|getting started guide]],
you can move on to more advanced topics.

## Object-Oriented Programming

Learn about classes and objects in Python.

## Functional Programming

Explore functional programming concepts like map, filter, and reduce.
"""
    }
    return posts


@pytest.fixture
def temp_blog_dir(sample_blog_posts):
    """Create temporary directory with sample blog posts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        blog_dir = Path(temp_dir)

        for filename, content in sample_blog_posts.items():
            post_file = blog_dir / filename
            post_file.write_text(content, encoding='utf-8')

        yield blog_dir


def create_mock_metrics(posts_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """Create mock graph metrics for testing."""
    mock_metrics = {}

    for i, slug in enumerate(posts_data.keys()):
        mock_metrics[slug] = {
            'pagerank': 0.1 + (i * 0.05),
            'betweenness_centrality': 0.2 + (i * 0.1),
            'in_degree': i + 1,
            'out_degree': (i % 2) + 1,
            'closeness_centrality': 0.5 + (i * 0.1),
            'hub_score': 0.3 + (i * 0.1),
            'authority_score': 0.25 + (i * 0.05),
            'clustering_coefficient': 0.4,
            'katz_centrality': 0.15 + (i * 0.03),
            'total_degree': (i + 1) + ((i % 2) + 1)
        }

    return mock_metrics


class TestMarkdownParser:
    """Test the MarkdownParser component."""

    def test_parser_initialization(self, temp_blog_dir):
        """Test parser initialization."""
        parser = MarkdownParser(temp_blog_dir)
        assert parser.content_dir == temp_blog_dir
        assert len(parser.link_patterns) > 0
        assert len(parser.compiled_patterns) > 0

    def test_parse_all_posts(self, temp_blog_dir):
        """Test parsing all posts in directory."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()

        assert len(posts) == 3
        assert "post1" in posts
        assert "post2" in posts
        assert "post3" in posts

    def test_post_metadata_extraction(self, temp_blog_dir):
        """Test metadata extraction from frontmatter."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()

        post1 = posts["post1"]
        assert post1["title"] == "Getting Started with Python"
        assert "python" in post1["keywords"]
        assert "programming" in post1["keywords"]
        assert post1["author"] == "Test Author"
        assert post1["published"] is True

    def test_internal_links_extraction(self, temp_blog_dir):
        """Test internal link extraction."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()

        post1 = posts["post1"]
        links = post1["internal_links"]

        assert len(links) == 2
        link_targets = {link["target_slug"] for link in links}
        assert "data-science" in link_targets
        assert "advanced-python" in link_targets

    def test_quality_metrics_calculation(self, temp_blog_dir):
        """Test content quality metrics calculation."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()

        post1 = posts["post1"]
        metrics = post1["quality_metrics"]

        assert "word_count" in metrics
        assert "header_count" in metrics
        assert "readability_score" in metrics
        assert metrics["word_count"] > 0
        assert metrics["header_count"] > 0

    def test_content_statistics(self, temp_blog_dir):
        """Test overall content statistics."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        stats = parser.get_content_statistics()

        assert stats["total_posts"] == 3
        assert stats["total_words"] > 0
        assert stats["total_internal_links"] > 0
        assert stats["avg_words_per_post"] > 0


class TestBlogAnalyzer:
    """Test the BlogAnalyzer component."""

    def test_analyzer_initialization(self, temp_blog_dir):
        """Test analyzer initialization."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        analyzer = BlogAnalyzer(posts, metrics)
        assert analyzer.posts_data == posts
        assert analyzer.metrics == metrics

    def test_comprehensive_analysis(self, temp_blog_dir):
        """Test comprehensive analysis generation."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        analyzer = BlogAnalyzer(posts, metrics)
        analysis = analyzer.generate_comprehensive_analysis()

        assert "summary" in analysis
        assert "pillar_pages" in analysis
        assert "underperforming_pages" in analysis
        assert "content_clusters" in analysis
        assert "link_opportunities" in analysis
        assert "recommendations" in analysis

    def test_pillar_pages_identification(self, temp_blog_dir):
        """Test pillar page identification."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        analyzer = BlogAnalyzer(posts, metrics)
        analysis = analyzer.generate_comprehensive_analysis()

        pillar_pages = analysis["pillar_pages"]
        assert len(pillar_pages) > 0

        # Check that pillar pages are sorted by score
        scores = [page["pillar_score"] for page in pillar_pages]
        assert scores == sorted(scores, reverse=True)

    def test_content_clusters_analysis(self, temp_blog_dir):
        """Test content clustering analysis."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        analyzer = BlogAnalyzer(posts, metrics)
        analysis = analyzer.generate_comprehensive_analysis()

        clusters = analysis["content_clusters"]
        # Should have at least one cluster for "python" keyword
        assert "python" in clusters
        assert len(clusters["python"]["pages"]) >= 2


class TestLinkOptimizer:
    """Test the LinkOptimizer component."""

    def test_optimizer_initialization(self, temp_blog_dir):
        """Test optimizer initialization."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        optimizer = LinkOptimizer(posts, metrics)
        assert optimizer.posts_data == posts
        assert optimizer.metrics == metrics
        assert len(optimizer.link_graph) > 0
        assert len(optimizer.keyword_index) > 0

    def test_link_opportunities_identification(self, temp_blog_dir):
        """Test link opportunities identification."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        optimizer = LinkOptimizer(posts, metrics)
        opportunities = optimizer.identify_link_opportunities(
            max_opportunities=10,
            min_relevance_score=0.1
        )

        assert len(opportunities) > 0

        # Check opportunity structure
        opp = opportunities[0]
        assert hasattr(opp, 'source_slug')
        assert hasattr(opp, 'target_slug')
        assert hasattr(opp, 'opportunity_score')
        assert hasattr(opp, 'shared_keywords')

    def test_cluster_opportunities_identification(self, temp_blog_dir):
        """Test cluster opportunities identification."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        optimizer = LinkOptimizer(posts, metrics)
        cluster_opportunities = optimizer.identify_cluster_opportunities(
            min_cluster_size=2,
            min_cluster_strength=0.01
        )

        # Should find at least one cluster (for "python" keyword)
        assert len(cluster_opportunities) > 0

        cluster = cluster_opportunities[0]
        assert hasattr(cluster, 'cluster_keyword')
        assert hasattr(cluster, 'cluster_pages')
        assert hasattr(cluster, 'optimization_potential')

    def test_link_equity_flow_analysis(self, temp_blog_dir):
        """Test link equity flow analysis."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        optimizer = LinkOptimizer(posts, metrics)
        flow_analysis = optimizer.analyze_link_equity_flow()

        assert "total_links" in flow_analysis
        assert "total_pages" in flow_analysis
        assert "page_flow" in flow_analysis
        assert "flow_health_score" in flow_analysis
        assert "recommendations" in flow_analysis

    def test_implementation_plan_generation(self, temp_blog_dir):
        """Test implementation plan generation."""
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()
        metrics = create_mock_metrics(posts)

        optimizer = LinkOptimizer(posts, metrics)
        link_opportunities = optimizer.identify_link_opportunities(max_opportunities=5)
        cluster_opportunities = optimizer.identify_cluster_opportunities(min_cluster_size=2)

        plan = optimizer.generate_implementation_plan(
            link_opportunities, cluster_opportunities
        )

        assert "implementation_phases" in plan
        assert "action_items" in plan
        assert "expected_roi" in plan
        assert "tracking_metrics" in plan


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self, temp_blog_dir):
        """Test the complete analysis workflow."""
        # Step 1: Parse content
        parser = MarkdownParser(temp_blog_dir)
        posts = parser.parse_all_posts()

        # Step 2: Create metrics
        metrics = create_mock_metrics(posts)

        # Step 3: Analyze content
        analyzer = BlogAnalyzer(posts, metrics)
        analysis = analyzer.generate_comprehensive_analysis()

        # Step 4: Optimize links
        optimizer = LinkOptimizer(posts, metrics)
        link_opportunities = optimizer.identify_link_opportunities()
        cluster_opportunities = optimizer.identify_cluster_opportunities()

        # Verify everything works together
        assert len(posts) > 0
        assert len(metrics) == len(posts)
        assert len(analysis) > 0
        assert len(link_opportunities) >= 0  # Might be 0 if no opportunities
        assert len(cluster_opportunities) >= 0

    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with empty posts data
        analyzer = BlogAnalyzer({}, {})
        analysis = analyzer.generate_comprehensive_analysis()

        # Should handle empty data gracefully
        assert analysis["summary"]["total_pages"] == 0
        assert len(analysis["pillar_pages"]) == 0


if __name__ == "__main__":
    pytest.main([__file__])