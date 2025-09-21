"""
Test suite for Enhanced content analysis components.

Tests the new content analysis functionality including MarkdownParser,
BlogAnalyzer, and LinkOptimizer with comprehensive coverage.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mcp_seo.content import MarkdownParser, BlogAnalyzer, LinkOptimizer
from mcp_seo.content.link_optimizer import LinkOpportunity, ClusterOpportunity


class TestMarkdownParser:
    """Test suite for MarkdownParser Enhanced functionality."""

    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing."""
        return """---
title: "Test Blog Post"
description: "A test blog post for SEO analysis"
keywords: ["seo", "testing", "content"]
date: "2024-01-15"
tags: ["seo", "blog"]
---

# Test Blog Post

This is a comprehensive test blog post that discusses various SEO topics.

## Introduction

SEO is crucial for website visibility. This section introduces the main concepts.

## Main Content

Here we discuss [internal linking](/internal-page) and [external resources](https://external.com).

### Subsection

More detailed content about [topic clusters](/topics/cluster) and optimization strategies.

## Conclusion

Final thoughts on SEO optimization and [best practices](/guides/best-practices).
"""

    @pytest.fixture
    def markdown_parser(self):
        """Create MarkdownParser instance."""
        return MarkdownParser()

    def test_parse_content_basic(self, markdown_parser, sample_markdown_content):
        """Test basic content parsing functionality."""
        result = markdown_parser.parse_content(sample_markdown_content)

        assert result is not None
        assert result['title'] == "Test Blog Post"
        assert result['description'] == "A test blog post for SEO analysis"
        assert 'seo' in result['keywords']
        assert len(result['internal_links']) > 0
        assert result['word_count'] > 50

    def test_extract_metadata(self, markdown_parser, sample_markdown_content):
        """Test metadata extraction from frontmatter."""
        result = markdown_parser.parse_content(sample_markdown_content)

        assert result['metadata']['title'] == "Test Blog Post"
        assert result['metadata']['date'] == "2024-01-15"
        assert "seo" in result['metadata']['tags']

    def test_extract_internal_links(self, markdown_parser, sample_markdown_content):
        """Test internal link extraction."""
        result = markdown_parser.parse_content(sample_markdown_content)

        internal_links = result['internal_links']
        assert len(internal_links) >= 3

        # Check specific links are found
        link_targets = [link['url'] for link in internal_links]
        assert '/internal-page' in link_targets
        assert '/topics/cluster' in link_targets
        assert '/guides/best-practices' in link_targets

    def test_content_quality_metrics(self, markdown_parser, sample_markdown_content):
        """Test content quality metric calculation."""
        result = markdown_parser.parse_content(sample_markdown_content)

        assert result['word_count'] > 0
        assert result['heading_count'] > 0
        assert result['link_count'] > 0
        assert 'keyword_density' in result['quality_metrics']

    def test_parse_file_with_missing_frontmatter(self, markdown_parser):
        """Test parsing content without frontmatter."""
        content_without_frontmatter = """
# Simple Post

This is just basic content without any frontmatter.

[Link to page](/simple-link)
"""
        result = markdown_parser.parse_content(content_without_frontmatter)

        assert result is not None
        assert result.get('title') is not None  # Should extract from first heading
        assert len(result['internal_links']) > 0

    def test_empty_content_handling(self, markdown_parser):
        """Test handling of empty or invalid content."""
        result = markdown_parser.parse_content("")

        assert result is not None
        assert result['word_count'] == 0
        assert len(result['internal_links']) == 0


class TestBlogAnalyzer:
    """Test suite for BlogAnalyzer Enhanced functionality."""

    @pytest.fixture
    def mock_kuzu_manager(self):
        """Mock KuzuManager for testing."""
        mock_manager = Mock()
        mock_manager.get_all_pages.return_value = [
            {"url": "/post-1", "title": "Post 1", "pagerank": 0.25},
            {"url": "/post-2", "title": "Post 2", "pagerank": 0.30},
            {"url": "/post-3", "title": "Post 3", "pagerank": 0.15},
        ]
        mock_manager.get_all_links.return_value = [
            {"source": "/post-1", "target": "/post-2", "anchor_text": "related post"},
            {"source": "/post-2", "target": "/post-3", "anchor_text": "another post"},
        ]
        return mock_manager

    @pytest.fixture
    def sample_content_data(self):
        """Sample content data for analysis."""
        return {
            "/post-1": {
                "title": "SEO Best Practices",
                "keywords": ["seo", "optimization", "ranking"],
                "word_count": 1200,
                "internal_links": ["/post-2"],
                "quality_metrics": {"readability": 0.8}
            },
            "/post-2": {
                "title": "Content Marketing Guide",
                "keywords": ["content", "marketing", "strategy"],
                "word_count": 800,
                "internal_links": ["/post-3"],
                "quality_metrics": {"readability": 0.7}
            },
            "/post-3": {
                "title": "Link Building Strategies",
                "keywords": ["links", "building", "seo"],
                "word_count": 600,
                "internal_links": [],
                "quality_metrics": {"readability": 0.6}
            }
        }

    @pytest.fixture
    def blog_analyzer(self, mock_kuzu_manager):
        """Create BlogAnalyzer instance."""
        return BlogAnalyzer(mock_kuzu_manager)

    def test_analyze_content_quality(self, blog_analyzer, sample_content_data):
        """Test content quality analysis."""
        result = blog_analyzer.analyze_content_quality(sample_content_data)

        assert 'quality_scores' in result
        assert 'recommendations' in result
        assert len(result['quality_scores']) == len(sample_content_data)

    def test_identify_pillar_pages(self, blog_analyzer, sample_content_data):
        """Test pillar page identification."""
        pillar_pages = blog_analyzer.identify_pillar_pages(sample_content_data)

        assert len(pillar_pages) > 0
        # Should identify pages with high word count and good metrics
        assert any(page['word_count'] > 1000 for page in pillar_pages)

    def test_detect_content_clusters(self, blog_analyzer, sample_content_data):
        """Test content cluster detection."""
        clusters = blog_analyzer.detect_content_clusters(sample_content_data)

        assert len(clusters) > 0
        # Should group related content by keywords
        for cluster in clusters:
            assert 'topic' in cluster
            assert 'pages' in cluster
            assert len(cluster['pages']) > 0

    def test_find_content_gaps(self, blog_analyzer, sample_content_data):
        """Test content gap identification."""
        gaps = blog_analyzer.find_content_gaps(sample_content_data)

        assert 'missing_topics' in gaps
        assert 'thin_content' in gaps
        assert 'orphaned_content' in gaps

    def test_generate_content_recommendations(self, blog_analyzer, sample_content_data):
        """Test content recommendation generation."""
        recommendations = blog_analyzer.generate_content_recommendations(sample_content_data)

        assert len(recommendations) > 0
        for rec in recommendations:
            assert 'type' in rec
            assert 'priority' in rec
            assert 'description' in rec
            assert 'action_items' in rec


class TestLinkOptimizer:
    """Test suite for LinkOptimizer Enhanced functionality."""

    @pytest.fixture
    def mock_kuzu_manager(self):
        """Mock KuzuManager for testing."""
        mock_manager = Mock()
        mock_manager.get_all_pages.return_value = [
            {"url": "/high-authority", "title": "High Authority Page", "pagerank": 0.35},
            {"url": "/medium-authority", "title": "Medium Authority Page", "pagerank": 0.20},
            {"url": "/low-authority", "title": "Low Authority Page", "pagerank": 0.10},
        ]
        mock_manager.get_page_links.return_value = [
            {"target": "/medium-authority", "anchor_text": "related content"},
        ]
        return mock_manager

    @pytest.fixture
    def sample_content_data(self):
        """Sample content data for link optimization."""
        return {
            "/high-authority": {
                "keywords": ["authority", "content", "seo"],
                "topics": ["seo", "optimization"],
                "word_count": 1500,
                "outbound_links": 3
            },
            "/medium-authority": {
                "keywords": ["content", "marketing", "strategy"],
                "topics": ["marketing", "content"],
                "word_count": 1200,
                "outbound_links": 2
            },
            "/low-authority": {
                "keywords": ["beginner", "seo", "basics"],
                "topics": ["seo", "basics"],
                "word_count": 800,
                "outbound_links": 1
            }
        }

    @pytest.fixture
    def link_optimizer(self, mock_kuzu_manager):
        """Create LinkOptimizer instance."""
        return LinkOptimizer(mock_kuzu_manager)

    def test_find_linking_opportunities(self, link_optimizer, sample_content_data):
        """Test link opportunity discovery."""
        opportunities = link_optimizer.find_linking_opportunities(sample_content_data)

        assert len(opportunities) > 0
        for opp in opportunities:
            assert isinstance(opp, LinkOpportunity)
            assert opp.source_url is not None
            assert opp.target_url is not None
            assert opp.relevance_score > 0

    def test_optimize_link_distribution(self, link_optimizer, sample_content_data):
        """Test link distribution optimization."""
        result = link_optimizer.optimize_link_distribution(sample_content_data)

        assert 'recommended_links' in result
        assert 'distribution_analysis' in result
        assert 'priority_actions' in result

    def test_analyze_content_clusters_for_linking(self, link_optimizer, sample_content_data):
        """Test cluster-based linking analysis."""
        clusters = link_optimizer.analyze_content_clusters(sample_content_data)

        assert len(clusters) > 0
        for cluster in clusters:
            assert isinstance(cluster, ClusterOpportunity)
            assert cluster.topic is not None
            assert len(cluster.pages) > 0

    def test_calculate_link_equity_flow(self, link_optimizer, sample_content_data):
        """Test link equity flow calculation."""
        flow_analysis = link_optimizer.calculate_link_equity_flow(sample_content_data)

        assert 'authority_distribution' in flow_analysis
        assert 'flow_recommendations' in flow_analysis
        assert 'bottlenecks' in flow_analysis

    def test_suggest_contextual_links(self, link_optimizer, sample_content_data):
        """Test contextual link suggestions."""
        suggestions = link_optimizer.suggest_contextual_links(
            source_url="/high-authority",
            content_data=sample_content_data
        )

        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert 'target_url' in suggestion
            assert 'suggested_anchor_text' in suggestion
            assert 'context_reason' in suggestion
            assert 'confidence_score' in suggestion


class TestContentAnalysisIntegration:
    """Integration tests for content analysis workflow."""

    @pytest.fixture
    def temp_markdown_files(self):
        """Create temporary markdown files for testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_content_")
        temp_path = Path(temp_dir)

        # Create test markdown files
        files = {}
        for i in range(3):
            file_path = temp_path / f"post-{i+1}.md"
            content = f"""---
title: "Test Post {i+1}"
keywords: ["test", "post", "seo"]
date: "2024-01-{15+i:02d}"
---

# Test Post {i+1}

This is test content for post {i+1}.

## Section

Content with [internal link](/test-post-{(i % 3) + 1}) and more text.
"""
            file_path.write_text(content)
            files[f"post-{i+1}"] = str(file_path)

        yield files

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_full_content_analysis_workflow(self, temp_markdown_files):
        """Test complete content analysis workflow."""
        # Initialize components
        parser = MarkdownParser()

        # Parse all files
        parsed_content = {}
        for post_id, file_path in temp_markdown_files.items():
            with open(file_path, 'r') as f:
                content = f.read()
            parsed_content[post_id] = parser.parse_content(content)

        # Verify parsed content
        assert len(parsed_content) == 3
        for post_id, data in parsed_content.items():
            assert data['title'].startswith("Test Post")
            assert data['word_count'] > 0
            assert len(data['internal_links']) > 0

    def test_content_analysis_with_link_optimization(self, mock_kuzu_manager, sample_content_data):
        """Test integrated content analysis with link optimization."""
        # Initialize analyzer and optimizer
        analyzer = BlogAnalyzer(mock_kuzu_manager)
        optimizer = LinkOptimizer(mock_kuzu_manager)

        # Analyze content quality
        quality_analysis = analyzer.analyze_content_quality(sample_content_data)

        # Find linking opportunities
        link_opportunities = optimizer.find_linking_opportunities(sample_content_data)

        # Verify integration
        assert 'quality_scores' in quality_analysis
        assert len(link_opportunities) > 0

        # Check that high-quality content gets more link opportunities
        high_quality_pages = [url for url, score in quality_analysis['quality_scores'].items()
                             if score > 0.7]
        if high_quality_pages:
            high_quality_opportunities = [opp for opp in link_opportunities
                                        if opp.target_url in high_quality_pages]
            assert len(high_quality_opportunities) > 0


# Additional test data and helper functions

@pytest.fixture
def complex_markdown_content():
    """Complex markdown content for advanced testing."""
    return """---
title: "Advanced SEO Strategies for 2024"
description: "Comprehensive guide to modern SEO techniques"
keywords: ["seo", "strategies", "2024", "optimization", "ranking"]
author: "SEO Expert"
date: "2024-01-20"
tags: ["seo", "advanced", "guide"]
category: "SEO"
featured: true
---

# Advanced SEO Strategies for 2024

In the ever-evolving landscape of search engine optimization, staying ahead requires mastering both traditional and cutting-edge techniques.

## Table of Contents

1. [Technical SEO Fundamentals](#technical-seo)
2. [Content Strategy Revolution](#content-strategy)
3. [Link Building in the Modern Era](#link-building)
4. [Emerging Trends](#emerging-trends)

## Technical SEO Fundamentals {#technical-seo}

Technical SEO remains the foundation of any successful optimization strategy. Key areas include:

### Core Web Vitals
Understanding and optimizing for [Core Web Vitals](/guides/core-web-vitals) is essential for modern SEO success.

### Schema Markup
Implementing comprehensive [schema markup](/technical/schema-guide) helps search engines understand your content.

## Content Strategy Revolution {#content-strategy}

Content strategy has evolved beyond simple keyword targeting. Modern approaches focus on:

- User intent optimization
- Topic clustering around [pillar content](/content/pillar-pages)
- Comprehensive coverage of [semantic keywords](/keywords/semantic-analysis)

### Content Clusters

Building effective [content clusters](/strategies/content-clusters) requires understanding topical authority and user journey mapping.

## Link Building in the Modern Era {#link-building}

Link building strategies have become more sophisticated, emphasizing:

1. [Digital PR techniques](/outreach/digital-pr)
2. [Resource page optimization](/link-building/resource-pages)
3. [Internal linking strategies](/internal-links/best-practices)

### Internal Linking Best Practices

Effective internal linking involves connecting related content through [strategic anchor text](/internal-links/anchor-text) and maintaining proper link equity distribution.

## Emerging Trends {#emerging-trends}

Several emerging trends are shaping the future of SEO:

- AI-powered content optimization
- Voice search optimization
- [Mobile-first indexing strategies](/mobile/indexing-guide)
- [Local SEO automation](/local/automation-tools)

## Conclusion

Mastering these advanced strategies requires continuous learning and adaptation. For implementation guidance, see our [SEO implementation checklist](/checklists/seo-implementation).

---

*This guide is part of our comprehensive [SEO mastery series](/series/seo-mastery).*
"""


def test_complex_content_parsing(complex_markdown_content):
    """Test parsing of complex markdown content with various elements."""
    parser = MarkdownParser()
    result = parser.parse_content(complex_markdown_content)

    # Test metadata extraction
    assert result['title'] == "Advanced SEO Strategies for 2024"
    assert result['metadata']['featured'] is True
    assert "seo" in result['keywords']

    # Test internal link extraction
    internal_links = result['internal_links']
    assert len(internal_links) >= 10  # Should find many internal links

    # Verify specific links are found
    link_urls = [link['url'] for link in internal_links]
    assert '/guides/core-web-vitals' in link_urls
    assert '/technical/schema-guide' in link_urls
    assert '/content/pillar-pages' in link_urls

    # Test content structure analysis
    assert result['heading_count'] >= 6  # Multiple headings
    assert result['word_count'] > 300  # Substantial content

    # Test quality metrics
    assert 'quality_metrics' in result
    quality = result['quality_metrics']
    assert quality['link_density'] > 0  # Should have good link density
    assert quality['heading_ratio'] > 0  # Good heading structure