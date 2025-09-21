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
    def markdown_parser(self, tmp_path):
        """Create MarkdownParser instance."""
        return MarkdownParser(content_dir=tmp_path)

    def test_parse_content_basic(self, markdown_parser, sample_markdown_content, tmp_path):
        """Test basic content parsing functionality."""
        # Create a temporary file with the content
        test_file = tmp_path / "test_post.md"
        test_file.write_text(sample_markdown_content)

        result = markdown_parser.parse_single_file(test_file)

        assert result is not None
        assert result['title'] == "Test Blog Post"
        assert result['description'] == "A test blog post for SEO analysis"
        assert 'seo' in result['keywords']
        assert len(result['internal_links']) > 0
        assert result['word_count'] > 50

    def test_extract_metadata(self, markdown_parser, sample_markdown_content, tmp_path):
        """Test metadata extraction from frontmatter."""
        # Create a temporary file with the content
        test_file = tmp_path / "test_post.md"
        test_file.write_text(sample_markdown_content)

        result = markdown_parser.parse_single_file(test_file)

        assert result['frontmatter']['title'] == "Test Blog Post"
        assert result['frontmatter']['date'] == "2024-01-15"
        assert "seo" in result['frontmatter']['tags']

    def test_extract_internal_links(self, markdown_parser, sample_markdown_content, tmp_path):
        """Test internal link extraction."""
        # Create a temporary file with the content
        test_file = tmp_path / "test_post.md"
        test_file.write_text(sample_markdown_content)

        result = markdown_parser.parse_single_file(test_file)

        internal_links = result['internal_links']
        assert len(internal_links) >= 3

        # Check specific links are found
        link_targets = [link['target_slug'] for link in internal_links]
        assert 'internal-page' in link_targets
        assert 'topics-cluster' in link_targets
        assert 'guides-best-practices' in link_targets

    def test_content_quality_metrics(self, markdown_parser, sample_markdown_content, tmp_path):
        """Test content quality metric calculation."""
        # Create a temporary file with the content
        test_file = tmp_path / "test_post.md"
        test_file.write_text(sample_markdown_content)

        result = markdown_parser.parse_single_file(test_file)

        assert result['word_count'] > 0
        assert result['quality_metrics']['header_count'] > 0
        assert result['quality_metrics']['link_count'] > 0
        assert 'readability_score' in result['quality_metrics']

    def test_parse_file_with_missing_frontmatter(self, markdown_parser, tmp_path):
        """Test parsing content without frontmatter."""
        content_without_frontmatter = """
# Simple Post

This is just basic content without any frontmatter.

[Link to page](/simple-link)
"""
        # Create a temporary file with the content
        test_file = tmp_path / "simple_post.md"
        test_file.write_text(content_without_frontmatter)

        result = markdown_parser.parse_single_file(test_file)

        assert result is not None
        assert result.get('title') is not None  # Should extract from first heading
        assert len(result['internal_links']) > 0

    def test_empty_content_handling(self, markdown_parser, tmp_path):
        """Test handling of empty or invalid content."""
        # Create a temporary file with empty content
        test_file = tmp_path / "empty_post.md"
        test_file.write_text("")

        result = markdown_parser.parse_single_file(test_file)

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
                "keywords": ["seo", "optimization", "content"],
                "word_count": 1200,
                "internal_links": [{"target_slug": "/post-2", "anchor_text": "content marketing"}],
                "quality_metrics": {"readability_score": 0.8, "header_count": 3, "image_count": 1}
            },
            "/post-2": {
                "title": "Content Marketing Guide",
                "keywords": ["content", "marketing", "seo"],
                "word_count": 800,
                "internal_links": [{"target_slug": "/post-3", "anchor_text": "link building"}],
                "quality_metrics": {"readability_score": 0.7, "header_count": 2, "image_count": 0}
            },
            "/post-3": {
                "title": "Link Building Strategies",
                "keywords": ["links", "building", "seo"],
                "word_count": 600,
                "internal_links": [],
                "quality_metrics": {"readability_score": 0.6, "header_count": 1, "image_count": 0}
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
            assert 'category' in rec
            assert 'priority' in rec
            assert 'action' in rec
            assert 'details' in rec


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
                "title": "High Authority Page",
                "keywords": ["authority", "content", "seo"],
                "topics": ["seo", "optimization"],
                "word_count": 1500,
                "internal_links": [
                    {"target_slug": "/medium-authority", "anchor_text": "related content"}
                ],
                "quality_metrics": {
                    "readability_score": 80,
                    "header_count": 5
                }
            },
            "/medium-authority": {
                "title": "Medium Authority Page",
                "keywords": ["content", "marketing", "strategy"],
                "topics": ["marketing", "content"],
                "word_count": 1200,
                "internal_links": [
                    {"target_slug": "/low-authority", "anchor_text": "basics guide"}
                ],
                "quality_metrics": {
                    "readability_score": 75,
                    "header_count": 4
                }
            },
            "/low-authority": {
                "title": "Low Authority Page",
                "keywords": ["beginner", "seo", "basics"],
                "topics": ["seo", "basics"],
                "word_count": 800,
                "internal_links": [],
                "quality_metrics": {
                    "readability_score": 70,
                    "header_count": 3
                }
            }
        }

    @pytest.fixture
    def link_optimizer(self, sample_content_data):
        """Create LinkOptimizer instance."""
        # Create sample metrics data
        sample_metrics = {
            "/high-authority": {
                "pagerank": 0.35,
                "authority_score": 0.8,
                "in_degree": 2,
                "out_degree": 1
            },
            "/medium-authority": {
                "pagerank": 0.20,
                "authority_score": 0.6,
                "in_degree": 1,
                "out_degree": 1
            },
            "/low-authority": {
                "pagerank": 0.10,
                "authority_score": 0.3,
                "in_degree": 1,
                "out_degree": 0
            }
        }
        return LinkOptimizer(sample_content_data, sample_metrics)

    def test_identify_link_opportunities(self, link_optimizer):
        """Test link opportunity discovery."""
        opportunities = link_optimizer.identify_link_opportunities(max_opportunities=10, min_relevance_score=0.1)

        assert isinstance(opportunities, list)
        for opp in opportunities:
            assert isinstance(opp, LinkOpportunity)
            assert opp.source_slug is not None
            assert opp.target_slug is not None
            assert opp.relevance_score >= 0.1

    def test_analyze_link_equity_flow(self, link_optimizer):
        """Test link equity flow analysis."""
        result = link_optimizer.analyze_link_equity_flow()

        assert 'total_links' in result
        assert 'total_pages' in result
        assert 'page_flow' in result
        assert 'flow_health_score' in result

    def test_identify_cluster_opportunities(self, link_optimizer):
        """Test cluster-based linking analysis."""
        clusters = link_optimizer.identify_cluster_opportunities(min_cluster_size=2, min_cluster_strength=0.01)

        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, ClusterOpportunity)
            assert cluster.cluster_keyword is not None
            assert len(cluster.cluster_pages) >= 2

    def test_generate_implementation_plan(self, link_optimizer):
        """Test implementation plan generation."""
        # First get some opportunities
        link_opportunities = link_optimizer.identify_link_opportunities(max_opportunities=5)
        cluster_opportunities = link_optimizer.identify_cluster_opportunities(min_cluster_size=2)

        plan = link_optimizer.generate_implementation_plan(link_opportunities, cluster_opportunities)

        assert 'implementation_phases' in plan
        assert 'action_items' in plan
        assert 'expected_roi' in plan

    def test_link_opportunity_analysis(self, link_optimizer):
        """Test link opportunity analysis methods."""
        # Test the private method that analyzes individual opportunities
        source_post = {
            "title": "Source Post",
            "keywords": ["seo", "content"],
            "word_count": 1000,
            "quality_metrics": {"readability_score": 80}
        }
        target_post = {
            "title": "Target Post",
            "keywords": ["seo", "optimization"],
            "word_count": 800,
            "quality_metrics": {"readability_score": 75}
        }
        source_keywords = {"seo", "content"}

        opportunity = link_optimizer._analyze_link_opportunity(
            "source", source_post, "target", target_post, source_keywords
        )

        if opportunity:  # Only test if opportunity is found
            assert opportunity.source_slug == "source"
            assert opportunity.target_slug == "target"
            assert opportunity.relevance_score >= 0


class TestContentAnalysisIntegration:
    """Integration tests for content analysis workflow."""

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
        # Initialize components - use the temp directory as content_dir
        temp_dir = Path(list(temp_markdown_files.values())[0]).parent
        parser = MarkdownParser(content_dir=temp_dir)

        # Parse all files
        parsed_content = {}
        for post_id, file_path in temp_markdown_files.items():
            parsed_content[post_id] = parser.parse_single_file(file_path)

        # Verify parsed content
        assert len(parsed_content) == 3
        for post_id, data in parsed_content.items():
            assert data['title'].startswith("Test Post")
            assert data['word_count'] > 0
            assert len(data['internal_links']) > 0

    @pytest.fixture
    def mock_kuzu_manager(self):
        """Mock KuzuManager for integration testing."""
        mock_manager = Mock()
        mock_manager.get_all_pages.return_value = [
            {"url": "/high-authority", "title": "High Authority Page", "pagerank": 0.35},
            {"url": "/medium-authority", "title": "Medium Authority Page", "pagerank": 0.20},
            {"url": "/low-authority", "title": "Low Authority Page", "pagerank": 0.10},
        ]
        mock_manager.get_all_links.return_value = [
            {"source": "/high-authority", "target": "/medium-authority", "anchor_text": "related content"},
            {"source": "/medium-authority", "target": "/low-authority", "anchor_text": "basics guide"},
        ]
        return mock_manager

    @pytest.fixture
    def sample_content_data(self):
        """Sample content data for integration testing."""
        return {
            "/high-authority": {
                "title": "High Authority Page",
                "keywords": ["authority", "content", "seo"],
                "word_count": 1500,
                "internal_links": [
                    {"target_slug": "/medium-authority", "anchor_text": "related content"}
                ],
                "quality_metrics": {
                    "readability_score": 80,
                    "header_count": 5
                }
            },
            "/medium-authority": {
                "title": "Medium Authority Page",
                "keywords": ["content", "marketing", "strategy"],
                "word_count": 1200,
                "internal_links": [
                    {"target_slug": "/low-authority", "anchor_text": "basics guide"}
                ],
                "quality_metrics": {
                    "readability_score": 75,
                    "header_count": 4
                }
            },
            "/low-authority": {
                "title": "Low Authority Page",
                "keywords": ["beginner", "seo", "basics"],
                "word_count": 800,
                "internal_links": [],
                "quality_metrics": {
                    "readability_score": 70,
                    "header_count": 3
                }
            }
        }

    def test_content_analysis_with_link_optimization(self, sample_content_data):
        """Test integrated content analysis with link optimization."""
        # Initialize analyzer and optimizer
        analyzer = BlogAnalyzer(sample_content_data)
        sample_metrics = {
            "/high-authority": {
                "pagerank": 0.35,
                "authority_score": 0.8,
                "in_degree": 2,
                "out_degree": 1
            },
            "/medium-authority": {
                "pagerank": 0.20,
                "authority_score": 0.6,
                "in_degree": 1,
                "out_degree": 1
            },
            "/low-authority": {
                "pagerank": 0.10,
                "authority_score": 0.3,
                "in_degree": 1,
                "out_degree": 0
            }
        }
        link_optimizer = LinkOptimizer(sample_content_data, sample_metrics)

        # Create sample graph metrics for the analyzer
        sample_nodes = [
            {"slug": slug, **data} for slug, data in sample_content_data.items()
        ]
        sample_edges = [
            {"source": "/high-authority", "target": "/medium-authority"},
            {"source": "/medium-authority", "target": "/low-authority"}
        ]

        # Calculate metrics and generate analysis
        metrics = analyzer.calculate_networkx_metrics(sample_nodes, sample_edges)
        analysis = analyzer.generate_comprehensive_analysis(metrics)

        # Find linking opportunities
        link_opportunities = link_optimizer.identify_link_opportunities(max_opportunities=5)

        # Verify integration
        assert 'pillar_pages' in analysis
        assert 'recommendations' in analysis
        assert isinstance(link_opportunities, list)


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


def test_complex_content_parsing(complex_markdown_content, tmp_path):
    """Test parsing of complex markdown content with various elements."""
    parser = MarkdownParser(content_dir=tmp_path)

    # Create a temporary file with the content
    test_file = tmp_path / "complex_post.md"
    test_file.write_text(complex_markdown_content)

    result = parser.parse_single_file(test_file)

    # Test metadata extraction
    assert result['title'] == "Advanced SEO Strategies for 2024"
    assert result['frontmatter']['featured'] is True
    assert "seo" in result['keywords']

    # Test internal link extraction
    internal_links = result['internal_links']
    assert len(internal_links) >= 10  # Should find many internal links

    # Verify specific links are found
    link_targets = [link['target_slug'] for link in internal_links]
    assert 'guides-core-web-vitals' in link_targets
    assert 'technical-schema-guide' in link_targets
    assert 'content-pillar-pages' in link_targets

    # Test content structure analysis
    assert result['quality_metrics']['header_count'] >= 6  # Multiple headings
    assert result['word_count'] > 200  # Substantial content

    # Test quality metrics
    assert 'quality_metrics' in result
    quality = result['quality_metrics']
    assert quality['link_count'] > 0  # Should have good link count
    assert quality['structure_score'] > 0  # Good heading structure