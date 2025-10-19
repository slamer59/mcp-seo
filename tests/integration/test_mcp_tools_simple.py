"""
Simplified integration tests for PageRank MCP tools.

These tests verify the core functionality without complex mocking.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder
from mcp_seo.tools.graph.pagerank_tools import (
    PageRankRequest, LinkGraphRequest, PillarPagesRequest, OrphanedPagesRequest
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPToolsSimple:
    """Simplified integration tests for MCP tools."""

    async def test_full_pagerank_workflow_integration(self):
        """Test the complete PageRank workflow with real components."""
        # Use temporary database
        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()
            
            # Add sample data
            sample_pages = [
                {"url": "https://example.com/", "title": "Home", "status_code": 200},
                {"url": "https://example.com/about", "title": "About", "status_code": 200},
                {"url": "https://example.com/contact", "title": "Contact", "status_code": 200},
            ]
            
            sample_links = [
                ("https://example.com/", "https://example.com/about", "About Us"),
                ("https://example.com/", "https://example.com/contact", "Contact"),
                ("https://example.com/about", "https://example.com/contact", "Get in touch"),
            ]
            
            kuzu_manager.add_pages_batch(sample_pages)
            kuzu_manager.add_links_batch(sample_links)
            kuzu_manager.calculate_degree_centrality()
            
            # Test PageRank calculation
            pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
            scores = pagerank_analyzer.calculate_pagerank(max_iterations=20)
            
            # Verify PageRank results
            assert len(scores) == 3
            assert all(score > 0 for score in scores.values())
            assert abs(sum(scores.values()) - 1.0) < 0.01  # Should sum to ~1.0
            
            # Test analysis summary
            summary = pagerank_analyzer.generate_analysis_summary()
            assert 'metrics' in summary
            assert 'insights' in summary
            assert 'recommendations' in summary
            
            # Test pillar pages identification  
            pillar_pages = pagerank_analyzer.get_pillar_pages(percentile=50, limit=2)
            assert len(pillar_pages) <= 2
            if pillar_pages:
                assert all('pagerank' in page for page in pillar_pages)
            
            # Test orphaned pages identification
            orphaned_pages = pagerank_analyzer.get_orphaned_pages()
            assert isinstance(orphaned_pages, list)
            
            # Test link opportunities
            builder = LinkGraphBuilder("https://example.com", kuzu_manager, 50)
            opportunities = builder.get_link_opportunities()
            assert 'orphaned_pages' in opportunities
            assert 'suggestions' in opportunities

    async def test_kuzu_manager_lifecycle(self):
        """Test KuzuManager proper lifecycle management."""
        # Test context manager
        with KuzuManager() as manager:
            manager.initialize_schema()
            
            # Add some data
            manager.add_page("https://test.com/", "Test Page")
            
            # Verify data exists
            pages = manager.get_page_data()
            assert len(pages) == 1
            assert pages[0]['url'] == "https://test.com/"
            
        # Manager should be properly cleaned up after context exit

    async def test_pagerank_mathematical_properties(self):
        """Test PageRank mathematical correctness."""
        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()
            
            # Create a simple graph: A -> B -> C -> A (cycle)
            pages = [
                {"url": "https://example.com/a", "title": "Page A"},
                {"url": "https://example.com/b", "title": "Page B"},
                {"url": "https://example.com/c", "title": "Page C"},
            ]
            
            links = [
                ("https://example.com/a", "https://example.com/b", "Link AB"),
                ("https://example.com/b", "https://example.com/c", "Link BC"),
                ("https://example.com/c", "https://example.com/a", "Link CA"),
            ]
            
            kuzu_manager.add_pages_batch(pages)
            kuzu_manager.add_links_batch(links)
            kuzu_manager.calculate_degree_centrality()
            
            analyzer = PageRankAnalyzer(kuzu_manager)
            
            # Test different damping factors
            scores_085 = analyzer.calculate_pagerank(damping_factor=0.85, max_iterations=50)
            scores_050 = analyzer.calculate_pagerank(damping_factor=0.50, max_iterations=50)
            
            # Both should converge to valid results
            assert len(scores_085) == 3
            assert len(scores_050) == 3
            
            # For symmetric graph, scores should be roughly equal
            values_085 = list(scores_085.values())
            values_050 = list(scores_050.values())
            
            # Check that scores are reasonable
            for score in values_085:
                assert 0.1 < score < 0.7  # Reasonable range for 3 pages
            
            # Sum should be approximately 1.0
            assert abs(sum(values_085) - 1.0) < 0.001
            assert abs(sum(values_050) - 1.0) < 0.001

    async def test_link_graph_builder_url_handling(self):
        """Test LinkGraphBuilder URL normalization and filtering."""
        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()
            
            builder = LinkGraphBuilder("https://example.com", kuzu_manager, 10)
            
            # Test URL normalization
            test_urls = [
                "https://example.com/page?param=1#fragment",
                "https://example.com/page/",
                "https://example.com/page",
            ]
            
            normalized = [builder.normalize_url(url) for url in test_urls]
            
            # All should normalize to the same URL
            expected = "https://example.com/page"
            assert all(url == expected for url in normalized)
            
            # Test internal/external URL detection
            internal_urls = [
                "https://example.com/internal",
                "/relative/path",
                "relative-link.html"
            ]
            
            external_urls = [
                "https://other-site.com/page",
                "mailto:test@example.com",
                "tel:+1234567890"
            ]
            
            for url in internal_urls:
                assert builder.is_internal_url(url), f"Should be internal: {url}"
            
            for url in external_urls:
                assert not builder.is_internal_url(url), f"Should be external: {url}"

    async def test_error_handling_resilience(self):
        """Test error handling and resilience of components."""
        # Test KuzuManager with invalid operations
        with KuzuManager() as manager:
            manager.initialize_schema()
            
            # Test adding invalid data
            try:
                manager.add_page("", "")  # Empty URL
                pages = manager.get_page_data()
                # Should handle gracefully
                assert isinstance(pages, list)
            except Exception:
                # Acceptable to raise exception for invalid input
                pass
            
            # Test PageRank with no data
            analyzer = PageRankAnalyzer(manager)
            scores = analyzer.calculate_pagerank()
            assert scores == {}  # Should return empty dict, not crash
            
            # Test analysis with no data
            summary = analyzer.generate_analysis_summary()
            assert 'error' in summary or 'metrics' in summary

    async def test_concurrent_operations(self):
        """Test that operations can run concurrently without conflicts."""
        async def analyze_sample_data(instance_id):
            with KuzuManager() as manager:
                manager.initialize_schema()
                
                # Add unique data for this instance
                pages = [
                    {"url": f"https://site{instance_id}.com/", "title": f"Site {instance_id}"},
                    {"url": f"https://site{instance_id}.com/about", "title": f"About {instance_id}"},
                ]
                
                links = [
                    (f"https://site{instance_id}.com/", f"https://site{instance_id}.com/about", "About")
                ]
                
                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()
                
                analyzer = PageRankAnalyzer(manager)
                scores = analyzer.calculate_pagerank(max_iterations=10)
                
                return len(scores)
        
        # Run multiple analyses concurrently
        tasks = [analyze_sample_data(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert all(result == 2 for result in results)

    async def test_pydantic_request_models(self):
        """Test Pydantic request model validation."""
        # Test valid requests
        pagerank_req = PageRankRequest(
            domain="https://example.com",
            max_pages=50,
            damping_factor=0.85
        )
        assert str(pagerank_req.domain) == "https://example.com"
        assert pagerank_req.max_pages == 50
        
        
        with pytest.raises(ValueError):
            PageRankRequest(domain="https://example.com", max_pages=2000)  # Too high
        
        with pytest.raises(ValueError):
            PageRankRequest(domain="https://example.com", damping_factor=1.5)  # Invalid range
        
        # Test other request types
        link_req = LinkGraphRequest(domain="https://example.com")
        assert link_req.use_sitemap is True  # Default
        
        pillar_req = PillarPagesRequest(domain="https://example.com")
        assert pillar_req.percentile == 90.0  # Default
        
        orphan_req = OrphanedPagesRequest(domain="https://example.com")
        assert str(orphan_req.domain).endswith("/")

    def test_graph_stats_accuracy(self):
        """Test that graph statistics are accurate."""
        with KuzuManager() as manager:
            manager.initialize_schema()
            
            # Add known data
            pages = [
                {"url": "https://example.com/1", "title": "Page 1"},
                {"url": "https://example.com/2", "title": "Page 2"},
                {"url": "https://different.com/3", "title": "Page 3"},  # Different domain
            ]
            
            links = [
                ("https://example.com/1", "https://example.com/2", "Link"),
                ("https://example.com/2", "https://different.com/3", "External"),
            ]
            
            manager.add_pages_batch(pages)
            manager.add_links_batch(links)
            
            stats = manager.get_graph_stats()
            
            assert stats['total_pages'] == 3
            assert stats['total_links'] == 2
            assert stats['total_domains'] == 2  # example.com and different.com
            
            # Test degree calculations
            manager.calculate_degree_centrality()
            page_data = manager.get_page_data()
            
            # Find degrees for known pages
            page1_data = next(p for p in page_data if p['url'] == "https://example.com/1")
            page2_data = next(p for p in page_data if p['url'] == "https://example.com/2")
            
            assert page1_data['out_degree'] == 1  # Links to page 2
            assert page1_data['in_degree'] == 0   # No incoming links
            assert page2_data['out_degree'] == 1  # Links to page 3
            assert page2_data['in_degree'] == 1   # Link from page 1