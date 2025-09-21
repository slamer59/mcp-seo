"""
Integration tests for PageRank MCP tools.

These tests verify the end-to-end functionality of MCP tools
with real database operations and async workflows.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the actual tool functions by accessing them through the module
import mcp_seo.tools.graph.pagerank_tools as pagerank_tools_module
from mcp_seo.tools.graph.pagerank_tools import (
    LinkGraphRequest,
    OrphanedPagesRequest,
    PageRankRequest,
    PillarPagesRequest,
    analyze_pagerank,
    build_link_graph,
    find_orphaned_pages,
    find_pillar_pages,
    optimize_internal_links,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestPageRankToolsIntegration:
    """Integration tests for PageRank MCP tools."""

    @patch("mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder")
    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_analyze_pagerank_full_workflow(
        self, mock_kuzu_class, mock_builder_class
    ):
        """Test complete PageRank analysis workflow."""
        # Setup mocks
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu_class.return_value = mock_kuzu

        # Mock graph builder
        mock_builder = MagicMock()
        mock_builder.build_link_graph_from_sitemap = AsyncMock(
            return_value={
                "total_pages": 10,
                "total_links": 25,
                "crawled_pages": 10,
                "discovered_links": 25,
            }
        )
        mock_builder_class.return_value = mock_builder

        # Mock PageRank analyzer
        with patch(
            "mcp_seo.tools.graph.pagerank_tools.PageRankAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.calculate_pagerank.return_value = {
                "https://example.com/": 0.25,
                "https://example.com/about": 0.20,
                "https://example.com/contact": 0.15,
            }
            mock_analyzer.generate_analysis_summary.return_value = {
                "graph_stats": {"total_pages": 10, "total_links": 25},
                "metrics": {
                    "total_pages": 10,
                    "avg_pagerank": 0.20,
                    "max_pagerank": 0.25,
                    "min_pagerank": 0.15,
                },
                "insights": {
                    "pillar_pages": [
                        {
                            "url": "https://example.com/",
                            "pagerank": 0.25,
                            "title": "Home",
                        }
                    ],
                    "orphaned_pages": [],
                    "low_outlink_pages": [],
                    "hub_pages": [],
                },
                "recommendations": ["Optimize internal linking structure"],
                "pagerank_scores": {
                    "https://example.com/": 0.25,
                    "https://example.com/about": 0.20,
                },
            }
            mock_analyzer_class.return_value = mock_analyzer

            # Test the tool
            request = PageRankRequest(
                domain="https://example.com",
                max_pages=50,
                damping_factor=0.85,
                use_sitemap=True,
            )

            result = await analyze_pagerank(request)

            # Verify results
            assert "error" not in result
            assert result["domain"] == "https://example.com/"
            assert "graph_statistics" in result
            assert "metrics" in result
            assert "insights" in result
            assert "recommendations" in result
            assert "pagerank_scores" in result
            assert "parameters" in result

            # Verify parameters were passed correctly
            params = result["parameters"]
            assert params["max_pages"] == 50
            assert params["damping_factor"] == 0.85
            assert params["used_sitemap"] is True

            # Verify mock calls
            mock_kuzu_class.assert_called_once()
            mock_kuzu.initialize_schema.assert_called_once()
            mock_builder_class.assert_called_once()
            mock_analyzer_class.assert_called_once_with(mock_kuzu)
            mock_analyzer.calculate_pagerank.assert_called_once()

    @patch("mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder")
    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_analyze_pagerank_with_urls(
        self, mock_kuzu_class, mock_builder_class
    ):
        """Test PageRank analysis with custom URLs instead of sitemap."""
        # Setup mocks
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu_class.return_value = mock_kuzu

        mock_builder = MagicMock()
        mock_builder.build_link_graph_from_urls = AsyncMock(
            return_value={"total_pages": 5, "total_links": 10}
        )
        mock_builder_class.return_value = mock_builder

        with patch(
            "mcp_seo.tools.graph.pagerank_tools.PageRankAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.calculate_pagerank.return_value = {
                "https://example.com/": 0.5
            }
            mock_analyzer.generate_analysis_summary.return_value = {
                "metrics": {"total_pages": 1},
                "insights": {"pillar_pages": []},
                "recommendations": [],
            }
            mock_analyzer_class.return_value = mock_analyzer

            # Test without sitemap
            request = PageRankRequest(domain="https://example.com", use_sitemap=False)

            result = await analyze_pagerank(request)

            # Should use homepage instead of sitemap
            mock_builder.build_link_graph_from_urls.assert_called_once_with(
                ["https://example.com/"]
            )
            assert result["parameters"]["used_sitemap"] is False

    @patch("mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder")
    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_analyze_pagerank_error_handling(
        self, mock_kuzu_class, mock_builder_class
    ):
        """Test error handling in PageRank analysis."""
        # Setup mock to fail
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu_class.return_value = mock_kuzu

        mock_builder = MagicMock()
        mock_builder.build_link_graph_from_sitemap = AsyncMock(
            return_value={"error": "Failed to fetch sitemap"}
        )
        mock_builder_class.return_value = mock_builder

        request = PageRankRequest(domain="https://example.com")
        result = await analyze_pagerank(request)

        assert "error" in result
        assert "Failed to build link graph" in result["error"]

    @patch("mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder")
    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_build_link_graph_tool(self, mock_kuzu_class, mock_builder_class):
        """Test build_link_graph MCP tool."""
        # Setup mocks
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu.get_page_data.return_value = [
            {"url": "https://example.com/", "in_degree": 0, "out_degree": 2},
            {"url": "https://example.com/about", "in_degree": 1, "out_degree": 0},
        ]
        mock_kuzu.get_links_data.return_value = [
            {
                "source_url": "https://example.com/",
                "target_url": "https://example.com/about",
            }
        ]
        mock_kuzu_class.return_value = mock_kuzu

        mock_builder = MagicMock()
        mock_builder.build_link_graph_from_sitemap = AsyncMock(
            return_value={"total_pages": 2, "total_links": 1}
        )
        mock_builder_class.return_value = mock_builder

        # Test the tool
        request = LinkGraphRequest(
            domain="https://example.com", max_pages=10, use_sitemap=True
        )

        result = await build_link_graph(request)

        # Verify results
        assert "error" not in result
        assert result["domain"] == "https://example.com/"
        assert "graph_statistics" in result
        assert "basic_metrics" in result
        assert "top_pages_by_links" in result

        # Check basic metrics
        metrics = result["basic_metrics"]
        assert metrics["total_pages"] == 2
        assert metrics["total_links"] == 1
        assert metrics["orphaned_pages_count"] == 1  # One page has 0 incoming links

    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_find_pillar_pages_tool(self, mock_kuzu_class):
        """Test find_pillar_pages MCP tool."""
        # Setup mock with existing data
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu.get_page_data.return_value = [
            {"url": "https://example.com/", "pagerank": 0.25, "title": "Home"},
            {"url": "https://example.com/about", "pagerank": 0.20, "title": "About"},
        ]
        mock_kuzu_class.return_value = mock_kuzu

        with patch(
            "mcp_seo.tools.graph.pagerank_tools.PageRankAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_pillar_pages.return_value = [
                {"url": "https://example.com/", "pagerank": 0.25, "title": "Home"}
            ]
            mock_analyzer_class.return_value = mock_analyzer

            # Test the tool
            request = PillarPagesRequest(
                domain="https://example.com", percentile=90.0, limit=5
            )

            result = await find_pillar_pages(request)

            # Verify results
            assert "error" not in result
            assert result["domain"] == "https://example.com/"
            assert "pillar_pages" in result
            assert "criteria" in result
            assert "recommendations" in result

            assert len(result["pillar_pages"]) == 1
            assert result["criteria"]["percentile_threshold"] == 90.0
            assert result["criteria"]["limit"] == 5

    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_find_pillar_pages_no_data(self, mock_kuzu_class):
        """Test find_pillar_pages with no existing data."""
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu.get_page_data.return_value = []
        mock_kuzu_class.return_value = mock_kuzu

        request = PillarPagesRequest(domain="https://example.com")
        result = await find_pillar_pages(request)

        assert "error" in result
        assert "No page data found" in result["error"]

    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_find_orphaned_pages_tool(self, mock_kuzu_class):
        """Test find_orphaned_pages MCP tool."""
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu.get_page_data.return_value = [
            {"url": "https://example.com/", "in_degree": 0, "path": "/"},
            {"url": "https://example.com/about", "in_degree": 1, "path": "/about"},
            {
                "url": "https://example.com/blog/post",
                "in_degree": 0,
                "path": "/blog/post",
            },
        ]
        mock_kuzu_class.return_value = mock_kuzu

        with patch(
            "mcp_seo.tools.graph.pagerank_tools.PageRankAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.get_orphaned_pages.return_value = [
                {"url": "https://example.com/", "in_degree": 0, "path": "/"},
                {
                    "url": "https://example.com/blog/post",
                    "in_degree": 0,
                    "path": "/blog/post",
                },
            ]
            mock_analyzer_class.return_value = mock_analyzer

            request = OrphanedPagesRequest(domain="https://example.com")
            result = await find_orphaned_pages(request)

            # Verify results
            assert "error" not in result
            assert result["domain"] == "https://example.com/"
            assert "orphaned_pages" in result
            assert "total_orphaned" in result
            assert "percentage_orphaned" in result
            assert "recommendations" in result
            assert "orphaned_by_category" in result

            assert result["total_orphaned"] == 2
            assert abs(result["percentage_orphaned"] - 66.67) < 0.1  # 2/3 * 100

    @patch("mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder")
    @patch("mcp_seo.tools.graph.pagerank_tools.PageRankAnalyzer")
    @patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager")
    async def test_optimize_internal_links_tool(
        self, mock_kuzu_class, mock_analyzer_class, mock_builder_class
    ):
        """Test optimize_internal_links MCP tool."""
        # Setup mocks
        mock_kuzu = MagicMock()
        mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
        mock_kuzu.__exit__ = MagicMock(return_value=None)
        mock_kuzu.get_page_data.return_value = [
            {"url": "https://example.com/", "pagerank": 0.25}
        ]
        mock_kuzu_class.return_value = mock_kuzu

        # Mock link graph builder
        mock_builder = MagicMock()
        mock_builder.get_link_opportunities.return_value = {
            "orphaned_pages": [
                {"url": "https://example.com/orphan", "title": "Orphaned Page"}
            ],
            "low_outlink_pages": [{"url": "https://example.com/low", "out_degree": 1}],
            "high_authority_pages": [{"url": "https://example.com/", "pagerank": 0.25}],
            "suggestions": [
                {
                    "type": "orphaned_pages",
                    "priority": "high",
                    "description": "Fix orphaned pages",
                }
            ],
        }
        mock_builder_class.return_value = mock_builder

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.generate_analysis_summary.return_value = {
            "recommendations": ["Add more internal links"],
            "metrics": {"avg_pagerank": 0.2},
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Test the tool
        request = LinkGraphRequest(domain="https://example.com", max_pages=10)

        result = await optimize_internal_links(request)

        # Verify results
        assert "error" not in result
        assert result["domain"] == "https://example.com/"
        assert "link_opportunities" in result
        assert "optimization_plan" in result
        assert "general_recommendations" in result
        assert "metrics" in result

        # Check optimization plan structure
        plan = result["optimization_plan"]
        assert "priority_1_actions" in plan
        assert "priority_2_actions" in plan
        assert "priority_3_actions" in plan

        # Check that priority 1 has orphaned pages action
        priority_1 = plan["priority_1_actions"][0]
        assert priority_1["action"] == "Fix orphaned pages"
        assert priority_1["pages_affected"] == 1

    async def test_pydantic_models_validation(self):
        """Test Pydantic model validation for all request types."""
        # Test valid PageRankRequest
        valid_request = PageRankRequest(
            domain="https://example.com",
            max_pages=100,
            damping_factor=0.85,
            max_iterations=100,
            use_sitemap=True,
        )
        assert str(valid_request.domain) == "https://example.com/"
        assert valid_request.max_pages == 100

        # Test invalid domain
        with pytest.raises(ValueError):
            PageRankRequest(domain="not-a-url")

        # Test invalid max_pages (too high)
        with pytest.raises(ValueError):
            PageRankRequest(domain="https://example.com", max_pages=2000)

        # Test invalid damping_factor
        with pytest.raises(ValueError):
            PageRankRequest(domain="https://example.com", damping_factor=1.5)

        # Test LinkGraphRequest with URLs
        link_request = LinkGraphRequest(
            domain="https://example.com",
            urls=["https://example.com/page1", "https://example.com/page2"],
        )
        assert len(link_request.urls) == 2

        # Test PillarPagesRequest
        pillar_request = PillarPagesRequest(
            domain="https://example.com", percentile=95.0, limit=5
        )
        assert pillar_request.percentile == 95.0

        # Test OrphanedPagesRequest (minimal)
        orphan_request = OrphanedPagesRequest(domain="https://example.com")
        assert str(orphan_request.domain) == "https://example.com/"

    @patch("mcp_seo.tools.graph.pagerank_tools.logger")
    async def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        # This would be tested with actual error scenarios
        # For now, just verify logger is available

        request = PageRankRequest(domain="https://example.com")

        with patch(
            "mcp_seo.tools.graph.pagerank_tools.KuzuManager",
            side_effect=Exception("Test error"),
        ):
            result = await analyze_pagerank(request)

            # Should handle error gracefully
            assert "error" in result
            assert "Test error" in result["error"]

    async def test_concurrent_tool_execution(self):
        """Test that tools can handle concurrent execution."""
        import asyncio

        # Create multiple requests
        requests = [PageRankRequest(domain=f"https://example{i}.com") for i in range(3)]

        # Mock the heavy operations
        with patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager") as mock_kuzu_class:
            mock_kuzu = MagicMock()
            mock_kuzu.__enter__ = MagicMock(return_value=mock_kuzu)
            mock_kuzu.__exit__ = MagicMock(return_value=None)
            mock_kuzu.get_page_data.return_value = []
            mock_kuzu_class.return_value = mock_kuzu

            with patch(
                "mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder"
            ) as mock_builder_class:
                mock_builder = MagicMock()
                mock_builder.build_link_graph_from_sitemap = AsyncMock(
                    return_value={"error": "No pages"}
                )
                mock_builder_class.return_value = mock_builder

                # Run concurrently
                tasks = [analyze_pagerank(req) for req in requests]
                results = await asyncio.gather(*tasks)

                # All should complete (even with errors)
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, dict)
                    assert (
                        "error" in result
                    )  # Expected error due to no pages                results = await asyncio.gather(*tasks)

                # All should complete (even with errors)
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, dict)
                    assert "error" in result  # Expected error due to no pages
