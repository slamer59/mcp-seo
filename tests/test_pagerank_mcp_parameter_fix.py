"""
Test for PageRank MCP tools parameter handling issue.

This test reproduces and verifies the fix for the parameter validation error:
'Input validation error: ... is not of type 'object''
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import FastMCP

from mcp_seo.tools.graph.pagerank_tools import (
    LinkGraphRequest,
    PageRankRequest,
    register_pagerank_tools,
)


class TestMCPParameterHandling:
    """Test MCP parameter handling and validation."""

    def test_link_graph_request_validation(self):
        """Test LinkGraphRequest Pydantic model validation."""
        # Test valid request
        valid_request = LinkGraphRequest(
            domain="https://www.gitalchemy.app", max_pages=100, use_sitemap=True
        )
        assert str(valid_request.domain) == "https://www.gitalchemy.app/"
        assert valid_request.max_pages == 100
        assert valid_request.use_sitemap is True

    def test_pagerank_request_validation(self):
        """Test PageRankRequest Pydantic model validation."""
        # Test valid request
        valid_request = PageRankRequest(
            domain="https://www.gitalchemy.app/blog", max_pages=100, use_sitemap=True
        )
        assert str(valid_request.domain) == "https://www.gitalchemy.app/blog"
        assert valid_request.max_pages == 100
        assert valid_request.use_sitemap is True

    def test_json_string_parameter_parsing(self):
        """Test that JSON string parameters can be properly parsed into Pydantic models."""
        # This reproduces the original error case
        json_params = {
            "domain": "https://www.gitalchemy.app",
            "max_pages": 100,
            "use_sitemap": True,
        }

        # Test that we can create the model from dict
        request = LinkGraphRequest(**json_params)
        assert str(request.domain) == "https://www.gitalchemy.app"
        assert request.max_pages == 100
        assert request.use_sitemap is True

    @pytest.mark.asyncio
    async def test_mcp_tool_registration(self):
        """Test that MCP tools register correctly."""
        mcp = FastMCP("Test SEO Server")

        # Register the tools
        register_pagerank_tools(mcp)

        # Check that tools are registered
        tools = await mcp.get_tools()
        tool_names = (
            list(tools.keys())
            if isinstance(tools, dict)
            else [tool.name for tool in tools]
        )
        expected_tools = [
            "analyze_pagerank",
            "build_link_graph",
            "find_pillar_pages",
            "find_orphaned_pages",
            "optimize_internal_links",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Tool {tool_name} not registered"

    @pytest.mark.asyncio
    async def test_build_link_graph_with_mock(self):
        """Test build_link_graph tool with mocked dependencies."""
        mcp = FastMCP("Test SEO Server")
        register_pagerank_tools(mcp)

        # Mock the dependencies
        with (
            patch("mcp_seo.tools.graph.pagerank_tools.KuzuManager") as mock_kuzu,
            patch(
                "mcp_seo.tools.graph.pagerank_tools.LinkGraphBuilder"
            ) as mock_builder,
        ):
            # Setup mocks
            mock_kuzu_instance = mock_kuzu.return_value.__enter__.return_value
            mock_kuzu_instance.initialize_schema.return_value = None
            mock_kuzu_instance.get_page_data.return_value = [
                {"url": "https://www.gitalchemy.app/", "in_degree": 5, "out_degree": 3},
                {
                    "url": "https://www.gitalchemy.app/blog",
                    "in_degree": 2,
                    "out_degree": 8,
                },
            ]
            mock_kuzu_instance.get_links_data.return_value = [
                {
                    "source": "https://www.gitalchemy.app/",
                    "target": "https://www.gitalchemy.app/blog",
                }
            ]

            mock_builder_instance = mock_builder.return_value
            mock_builder_instance.build_link_graph_from_sitemap = AsyncMock(
                return_value={
                    "pages_crawled": 2,
                    "links_found": 1,
                    "crawl_time": 10.5,
                    "success": True,
                }
            )

            # Get the build_link_graph tool
            build_link_graph_tool = await mcp.get_tool("build_link_graph")

            assert build_link_graph_tool is not None, "build_link_graph tool not found"

            # Call the tool with individual parameters (new approach)
            result = await build_link_graph_tool.fn(
                domain="https://www.gitalchemy.app", max_pages=100, use_sitemap=True
            )

            # Verify result structure
            assert "domain" in result
            assert "graph_statistics" in result
            assert "basic_metrics" in result
            assert result["domain"] == "https://www.gitalchemy.app"
            assert result["graph_statistics"]["success"] is True

    def test_parameter_type_conversion(self):
        """Test parameter type handling that caused the original error."""
        # Simulate the original error scenario - parameters as JSON string
        json_string = '{"domain": "https://www.gitalchemy.app", "max_pages": 100, "use_sitemap": true}'

        # Parse JSON string to dict (this is what should happen in MCP handling)
        params_dict = json.loads(json_string)

        # Create request from parsed dict
        request = LinkGraphRequest(**params_dict)

        # Verify it works
        assert str(request.domain) == "https://www.gitalchemy.app"
        assert request.max_pages == 100
        assert request.use_sitemap is True

    def test_edge_cases_validation(self):
        """Test edge cases and validation errors."""

        # Test max_pages bounds
        with pytest.raises(ValueError):
            LinkGraphRequest(domain="https://example.com", max_pages=0)

        with pytest.raises(ValueError):
            LinkGraphRequest(domain="https://example.com", max_pages=1001)

        # Test required fields
        with pytest.raises(ValueError):
            LinkGraphRequest()  # Missing required domain            LinkGraphRequest()  # Missing required domain
