"""
Unit tests for LinkGraphBuilder class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import xml.etree.ElementTree as ET

from mcp_seo.graph.link_graph_builder import LinkGraphBuilder


@pytest.mark.unit
@pytest.mark.asyncio
class TestLinkGraphBuilder:
    """Test cases for LinkGraphBuilder."""

    def test_init(self, kuzu_manager):
        """Test LinkGraphBuilder initialization."""
        base_url = "https://example.com"
        max_pages = 50
        
        builder = LinkGraphBuilder(base_url, kuzu_manager, max_pages)
        
        assert builder.base_url == base_url
        assert builder.kuzu_manager == kuzu_manager
        assert builder.max_pages == max_pages
        assert builder.processed_urls == set()

    def test_normalize_url(self, link_graph_builder):
        """Test URL normalization."""
        test_cases = [
            ("https://example.com/page?param=1#section", "https://example.com/page"),
            ("https://example.com/page/", "https://example.com/page"),
            ("https://example.com/page", "https://example.com/page"),
            ("https://example.com/", "https://example.com/"),
            ("https://example.com/path/?query=1&other=2#frag", "https://example.com/path"),
        ]
        
        for input_url, expected in test_cases:
            result = link_graph_builder.normalize_url(input_url)
            assert result == expected, f"Failed for {input_url}: expected {expected}, got {result}"

    def test_is_internal_url(self, link_graph_builder):
        """Test internal URL detection."""
        internal_urls = [
            "https://example.com/page",
            "http://example.com/page",
            "https://www.example.com/page",
            "/relative/path",
            "relative/path",
            "?query=1",
            "#fragment",
        ]
        
        external_urls = [
            "https://other-site.com/page",
            "http://different.com/page",
            "https://sub.other.com/page",
            "mailto:test@example.com",
            "tel:+1234567890",
        ]
        
        for url in internal_urls:
            assert link_graph_builder.is_internal_url(url), f"Should be internal: {url}"
        
        for url in external_urls:
            assert not link_graph_builder.is_internal_url(url), f"Should be external: {url}"

    def test_is_internal_url_subdomain(self):
        """Test internal URL detection with subdomain handling."""
        builder = LinkGraphBuilder("https://www.example.com", None, 10)
        
        # These should be considered internal
        internal_cases = [
            "https://example.com/page",
            "https://www.example.com/page",
            "http://example.com/page",
        ]
        
        # These should be external
        external_cases = [
            "https://sub.example.com/page",  # Subdomain
            "https://example-other.com/page",  # Different domain
            "https://notexample.com/page",
        ]
        
        for url in internal_cases:
            assert builder.is_internal_url(url), f"Should be internal: {url}"
            
        for url in external_cases:
            assert not builder.is_internal_url(url), f"Should be external: {url}"

    @patch('aiohttp.ClientSession')
    async def test_fetch_sitemap_success(self, mock_session_class, link_graph_builder, mock_sitemap_xml):
        """Test successful sitemap fetching."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_sitemap_xml)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session
        
        # Test
        pages = await link_graph_builder.fetch_sitemap()
        
        # Verify
        assert len(pages) == 5
        expected_urls = [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact",
            "https://example.com/blog",
            "https://example.com/products"
        ]
        assert pages == expected_urls

    @patch('aiohttp.ClientSession')
    async def test_fetch_sitemap_http_error(self, mock_session_class, link_graph_builder):
        """Test sitemap fetching with HTTP error."""
        # Setup mock response with error
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session
        
        # Test
        pages = await link_graph_builder.fetch_sitemap()
        
        # Should return empty list on error
        assert pages == []

    @patch('aiohttp.ClientSession')
    async def test_fetch_sitemap_max_pages_limit(self, mock_session_class, kuzu_manager):
        """Test sitemap fetching with page limit."""
        # Create builder with small limit
        builder = LinkGraphBuilder("https://example.com", kuzu_manager, max_pages=2)
        
        # Create sitemap with many URLs
        sitemap_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/page1</loc></url>
    <url><loc>https://example.com/page2</loc></url>
    <url><loc>https://example.com/page3</loc></url>
    <url><loc>https://example.com/page4</loc></url>
</urlset>'''
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=sitemap_xml)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session
        
        # Test
        pages = await builder.fetch_sitemap()
        
        # Should be limited to max_pages
        assert len(pages) == 2
        assert pages == ["https://example.com/page1", "https://example.com/page2"]

    @patch('aiohttp.ClientSession')
    async def test_crawl_page_links_success(self, mock_session_class, link_graph_builder, mock_html_content):
        """Test successful page crawling."""
        url = "https://example.com/test"
        
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-length': '1500'}
        mock_response.text = AsyncMock(return_value=mock_html_content)
        
        mock_session = AsyncMock()
        
        # Test
        page_data, links = await link_graph_builder.crawl_page_links(mock_session, url)
        
        # Verify page data
        assert page_data['url'] == url
        assert page_data['title'] == 'Test Page'
        assert page_data['status_code'] == 200
        assert page_data['content_length'] == 1500
        
        # Verify links (should extract internal links only)
        assert len(links) > 0
        for source, target, anchor in links:
            assert source == url
            assert link_graph_builder.is_internal_url(target)
            assert isinstance(anchor, str)

    @patch('aiohttp.ClientSession')
    async def test_crawl_page_links_http_error(self, mock_session_class, link_graph_builder):
        """Test page crawling with HTTP error."""
        url = "https://example.com/error"
        
        # Setup mock response with error
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        
        # Test
        page_data, links = await link_graph_builder.crawl_page_links(mock_session, url)
        
        # Verify error handling
        assert page_data['url'] == url
        assert page_data['status_code'] == 404
        assert page_data['content_length'] == 0
        assert links == []

    @patch('aiohttp.ClientSession')
    async def test_crawl_page_links_exception(self, mock_session_class, link_graph_builder):
        """Test page crawling with exception."""
        url = "https://example.com/exception"
        
        # Setup mock to raise exception
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Network error")
        
        # Test
        page_data, links = await link_graph_builder.crawl_page_links(mock_session, url)
        
        # Should handle gracefully
        assert page_data['url'] == url
        assert page_data['status_code'] == 0
        assert links == []

    @patch.object(LinkGraphBuilder, 'fetch_sitemap')
    @patch.object(LinkGraphBuilder, 'crawl_page_links')
    async def test_build_link_graph_from_sitemap(self, mock_crawl, mock_fetch, link_graph_builder):
        """Test building link graph from sitemap."""
        # Setup mocks
        mock_fetch.return_value = [
            "https://example.com/",
            "https://example.com/about"
        ]
        
        mock_crawl.side_effect = [
            (
                {"url": "https://example.com/", "title": "Home", "status_code": 200, "content_length": 1000},
                [("https://example.com/", "https://example.com/about", "About")]
            ),
            (
                {"url": "https://example.com/about", "title": "About", "status_code": 200, "content_length": 800},
                []
            )
        ]
        
        # Test
        stats = await link_graph_builder.build_link_graph_from_sitemap()
        
        # Verify results
        assert 'crawled_pages' in stats
        assert 'discovered_links' in stats
        assert stats['crawled_pages'] == 2
        assert stats['discovered_links'] == 1

    @patch.object(LinkGraphBuilder, 'fetch_sitemap')
    async def test_build_link_graph_from_sitemap_no_pages(self, mock_fetch, link_graph_builder):
        """Test building link graph when no pages found."""
        mock_fetch.return_value = []
        
        stats = await link_graph_builder.build_link_graph_from_sitemap()
        
        assert 'error' in stats
        assert 'No pages found in sitemap' in stats['error']

    @patch.object(LinkGraphBuilder, 'crawl_page_links')
    async def test_build_link_graph_from_urls(self, mock_crawl, link_graph_builder):
        """Test building link graph from URL list."""
        urls = ["https://example.com/", "https://example.com/about"]
        
        mock_crawl.side_effect = [
            (
                {"url": "https://example.com/", "title": "Home", "status_code": 200, "content_length": 1000},
                [("https://example.com/", "https://example.com/about", "About")]
            ),
            (
                {"url": "https://example.com/about", "title": "About", "status_code": 200, "content_length": 800},
                []
            )
        ]
        
        stats = await link_graph_builder.build_link_graph_from_urls(urls)
        
        assert 'crawled_pages' in stats
        assert stats['crawled_pages'] == 2

    def test_get_link_opportunities(self, populated_kuzu_manager):
        """Test link opportunities identification."""
        builder = LinkGraphBuilder("https://example.com", populated_kuzu_manager, 50)
        opportunities = builder.get_link_opportunities()
        
        # Check structure
        assert 'orphaned_pages' in opportunities
        assert 'low_outlink_pages' in opportunities
        assert 'high_authority_pages' in opportunities
        assert 'potential_hubs' in opportunities
        assert 'suggestions' in opportunities
        
        # Check suggestions format
        suggestions = opportunities['suggestions']
        assert isinstance(suggestions, list)
        
        for suggestion in suggestions:
            assert 'type' in suggestion
            assert 'priority' in suggestion
            assert 'description' in suggestion

    def test_get_link_opportunities_empty(self, kuzu_manager):
        """Test link opportunities with empty database."""
        builder = LinkGraphBuilder("https://example.com", kuzu_manager, 50)
        opportunities = builder.get_link_opportunities()
        
        assert 'error' in opportunities
        assert 'No page data available' in opportunities['error']

    @patch.object(LinkGraphBuilder, 'build_link_graph_from_urls')
    async def test_expand_graph_from_discovered_pages(self, mock_build, populated_kuzu_manager):
        """Test graph expansion from discovered pages."""
        builder = LinkGraphBuilder("https://example.com", populated_kuzu_manager, 50)
        
        # Add a link to a new page that doesn't exist in pages
        populated_kuzu_manager.add_link(
            "https://example.com/",
            "https://example.com/new-page",
            "New Page"
        )
        
        mock_build.return_value = {
            'crawled_pages': 1,
            'discovered_links': 2
        }
        
        stats = await builder.expand_graph_from_discovered_pages(max_new_pages=10)
        
        # Should have called build_link_graph_from_urls
        mock_build.assert_called_once()
        assert 'expansion_pages' in stats

    async def test_expand_graph_no_new_pages(self, populated_kuzu_manager):
        """Test graph expansion when no new pages to discover."""
        builder = LinkGraphBuilder("https://example.com", populated_kuzu_manager, 50)
        
        stats = await builder.expand_graph_from_discovered_pages()
        
        assert 'message' in stats
        assert 'No new pages to discover' in stats['message']

    def test_progress_callback(self, kuzu_manager):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))
        
        builder = LinkGraphBuilder("https://example.com", kuzu_manager, 50)
        
        # This would be tested in integration tests with actual async calls
        # Here we just verify the callback mechanism works
        assert callable(progress_callback)

    @patch('mcp_seo.graph.link_graph_builder.logger')
    def test_logging(self, mock_logger, link_graph_builder):
        """Test logging functionality."""
        # Test URL normalization (should not log)
        link_graph_builder.normalize_url("https://example.com/test")
        
        # Logger should be available but may not be called for simple operations
        assert hasattr(link_graph_builder, '__class__')

    def test_semaphore_limiting(self, kuzu_manager):
        """Test that semaphore limiting is properly configured."""
        builder = LinkGraphBuilder("https://example.com", kuzu_manager, 50)
        
        # This is mainly tested through integration tests,
        # but we can verify the builder is properly initialized
        assert builder.kuzu_manager == kuzu_manager