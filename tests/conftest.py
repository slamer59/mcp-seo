"""
Pytest configuration and fixtures for MCP Data4SEO tests.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path that gets cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="test_kuzu_")
    db_path = str(Path(temp_dir) / "test.db")
    yield db_path
    # Cleanup is handled by KuzuManager context manager


@pytest.fixture
def kuzu_manager(temp_db_path):
    """Create a KuzuManager instance with temporary database."""
    with KuzuManager(temp_db_path) as manager:
        manager.initialize_schema()
        yield manager


@pytest.fixture
def sample_pages_data():
    """Sample page data for testing."""
    return [
        {
            "url": "https://example.com/",
            "title": "Home Page",
            "status_code": 200,
            "content_length": 1000
        },
        {
            "url": "https://example.com/about",
            "title": "About Us",
            "status_code": 200,
            "content_length": 800
        },
        {
            "url": "https://example.com/contact",
            "title": "Contact",
            "status_code": 200,
            "content_length": 500
        },
        {
            "url": "https://example.com/blog",
            "title": "Blog",
            "status_code": 200,
            "content_length": 1200
        },
        {
            "url": "https://example.com/products",
            "title": "Products",
            "status_code": 200,
            "content_length": 900
        }
    ]


@pytest.fixture
def sample_links_data():
    """Sample links data for testing."""
    return [
        ("https://example.com/", "https://example.com/about", "About Us"),
        ("https://example.com/", "https://example.com/contact", "Contact"),
        ("https://example.com/", "https://example.com/blog", "Blog"),
        ("https://example.com/", "https://example.com/products", "Products"),
        ("https://example.com/about", "https://example.com/contact", "Get in touch"),
        ("https://example.com/blog", "https://example.com/about", "Learn more"),
        ("https://example.com/blog", "https://example.com/products", "Our products"),
        ("https://example.com/products", "https://example.com/contact", "Contact us"),
    ]


@pytest.fixture
def populated_kuzu_manager(kuzu_manager, sample_pages_data, sample_links_data):
    """KuzuManager populated with sample data."""
    kuzu_manager.add_pages_batch(sample_pages_data)
    kuzu_manager.add_links_batch(sample_links_data)
    kuzu_manager.calculate_degree_centrality()
    return kuzu_manager


@pytest.fixture
def pagerank_analyzer(populated_kuzu_manager):
    """PageRankAnalyzer with populated data."""
    return PageRankAnalyzer(populated_kuzu_manager)


@pytest.fixture
def link_graph_builder(kuzu_manager):
    """LinkGraphBuilder instance."""
    return LinkGraphBuilder(
        base_url="https://example.com",
        kuzu_manager=kuzu_manager,
        max_pages=50
    )


@pytest.fixture
def mock_sitemap_xml():
    """Mock sitemap XML content."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/</loc>
    </url>
    <url>
        <loc>https://example.com/about</loc>
    </url>
    <url>
        <loc>https://example.com/contact</loc>
    </url>
    <url>
        <loc>https://example.com/blog</loc>
    </url>
    <url>
        <loc>https://example.com/products</loc>
    </url>
</urlset>'''


@pytest.fixture
def mock_html_content():
    """Mock HTML content with links."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <meta name="description" content="Test page description">
    </head>
    <body>
        <nav>
            <a href="/about" title="About Us">About</a>
            <a href="/contact">Contact Us</a>
            <a href="/blog">Blog</a>
        </nav>
        <main>
            <h1>Welcome to Test Page</h1>
            <p>This is a test page with <a href="/products">products</a>.</p>
            <a href="https://external-site.com/page">External Link</a>
        </main>
        <footer>
            <a href="/privacy">Privacy Policy</a>
        </footer>
    </body>
    </html>
    '''


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing web requests."""
    session_mock = AsyncMock()
    
    # Mock response for sitemap
    sitemap_response_mock = AsyncMock()
    sitemap_response_mock.status = 200
    sitemap_response_mock.text = AsyncMock(return_value='''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/</loc></url>
    <url><loc>https://example.com/about</loc></url>
</urlset>''')
    
    # Mock response for HTML pages
    html_response_mock = AsyncMock()
    html_response_mock.status = 200
    html_response_mock.text = AsyncMock(return_value='''
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </nav>
    </body>
    </html>
    ''')
    
    # Configure the session mock
    session_mock.get.return_value.__aenter__.return_value = html_response_mock
    
    return session_mock


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status=200, text="", headers=None):
        self.status = status
        self._text = text
        self.headers = headers or {}
    
    async def text(self):
        return self._text
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def expected_pagerank_results():
    """Expected PageRank results for validation."""
    return {
        "total_pages_min": 4,
        "total_links_min": 5,
        "max_pagerank_threshold": 0.5,  # Should be reasonable
        "min_pagerank_threshold": 0.0,  # Should be non-negative
    }


# Test categories
pytestmark = pytest.mark.asyncio