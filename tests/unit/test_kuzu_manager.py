"""
Unit tests for KuzuManager class.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mcp_seo.graph.kuzu_manager import KuzuManager


@pytest.mark.unit
class TestKuzuManager:
    """Test cases for KuzuManager."""

    def test_init_with_path(self, temp_db_path):
        """Test initialization with specific database path."""
        manager = KuzuManager(temp_db_path)
        assert manager.db_path == temp_db_path
        assert not manager.is_temp
        manager.close()

    def test_init_without_path(self):
        """Test initialization with temporary database."""
        manager = KuzuManager()
        assert manager.db_path is not None
        assert manager.is_temp
        assert Path(manager.db_path).name == "seo_graph.db"
        manager.close()

    def test_context_manager(self, temp_db_path):
        """Test context manager functionality."""
        with KuzuManager(temp_db_path) as manager:
            assert manager.connection is not None
            assert manager.database is not None

    def test_initialize_schema(self, kuzu_manager):
        """Test database schema initialization."""
        # Schema should be initialized by fixture
        assert kuzu_manager._schema_initialized
        
        # Test that pages can be queried (schema exists)
        result = kuzu_manager.connection.execute("MATCH (p:Page) RETURN COUNT(p) as count")
        count = next(result)[0]
        assert count == 0  # No pages added yet

    def test_add_page(self, kuzu_manager):
        """Test adding a single page."""
        url = "https://example.com/test"
        title = "Test Page"
        status_code = 200
        content_length = 1000

        kuzu_manager.add_page(url, title, status_code, content_length)

        # Verify page was added
        result = kuzu_manager.connection.execute(
            "MATCH (p:Page {url: $url}) RETURN p.title, p.status_code, p.content_length",
            parameters={"url": url}
        )
        row = next(result)
        assert row[0] == title
        assert row[1] == status_code
        assert row[2] == content_length

    def test_add_page_minimal(self, kuzu_manager):
        """Test adding page with minimal data."""
        url = "https://example.com/minimal"
        kuzu_manager.add_page(url)

        # Verify page was added with defaults
        result = kuzu_manager.connection.execute(
            "MATCH (p:Page {url: $url}) RETURN p.title, p.status_code",
            parameters={"url": url}
        )
        row = next(result)
        assert row[0] == "minimal"  # Should derive from path
        assert row[1] == 0

    def test_add_pages_batch(self, kuzu_manager, sample_pages_data):
        """Test adding multiple pages in batch."""
        kuzu_manager.add_pages_batch(sample_pages_data)

        # Verify all pages were added
        result = kuzu_manager.connection.execute("MATCH (p:Page) RETURN COUNT(p) as count")
        count = next(result)[0]
        assert count == len(sample_pages_data)

    def test_add_link(self, kuzu_manager):
        """Test adding a single link."""
        source = "https://example.com/page1"
        target = "https://example.com/page2"
        anchor_text = "Link text"

        kuzu_manager.add_link(source, target, anchor_text)

        # Verify link was added
        result = kuzu_manager.connection.execute(
            """MATCH (s:Page)-[l:Links]->(t:Page) 
               WHERE s.url = $source AND t.url = $target
               RETURN l.anchor_text""",
            parameters={"source": source, "target": target}
        )
        row = next(result)
        assert row[0] == anchor_text

    def test_add_links_batch(self, kuzu_manager, sample_links_data):
        """Test adding multiple links in batch."""
        # First add some pages
        pages = [
            {"url": "https://example.com/", "title": "Home"},
            {"url": "https://example.com/about", "title": "About"},
            {"url": "https://example.com/contact", "title": "Contact"},
        ]
        kuzu_manager.add_pages_batch(pages)

        # Add links
        links_subset = [link for link in sample_links_data if 
                       all(url in ["https://example.com/", "https://example.com/about", "https://example.com/contact"] 
                           for url in [link[0], link[1]])]
        kuzu_manager.add_links_batch(links_subset)

        # Verify links were added
        result = kuzu_manager.connection.execute("MATCH ()-[l:Links]->() RETURN COUNT(l) as count")
        count = next(result)[0]
        assert count == len(links_subset)

    def test_calculate_degree_centrality(self, kuzu_manager, sample_pages_data, sample_links_data):
        """Test degree centrality calculation."""
        # Add data
        kuzu_manager.add_pages_batch(sample_pages_data)
        kuzu_manager.add_links_batch(sample_links_data)

        # Calculate centrality
        kuzu_manager.calculate_degree_centrality()

        # Verify degree calculations
        result = kuzu_manager.connection.execute(
            """MATCH (p:Page) 
               RETURN p.url, p.in_degree, p.out_degree
               ORDER BY p.in_degree DESC"""
        )
        
        pages_with_degrees = list(result)
        assert len(pages_with_degrees) == len(sample_pages_data)
        
        # Check that degrees are calculated (not all zero)
        total_in_degrees = sum(row[1] for row in pages_with_degrees)
        total_out_degrees = sum(row[2] for row in pages_with_degrees)
        assert total_in_degrees > 0
        assert total_out_degrees > 0
        
        # Total in-degrees should equal total out-degrees (number of links)
        assert total_in_degrees == total_out_degrees == len(sample_links_data)

    def test_get_page_data(self, populated_kuzu_manager):
        """Test getting page data."""
        pages = populated_kuzu_manager.get_page_data()
        
        assert len(pages) == 5
        for page in pages:
            assert 'url' in page
            assert 'title' in page
            assert 'pagerank' in page
            assert 'in_degree' in page
            assert 'out_degree' in page
            assert page['pagerank'] >= 0.0

    def test_get_links_data(self, populated_kuzu_manager):
        """Test getting links data."""
        links = populated_kuzu_manager.get_links_data()
        
        assert len(links) == 8  # From sample_links_data
        for link in links:
            assert 'source_url' in link
            assert 'target_url' in link
            assert 'anchor_text' in link
            assert 'position' in link

    def test_get_incoming_links(self, populated_kuzu_manager):
        """Test getting incoming links for a page."""
        # Test for a page that should have incoming links
        incoming = populated_kuzu_manager.get_incoming_links("https://example.com/contact")
        
        assert len(incoming) > 0
        for link in incoming:
            assert 'source_url' in link
            assert 'source_pagerank' in link
            assert 'source_out_degree' in link
            assert 'anchor_text' in link

    def test_update_pagerank_scores(self, populated_kuzu_manager):
        """Test updating PageRank scores."""
        scores = {
            "https://example.com/": 0.25,
            "https://example.com/about": 0.20,
            "https://example.com/contact": 0.15,
        }
        
        populated_kuzu_manager.update_pagerank_scores(scores)
        
        # Verify scores were updated
        for url, expected_score in scores.items():
            result = populated_kuzu_manager.connection.execute(
                "MATCH (p:Page {url: $url}) RETURN p.pagerank",
                parameters={"url": url}
            )
            actual_score = next(result)[0]
            assert abs(actual_score - expected_score) < 1e-10

    def test_get_graph_stats(self, populated_kuzu_manager):
        """Test getting graph statistics."""
        stats = populated_kuzu_manager.get_graph_stats()
        
        assert 'total_pages' in stats
        assert 'total_links' in stats
        assert 'total_domains' in stats
        
        assert stats['total_pages'] == 5
        assert stats['total_links'] == 8
        assert stats['total_domains'] == 1

    def test_close_cleanup(self, temp_db_path):
        """Test proper cleanup on close."""
        # Test with temporary database
        manager = KuzuManager()
        temp_path = Path(manager.db_path)
        temp_dir = temp_path.parent
        
        manager.connect()
        assert manager.connection is not None
        
        manager.close()
        assert manager.connection is None
        assert manager.database is None

    def test_error_handling_invalid_operation(self, kuzu_manager):
        """Test error handling for invalid operations."""
        with pytest.raises(Exception):
            # Try to execute invalid Cypher
            kuzu_manager.connection.execute("INVALID CYPHER QUERY")

    def test_connection_not_established_error(self):
        """Test error when operations are called without connection."""
        manager = KuzuManager()  # Not connected
        
        with pytest.raises(RuntimeError, match="Database connection not established"):
            manager.add_page("https://example.com/test")

    @patch('mcp_seo.graph.kuzu_manager.kuzu')
    def test_connection_failure(self, mock_kuzu, temp_db_path):
        """Test handling of connection failures."""
        mock_kuzu.Database.side_effect = Exception("Connection failed")
        
        manager = KuzuManager(temp_db_path)
        with pytest.raises(Exception, match="Connection failed"):
            manager.connect()