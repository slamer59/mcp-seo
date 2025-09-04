"""
Tests for NetworkX implementation fixes.

Tests the specific issues that were identified and fixed:
1. Import errors for community detection
2. NoneType errors with null checks
3. Complex function refactoring
4. Empty graph handling
5. Type issues
"""

import pytest
from unittest.mock import Mock, patch
import networkx as nx

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.networkx_analyzer import NetworkXAnalyzer


class TestNetworkXFixes:
    """Test NetworkX analyzer fixes."""

    def test_community_detection_imports(self):
        """Test that community detection imports work correctly."""
        # This should not raise ImportError
        from networkx.algorithms.community import louvain_communities, greedy_modularity_communities, modularity
        
        # Test with a simple graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        
        # Test Louvain algorithm
        communities_louvain = list(louvain_communities(G, seed=42))
        assert len(communities_louvain) > 0
        
        # Test greedy modularity
        communities_greedy = list(greedy_modularity_communities(G))
        assert len(communities_greedy) > 0
        
        # Test modularity calculation
        mod_score = modularity(G, communities_louvain)
        assert isinstance(mod_score, float)
        assert -1 <= mod_score <= 1
    
    def test_empty_graph_handling(self):
        """Test that empty graphs are handled gracefully."""
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = []
        mock_manager.get_links_data.return_value = []
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        # Test centrality analysis with empty graph
        result = analyzer.analyze_centrality()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
        
        # Test community detection with empty graph
        result = analyzer.detect_communities()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
        
        # Test path analysis with empty graph
        result = analyzer.analyze_paths()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
        
        # Test structural analysis with empty graph
        result = analyzer.analyze_structure()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
        
        # Test connector analysis with empty graph
        result = analyzer.find_connector_pages()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
    
    def test_single_node_graph_handling(self):
        """Test that single node graphs are handled gracefully."""
        pages_data = [
            {
                'url': 'https://example.com/',
                'title': 'Homepage',
                'path': '/',
                'status_code': 200,
                'in_degree': 0,
                'out_degree': 0,
                'pagerank': 1.0
            }
        ]
        links_data = []  # No links for single node
        
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = pages_data
        mock_manager.get_links_data.return_value = links_data
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        # Test centrality analysis with single node
        result = analyzer.analyze_centrality()
        assert 'error' not in result
        assert 'centrality_analysis' in result
        assert len(result['centrality_analysis']) == 1
        
        # Test community detection with single node (should create one community)
        result = analyzer.detect_communities()
        assert 'error' not in result
        assert 'communities' in result
        assert len(result['communities']) == 1
        assert result['communities'][0]['size'] == 1
    
    def test_graph_with_no_edges_handling(self):
        """Test graphs with nodes but no edges."""
        pages_data = [
            {
                'url': 'https://example.com/page1',
                'title': 'Page 1',
                'path': '/page1',
                'status_code': 200,
                'in_degree': 0,
                'out_degree': 0,
                'pagerank': 0.5
            },
            {
                'url': 'https://example.com/page2',
                'title': 'Page 2',
                'path': '/page2',
                'status_code': 200,
                'in_degree': 0,
                'out_degree': 0,
                'pagerank': 0.5
            }
        ]
        links_data = []  # No links between nodes
        
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = pages_data
        mock_manager.get_links_data.return_value = links_data
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        # Test community detection with isolated nodes
        result = analyzer.detect_communities()
        assert 'error' not in result
        assert 'communities' in result
        assert len(result['communities']) == 2  # Each node is its own community
        assert result['modularity_score'] == 0.0  # No edges means 0 modularity
    
    def test_null_graph_protection(self):
        """Test protection against None graphs."""
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = None
        mock_manager.get_links_data.return_value = None
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        # Test that build_networkx_graph handles None data
        result = analyzer.build_networkx_graph()
        assert result is False
        
        # Test that analysis functions handle failed graph building
        result = analyzer.analyze_centrality()
        assert 'error' in result
        assert 'failed to build graph' in result['error'].lower()
    
    def test_centrality_helper_functions(self):
        """Test the refactored centrality helper functions."""
        pages_data = [
            {
                'url': 'https://example.com/',
                'title': 'Homepage',
                'path': '/',
                'status_code': 200,
                'in_degree': 2,
                'out_degree': 2,
                'pagerank': 0.4
            },
            {
                'url': 'https://example.com/about',
                'title': 'About',
                'path': '/about',
                'status_code': 200,
                'in_degree': 1,
                'out_degree': 1,
                'pagerank': 0.3
            },
            {
                'url': 'https://example.com/contact',
                'title': 'Contact',
                'path': '/contact',
                'status_code': 200,
                'in_degree': 1,
                'out_degree': 1,
                'pagerank': 0.3
            }
        ]
        
        links_data = [
            {'source_url': 'https://example.com/', 'target_url': 'https://example.com/about'},
            {'source_url': 'https://example.com/', 'target_url': 'https://example.com/contact'},
            {'source_url': 'https://example.com/about', 'target_url': 'https://example.com/'},
            {'source_url': 'https://example.com/contact', 'target_url': 'https://example.com/'}
        ]
        
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = pages_data
        mock_manager.get_links_data.return_value = links_data
        
        analyzer = NetworkXAnalyzer(mock_manager)
        analyzer.build_networkx_graph()
        
        # Test individual helper functions
        centralities = analyzer._calculate_all_centralities()
        assert 'in_degree' in centralities
        assert 'out_degree' in centralities
        assert 'betweenness' in centralities
        assert 'closeness' in centralities
        assert 'eigenvector' in centralities
        assert 'katz' in centralities
        
        # Test combining results
        results = analyzer._combine_centrality_results(centralities)
        assert len(results) == 3
        for url, data in results.items():
            assert 'url' in data
            assert 'title' in data
            assert 'pagerank' in data
            assert 'betweenness_centrality' in data
    
    def test_community_edge_counting_with_nulls(self):
        """Test community edge counting with null protection."""
        pages_data = [
            {
                'url': 'https://example.com/page1',
                'title': 'Page 1',
                'path': '/page1',
                'status_code': 200,
                'in_degree': 1,
                'out_degree': 1,
                'pagerank': 0.5
            },
            {
                'url': 'https://example.com/page2',
                'title': 'Page 2',
                'path': '/page2',
                'status_code': 200,
                'in_degree': 1,
                'out_degree': 1,
                'pagerank': 0.5
            }
        ]
        
        links_data = [
            {'source_url': 'https://example.com/page1', 'target_url': 'https://example.com/page2'},
            {'source_url': 'https://example.com/page2', 'target_url': 'https://example.com/page1'}
        ]
        
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = pages_data
        mock_manager.get_links_data.return_value = links_data
        
        analyzer = NetworkXAnalyzer(mock_manager)
        analyzer.build_networkx_graph()
        
        # Test edge counting with valid community
        community = {'https://example.com/page1', 'https://example.com/page2'}
        internal_edges = analyzer._count_internal_edges(community)
        external_edges = analyzer._count_external_edges(community)
        
        assert internal_edges >= 0
        assert external_edges >= 0
        
        # Test edge counting with empty community
        empty_community = set()
        internal_edges = analyzer._count_internal_edges(empty_community)
        external_edges = analyzer._count_external_edges(empty_community)
        
        assert internal_edges == 0
        assert external_edges == 0
    
    def test_error_resilience(self):
        """Test that the analyzer is resilient to various error conditions."""
        pages_data = [
            {
                'url': 'https://example.com/page1',
                'title': 'Page 1',
                'path': '/page1',
                'status_code': 200,
                'in_degree': 1,
                'out_degree': 1,
                'pagerank': 0.5
            }
        ]
        
        links_data = [
            # This link references a non-existent target node
            {'source_url': 'https://example.com/page1', 'target_url': 'https://example.com/nonexistent'}
        ]
        
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = pages_data
        mock_manager.get_links_data.return_value = links_data
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        # The graph building should handle missing target nodes gracefully
        result = analyzer.build_networkx_graph()
        assert result is True
        
        # Analysis should still work with the valid nodes
        centrality_result = analyzer.analyze_centrality()
        assert 'error' not in centrality_result
        assert len(centrality_result['centrality_analysis']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])