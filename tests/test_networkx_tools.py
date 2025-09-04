"""
Tests for NetworkX analysis tools.

Tests comprehensive graph analysis functionality including centrality analysis,
community detection, path optimization, structural analysis, and connector identification.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.networkx_analyzer import NetworkXAnalyzer


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    pages_data = [
        {
            'url': 'https://example.com/',
            'title': 'Homepage',
            'path': '/',
            'status_code': 200,
            'in_degree': 3,
            'out_degree': 5,
            'pagerank': 0.25
        },
        {
            'url': 'https://example.com/about',
            'title': 'About Us',
            'path': '/about',
            'status_code': 200,
            'in_degree': 2,
            'out_degree': 3,
            'pagerank': 0.15
        },
        {
            'url': 'https://example.com/blog',
            'title': 'Blog',
            'path': '/blog',
            'status_code': 200,
            'in_degree': 4,
            'out_degree': 2,
            'pagerank': 0.20
        },
        {
            'url': 'https://example.com/blog/post1',
            'title': 'Blog Post 1',
            'path': '/blog/post1',
            'status_code': 200,
            'in_degree': 1,
            'out_degree': 1,
            'pagerank': 0.10
        },
        {
            'url': 'https://example.com/contact',
            'title': 'Contact',
            'path': '/contact',
            'status_code': 200,
            'in_degree': 1,
            'out_degree': 0,
            'pagerank': 0.08
        },
        {
            'url': 'https://example.com/products',
            'title': 'Products',
            'path': '/products',
            'status_code': 200,
            'in_degree': 2,
            'out_degree': 4,
            'pagerank': 0.18
        }
    ]
    
    links_data = [
        {'source_url': 'https://example.com/', 'target_url': 'https://example.com/about'},
        {'source_url': 'https://example.com/', 'target_url': 'https://example.com/blog'},
        {'source_url': 'https://example.com/', 'target_url': 'https://example.com/products'},
        {'source_url': 'https://example.com/', 'target_url': 'https://example.com/contact'},
        {'source_url': 'https://example.com/about', 'target_url': 'https://example.com/'},
        {'source_url': 'https://example.com/about', 'target_url': 'https://example.com/contact'},
        {'source_url': 'https://example.com/blog', 'target_url': 'https://example.com/'},
        {'source_url': 'https://example.com/blog', 'target_url': 'https://example.com/blog/post1'},
        {'source_url': 'https://example.com/blog/post1', 'target_url': 'https://example.com/blog'},
        {'source_url': 'https://example.com/products', 'target_url': 'https://example.com/'},
        {'source_url': 'https://example.com/products', 'target_url': 'https://example.com/about'}
    ]
    
    return pages_data, links_data


@pytest.fixture
def mock_kuzu_manager(sample_graph_data):
    """Mock KuzuManager with sample data."""
    pages_data, links_data = sample_graph_data
    
    mock_manager = Mock()
    mock_manager.get_page_data.return_value = pages_data
    mock_manager.get_links_data.return_value = links_data
    mock_manager.get_graph_stats.return_value = {
        'total_pages': len(pages_data),
        'total_links': len(links_data)
    }
    
    return mock_manager


class TestNetworkXAnalyzer:
    """Test NetworkXAnalyzer functionality."""
    
    def test_build_networkx_graph_success(self, mock_kuzu_manager):
        """Test successful graph building."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.build_networkx_graph()
        
        assert result is True
        assert analyzer.graph is not None
        assert analyzer.undirected_graph is not None
        assert analyzer.graph.number_of_nodes() == 6
        assert analyzer.graph.number_of_edges() == 11
    
    def test_build_networkx_graph_no_data(self, mock_kuzu_manager):
        """Test graph building with no data."""
        mock_kuzu_manager.get_page_data.return_value = []
        mock_kuzu_manager.get_links_data.return_value = []
        
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        result = analyzer.build_networkx_graph()
        
        assert result is False
    
    def test_analyze_centrality_success(self, mock_kuzu_manager):
        """Test centrality analysis."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.analyze_centrality()
        
        assert 'error' not in result
        assert 'centrality_analysis' in result
        assert 'insights' in result
        assert 'total_pages' in result
        assert result['total_pages'] == 6
        
        # Check that centrality metrics are calculated
        centrality_data = result['centrality_analysis']
        assert len(centrality_data) == 6
        
        # Check that insights are generated
        insights = result['insights']
        assert isinstance(insights, list)
        assert len(insights) > 0
    
    def test_detect_communities_success(self, mock_kuzu_manager):
        """Test community detection."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.detect_communities('greedy_modularity')
        
        assert 'error' not in result
        assert 'communities' in result
        assert 'modularity_score' in result
        assert 'num_communities' in result
        assert 'insights' in result
        
        # Check communities structure
        communities = result['communities']
        assert isinstance(communities, list)
        assert len(communities) > 0
        
        # Each community should have required fields
        for community in communities:
            assert 'community_id' in community
            assert 'size' in community
            assert 'pages' in community
            assert 'total_authority' in community
    
    def test_analyze_paths_success(self, mock_kuzu_manager):
        """Test path analysis."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.analyze_paths()
        
        assert 'error' not in result
        assert 'path_metrics' in result
        assert 'strongly_connected_components' in result
        assert 'connectivity_ratio' in result
        assert 'hard_to_reach_pages' in result
        assert 'easy_to_reach_pages' in result
        assert 'insights' in result
    
    def test_analyze_structure_success(self, mock_kuzu_manager):
        """Test structural analysis."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.analyze_structure()
        
        assert 'error' not in result
        assert 'k_core_decomposition' in result
        assert 'clustering_analysis' in result
        assert 'critical_pages' in result
        assert 'graph_density' in result
        assert 'structural_insights' in result
        
        # Check k-core structure
        k_core = result['k_core_decomposition']
        assert 'max_core_number' in k_core
        assert 'cores' in k_core
        
        # Check clustering analysis
        clustering = result['clustering_analysis']
        assert 'average_clustering' in clustering
        assert 'transitivity' in clustering
    
    def test_find_connector_pages_success(self, mock_kuzu_manager):
        """Test connector pages identification."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        
        result = analyzer.find_connector_pages()
        
        assert 'error' not in result
        assert 'connector_pages' in result
        assert 'bridge_opportunities' in result
        assert 'cross_section_connections' in result
        assert 'insights' in result
        
        # Check connector pages structure
        connectors = result['connector_pages']
        for connector in connectors:
            assert 'url' in connector
            assert 'betweenness_centrality' in connector
            assert 'pagerank' in connector
    
    def test_error_handling_no_graph(self):
        """Test error handling when graph building fails."""
        mock_manager = Mock()
        mock_manager.get_page_data.return_value = None
        mock_manager.get_links_data.return_value = None
        
        analyzer = NetworkXAnalyzer(mock_manager)
        
        result = analyzer.analyze_centrality()
        assert 'error' in result
        
        result = analyzer.detect_communities()
        assert 'error' in result
        
        result = analyzer.analyze_paths()
        assert 'error' in result
        
        result = analyzer.analyze_structure()
        assert 'error' in result
        
        result = analyzer.find_connector_pages()
        assert 'error' in result


class TestNetworkXTools:
    """Test NetworkX MCP tools integration."""
    
    def test_tools_registration_successful(self):
        """Test that NetworkX tools can be registered without errors."""
        from mcp_seo.tools.graph.networkx_tools import register_networkx_tools
        from fastmcp import FastMCP
        
        # Should not raise any exceptions
        mcp = FastMCP("test")
        register_networkx_tools(mcp)
        
        # Test that we can import required models
        from mcp_seo.tools.graph.networkx_tools import (
            CentralityAnalysisRequest,
            CommunityDetectionRequest,
            ConnectorAnalysisRequest,
            NetworkXAnalysisRequest
        )
        
        # Basic validation that models work
        centrality_req = CentralityAnalysisRequest(domain="https://example.com")
        assert str(centrality_req.domain) == "https://example.com/"
        
        community_req = CommunityDetectionRequest(domain="https://example.com")
        assert community_req.algorithm == "louvain"
        
        connector_req = ConnectorAnalysisRequest(domain="https://example.com")
        assert connector_req.min_betweenness == 0.0


class TestInsightGeneration:
    """Test insight generation functions."""
    
    def test_centrality_insights(self, mock_kuzu_manager):
        """Test centrality insights generation."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        result = analyzer.analyze_centrality()
        
        insights = result.get('insights', [])
        
        # Should have insights
        assert len(insights) > 0
        
        # Insights should be strings
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0
    
    def test_community_insights(self, mock_kuzu_manager):
        """Test community detection insights."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        result = analyzer.detect_communities()
        
        insights = result.get('insights', [])
        
        # Should have insights about communities
        assert len(insights) > 0
        
        # Should mention communities or clustering
        insight_text = ' '.join(insights).lower()
        assert any(keyword in insight_text for keyword in ['communit', 'cluster', 'modular'])
    
    def test_path_insights(self, mock_kuzu_manager):
        """Test path analysis insights."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        result = analyzer.analyze_paths()
        
        insights = result.get('insights', [])
        
        # Should have insights about navigation
        assert len(insights) > 0
        
        # Should mention navigation or paths
        insight_text = ' '.join(insights).lower()
        assert any(keyword in insight_text for keyword in ['navigation', 'path', 'click', 'reach'])
    
    def test_structural_insights(self, mock_kuzu_manager):
        """Test structural analysis insights."""
        analyzer = NetworkXAnalyzer(mock_kuzu_manager)
        result = analyzer.analyze_structure()
        
        insights = result.get('structural_insights', [])
        
        # Should have structural insights
        assert len(insights) > 0
        
        # Should mention structure or architecture
        insight_text = ' '.join(insights).lower()
        assert any(keyword in insight_text for keyword in ['structure', 'core', 'cluster', 'densit'])


@pytest.mark.integration
class TestIntegrationWithRealData:
    """Integration tests with real Kuzu database."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        # This would require actual Kuzu database setup
        # For now, we'll skip this test unless specifically enabled
        pytest.skip("Integration test requires real database setup")
    
    def test_performance_with_large_graph(self):
        """Test performance with larger graphs."""
        # This would test scalability with larger datasets
        pytest.skip("Performance test requires large dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])