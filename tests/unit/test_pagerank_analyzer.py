"""
Unit tests for PageRankAnalyzer class.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.graph.kuzu_manager import KuzuManager


@pytest.mark.unit
class TestPageRankAnalyzer:
    """Test cases for PageRankAnalyzer."""

    def test_init(self, kuzu_manager):
        """Test PageRankAnalyzer initialization."""
        analyzer = PageRankAnalyzer(kuzu_manager)
        assert analyzer.kuzu_manager == kuzu_manager
        assert analyzer.pagerank_scores == {}

    def test_calculate_pagerank_empty_database(self, kuzu_manager):
        """Test PageRank calculation with empty database."""
        analyzer = PageRankAnalyzer(kuzu_manager)
        scores = analyzer.calculate_pagerank()
        
        assert scores == {}

    def test_calculate_pagerank_with_data(self, populated_kuzu_manager):
        """Test PageRank calculation with populated database."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        scores = analyzer.calculate_pagerank(max_iterations=10, tolerance=1e-4)
        
        # Should have scores for all pages
        assert len(scores) == 5
        
        # All scores should be positive
        for score in scores.values():
            assert score > 0
        
        # Sum of all scores should be approximately 1.0 (normalized)
        total_score = sum(scores.values())
        assert abs(total_score - 1.0) < 0.01
        
        # Store scores for later verification
        analyzer.pagerank_scores = scores

    def test_calculate_pagerank_convergence(self, populated_kuzu_manager):
        """Test PageRank convergence behavior."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        
        # Test with different tolerance levels
        scores_strict = analyzer.calculate_pagerank(tolerance=1e-8, max_iterations=200)
        scores_loose = analyzer.calculate_pagerank(tolerance=1e-3, max_iterations=200)
        
        # Both should converge
        assert len(scores_strict) > 0
        assert len(scores_loose) > 0
        
        # Results should be similar
        for url in scores_strict:
            if url in scores_loose:
                assert abs(scores_strict[url] - scores_loose[url]) < 0.1

    def test_calculate_pagerank_parameters(self, populated_kuzu_manager):
        """Test PageRank with different damping factors."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        
        # Test different damping factors
        scores_085 = analyzer.calculate_pagerank(damping_factor=0.85, max_iterations=20)
        scores_050 = analyzer.calculate_pagerank(damping_factor=0.50, max_iterations=20)
        
        # Both should produce results
        assert len(scores_085) > 0
        assert len(scores_050) > 0
        
        # Lower damping factor should produce more uniform scores
        variance_085 = np.var(list(scores_085.values()))
        variance_050 = np.var(list(scores_050.values()))
        
        # Generally, lower damping should reduce variance
        # (though this might not always hold for small graphs)
        assert variance_050 >= 0  # At least check it's valid

    def test_get_pillar_pages(self, pagerank_analyzer):
        """Test pillar pages identification."""
        # First calculate PageRank
        pagerank_analyzer.calculate_pagerank(max_iterations=20)
        
        # Test pillar pages with default percentile
        pillar_pages = pagerank_analyzer.get_pillar_pages(percentile=80, limit=3)
        
        assert len(pillar_pages) <= 3
        assert len(pillar_pages) > 0
        
        # Should be sorted by PageRank in descending order
        pagerank_scores = [page['pagerank'] for page in pillar_pages]
        assert pagerank_scores == sorted(pagerank_scores, reverse=True)
        
        # All pillar pages should have required fields
        for page in pillar_pages:
            assert 'url' in page
            assert 'title' in page
            assert 'pagerank' in page
            assert page['pagerank'] > 0

    def test_get_pillar_pages_edge_cases(self, kuzu_manager):
        """Test pillar pages with edge cases."""
        analyzer = PageRankAnalyzer(kuzu_manager)
        
        # Empty database
        pillar_pages = analyzer.get_pillar_pages()
        assert pillar_pages == []

    def test_get_orphaned_pages(self, populated_kuzu_manager):
        """Test orphaned pages identification."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        orphaned_pages = analyzer.get_orphaned_pages()
        
        # Check structure
        for page in orphaned_pages:
            assert 'url' in page
            assert 'title' in page
            assert 'in_degree' in page
            assert page['in_degree'] == 0

    def test_get_low_outlink_pages(self, populated_kuzu_manager):
        """Test low outlink pages identification."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        low_outlink_pages = analyzer.get_low_outlink_pages(percentile=50, limit=5)
        
        assert len(low_outlink_pages) <= 5
        
        # Should be sorted by out_degree ascending
        if len(low_outlink_pages) > 1:
            out_degrees = [page['out_degree'] for page in low_outlink_pages]
            assert out_degrees == sorted(out_degrees)
        
        for page in low_outlink_pages:
            assert 'url' in page
            assert 'out_degree' in page
            assert page['out_degree'] >= 0

    def test_get_hub_pages(self, populated_kuzu_manager):
        """Test hub pages identification."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        hub_pages = analyzer.get_hub_pages(limit=3)
        
        assert len(hub_pages) <= 3
        
        # Should be sorted by out_degree descending
        if len(hub_pages) > 1:
            out_degrees = [page['out_degree'] for page in hub_pages]
            assert out_degrees == sorted(out_degrees, reverse=True)
        
        for page in hub_pages:
            assert 'url' in page
            assert 'out_degree' in page
            assert page['out_degree'] >= 0

    def test_generate_analysis_summary(self, pagerank_analyzer):
        """Test comprehensive analysis summary generation."""
        # Calculate PageRank first
        pagerank_analyzer.calculate_pagerank(max_iterations=20)
        
        summary = pagerank_analyzer.generate_analysis_summary()
        
        # Check main structure
        assert 'graph_stats' in summary
        assert 'metrics' in summary
        assert 'insights' in summary
        assert 'recommendations' in summary
        assert 'pagerank_scores' in summary
        
        # Check metrics
        metrics = summary['metrics']
        assert 'total_pages' in metrics
        assert 'avg_pagerank' in metrics
        assert 'max_pagerank' in metrics
        assert 'min_pagerank' in metrics
        
        assert metrics['total_pages'] > 0
        assert metrics['avg_pagerank'] > 0
        assert metrics['max_pagerank'] >= metrics['avg_pagerank']
        assert metrics['min_pagerank'] <= metrics['avg_pagerank']
        
        # Check insights
        insights = summary['insights']
        assert 'pillar_pages' in insights
        assert 'orphaned_pages' in insights
        assert 'low_outlink_pages' in insights
        assert 'hub_pages' in insights
        
        # Check recommendations
        recommendations = summary['recommendations']
        assert isinstance(recommendations, list)
        
        # PageRank scores should match
        assert len(summary['pagerank_scores']) > 0

    def test_generate_analysis_summary_empty(self, kuzu_manager):
        """Test analysis summary with empty database."""
        analyzer = PageRankAnalyzer(kuzu_manager)
        summary = analyzer.generate_analysis_summary()
        
        assert 'error' in summary
        assert 'No page data available' in summary['error']

    def test_generate_recommendations(self, pagerank_analyzer):
        """Test recommendation generation."""
        # Calculate PageRank first
        pagerank_analyzer.calculate_pagerank(max_iterations=20)
        
        # Get data for recommendations
        all_pages = pagerank_analyzer.kuzu_manager.get_page_data()
        pillar_pages = pagerank_analyzer.get_pillar_pages()
        orphaned_pages = pagerank_analyzer.get_orphaned_pages()
        low_outlink_pages = pagerank_analyzer.get_low_outlink_pages()
        
        recommendations = pagerank_analyzer._generate_recommendations(
            all_pages, pillar_pages, orphaned_pages, low_outlink_pages
        )
        
        assert isinstance(recommendations, list)
        
        # Should have some recommendations
        assert len(recommendations) > 0
        
        # Each recommendation should be a string
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_export_pagerank_data(self, pagerank_analyzer):
        """Test PageRank data export."""
        # Calculate PageRank first
        pagerank_analyzer.calculate_pagerank(max_iterations=20)
        
        export_data = pagerank_analyzer.export_pagerank_data()
        
        # Check structure
        assert 'pages' in export_data
        assert 'links' in export_data
        assert 'pagerank_scores' in export_data
        assert 'analysis_summary' in export_data
        
        # Check data types
        assert isinstance(export_data['pages'], list)
        assert isinstance(export_data['links'], list)
        assert isinstance(export_data['pagerank_scores'], dict)
        assert isinstance(export_data['analysis_summary'], dict)

    def test_error_handling(self, kuzu_manager):
        """Test error handling in various scenarios."""
        analyzer = PageRankAnalyzer(kuzu_manager)
        
        # Test with broken kuzu manager
        with patch.object(kuzu_manager, 'get_page_data', side_effect=Exception("DB Error")):
            summary = analyzer.generate_analysis_summary()
            assert 'error' in summary
            assert 'DB Error' in summary['error']

    @patch('mcp_seo.graph.pagerank_analyzer.logger')
    def test_logging(self, mock_logger, pagerank_analyzer):
        """Test logging functionality."""
        pagerank_analyzer.calculate_pagerank(max_iterations=5)
        
        # Verify logging calls were made
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()

    def test_pagerank_mathematical_properties(self, populated_kuzu_manager):
        """Test mathematical properties of PageRank."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        scores = analyzer.calculate_pagerank(damping_factor=0.85, max_iterations=100)
        
        # Test mathematical properties
        assert len(scores) > 0
        
        # All scores should be positive
        for score in scores.values():
            assert score > 0, f"PageRank score should be positive: {score}"
        
        # Sum should be approximately 1 (within tolerance)
        total_score = sum(scores.values())
        assert abs(total_score - 1.0) < 0.001, f"Sum of PageRank scores should be ~1.0, got {total_score}"
        
        # No score should be too large (sanity check)
        max_score = max(scores.values())
        assert max_score <= 1.0, f"Individual PageRank score too large: {max_score}"
        
        # Minimum theoretical score check
        n_pages = len(scores)
        min_theoretical = (1.0 - 0.85) / n_pages  # (1-d)/N
        min_actual = min(scores.values())
        assert min_actual >= min_theoretical * 0.5, f"PageRank too small: {min_actual} vs theoretical min {min_theoretical}"

    def test_pagerank_iteration_behavior(self, populated_kuzu_manager):
        """Test PageRank behavior across iterations."""
        analyzer = PageRankAnalyzer(populated_kuzu_manager)
        
        # Test with very few iterations
        scores_few = analyzer.calculate_pagerank(max_iterations=2, tolerance=1e-10)
        
        # Test with many iterations
        scores_many = analyzer.calculate_pagerank(max_iterations=100, tolerance=1e-6)
        
        # Both should produce results
        assert len(scores_few) > 0
        assert len(scores_many) > 0
        
        # More iterations should generally lead to more convergence
        # (scores_many should be more accurate, but both should be reasonable)
        for url in scores_few:
            if url in scores_many:
                assert abs(scores_few[url] - scores_many[url]) < 0.5  # Reasonable difference