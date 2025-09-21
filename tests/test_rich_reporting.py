"""
Test suite for Rich reporting integration.

Tests the enhanced Rich console reporting functionality,
formatted output generation, and reporting integration.
"""

import pytest
import io
import sys
from unittest.mock import Mock, patch, MagicMock
from contextlib import redirect_stdout

from mcp_seo.reporting.seo_reporter import SEOReporter
from mcp_seo.utils.rich_reporter import SEOReporter as RichReporter
from rich.console import Console


class TestSEOReporter:
    """Test suite for SEO Reporter with Rich integration."""

    @pytest.fixture
    def seo_reporter(self):
        """Create SEO Reporter instance with Rich enabled."""
        return SEOReporter(use_rich=True)

    @pytest.fixture
    def seo_reporter_no_rich(self):
        """Create SEO Reporter instance without Rich."""
        return SEOReporter(use_rich=False)

    @pytest.fixture
    def sample_keyword_data(self):
        """Sample keyword analysis data for reporting."""
        return {
            "keywords_data": [
                {
                    "keyword": "seo tools",
                    "search_volume": 5400,
                    "cpc": 2.50,
                    "competition": 0.85,
                    "difficulty_score": 75
                },
                {
                    "keyword": "keyword research",
                    "search_volume": 8100,
                    "cpc": 3.20,
                    "competition": 0.75,
                    "difficulty_score": 68
                }
            ],
            "keyword_suggestions": [
                {
                    "keyword": "free seo tools",
                    "search_volume": 3300,
                    "cpc": 1.80,
                    "competition": 0.55
                }
            ],
            "analysis_summary": {
                "total_keywords": 2,
                "avg_search_volume": 6750,
                "avg_competition": 0.80,
                "high_volume_count": 2
            }
        }

    @pytest.fixture
    def sample_pagerank_data(self):
        """Sample PageRank analysis data for reporting."""
        return {
            "basic_metrics": {
                "total_pages": 50,
                "total_links": 125,
                "avg_pagerank": 0.25,
                "max_pagerank": 0.45
            },
            "top_pages": [
                {"url": "/home", "pagerank": 0.45, "inbound_links": 15},
                {"url": "/about", "pagerank": 0.35, "inbound_links": 8},
                {"url": "/services", "pagerank": 0.30, "inbound_links": 12}
            ],
            "orphaned_pages": ["/orphan-1", "/orphan-2"],
            "recommendations": [
                {
                    "type": "link_building",
                    "priority": "high",
                    "description": "Add internal links to orphaned pages",
                    "affected_pages": 2
                }
            ]
        }

    @pytest.fixture
    def sample_onpage_data(self):
        """Sample on-page analysis data for reporting."""
        return {
            "summary": {
                "total_pages_analyzed": 45,
                "critical_issues": 3,
                "high_priority_issues": 8,
                "medium_priority_issues": 15,
                "low_priority_issues": 22
            },
            "issues_by_type": {
                "missing_meta_descriptions": 12,
                "duplicate_title_tags": 5,
                "large_page_size": 8,
                "slow_response_time": 3
            },
            "top_issues": [
                {
                    "type": "missing_meta_description",
                    "severity": "medium",
                    "count": 12,
                    "pages": ["/page1", "/page2", "/page3"]
                }
            ]
        }

    def test_generate_keyword_report_with_rich(self, seo_reporter, sample_keyword_data):
        """Test keyword report generation with Rich formatting."""
        with patch.object(seo_reporter.rich_reporter, 'generate_keyword_analysis_report') as mock_rich:
            mock_rich.return_value = "Rich formatted keyword report"

            result = seo_reporter.generate_keyword_report(sample_keyword_data)

            assert result == "Rich formatted keyword report"
            mock_rich.assert_called_once_with(sample_keyword_data)

    def test_generate_keyword_report_without_rich(self, seo_reporter_no_rich, sample_keyword_data):
        """Test keyword report generation without Rich formatting."""
        result = seo_reporter_no_rich.generate_keyword_report(sample_keyword_data)

        assert isinstance(result, str)
        assert "seo tools" in result
        assert "keyword research" in result
        assert "5400" in result  # Search volume
        assert "8100" in result  # Search volume

    def test_generate_pagerank_report_with_rich(self, seo_reporter, sample_pagerank_data):
        """Test PageRank report generation with Rich formatting."""
        with patch.object(seo_reporter.rich_reporter, 'generate_pagerank_analysis_report') as mock_rich:
            mock_rich.return_value = "Rich formatted PageRank report"

            result = seo_reporter.generate_pagerank_report(sample_pagerank_data)

            assert result == "Rich formatted PageRank report"
            mock_rich.assert_called_once_with(sample_pagerank_data)

    def test_generate_pagerank_report_without_rich(self, seo_reporter_no_rich, sample_pagerank_data):
        """Test PageRank report generation without Rich formatting."""
        result = seo_reporter_no_rich.generate_pagerank_report(sample_pagerank_data)

        assert isinstance(result, str)
        assert "50" in result  # Total pages
        assert "125" in result  # Total links
        assert "/home" in result  # Top page
        assert "orphan" in result.lower()  # Orphaned pages

    def test_generate_onpage_report_with_rich(self, seo_reporter, sample_onpage_data):
        """Test on-page analysis report generation with Rich formatting."""
        with patch.object(seo_reporter.rich_reporter, 'generate_onpage_analysis_report') as mock_rich:
            mock_rich.return_value = "Rich formatted on-page report"

            result = seo_reporter.generate_onpage_report(sample_onpage_data)

            assert result == "Rich formatted on-page report"
            mock_rich.assert_called_once_with(sample_onpage_data)

    def test_generate_onpage_report_without_rich(self, seo_reporter_no_rich, sample_onpage_data):
        """Test on-page analysis report generation without Rich formatting."""
        result = seo_reporter_no_rich.generate_onpage_report(sample_onpage_data)

        assert isinstance(result, str)
        assert "45" in result  # Total pages analyzed
        assert "3" in result  # Critical issues
        assert "meta" in result.lower()  # Issue types

    def test_generate_comprehensive_report(self, seo_reporter, sample_keyword_data,
                                         sample_pagerank_data, sample_onpage_data):
        """Test comprehensive report generation combining multiple analyses."""
        with patch.object(seo_reporter.rich_reporter, 'generate_comprehensive_seo_report') as mock_rich:
            mock_rich.return_value = "Rich comprehensive SEO report"

            result = seo_reporter.generate_comprehensive_report(
                keyword_data=sample_keyword_data,
                pagerank_data=sample_pagerank_data,
                onpage_data=sample_onpage_data
            )

            assert result == "Rich comprehensive SEO report"
            mock_rich.assert_called_once()

    def test_fallback_when_rich_fails(self, seo_reporter, sample_keyword_data):
        """Test fallback to plain text when Rich reporting fails."""
        # Mock Rich reporter to raise an exception
        seo_reporter.rich_reporter.generate_keyword_analysis_report.side_effect = Exception("Rich error")

        result = seo_reporter.generate_keyword_report(sample_keyword_data)

        # Should fallback to plain text
        assert isinstance(result, str)
        assert "seo tools" in result

    def test_progress_tracking_integration(self, seo_reporter):
        """Test progress tracking integration with Rich console."""
        with patch.object(seo_reporter.rich_reporter, 'create_progress_tracker') as mock_progress:
            mock_tracker = Mock()
            mock_progress.return_value = mock_tracker

            tracker = seo_reporter.create_progress_tracker("Test Analysis", 10)

            assert tracker == mock_tracker
            mock_progress.assert_called_once_with("Test Analysis", 10)


class TestRichReporter:
    """Test suite for Rich Reporter utility functions."""

    @pytest.fixture
    def rich_reporter(self):
        """Create Rich Reporter instance."""
        return RichReporter()

    @pytest.fixture
    def sample_keyword_analysis(self):
        """Sample keyword analysis data for Rich formatting."""
        return {
            "keywords_data": [
                {
                    "keyword": "seo tools",
                    "search_volume": 5400,
                    "cpc": 2.50,
                    "competition": 0.85,
                    "difficulty_score": 75
                },
                {
                    "keyword": "keyword research",
                    "search_volume": 8100,
                    "cpc": 3.20,
                    "competition": 0.75,
                    "difficulty_score": 68
                }
            ],
            "analysis_summary": {
                "total_keywords": 2,
                "avg_search_volume": 6750,
                "avg_competition": 0.80
            }
        }

    def test_generate_keyword_analysis_report(self, rich_reporter, sample_keyword_analysis):
        """Test Rich keyword analysis report generation."""
        # Capture console output
        console_output = io.StringIO()

        with patch.object(rich_reporter, 'console', Console(file=console_output, width=120)):
            result = rich_reporter.generate_keyword_analysis_report(sample_keyword_analysis)

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify content is included
        assert "seo tools" in result
        assert "5400" in result

    def test_create_keyword_table(self, rich_reporter, sample_keyword_analysis):
        """Test keyword data table creation."""
        table = rich_reporter._create_keyword_table(sample_keyword_analysis["keywords_data"])

        assert table is not None
        assert hasattr(table, 'columns')
        # Table should have keyword, volume, competition, CPC columns

    def test_create_summary_panel(self, rich_reporter, sample_keyword_analysis):
        """Test summary panel creation."""
        panel = rich_reporter._create_summary_panel(sample_keyword_analysis["analysis_summary"])

        assert panel is not None
        assert hasattr(panel, 'title')

    def test_generate_pagerank_analysis_report(self, rich_reporter):
        """Test PageRank analysis report generation."""
        pagerank_data = {
            "basic_metrics": {
                "total_pages": 50,
                "total_links": 125,
                "avg_pagerank": 0.25
            },
            "top_pages": [
                {"url": "/home", "pagerank": 0.45, "inbound_links": 15}
            ]
        }

        console_output = io.StringIO()

        with patch.object(rich_reporter, 'console', Console(file=console_output, width=120)):
            result = rich_reporter.generate_pagerank_analysis_report(pagerank_data)

        assert isinstance(result, str)
        assert "/home" in result
        assert "50" in result

    def test_generate_onpage_analysis_report(self, rich_reporter):
        """Test on-page analysis report generation."""
        onpage_data = {
            "summary": {
                "total_pages_analyzed": 45,
                "critical_issues": 3,
                "high_priority_issues": 8
            },
            "issues_by_type": {
                "missing_meta_descriptions": 12,
                "duplicate_title_tags": 5
            }
        }

        console_output = io.StringIO()

        with patch.object(rich_reporter, 'console', Console(file=console_output, width=120)):
            result = rich_reporter.generate_onpage_analysis_report(onpage_data)

        assert isinstance(result, str)
        assert "45" in result
        assert "3" in result

    def test_generate_comprehensive_seo_report(self, rich_reporter, sample_keyword_analysis):
        """Test comprehensive SEO report generation."""
        pagerank_data = {"basic_metrics": {"total_pages": 50}}
        onpage_data = {"summary": {"critical_issues": 3}}

        console_output = io.StringIO()

        with patch.object(rich_reporter, 'console', Console(file=console_output, width=120)):
            result = rich_reporter.generate_comprehensive_seo_report(
                keyword_data=sample_keyword_analysis,
                pagerank_data=pagerank_data,
                onpage_data=onpage_data
            )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_progress_tracker(self, rich_reporter):
        """Test progress tracker creation."""
        tracker = rich_reporter.create_progress_tracker("Test Task", 10)

        assert tracker is not None
        # Should return a Rich Progress instance or compatible object

    def test_format_recommendations_table(self, rich_reporter):
        """Test recommendation table formatting."""
        recommendations = [
            {
                "title": "Fix Critical Issues",
                "priority": "critical",
                "category": "technical",
                "impact": "High",
                "effort": "High",
                "affected_pages": 10
            },
            {
                "title": "Optimize Content",
                "priority": "medium",
                "category": "content",
                "impact": "Medium",
                "effort": "Low",
                "affected_pages": 5
            }
        ]

        table = rich_reporter._create_recommendations_table(recommendations)

        assert table is not None
        assert hasattr(table, 'columns')

    def test_format_metrics_panel(self, rich_reporter):
        """Test metrics panel formatting."""
        metrics = {
            "overall_score": 75,
            "technical_score": 80,
            "content_score": 70,
            "keywords_score": 75,
            "links_score": 80
        }

        panel = rich_reporter._create_metrics_panel(metrics)

        assert panel is not None
        assert hasattr(panel, 'title')

    def test_color_coding_by_priority(self, rich_reporter):
        """Test color coding based on priority levels."""
        # Test critical priority
        critical_color = rich_reporter._get_priority_color("critical")
        assert critical_color is not None

        # Test high priority
        high_color = rich_reporter._get_priority_color("high")
        assert high_color is not None

        # Test medium priority
        medium_color = rich_reporter._get_priority_color("medium")
        assert medium_color is not None

        # Test low priority
        low_color = rich_reporter._get_priority_color("low")
        assert low_color is not None

    def test_export_report_functionality(self, rich_reporter, sample_keyword_analysis):
        """Test report export functionality."""
        with patch("builtins.open", create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            rich_reporter.export_report_to_file(
                sample_keyword_analysis,
                "/tmp/test_report.html",
                format="html"
            )

            mock_open.assert_called_once_with("/tmp/test_report.html", "w", encoding="utf-8")

    def test_console_width_adaptation(self, rich_reporter):
        """Test console width adaptation for different terminals."""
        # Test with narrow console
        narrow_console = Console(width=80)
        with patch.object(rich_reporter, 'console', narrow_console):
            # Should adapt table widths appropriately
            table = rich_reporter._create_keyword_table([
                {"keyword": "test", "search_volume": 1000, "competition": 0.5, "cpc": 1.0}
            ])
            assert table is not None

        # Test with wide console
        wide_console = Console(width=200)
        with patch.object(rich_reporter, 'console', wide_console):
            table = rich_reporter._create_keyword_table([
                {"keyword": "test", "search_volume": 1000, "competition": 0.5, "cpc": 1.0}
            ])
            assert table is not None


class TestReportingIntegration:
    """Integration tests for reporting system."""

    def test_end_to_end_keyword_reporting(self):
        """Test complete keyword analysis to report generation flow."""
        # Mock keyword analysis results
        keyword_data = {
            "keywords_data": [
                {"keyword": "test seo", "search_volume": 1000, "competition": 0.6, "cpc": 2.0}
            ],
            "analysis_summary": {"total_keywords": 1, "avg_search_volume": 1000}
        }

        # Test with Rich enabled
        reporter = SEOReporter(use_rich=True)
        with patch.object(reporter.rich_reporter, 'generate_keyword_analysis_report') as mock_rich:
            mock_rich.return_value = "Rich report"
            result = reporter.generate_keyword_report(keyword_data)
            assert result == "Rich report"

        # Test fallback to plain text
        reporter = SEOReporter(use_rich=False)
        result = reporter.generate_keyword_report(keyword_data)
        assert "test seo" in result

    def test_error_handling_in_report_generation(self):
        """Test error handling during report generation."""
        reporter = SEOReporter(use_rich=True)

        # Test with invalid data
        invalid_data = {"invalid": "data"}

        with patch.object(reporter.rich_reporter, 'generate_keyword_analysis_report') as mock_rich:
            mock_rich.side_effect = Exception("Formatting error")

            # Should fallback gracefully
            result = reporter.generate_keyword_report(invalid_data)
            assert isinstance(result, str)

    def test_performance_with_large_datasets(self):
        """Test reporting performance with large datasets."""
        # Generate large dataset
        large_keyword_data = {
            "keywords_data": [
                {"keyword": f"keyword_{i}", "search_volume": 1000+i, "competition": 0.5, "cpc": 1.0}
                for i in range(1000)
            ],
            "analysis_summary": {"total_keywords": 1000, "avg_search_volume": 1500}
        }

        reporter = SEOReporter(use_rich=False)  # Use plain text for performance

        import time
        start_time = time.time()
        result = reporter.generate_keyword_report(large_keyword_data)
        end_time = time.time()

        # Should complete within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 5.0  # 5 seconds threshold
        assert len(result) > 0

    def test_report_consistency_across_formats(self):
        """Test that reports contain consistent information across formats."""
        keyword_data = {
            "keywords_data": [
                {"keyword": "seo test", "search_volume": 1500, "competition": 0.7, "cpc": 2.5}
            ],
            "analysis_summary": {"total_keywords": 1, "avg_search_volume": 1500}
        }

        # Generate Rich report
        rich_reporter = SEOReporter(use_rich=True)
        with patch.object(rich_reporter.rich_reporter, 'generate_keyword_analysis_report') as mock_rich:
            mock_rich.return_value = "Rich: seo test, 1500, 0.7, 2.5"
            rich_result = rich_reporter.generate_keyword_report(keyword_data)

        # Generate plain text report
        plain_reporter = SEOReporter(use_rich=False)
        plain_result = plain_reporter.generate_keyword_report(keyword_data)

        # Both should contain the same key information
        key_info = ["seo test", "1500", "0.7", "2.5"]
        for info in key_info:
            assert info in rich_result
            assert info in plain_result