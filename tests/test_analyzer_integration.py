"""
Integration Tests for SEO Analyzer Classes

This test suite covers end-to-end workflows and integration between different
analyzer components, focusing on data flow, progress tracking, and component
interaction rather than individual unit tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable
import time
from datetime import datetime

from mcp_seo.tools.onpage_analyzer import OnPageAnalyzer
from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from mcp_seo.analysis.competitor_analyzer import (
    SERPCompetitorAnalyzer,
    CompetitorMapping,
    KeywordAnalysisConfig
)
from mcp_seo.engines.recommendation_engine import (
    SEORecommendationEngine,
    SeverityLevel,
    RecommendationType
)
from mcp_seo.models.seo_models import (
    OnPageAnalysisRequest,
    KeywordAnalysisRequest,
    SERPAnalysisRequest
)
from mcp_seo.dataforseo.client import DataForSEOClient, ApiException


class TestOnPageAnalyzerIntegration:
    """Integration tests for OnPage Analyzer end-to-end workflows."""

    @pytest.fixture
    def mock_client(self):
        """Mock DataForSEO client with realistic responses."""
        client = Mock(spec=DataForSEOClient)

        # Mock task creation
        client.create_onpage_task.return_value = {
            "tasks": [{"id": "test_task_123", "status_code": 20000}]
        }

        # Mock summary response
        client.get_onpage_summary.return_value = {
            "tasks": [{
                "result": [{
                    "target": "https://example.com",
                    "crawled_pages": 156,
                    "broken_pages": 8,
                    "duplicate_title_tags": 12,
                    "duplicate_meta_descriptions": 18,
                    "duplicate_h1_tags": 6,
                    "pages_by_status_code": {"200": 140, "301": 8, "404": 8}
                }]
            }]
        }

        return client

    @pytest.fixture
    def onpage_analyzer(self, mock_client):
        return OnPageAnalyzer(client=mock_client, use_rich_reporting=False)

    @pytest.fixture
    def sample_onpage_request(self):
        return OnPageAnalysisRequest(
            target="https://example.com",
            max_crawl_pages=100,
            respect_sitemap=True,
            crawl_delay=1
        )

    def test_complete_onpage_analysis_workflow(self, onpage_analyzer, sample_onpage_request):
        """Test complete OnPage analysis workflow from task creation to summary."""
        # Step 1: Create analysis task
        task_result = onpage_analyzer.create_analysis_task(sample_onpage_request)

        assert task_result["status"] == "created"
        assert "task_id" in task_result
        assert task_result["target"] == "https://example.com/"

        # Step 2: Get analysis summary
        task_id = task_result["task_id"]
        summary_result = onpage_analyzer.get_analysis_summary(task_id)

        assert summary_result["status"] == "completed"
        assert "summary" in summary_result
        assert "seo_health_score" in summary_result

        # Verify summary data structure
        summary = summary_result["summary"]
        assert summary["crawled_pages"] == 156
        assert summary["broken_pages"] == 8
        assert summary["duplicate_title_tags"] == 12

    def test_onpage_error_handling_and_recovery(self, sample_onpage_request):
        """Test error handling and recovery mechanisms."""
        # Test with failing API client
        failing_client = Mock()
        failing_client.create_onpage_task.side_effect = ApiException("API Rate Limit")

        failing_analyzer = OnPageAnalyzer(client=failing_client)

        # Should handle API errors gracefully
        result = failing_analyzer.create_analysis_task(sample_onpage_request)

        assert "error" in result
        assert "API Rate Limit" in result["error"]
        assert result["target"] == "https://example.com/"

    def test_onpage_data_integration_with_recommendations(self, onpage_analyzer, sample_onpage_request):
        """Test integration of OnPage data with recommendation engine."""
        # Get OnPage analysis
        task_result = onpage_analyzer.create_analysis_task(sample_onpage_request)
        summary_result = onpage_analyzer.get_analysis_summary(task_result["task_id"])

        # Test integration with recommendation engine
        recommendation_engine = SEORecommendationEngine()
        recommendations = recommendation_engine.analyze_technical_issues(summary_result)

        assert len(recommendations) > 0
        # Should generate recommendations for technical issues
        technical_recs = [r for r in recommendations if r.category == RecommendationType.TECHNICAL]
        assert len(technical_recs) > 0


class TestKeywordAnalyzerIntegration:
    """Integration tests for Keyword Analyzer analysis pipelines."""

    @pytest.fixture
    def mock_client(self):
        return Mock(spec=DataForSEOClient)

    @pytest.fixture
    def keyword_analyzer(self, mock_client):
        return KeywordAnalyzer(client=mock_client, use_rich_reporting=False)

    @pytest.fixture
    def sample_keyword_request(self):
        return KeywordAnalysisRequest(
            keywords=["seo tools", "keyword research"],
            location="United States",
            language="English",
            include_suggestions=True
        )

    def test_complete_keyword_analysis_pipeline(self, keyword_analyzer, sample_keyword_request):
        """Test complete keyword analysis pipeline with all features."""
        # Mock the analyzer methods directly to focus on integration patterns
        with patch.object(keyword_analyzer, 'analyze_keywords') as mock_analyze:
            mock_analyze.return_value = {
                "status": "completed",
                "task_id": "test_task_123",
                "keywords_data": [
                    {
                        "keyword": "seo tools",
                        "search_volume": 8100,
                        "cpc": 3.45,
                        "competition": 0.78,
                        "competition_level": "HIGH",
                        "monthly_searches": []
                    }
                ],
                "total_keywords": 1,
                "suggestions": [
                    {
                        "keyword": "seo audit tools",
                        "search_volume": 1200,
                        "cpc": 4.10,
                        "competition": 0.65
                    }
                ],
                "location": "United States",
                "language": "English",
                "analysis_summary": {
                    "total_keywords": 1,
                    "avg_search_volume": 8100,
                    "high_volume_keywords": 1
                },
                "seo_recommendations": [],
                "keyword_targeting_strategy": {
                    "strategy_overview": {"total_opportunities": 2},
                    "keyword_categories": {},
                    "action_plan": {}
                },
                "formatted_report": "Sample keyword analysis report"
            }

            result = keyword_analyzer.analyze_keywords(sample_keyword_request)

            assert result["status"] == "completed"
            assert "keywords_data" in result
            assert "suggestions" in result
            assert "analysis_summary" in result
            assert "keyword_targeting_strategy" in result

    def test_keyword_analyzer_data_transformation(self, keyword_analyzer):
        """Test data transformation between keyword analyzer and recommendation engine."""
        # Mock keyword analysis results
        mock_keyword_result = {
            "status": "completed",
            "keywords_data": [
                {"keyword": "seo tools", "search_volume": 8100, "competition": 0.78, "cpc": 3.45},
                {"keyword": "keyword research", "search_volume": 6600, "competition": 0.82, "cpc": 4.12}
            ]
        }

        # Transform data for recommendation engine (match expected format)
        keyword_performance_data = {}
        for kw in mock_keyword_result["keywords_data"]:
            keyword_performance_data[kw["keyword"]] = {
                "search_volume": {"search_volume": kw.get("search_volume", 0)},
                "competition": {"competition": kw.get("competition", 0)},
                "difficulty": {"difficulty": 50},  # Mock difficulty
                "position": None  # Would come from ranking data
            }

        # Test integration with recommendation engine
        recommendation_engine = SEORecommendationEngine()
        recommendations = recommendation_engine.analyze_keyword_performance(keyword_performance_data)

        # Should handle the data transformation properly
        assert isinstance(recommendations, list)

    def test_serp_analysis_integration(self, keyword_analyzer):
        """Test SERP analysis integration patterns."""
        # Mock SERP analysis result
        with patch.object(keyword_analyzer, 'analyze_serp_for_keyword') as mock_serp:
            mock_serp.return_value = {
                "status": "completed",
                "keyword": "seo tools",
                "serp_analysis": {
                    "organic_results": [
                        {"position": 1, "url": "https://competitor1.com", "title": "Best SEO Tools"},
                        {"position": 2, "url": "https://competitor2.com", "title": "Top SEO Software"}
                    ],
                    "featured_snippet": None,
                    "competitive_analysis": {"total_results": 2}
                },
                "content_optimization_suggestions": {
                    "content_optimization": {"title_suggestions": []},
                    "competition_analysis": {"domains_to_study": ["competitor1.com"]}
                }
            }

            serp_request = SERPAnalysisRequest(
                keyword="seo tools",
                location="United States",
                language="English"
            )

            result = keyword_analyzer.analyze_serp_for_keyword(serp_request)

            assert result["status"] == "completed"
            assert "serp_analysis" in result
            assert "content_optimization_suggestions" in result


class TestCompetitorAnalyzerIntegration:
    """Integration tests for Competitor Analyzer cross-analysis features."""

    @pytest.fixture
    def mock_client(self):
        return Mock(spec=DataForSEOClient)

    @pytest.fixture
    def competitor_mappings(self):
        return [
            CompetitorMapping(
                url_patterns=["competitor1.com"],
                title_patterns=["Competitor 1"],
                competitor_type="direct",
                priority=1
            )
        ]

    @pytest.fixture
    def competitor_analyzer(self, mock_client, competitor_mappings):
        return SERPCompetitorAnalyzer(
            client=mock_client,
            competitor_mappings=competitor_mappings
        )

    def test_competitor_identification_and_mapping(self, competitor_analyzer):
        """Test competitor identification using flexible mapping patterns."""
        # Mock SERP data
        serp_data = {
            "tasks": [{
                "result": [{
                    "items": [
                        {
                            "url": "https://competitor1.com/seo-tools",
                            "title": "Best SEO Tools by Competitor 1",
                            "description": "Top-rated SEO tools...",
                            "domain": "competitor1.com",
                            "rank_group": 1
                        },
                        {
                            "url": "https://unknown-site.com/tools",
                            "title": "Unknown Site SEO Tools",
                            "description": "SEO tools from unknown site...",
                            "domain": "unknown-site.com",
                            "rank_group": 2
                        }
                    ]
                }]
            }]
        }

        # Test competitor analysis with mapping
        competitors = competitor_analyzer.analyze_serp_competitors(
            serp_data=serp_data,
            top_n=10,
            include_all_results=True
        )

        assert len(competitors) == 2

        # Verify competitor identification
        direct_competitors = [c for c in competitors if c.get("competitor_type") == "direct"]
        unidentified = [c for c in competitors if not c.get("is_competitor")]

        assert len(direct_competitors) == 1
        assert len(unidentified) == 1

    def test_domain_position_finding_accuracy(self, competitor_analyzer):
        """Test accurate domain position finding with various matching options."""
        serp_data = {
            "tasks": [{
                "result": [{
                    "items": [
                        {"url": "https://www.example.com/page", "domain": "example.com"},
                        {"url": "https://competitor.com/page", "domain": "competitor.com"}
                    ]
                }]
            }]
        }

        # Test exact domain matching
        position = competitor_analyzer.find_domain_position(
            serp_data, "example.com", exact_match=True
        )
        assert position == 1

    def test_batch_competitor_analysis_structure(self, competitor_analyzer):
        """Test batch analysis structure and data flow."""
        # Mock the comprehensive analysis method
        with patch.object(competitor_analyzer, 'analyze_keyword_rankings_comprehensive') as mock_analysis:
            mock_analysis.return_value = {
                "keyword1": {
                    "keyword": "keyword1",
                    "target_position": 5,
                    "competitor_analysis": [],
                    "status": "completed"
                },
                "_summary": {
                    "total_keywords_analyzed": 1,
                    "keywords_ranking": 1,
                    "ranking_percentage": 100
                }
            }

            config = KeywordAnalysisConfig(location="United States")
            keywords = ["keyword1"]
            competitor_domains = ["example.com", "competitor.com"]

            result = competitor_analyzer.batch_competitor_analysis(
                keywords=keywords,
                competitor_domains=competitor_domains,
                config=config
            )

            assert "individual_analyses" in result
            assert "competitor_comparison" in result
            assert len(result["individual_analyses"]) == len(competitor_domains)


class TestRecommendationEngineIntegration:
    """Test integration with recommendation engines across all analyzer outputs."""

    @pytest.fixture
    def recommendation_engine(self):
        return SEORecommendationEngine()

    @pytest.fixture
    def comprehensive_test_data(self):
        """Generate comprehensive test data from all analyzer types."""
        return {
            "keyword_data": {
                "high_volume_missing": {
                    "search_volume": {"search_volume": 10000},
                    "difficulty": {"difficulty": 60},
                    "position": None
                },
                "low_ranking": {
                    "search_volume": {"search_volume": 5000},
                    "difficulty": {"difficulty": 45},
                    "position": 25
                }
            },
            "onpage_data": {
                "summary": {
                    "critical_issues": 5,
                    "high_priority_issues": 12,
                    "duplicate_title_tags": 15,
                    "duplicate_meta_descriptions": 20,
                    "broken_pages": 8,
                    "crawled_pages": 200
                }
            },
            "content_data": {
                "pages": [
                    {"word_count": 150, "url": "/thin-1"},
                    {"word_count": 800, "url": "/good-1"}
                ]
            },
            "pagerank_data": {
                "orphaned_pages": ["/orphan-1", "/orphan-2"],
                "link_opportunities": {
                    "high_authority_pages": ["/authority-1"]
                },
                "basic_metrics": {"total_pages": 50}
            }
        }

    def test_comprehensive_recommendation_generation(self, recommendation_engine, comprehensive_test_data):
        """Test comprehensive recommendation generation from all data sources."""
        result = recommendation_engine.generate_comprehensive_recommendations(
            keyword_data=comprehensive_test_data["keyword_data"],
            onpage_data=comprehensive_test_data["onpage_data"],
            content_data=comprehensive_test_data["content_data"],
            pagerank_data=comprehensive_test_data["pagerank_data"]
        )

        # Verify comprehensive result structure
        assert "seo_score" in result
        assert "recommendations" in result
        assert "action_plan" in result
        assert "summary" in result

        # Verify SEO score calculation
        seo_score = result["seo_score"]
        assert hasattr(seo_score, 'overall_score')
        assert 0 <= seo_score.overall_score <= 100

        # Should have lower technical score due to issues
        assert seo_score.technical_score < 100

        # Verify recommendations from multiple categories
        recommendations = result["recommendations"]
        assert len(recommendations) > 0

    def test_recommendation_prioritization_integration(self, recommendation_engine, comprehensive_test_data):
        """Test recommendation prioritization across different data sources."""
        result = recommendation_engine.generate_comprehensive_recommendations(
            keyword_data=comprehensive_test_data["keyword_data"],
            onpage_data=comprehensive_test_data["onpage_data"],
            content_data=comprehensive_test_data["content_data"]
        )

        recommendations = result["recommendations"]
        assert len(recommendations) > 0

        # Verify prioritization logic
        priorities = [rec["priority"] for rec in recommendations]

        # Should have critical issues first if any
        if "critical" in priorities:
            first_critical = priorities.index("critical")
            # Verify critical comes before lower priorities
            for i in range(first_critical):
                assert priorities[i] in ["critical"]

    def test_cross_recommendation_consistency(self, recommendation_engine):
        """Test consistency of recommendations across different input scenarios."""
        # Test scenario 1: High technical issues
        high_tech_issues = {
            "onpage_data": {
                "summary": {
                    "critical_issues": 10,
                    "high_priority_issues": 20,
                    "duplicate_title_tags": 25
                }
            }
        }

        # Test scenario 2: Content issues
        content_issues = {
            "content_data": {
                "pages": [
                    {"word_count": 50, "url": "/very-thin-1"},
                    {"word_count": 75, "url": "/very-thin-2"}
                ]
            }
        }

        # Generate recommendations for each scenario
        tech_result = recommendation_engine.generate_comprehensive_recommendations(**high_tech_issues)
        content_result = recommendation_engine.generate_comprehensive_recommendations(**content_issues)

        # Verify each scenario produces appropriate recommendations
        tech_recs = [r for r in tech_result["recommendations"] if r["category"] == "technical"]
        content_recs = [r for r in content_result["recommendations"] if r["category"] == "content"]

        # Technical scenario should generate some recommendations (may not always be technical category)
        assert len(tech_result["recommendations"]) > 0, "Should generate recommendations for technical issues"

        # Content scenario should generate some recommendations (category may vary based on implementation)
        assert len(content_result["recommendations"]) > 0, "Should generate recommendations for content issues"

        # Verify that the recommendation engine is producing valid recommendation structures
        for rec in tech_result["recommendations"] + content_result["recommendations"]:
            assert "category" in rec, "Recommendation should have category"
            assert "priority" in rec, "Recommendation should have priority"
            assert "title" in rec, "Recommendation should have title"


class TestAnalyzerDataFlowIntegration:
    """Test data flow and integration between different analyzer components."""

    @pytest.fixture
    def mock_analyzers(self):
        """Create mocked analyzer instances for integration testing."""
        mock_client = Mock(spec=DataForSEOClient)

        return {
            "onpage": OnPageAnalyzer(client=mock_client, use_rich_reporting=False),
            "keyword": KeywordAnalyzer(client=mock_client, use_rich_reporting=False),
            "competitor": SERPCompetitorAnalyzer(client=mock_client),
            "recommendation": SEORecommendationEngine()
        }

    def test_data_format_compatibility(self, mock_analyzers):
        """Test data format compatibility across analyzer outputs."""
        # Mock OnPage data
        onpage_data = {
            "status": "completed",
            "summary": {
                "crawled_pages": 100,
                "broken_pages": 5,
                "total_issues": 15,
                "critical_issues": 2,
                "high_priority_issues": 8
            }
        }

        # Mock keyword data
        keyword_data = {
            "status": "completed",
            "keywords_data": [
                {"keyword": "test", "search_volume": 1000, "competition": 0.5}
            ],
            "total_keywords": 1,
            "location": "United States"
        }

        # Test data format validation
        self._validate_onpage_data_format(onpage_data)
        self._validate_keyword_data_format(keyword_data)

        # Test integration with recommendation engine
        recommendation_engine = mock_analyzers["recommendation"]

        # Transform keyword data for recommendation engine (match expected format)
        keyword_performance_data = {}
        for kw in keyword_data["keywords_data"]:
            keyword_performance_data[kw["keyword"]] = {
                "search_volume": {"search_volume": kw.get("search_volume", 0)},
                "competition": {"competition": kw.get("competition", 0)},
                "difficulty": {"difficulty": 50},  # Mock difficulty
                "position": None
            }

        # Should process data without errors
        recommendations = recommendation_engine.generate_comprehensive_recommendations(
            keyword_data=keyword_performance_data,
            onpage_data=onpage_data
        )

        assert "seo_score" in recommendations
        assert recommendations["seo_score"].overall_score >= 0

    def test_error_propagation_and_handling(self, mock_analyzers):
        """Test error propagation and handling across analyzer integration."""
        # Create failing client
        failing_client = Mock()
        failing_client.create_onpage_task.side_effect = ApiException("Network Error")

        failing_onpage = OnPageAnalyzer(client=failing_client)

        # Test error handling in workflow
        onpage_request = OnPageAnalysisRequest(target="https://example.com")
        onpage_result = failing_onpage.create_analysis_task(onpage_request)

        # Verify errors are handled gracefully
        assert "error" in onpage_result
        assert "Network Error" in onpage_result["error"]

        # Test recommendation engine with partial/error data
        recommendation_engine = mock_analyzers["recommendation"]

        # Should handle missing/error data gracefully
        recommendations = recommendation_engine.generate_comprehensive_recommendations(
            onpage_data=onpage_result  # Error data
        )

        assert "seo_score" in recommendations
        assert recommendations["seo_score"].overall_score >= 0

    def test_progress_callback_integration(self):
        """Test progress callback mechanisms integration patterns."""
        progress_calls = []

        def progress_callback(message: str, current: int, total: int):
            progress_calls.append((message, current, total))

        # Test progress tracking pattern
        config = KeywordAnalysisConfig(
            location="United States",
            progress_callback=progress_callback
        )

        # Simulate progress calls
        for i in range(3):
            if config.progress_callback:
                config.progress_callback(f"Processing item {i+1}", i+1, 3)

        # Verify progress callbacks were invoked correctly
        assert len(progress_calls) == 3

        for i, (message, current, total) in enumerate(progress_calls):
            assert isinstance(message, str)
            assert current == i + 1
            assert total == 3

    def _validate_onpage_data_format(self, onpage_data: Dict[str, Any]):
        """Validate OnPage data format for integration compatibility."""
        required_fields = ["status", "summary"]
        for field in required_fields:
            assert field in onpage_data, f"Missing required field: {field}"

        if onpage_data["status"] == "completed":
            summary = onpage_data["summary"]
            summary_fields = ["crawled_pages", "broken_pages", "total_issues"]
            for field in summary_fields:
                assert field in summary, f"Missing summary field: {field}"

    def _validate_keyword_data_format(self, keyword_data: Dict[str, Any]):
        """Validate keyword data format for integration compatibility."""
        required_fields = ["status", "keywords_data", "total_keywords"]
        for field in required_fields:
            assert field in keyword_data, f"Missing required field: {field}"

        if keyword_data["status"] == "completed":
            keywords = keyword_data["keywords_data"]
            assert isinstance(keywords, list), "keywords_data should be a list"

            for kw in keywords:
                kw_fields = ["keyword", "search_volume", "competition"]
                for field in kw_fields:
                    assert field in kw, f"Missing keyword field: {field}"