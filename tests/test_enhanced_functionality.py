"""
Test suite for Enhanced functionality that actually exists.

Tests the enhanced components that are available and working,
focusing on achievable test coverage goals.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_seo.engines.recommendation_engine import (
    RecommendationType,
    SEORecommendation,
    SEORecommendationEngine,
    SEOScore,
    SeverityLevel,
)
from mcp_seo.reporting import SEOReporter
from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from mcp_seo.utils.rich_reporter import SEOReporter as RichReporter


class TestEnhancedKeywordAnalyzer:
    """Test suite for Enhanced Keyword Analyzer functionality."""

    @pytest.fixture
    def mock_dataforseo_client(self):
        """Mock DataForSEO client."""
        client = Mock()

        # Mock keyword data response (live API - returns immediately with results)
        client.get_keyword_data.return_value = {
            "tasks": [
                {
                    "id": "test_task_123",
                    "status_code": 20000,
                    "result": [
                        {
                            "keyword": "seo tools",
                            "search_volume": 5400,
                            "cpc": 2.50,
                            "competition": 0.85,
                            "competition_level": "HIGH",
                            "monthly_searches": [],
                        },
                        {
                            "keyword": "keyword research",
                            "search_volume": 8100,
                            "cpc": 3.20,
                            "competition": 0.75,
                            "competition_level": "HIGH",
                            "monthly_searches": [],
                        },
                    ]
                }
            ]
        }

        # Mock suggestions response (live API - returns immediately with results)
        client.get_keyword_suggestions.return_value = {
            "tasks": [
                {
                    "id": "suggestion_task_456",
                    "status_code": 20000,
                    "result": [
                        {
                            "keyword": "seo audit tools",
                            "search_volume": 1200,
                            "cpc": 4.10,
                            "competition": 0.65,
                        }
                    ]
                }
            ]
        }

        return client

    @pytest.fixture
    def keyword_analyzer(self, mock_dataforseo_client):
        """Create Enhanced Keyword Analyzer instance."""
        return KeywordAnalyzer(client=mock_dataforseo_client, use_rich_reporting=False)

    @pytest.fixture
    def sample_keyword_request(self):
        """Sample keyword analysis request."""
        from mcp_seo.models.seo_models import KeywordAnalysisRequest

        return KeywordAnalysisRequest(
            keywords=["seo tools", "keyword research"],
            location="United States",
            language="English",
            include_suggestions=True,
        )

    def test_keyword_analyzer_initialization(self, keyword_analyzer):
        """Test Enhanced Keyword Analyzer initialization."""
        assert keyword_analyzer is not None
        assert hasattr(keyword_analyzer, "client")
        assert hasattr(keyword_analyzer, "recommendation_engine")
        assert hasattr(keyword_analyzer, "reporter")

    def test_keyword_analysis_workflow(self, keyword_analyzer, sample_keyword_request):
        """Test keyword analysis workflow functionality."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert result["status"] == "completed"
        assert "keywords_data" in result
        assert len(result["keywords_data"]) == 2

        # Verify keyword data structure
        for keyword_data in result["keywords_data"]:
            assert "keyword" in keyword_data
            assert "search_volume" in keyword_data
            assert "competition" in keyword_data
            assert "cpc" in keyword_data

    def test_keyword_targeting_strategy(self, keyword_analyzer, sample_keyword_request):
        """Test keyword targeting strategy generation."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert "keyword_targeting_strategy" in result
        strategy = result["keyword_targeting_strategy"]

        assert "keyword_categories" in strategy
        assert "strategy_overview" in strategy
        assert "action_plan" in strategy

        # Check keyword categories structure
        categories = strategy["keyword_categories"]
        assert "high_volume_low_competition" in categories
        assert "long_tail_opportunities" in categories
        assert "quick_wins" in categories
        assert "competitive_targets" in categories

    def test_error_handling(self, keyword_analyzer, sample_keyword_request):
        """Test error handling in keyword analysis."""
        from mcp_seo.dataforseo.client import ApiException

        keyword_analyzer.client.get_keyword_data.side_effect = ApiException("API Error")

        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert "error" in result
        assert "API Error" in result["error"]
        assert result["keywords"] == sample_keyword_request.keywords


class TestSEORecommendationEngine:
    """Test suite for SEO Recommendation Engine."""

    @pytest.fixture
    def recommendation_engine(self):
        """Create SEO Recommendation Engine instance."""
        return SEORecommendationEngine()

    @pytest.fixture
    def sample_keyword_data(self):
        """Sample keyword performance data."""
        return {
            "high volume keyword": {
                "search_volume": {"search_volume": 5000},
                "difficulty": {"difficulty": 60},
                "position": None,  # Not ranking
            },
            "ranking keyword": {
                "search_volume": {"search_volume": 1200},
                "difficulty": {"difficulty": 45},
                "position": 25,  # Low ranking (>20)
            },
            "good ranking keyword": {
                "search_volume": {"search_volume": 800},
                "difficulty": {"difficulty": 40},
                "position": 5,  # Good ranking
            },
        }

    @pytest.fixture
    def sample_onpage_data(self):
        """Sample on-page SEO data."""
        return {
            "summary": {
                "critical_issues": 3,
                "high_priority_issues": 5,
                "duplicate_title_tags": 8,
                "duplicate_meta_descriptions": 12,
            }
        }

    def test_recommendation_engine_initialization(self, recommendation_engine):
        """Test recommendation engine initialization."""
        assert recommendation_engine is not None
        assert hasattr(recommendation_engine, "recommendations")
        assert hasattr(recommendation_engine, "score_weights")

    def test_keyword_performance_analysis(
        self, recommendation_engine, sample_keyword_data
    ):
        """Test keyword performance analysis and recommendations."""
        recommendations = recommendation_engine.analyze_keyword_performance(
            sample_keyword_data
        )

        assert len(recommendations) > 0

        # Should identify missing high-volume keywords
        missing_keyword_recs = [
            r for r in recommendations if "Missing Keywords" in r.title
        ]
        assert len(missing_keyword_recs) > 0

        # Should identify low rankings
        low_ranking_recs = [r for r in recommendations if "Low-Ranking" in r.title]
        assert len(low_ranking_recs) > 0

    def test_technical_issues_analysis(self, recommendation_engine, sample_onpage_data):
        """Test technical SEO issue analysis."""
        recommendations = recommendation_engine.analyze_technical_issues(
            sample_onpage_data
        )

        assert len(recommendations) >= 2

        # Should identify critical issues
        critical_recs = [
            r for r in recommendations if r.priority == SeverityLevel.CRITICAL
        ]
        assert len(critical_recs) > 0

        # Should identify duplicate content issues
        duplicate_recs = [r for r in recommendations if "Duplicate Content" in r.title]
        assert len(duplicate_recs) > 0

    def test_comprehensive_recommendations(
        self, recommendation_engine, sample_keyword_data, sample_onpage_data
    ):
        """Test comprehensive recommendation generation."""
        result = recommendation_engine.generate_comprehensive_recommendations(
            keyword_data=sample_keyword_data, onpage_data=sample_onpage_data
        )

        assert "seo_score" in result
        assert "recommendations" in result
        assert "action_plan" in result
        assert "summary" in result

        # Verify SEO score structure
        seo_score = result["seo_score"]
        assert hasattr(seo_score, "overall_score")
        assert 0 <= seo_score.overall_score <= 100

        # Verify action plan structure
        action_plan = result["action_plan"]
        assert "immediate_actions" in action_plan
        assert "short_term_actions" in action_plan
        assert "long_term_actions" in action_plan

    def test_recommendation_prioritization(self, recommendation_engine):
        """Test recommendation prioritization logic."""
        recommendations = [
            SEORecommendation(
                title="Critical Issue",
                description="Fix immediately",
                priority=SeverityLevel.CRITICAL,
                category=RecommendationType.TECHNICAL,
                impact="High",
                effort="High",
                affected_pages=10,
            ),
            SEORecommendation(
                title="Medium Issue",
                description="Address soon",
                priority=SeverityLevel.MEDIUM,
                category=RecommendationType.CONTENT,
                impact="Medium",
                effort="Low",
                affected_pages=5,
            ),
            SEORecommendation(
                title="High Issue",
                description="Important fix",
                priority=SeverityLevel.HIGH,
                category=RecommendationType.KEYWORDS,
                impact="High",
                effort="Medium",
                affected_pages=15,
            ),
        ]

        prioritized = recommendation_engine._prioritize_recommendations(recommendations)

        assert len(prioritized) == 3
        assert prioritized[0].priority == SeverityLevel.CRITICAL
        assert prioritized[1].priority == SeverityLevel.HIGH
        assert prioritized[2].priority == SeverityLevel.MEDIUM


class TestEnhancedReporting:
    """Test suite for Enhanced Reporting functionality."""

    @pytest.fixture
    def seo_reporter(self):
        """Create SEO Reporter instance."""
        return SEOReporter(use_rich=False)  # Use plain text for testing

    @pytest.fixture
    def rich_reporter(self):
        """Create Rich Reporter instance."""
        return RichReporter()

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
                    "difficulty_score": 75,
                },
                {
                    "keyword": "keyword research",
                    "search_volume": 8100,
                    "cpc": 3.20,
                    "competition": 0.75,
                    "difficulty_score": 68,
                },
            ],
            "analysis_summary": {
                "total_keywords": 2,
                "avg_search_volume": 6750,
                "avg_competition": 0.80,
                "high_volume_count": 2,
            },
        }

    def test_plain_text_keyword_report(self, seo_reporter, sample_keyword_data):
        """Test plain text keyword report generation."""
        result = seo_reporter.generate_keyword_report(sample_keyword_data)

        assert isinstance(result, str)
        assert "seo tools" in result
        assert "keyword research" in result
        assert "5400" in result  # Search volume
        assert "8100" in result  # Search volume

    def test_rich_reporter_initialization(self, rich_reporter):
        """Test Rich Reporter initialization."""
        assert rich_reporter is not None
        assert hasattr(rich_reporter, "console")

    def test_report_generation_workflow(self, seo_reporter):
        """Test report generation workflow doesn't crash."""
        # Test with minimal data
        minimal_data = {
            "keywords_data": [{"keyword": "test", "search_volume": 100}],
            "analysis_summary": {"total_keywords": 1},
        }

        result = seo_reporter.generate_keyword_report(minimal_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_error_handling_in_reporting(self, seo_reporter):
        """Test error handling in report generation."""
        # Test with empty data
        empty_data = {}

        result = seo_reporter.generate_keyword_report(empty_data)
        assert isinstance(result, str)

        # Test with invalid data structure
        invalid_data = {"invalid": "structure"}

        result = seo_reporter.generate_keyword_report(invalid_data)
        assert isinstance(result, str)


class TestEnhancedWorkflow:
    """Integration tests for Enhanced workflow."""

    def test_keyword_to_recommendation_workflow(self):
        """Test complete workflow from keyword analysis to recommendations."""
        # Mock keyword analyzer results
        keyword_results = {
            "keywords_data": [
                {
                    "keyword": "seo tools",
                    "search_volume": 5000,
                    "competition": 0.8,
                    "position": None,
                },
                {
                    "keyword": "keyword research",
                    "search_volume": 3000,
                    "competition": 0.6,
                    "position": 15,
                },
            ]
        }

        # Convert to recommendation engine format
        keyword_performance_data = {}
        for kw in keyword_results["keywords_data"]:
            keyword_performance_data[kw["keyword"]] = {
                "search_volume": {"search_volume": kw["search_volume"]},
                "competition": {"competition": kw["competition"]},
                "position": kw.get("position"),
            }

        # Generate recommendations
        engine = SEORecommendationEngine()
        recommendations = engine.analyze_keyword_performance(keyword_performance_data)

        assert len(recommendations) > 0
        # Should generate recommendations for missing and low rankings
        rec_titles = [r.title for r in recommendations]
        assert any("Missing Keywords" in title for title in rec_titles)

    def test_reporting_with_recommendations(self):
        """Test reporting integration with recommendations."""
        # Create sample recommendation data
        engine = SEORecommendationEngine()
        recommendations = [
            SEORecommendation(
                title="Test Recommendation",
                description="Test description",
                priority=SeverityLevel.HIGH,
                category=RecommendationType.TECHNICAL,
                impact="High",
                effort="Medium",
            )
        ]

        # Test that recommendations can be serialized for reporting
        serialized = [rec.__dict__ for rec in recommendations]
        assert len(serialized) == 1
        assert serialized[0]["title"] == "Test Recommendation"

    def test_enhanced_functionality_integration(self):
        """Test that Enhanced components work together."""
        # Initialize components
        engine = SEORecommendationEngine()
        reporter = SEOReporter(use_rich=False)

        # Test data flow
        sample_data = {
            "keywords_data": [{"keyword": "test", "search_volume": 1000}],
            "analysis_summary": {"total_keywords": 1},
        }

        # Generate report
        report = reporter.generate_keyword_report(sample_data)
        assert isinstance(report, str)
        assert "test" in report

        # Generate recommendations (this should not crash)
        keyword_perf_data = {
            "test": {"search_volume": {"search_volume": 1000}, "position": None}
        }
        recommendations = engine.analyze_keyword_performance(keyword_perf_data)
        assert isinstance(recommendations, list)


# Test the actual Enhanced functionality we have
def test_enhanced_imports():
    """Test that Enhanced modules can be imported successfully."""
    # Test that we can import the enhanced components
    from mcp_seo.engines.recommendation_engine import SEORecommendationEngine
    from mcp_seo.reporting import SEOReporter
    from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer

    assert KeywordAnalyzer is not None
    assert SEORecommendationEngine is not None
    assert SEOReporter is not None


def test_recommendation_data_structures():
    """Test Enhanced recommendation data structures."""
    # Test SEORecommendation creation
    rec = SEORecommendation(
        title="Test Recommendation",
        description="Test description",
        priority=SeverityLevel.HIGH,
        category=RecommendationType.TECHNICAL,
        impact="High",
        effort="Medium",
    )

    assert rec.title == "Test Recommendation"
    assert rec.priority == SeverityLevel.HIGH
    assert rec.category == RecommendationType.TECHNICAL

    # Test SEOScore creation
    score = SEOScore(
        overall_score=75,
        technical_score=80,
        content_score=70,
        keywords_score=75,
        links_score=80,
        performance_score=85,
        breakdown={"technical": {"score": 80, "weight": 0.25}},
    )

    assert score.overall_score == 75
    assert score.technical_score == 80
    assert "technical" in score.breakdown


def test_enhanced_enums():
    """Test Enhanced enumeration types."""
    # Test SeverityLevel enum
    assert SeverityLevel.CRITICAL.value == "critical"
    assert SeverityLevel.HIGH.value == "high"
    assert SeverityLevel.MEDIUM.value == "medium"

    # Test RecommendationType enum
    assert RecommendationType.TECHNICAL.value == "technical"
    assert RecommendationType.CONTENT.value == "content"
    assert RecommendationType.KEYWORDS.value == "keywords"
    # Test RecommendationType enum
    assert RecommendationType.TECHNICAL.value == "technical"
    assert RecommendationType.CONTENT.value == "content"
    assert RecommendationType.KEYWORDS.value == "keywords"
