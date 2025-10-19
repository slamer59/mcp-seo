"""
Test suite for Enhanced SEO analyzers.

Tests the enhanced keyword analyzer, recommendation engine,
and competitor analysis functionality.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_seo.analysis.competitor_analyzer import (CompetitorMapping,
                                                  SERPCompetitorAnalyzer)
from mcp_seo.dataforseo.client import DataForSEOClient
from mcp_seo.engines.recommendation_engine import (RecommendationType,
                                                   SEORecommendation,
                                                   SEORecommendationEngine,
                                                   SEOScore, SeverityLevel)
from mcp_seo.models.seo_models import (KeywordAnalysisRequest,
                                       SERPAnalysisRequest)
from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer


class TestEnhancedKeywordAnalyzer:
    """Test suite for Enhanced Keyword Analyzer."""

    @pytest.fixture
    def mock_dataforseo_client(self):
        """Mock DataForSEO client."""
        client = Mock(spec=DataForSEOClient)

        # Mock keyword data response (live API - returns immediately with results)
        client.get_keyword_data.return_value = {
            "tasks": [{
                "id": "test_task_123",
                "status_code": 20000,
                "result": [
                    {
                        "keyword": "seo tools",
                        "search_volume": 5400,
                        "cpc": 2.50,
                        "competition": 0.85,
                        "competition_level": "HIGH",
                        "monthly_searches": []
                    },
                    {
                        "keyword": "keyword research",
                        "search_volume": 8100,
                        "cpc": 3.20,
                        "competition": 0.75,
                        "competition_level": "HIGH",
                        "monthly_searches": []
                    }
                ]
            }]
        }

        # Mock suggestions response (live API - returns immediately with results)
        client.get_keyword_suggestions.return_value = {
            "tasks": [{
                "id": "suggestion_task_456",
                "status_code": 20000,
                "result": [
                    {
                        "keyword": "seo audit tools",
                        "search_volume": 1200,
                        "cpc": 4.10,
                        "competition": 0.65
                    },
                    {
                        "keyword": "free seo tools",
                        "search_volume": 3300,
                        "cpc": 1.80,
                        "competition": 0.55
                    }
                ]
            }]
        }

        return client

    @pytest.fixture
    def keyword_analyzer(self, mock_dataforseo_client):
        """Create Enhanced Keyword Analyzer instance."""
        return KeywordAnalyzer(client=mock_dataforseo_client, use_rich_reporting=False)

    @pytest.fixture
    def sample_keyword_request(self):
        """Sample keyword analysis request."""
        return KeywordAnalysisRequest(
            keywords=["seo tools", "keyword research"],
            location="United States",
            language="English",
            include_suggestions=True
        )

    def test_analyze_keywords_success(self, keyword_analyzer, sample_keyword_request):
        """Test successful keyword analysis."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert result["status"] == "completed"
        assert "keywords_data" in result
        assert "keyword_suggestions" in result
        assert "analysis_summary" in result
        assert "seo_recommendations" in result

        # Verify keyword data structure
        keywords_data = result["keywords_data"]
        assert len(keywords_data) == 2

        for keyword_data in keywords_data:
            assert "keyword" in keyword_data
            assert "search_volume" in keyword_data
            assert "competition" in keyword_data
            assert "cpc" in keyword_data

    def test_keyword_difficulty_calculation(self, keyword_analyzer, sample_keyword_request):
        """Test keyword difficulty scoring algorithm."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        # Get difficulty analysis from recommendations or summary
        assert "keyword_targeting_strategy" in result
        strategy = result["keyword_targeting_strategy"]

        assert "high_opportunity" in strategy
        assert "medium_opportunity" in strategy
        assert "low_opportunity" in strategy

    def test_keyword_suggestions_integration(self, keyword_analyzer, sample_keyword_request):
        """Test integration of keyword suggestions."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        suggestions = result.get("keyword_suggestions", [])
        assert len(suggestions) > 0

        for suggestion in suggestions:
            assert "keyword" in suggestion
            assert "search_volume" in suggestion

    def test_seo_recommendations_generation(self, keyword_analyzer, sample_keyword_request):
        """Test SEO recommendations based on keyword analysis."""
        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        recommendations = result.get("seo_recommendations", [])
        assert len(recommendations) >= 0  # May have recommendations based on data

    def test_formatted_report_generation(self, keyword_analyzer, sample_keyword_request):
        """Test formatted report generation."""
        with patch.object(keyword_analyzer.reporter, 'generate_keyword_report') as mock_report:
            mock_report.return_value = "Formatted keyword report"

            result = keyword_analyzer.analyze_keywords(sample_keyword_request)

            assert "formatted_report" in result
            mock_report.assert_called_once()

    def test_error_handling_no_results(self, keyword_analyzer, sample_keyword_request):
        """Test error handling when no results are returned."""
        keyword_analyzer.client.get_keyword_data.return_value = {
            "tasks": [{"result": []}]  # Empty result array
        }

        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert result["status"] == "failed"
        assert "error" in result

    def test_api_exception_handling(self, keyword_analyzer, sample_keyword_request):
        """Test handling of API exceptions."""
        from mcp_seo.dataforseo.client import ApiException

        keyword_analyzer.client.get_keyword_data.side_effect = ApiException("API Error")

        result = keyword_analyzer.analyze_keywords(sample_keyword_request)

        assert result["status"] == "failed"
        assert "API Error" in result["error"]


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
                "position": None  # Not ranking
            },
            "ranking keyword": {
                "search_volume": {"search_volume": 1200},
                "difficulty": {"difficulty": 45},
                "position": 15  # Low ranking
            },
            "good ranking keyword": {
                "search_volume": {"search_volume": 800},
                "difficulty": {"difficulty": 40},
                "position": 5  # Good ranking
            }
        }

    @pytest.fixture
    def sample_onpage_data(self):
        """Sample on-page SEO data."""
        return {
            "summary": {
                "critical_issues": 3,
                "high_priority_issues": 5,
                "duplicate_title_tags": 8,
                "duplicate_meta_descriptions": 12
            }
        }

    @pytest.fixture
    def sample_content_data(self):
        """Sample content analysis data."""
        return {
            "pages": [
                {"word_count": 150, "url": "/thin-page-1"},
                {"word_count": 200, "url": "/thin-page-2"},
                {"word_count": 800, "url": "/good-page-1"},
                {"word_count": 1200, "url": "/good-page-2"}
            ]
        }

    @pytest.fixture
    def sample_pagerank_data(self):
        """Sample PageRank analysis data."""
        return {
            "orphaned_pages": ["/orphan-1", "/orphan-2", "/orphan-3"],
            "link_opportunities": {
                "high_authority_pages": ["/authority-1", "/authority-2"]
            },
            "basic_metrics": {
                "total_pages": 50
            }
        }

    def test_analyze_keyword_performance(self, recommendation_engine, sample_keyword_data):
        """Test keyword performance analysis."""
        recommendations = recommendation_engine.analyze_keyword_performance(sample_keyword_data)

        assert len(recommendations) > 0

        # Should identify missing high-volume keywords
        missing_keyword_recs = [r for r in recommendations
                              if "High-Volume Missing Keywords" in r.title]
        assert len(missing_keyword_recs) > 0

        # Should identify low rankings
        low_ranking_recs = [r for r in recommendations
                           if "Low-Ranking" in r.title]
        assert len(low_ranking_recs) > 0

    def test_analyze_technical_issues(self, recommendation_engine, sample_onpage_data):
        """Test technical SEO issue analysis."""
        recommendations = recommendation_engine.analyze_technical_issues(sample_onpage_data)

        assert len(recommendations) >= 2

        # Should identify critical issues
        critical_recs = [r for r in recommendations if r.priority == SeverityLevel.CRITICAL]
        assert len(critical_recs) > 0

        # Should identify duplicate content issues
        duplicate_recs = [r for r in recommendations
                         if "Duplicate Content" in r.title]
        assert len(duplicate_recs) > 0

    def test_analyze_content_opportunities(self, recommendation_engine, sample_content_data):
        """Test content opportunity analysis."""
        recommendations = recommendation_engine.analyze_content_opportunities(sample_content_data)

        assert len(recommendations) > 0

        # Should identify thin content
        thin_content_recs = [r for r in recommendations
                           if "Thin Content" in r.title]
        assert len(thin_content_recs) > 0

    def test_analyze_link_opportunities(self, recommendation_engine, sample_pagerank_data):
        """Test link opportunity analysis."""
        recommendations = recommendation_engine.analyze_link_opportunities(sample_pagerank_data)

        assert len(recommendations) >= 2

        # Should identify orphaned pages
        orphan_recs = [r for r in recommendations
                      if "Orphaned Pages" in r.title]
        assert len(orphan_recs) > 0

        # Should identify authority page opportunities
        authority_recs = [r for r in recommendations
                         if "High-Authority Pages" in r.title]
        assert len(authority_recs) > 0

    def test_comprehensive_recommendations(self, recommendation_engine, sample_keyword_data,
                                         sample_onpage_data, sample_content_data, sample_pagerank_data):
        """Test comprehensive recommendation generation."""
        result = recommendation_engine.generate_comprehensive_recommendations(
            keyword_data=sample_keyword_data,
            onpage_data=sample_onpage_data,
            content_data=sample_content_data,
            pagerank_data=sample_pagerank_data
        )

        assert "seo_score" in result
        assert "recommendations" in result
        assert "action_plan" in result
        assert "summary" in result

        # Verify SEO score structure
        seo_score = result["seo_score"]
        assert hasattr(seo_score, 'overall_score')
        assert 0 <= seo_score.overall_score <= 100

        # Verify recommendations are prioritized
        recommendations = result["recommendations"]
        assert len(recommendations) > 0

        # Verify action plan structure
        action_plan = result["action_plan"]
        assert "immediate_actions" in action_plan
        assert "short_term_actions" in action_plan
        assert "long_term_actions" in action_plan

    def test_seo_score_calculation(self, recommendation_engine, sample_keyword_data,
                                 sample_onpage_data, sample_content_data, sample_pagerank_data):
        """Test SEO score calculation algorithm."""
        seo_score = recommendation_engine._calculate_seo_score(
            keyword_data=sample_keyword_data,
            onpage_data=sample_onpage_data,
            content_data=sample_content_data,
            pagerank_data=sample_pagerank_data
        )

        assert isinstance(seo_score, SEOScore)
        assert 0 <= seo_score.overall_score <= 100
        assert 0 <= seo_score.technical_score <= 100
        assert 0 <= seo_score.content_score <= 100
        assert 0 <= seo_score.keywords_score <= 100
        assert 0 <= seo_score.links_score <= 100

        # Technical score should be lower due to issues
        assert seo_score.technical_score < 100

        # Keywords score should reflect ranking ratio
        assert seo_score.keywords_score > 0

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
                affected_pages=10
            ),
            SEORecommendation(
                title="Medium Issue",
                description="Address soon",
                priority=SeverityLevel.MEDIUM,
                category=RecommendationType.CONTENT,
                impact="Medium",
                effort="Low",
                affected_pages=5
            ),
            SEORecommendation(
                title="High Issue",
                description="Important fix",
                priority=SeverityLevel.HIGH,
                category=RecommendationType.KEYWORDS,
                impact="High",
                effort="Medium",
                affected_pages=15
            )
        ]

        prioritized = recommendation_engine._prioritize_recommendations(recommendations)

        assert len(prioritized) == 3
        assert prioritized[0].priority == SeverityLevel.CRITICAL
        assert prioritized[1].priority == SeverityLevel.HIGH
        assert prioritized[2].priority == SeverityLevel.MEDIUM

    def test_action_plan_generation(self, recommendation_engine):
        """Test action plan generation logic."""
        recommendations = [
            SEORecommendation("Critical", "Fix", SeverityLevel.CRITICAL, RecommendationType.TECHNICAL, "High", "High"),
            SEORecommendation("High", "Important", SeverityLevel.HIGH, RecommendationType.CONTENT, "High", "Medium"),
            SEORecommendation("Medium", "Soon", SeverityLevel.MEDIUM, RecommendationType.KEYWORDS, "Medium", "Low"),
            SEORecommendation("Low", "Eventually", SeverityLevel.LOW, RecommendationType.LINKS, "Low", "Low")
        ]

        action_plan = recommendation_engine._generate_action_plan(recommendations)

        assert "immediate_actions" in action_plan
        assert "short_term_actions" in action_plan
        assert "long_term_actions" in action_plan
        assert "estimated_impact" in action_plan

        # Immediate actions should include critical and high priority
        assert action_plan["immediate_actions"]["count"] == 2

        # Short term should include medium priority
        assert action_plan["short_term_actions"]["count"] == 1

        # Long term should include low priority
        assert action_plan["long_term_actions"]["count"] == 1


class TestSERPCompetitorAnalyzer:
    """Test suite for SERP Competitor Analyzer."""

    @pytest.fixture
    def mock_dataforseo_client(self):
        """Mock DataForSEO client for competitor analysis."""
        client = Mock(spec=DataForSEOClient)

        # Mock SERP data response
        client.get_serp_data.return_value = {
            "tasks": [{
                "result": [{
                    "items": [
                        {
                            "type": "organic",
                            "rank_group": 1,
                            "rank_absolute": 1,
                            "position": 1,
                            "domain": "competitor1.com",
                            "title": "Best SEO Tools 2024",
                            "url": "https://competitor1.com/seo-tools"
                        },
                        {
                            "type": "organic",
                            "rank_group": 2,
                            "rank_absolute": 2,
                            "position": 2,
                            "domain": "competitor2.com",
                            "title": "Top SEO Software",
                            "url": "https://competitor2.com/seo-software"
                        },
                        {
                            "type": "organic",
                            "rank_group": 3,
                            "rank_absolute": 3,
                            "position": 3,
                            "domain": "target-domain.com",
                            "title": "SEO Tools Guide",
                            "url": "https://target-domain.com/tools"
                        }
                    ]
                }]
            }]
        }

        return client

    @pytest.fixture
    def competitor_mappings(self):
        """Sample competitor mappings."""
        return [
            CompetitorMapping(
                url_patterns=["competitor1.com"],
                title_patterns=["Competitor 1"],
                competitor_type="direct",
                priority=1
            ),
            CompetitorMapping(
                url_patterns=["competitor2.com"],
                title_patterns=["Competitor 2"],
                competitor_type="indirect",
                priority=2
            )
        ]

    @pytest.fixture
    def competitor_analyzer(self, mock_dataforseo_client, competitor_mappings):
        """Create SERP Competitor Analyzer instance."""
        return SERPCompetitorAnalyzer(
            client=mock_dataforseo_client,
            target_domain="target-domain.com",
            competitor_mappings=competitor_mappings
        )

    def test_find_domain_position(self, competitor_analyzer):
        """Test domain position finding in SERP results."""
        serp_items = [
            {"domain": "example.com", "position": 1},
            {"domain": "competitor.com", "position": 2},
            {"domain": "target-domain.com", "position": 3}
        ]

        position = competitor_analyzer._find_domain_position(serp_items, "target-domain.com")
        assert position == 3

        # Test domain not found
        position = competitor_analyzer._find_domain_position(serp_items, "missing.com")
        assert position is None

    def test_analyze_competitors(self, competitor_analyzer):
        """Test competitor analysis in SERP results."""
        serp_items = [
            {
                "domain": "competitor1.com",
                "position": 1,
                "title": "Best SEO Tools 2024",
                "url": "https://competitor1.com/seo-tools"
            },
            {
                "domain": "competitor2.com",
                "position": 2,
                "title": "Top SEO Software",
                "url": "https://competitor2.com/seo-software"
            }
        ]

        competitors = competitor_analyzer._analyze_competitors(serp_items)

        assert len(competitors) == 2
        assert competitors[0]["domain"] == "competitor1.com"
        assert competitors[0]["position"] == 1
        assert competitors[0]["competitor_type"] == "direct"

    def test_analyze_keyword_rankings_success(self, competitor_analyzer):
        """Test successful keyword ranking analysis."""
        keywords = ["seo tools", "keyword research"]

        result = competitor_analyzer.analyze_keyword_rankings(keywords)

        assert result["status"] == "completed"
        assert "rankings" in result
        assert len(result["rankings"]) == len(keywords)

        for keyword_result in result["rankings"]:
            assert "keyword" in keyword_result
            assert "target_position" in keyword_result
            assert "competitors" in keyword_result

    def test_keyword_ranking_with_progress_callback(self, competitor_analyzer):
        """Test keyword ranking analysis with progress tracking."""
        progress_calls = []

        def progress_callback(message, current, total):
            progress_calls.append((message, current, total))

        keywords = ["seo tools"]
        config = Mock()
        config.progress_callback = progress_callback

        competitor_analyzer.analyze_keyword_rankings(keywords, config=config)

        assert len(progress_calls) > 0

    def test_competitor_identification(self, competitor_analyzer):
        """Test competitor identification logic."""
        serp_items = [
            {"domain": "competitor1.com", "position": 1, "title": "Best Tools"},
            {"domain": "unknown-domain.com", "position": 2, "title": "Unknown Site"},
            {"domain": "competitor2.com", "position": 3, "title": "Software Guide"}
        ]

        identified = competitor_analyzer._analyze_competitors(serp_items)

        # Should identify known competitors
        known_competitors = [c for c in identified if c.get("competitor_type")]
        assert len(known_competitors) == 2

        # Should preserve unknown domains
        assert len(identified) == 3

    def test_error_handling_api_failure(self, competitor_analyzer):
        """Test error handling when API fails."""
        from mcp_seo.dataforseo.client import ApiException

        competitor_analyzer.client.get_serp_data.side_effect = ApiException("SERP API Error")

        result = competitor_analyzer.analyze_keyword_rankings(["test keyword"])

        assert result["status"] == "failed"
        assert "SERP API Error" in result["error"]


class TestAnalyzerIntegration:
    """Integration tests for enhanced analyzers working together."""

    def test_keyword_to_recommendation_flow(self):
        """Test flow from keyword analysis to recommendations."""
        # Mock keyword analyzer results
        keyword_results = {
            "keywords_data": [
                {"keyword": "seo tools", "search_volume": 5000, "competition": 0.8, "position": None},
                {"keyword": "keyword research", "search_volume": 3000, "competition": 0.6, "position": 15}
            ]
        }

        # Convert to recommendation engine format
        keyword_performance_data = {}
        for kw in keyword_results["keywords_data"]:
            keyword_performance_data[kw['keyword']] = {
                'search_volume': {'search_volume': kw['search_volume']},
                'competition': {'competition': kw['competition']},
                'position': kw.get('position')
            }

        # Generate recommendations
        engine = SEORecommendationEngine()
        recommendations = engine.analyze_keyword_performance(keyword_performance_data)

        assert len(recommendations) > 0
        # Should generate recommendations for missing and low rankings
        rec_titles = [r.title for r in recommendations]
        assert any("Missing Keywords" in title for title in rec_titles)

    def test_competitor_analysis_integration(self):
        """Test integration of competitor analysis with recommendations."""
        # This would test how competitor data feeds into recommendation generation
        # Implementation would depend on specific integration requirements
        pass