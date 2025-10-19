#!/usr/bin/env python3
"""
Integration tests for live API endpoints.
Tests the new synchronous live endpoints for keyword analysis, suggestions, and SERP.
"""

import pytest
import time
from src.mcp_seo.dataforseo.client import DataForSEOClient
from src.mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from src.mcp_seo.models.seo_models import (
    KeywordAnalysisRequest,
    SERPAnalysisRequest,
    DeviceType,
)


@pytest.fixture
def client():
    """Create DataForSEO client instance."""
    return DataForSEOClient()


@pytest.fixture
def analyzer(client):
    """Create KeywordAnalyzer instance."""
    return KeywordAnalyzer(client, use_rich_reporting=False)


class TestLiveKeywordAnalysis:
    """Test live keyword analysis endpoint."""

    def test_keyword_analysis_returns_data(self, analyzer):
        """Test that keyword analysis returns data quickly."""
        request = KeywordAnalysisRequest(
            keywords=["python code analyzer", "seo tools"],
            location="United States",
            language="en",
            include_suggestions=False,
        )

        start_time = time.time()
        result = analyzer.analyze_keywords(request)
        elapsed = time.time() - start_time

        # Verify response structure
        assert result["status"] == "completed"
        assert "keywords_data" in result
        assert len(result["keywords_data"]) > 0

        # Verify keyword data fields
        first_keyword = result["keywords_data"][0]
        assert "keyword" in first_keyword
        assert "search_volume" in first_keyword
        assert "competition" in first_keyword
        assert "cpc" in first_keyword

        # Verify performance - should be instant (< 3 seconds)
        assert elapsed < 3.0, f"Keyword analysis took {elapsed:.2f}s (expected < 3s)"
        print(f"✓ Keyword analysis completed in {elapsed:.2f}s")

    def test_keyword_analysis_with_suggestions(self, analyzer):
        """Test keyword analysis with suggestions included."""
        request = KeywordAnalysisRequest(
            keywords=["python"],
            location="United States",
            language="en",
            include_suggestions=True,
            suggestion_limit=10,
        )

        start_time = time.time()
        result = analyzer.analyze_keywords(request)
        elapsed = time.time() - start_time

        assert result["status"] == "completed"
        assert "keyword_suggestions" in result
        assert len(result["keyword_suggestions"]) > 0

        # Should still be fast even with suggestions
        assert elapsed < 5.0, f"Analysis with suggestions took {elapsed:.2f}s (expected < 5s)"
        print(f"✓ Keyword analysis with suggestions completed in {elapsed:.2f}s")

    def test_keyword_analysis_multiple_keywords(self, analyzer):
        """Test analysis with multiple keywords."""
        request = KeywordAnalysisRequest(
            keywords=[
                "python code analyzer",
                "python dependency analysis",
                "code dependency graph",
                "static analysis tool",
                "refactoring tool python",
            ],
            location="United States",
            language="en",
            include_suggestions=False,
        )

        result = analyzer.analyze_keywords(request)

        assert result["status"] == "completed"
        assert result["total_keywords"] == 5
        assert len(result["keywords_data"]) == 5

        # Verify each keyword has data
        for kw_data in result["keywords_data"]:
            assert kw_data["keyword"] is not None
            assert isinstance(kw_data.get("search_volume"), (int, type(None)))


class TestLiveKeywordSuggestions:
    """Test live keyword suggestions endpoint."""

    def test_keyword_suggestions_returns_data(self, analyzer):
        """Test that keyword suggestions return quickly."""
        start_time = time.time()
        result = analyzer.get_keyword_suggestions(
            seed_keyword="python seo",
            location="United States",
            language="en",
            limit=20,
        )
        elapsed = time.time() - start_time

        # Verify response
        assert result["status"] == "completed"
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

        # Verify suggestion structure
        first_suggestion = result["suggestions"][0]
        assert "keyword" in first_suggestion
        assert "search_volume" in first_suggestion

        # Verify performance - should be instant
        assert elapsed < 3.0, f"Suggestions took {elapsed:.2f}s (expected < 3s)"
        print(f"✓ Keyword suggestions completed in {elapsed:.2f}s")

    def test_keyword_suggestions_categorization(self, analyzer):
        """Test that suggestions are properly categorized."""
        result = analyzer.get_keyword_suggestions(
            seed_keyword="best python tools",
            location="United States",
            language="en",
            limit=50,
        )

        assert "suggestion_categories" in result
        categories = result["suggestion_categories"]

        # Verify category structure
        assert "high_volume" in categories
        assert "low_competition" in categories
        assert "long_tail" in categories
        assert "commercial_intent" in categories


class TestLiveSERPAnalysis:
    """Test live SERP analysis endpoint."""

    def test_serp_analysis_returns_quickly(self, analyzer):
        """Test that SERP analysis returns within 10 seconds."""
        request = SERPAnalysisRequest(
            keyword="python seo tools",
            location="United States",
            language="en",
            device=DeviceType.DESKTOP,
            depth=10,
        )

        start_time = time.time()
        result = analyzer.analyze_serp_for_keyword(request)
        elapsed = time.time() - start_time

        # Verify response
        assert result["status"] == "completed"
        assert "serp_analysis" in result

        # Verify SERP data
        serp = result["serp_analysis"]
        assert "organic_results" in serp
        assert len(serp["organic_results"]) > 0

        # Verify performance - should be fast (< 10s with live API)
        assert elapsed < 10.0, f"SERP analysis took {elapsed:.2f}s (expected < 10s)"
        print(f"✓ SERP analysis completed in {elapsed:.2f}s")

    def test_serp_analysis_structure(self, analyzer):
        """Test SERP result structure."""
        request = SERPAnalysisRequest(
            keyword="seo",
            location="United States",
            language="en",
            device=DeviceType.DESKTOP,
            depth=20,
        )

        result = analyzer.analyze_serp_for_keyword(request)
        serp = result["serp_analysis"]

        # Verify organic results structure
        assert "organic_results" in serp
        assert "total_organic_results" in serp

        if len(serp["organic_results"]) > 0:
            first_result = serp["organic_results"][0]
            assert "url" in first_result
            assert "title" in first_result
            assert "description" in first_result
            assert "position" in first_result

        # Verify competitive analysis
        assert "competitive_analysis" in serp
        comp = serp["competitive_analysis"]
        assert "total_results" in comp
        assert "unique_domains" in comp


class TestPerformanceComparison:
    """Performance benchmarks for live vs task-based APIs."""

    def test_keyword_analysis_performance(self, analyzer):
        """Benchmark keyword analysis performance."""
        request = KeywordAnalysisRequest(
            keywords=["seo tools", "seo software"],
            location="United States",
            language="en",
            include_suggestions=False,
        )

        # Run multiple times to get average
        times = []
        for i in range(3):
            start = time.time()
            result = analyzer.analyze_keywords(request)
            elapsed = time.time() - start
            times.append(elapsed)
            assert result["status"] == "completed"

        avg_time = sum(times) / len(times)
        print(f"\n✓ Average keyword analysis time: {avg_time:.2f}s (n=3)")
        print(f"  Min: {min(times):.2f}s, Max: {max(times):.2f}s")

        # Live API should average under 2 seconds
        assert avg_time < 2.0, f"Average time {avg_time:.2f}s exceeds 2s threshold"

    def test_serp_analysis_performance(self, analyzer):
        """Benchmark SERP analysis performance."""
        request = SERPAnalysisRequest(
            keyword="python",
            location="United States",
            language="en",
            device=DeviceType.DESKTOP,
            depth=10,
        )

        # Run multiple times
        times = []
        for i in range(3):
            start = time.time()
            result = analyzer.analyze_serp_for_keyword(request)
            elapsed = time.time() - start
            times.append(elapsed)
            assert result["status"] == "completed"

        avg_time = sum(times) / len(times)
        print(f"\n✓ Average SERP analysis time: {avg_time:.2f}s (n=3)")
        print(f"  Min: {min(times):.2f}s, Max: {max(times):.2f}s")

        # Live API should average under 8 seconds
        assert avg_time < 8.0, f"Average time {avg_time:.2f}s exceeds 8s threshold"


class TestErrorHandling:
    """Test error handling for live APIs."""

    def test_invalid_location(self, analyzer):
        """Test handling of invalid location."""
        request = KeywordAnalysisRequest(
            keywords=["test"],
            location="Invalid Location XYZ",
            language="en",
        )

        # Should still work with default location
        result = analyzer.analyze_keywords(request)
        assert "status" in result

    def test_empty_keywords(self, analyzer):
        """Test handling of empty keywords list."""
        request = KeywordAnalysisRequest(
            keywords=[],
            location="United States",
            language="en",
        )

        result = analyzer.analyze_keywords(request)
        # Should fail gracefully
        assert result["status"] == "failed" or result.get("total_keywords", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
