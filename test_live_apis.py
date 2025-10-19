#!/usr/bin/env python3
"""
Simple test script for live DataForSEO APIs without MCP.
Tests keyword analysis, suggestions, and SERP analysis.
"""

from src.mcp_seo.dataforseo.client import DataForSEOClient
from src.mcp_seo.models.seo_models import (
    DeviceType,
    KeywordAnalysisRequest,
    SERPAnalysisRequest,
)
from src.mcp_seo.tools.keyword_analyzer import KeywordAnalyzer


def main():
    # Initialize client and analyzer
    client = DataForSEOClient()
    analyzer = KeywordAnalyzer(client, use_rich_reporting=False)

    print("=" * 80)
    print("DataForSEO Live API Tests")
    print("=" * 80)

    # Test 1: Keyword Analysis
    print("\n1. Testing Keyword Analysis...")
    print("-" * 80)

    keyword_request = KeywordAnalysisRequest(
        keywords=["python code analyzer", "seo tools"],
        location="United States",
        language="en",
        include_suggestions=True,
    )

    keyword_result = analyzer.analyze_keywords(keyword_request)

    if keyword_result.get("status") == "completed":
        print(f"✓ Success! Found {len(keyword_result['keywords_data'])} keywords")
        for kw in keyword_result["keywords_data"]:
            print(
                f"  - {kw['keyword']}: {kw['search_volume']:,} monthly searches, CPC: ${kw['cpc']:.2f}"
            )
    else:
        print(f"✗ Failed: {keyword_result.get('error')}")

    # Test 2: Keyword Suggestions
    print("\n2. Testing Keyword Suggestions...")
    print("-" * 80)

    suggestions_result = analyzer.get_keyword_suggestions(
        seed_keyword="python seo",
        location="United States",
        language="en",
        limit=10,
    )

    if suggestions_result.get("status") == "completed":
        print(f"✓ Success! Found {len(suggestions_result['suggestions'])} suggestions")
        for i, sugg in enumerate(suggestions_result["suggestions"][:5], 1):
            volume = sugg.get("search_volume", 0) or 0
            print(f"  {i}. {sugg['keyword']}: {volume:,} searches/month")
    else:
        print(f"✗ Failed: {suggestions_result.get('error')}")

    # Test 3: SERP Analysis
    print("\n3. Testing SERP Analysis...")
    print("-" * 80)

    serp_request = SERPAnalysisRequest(
        keyword="python code analyzer",
        location="United States",
        language="en",
        device=DeviceType.DESKTOP,
        depth=10,
    )

    serp_result = analyzer.analyze_serp_for_keyword(serp_request)

    if "serp_analysis" in serp_result:
        serp = serp_result["serp_analysis"]
        print(f"✓ Success! Found {len(serp['organic_results'])} organic results")
        print(f"\nTop 3 Results:")
        for i, result in enumerate(serp["organic_results"][:3], 1):
            print(f"  {i}. {result['title']}")
            print(f"     {result['url']}")
            desc = result.get('description', '') or ''
            print(f"     {desc[:100]}...")
    else:
        print(f"✗ Failed: {serp_result.get('error')}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
    main()
