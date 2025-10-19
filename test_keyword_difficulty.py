#!/usr/bin/env python3
"""Test the keyword difficulty function (new fast version)."""

from src.mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from src.mcp_seo.dataforseo.client import DataForSEOClient
import time

def main():
    # Initialize client and analyzer
    client = DataForSEOClient()
    analyzer = KeywordAnalyzer(client, use_rich_reporting=False)

    print("=" * 80)
    print("Testing Fast Keyword Difficulty Analysis")
    print("=" * 80)

    keywords = ["python code analyzer", "code dependency tool", "refactoring analyzer"]
    print(f"\nAnalyzing {len(keywords)} keywords:")
    for kw in keywords:
        print(f"  - {kw}")

    print("\nStarting analysis...")
    start_time = time.time()

    result = analyzer.get_keyword_difficulty(
        keywords=keywords,
        location="usa",
        language="english"
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n✓ Completed in {elapsed:.2f} seconds!")
    print(f"\nResults:")
    print(f"  Status: {result.get('status', 'unknown')}")

    if result.get("status") == "completed":
        for kw_data in result["keyword_difficulty"]:
            print(f"\n  Keyword: {kw_data['keyword']}")
            print(f"    Difficulty: {kw_data['difficulty_level']} ({kw_data['difficulty_score']}/100)")
            print(f"    Competition: {kw_data['competition']:.2f}")
            print(f"    Search Volume: {kw_data['search_volume']:,}")
    else:
        print(f"  Error: {result.get('error')}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
