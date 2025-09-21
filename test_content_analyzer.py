#!/usr/bin/env python3
"""
Quick test for the content analyzer functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mcp_seo.tools.content_analyzer import ContentKeywordAnalyzer

def test_content_analyzer():
    """Simple test of the content keyword analyzer."""

    # Sample content similar to your GitLab example
    content = """
    GitLab Android clients provide mobile access to your repositories and CI/CD pipelines.
    The best GitLab Android app offers native performance for mobile development teams.
    Enterprise users need GitLab mobile app features like offline mode and push notifications.
    Mobile code review capabilities make GitLab Android superior to GitHub mobile.
    Native GitLab Android client supports self-hosted enterprise installations.
    """

    # Target keywords and semantic groups (LLM would provide these)
    target_keywords = ["gitlab android", "mobile app", "enterprise"]
    semantic_groups = {
        "technology": ["android", "mobile", "app", "native", "client"],
        "devops": ["gitlab", "ci/cd", "pipeline", "repository"],
        "business": ["enterprise", "team", "self-hosted"],
        "features": ["offline", "notifications", "code review"]
    }

    # Test the analyzer
    analyzer = ContentKeywordAnalyzer()

    result = analyzer.analyze_content(
        content=content,
        title="Top GitLab Android Clients",
        target_keywords=target_keywords,
        semantic_groups=semantic_groups
    )

    # Print results
    print("🎯 CONTENT KEYWORD ANALYSIS TEST")
    print("=" * 50)
    print(f"Word Count: {result.word_count}")
    print(f"Optimization Score: {result.optimization_score:.1f}/100")

    print(f"\n📊 TOP KEYWORDS:")
    for i, (keyword, density) in enumerate(list(result.keyword_density.items())[:5], 1):
        status = "✅" if 0.5 <= density <= 3.0 else "⚠️" if density > 3.0 else "🔻"
        print(f"   {i}. {status} \"{keyword}\": {density:.2f}%")

    print(f"\n📈 SEMANTIC COVERAGE:")
    for group, count in result.semantic_coverage.items():
        status = "✅" if count >= 3 else "⚠️" if count >= 1 else "🔻"
        print(f"   {status} {group}: {count} mentions")

    print(f"\n🔍 LONG-TAIL OPPORTUNITIES ({len(result.long_tail_opportunities)} found):")
    for i, opp in enumerate(result.long_tail_opportunities[:3], 1):
        print(f"   {i}. \"{opp.keyword}\" ({opp.opportunity_type})")

    print(f"\n💡 RECOMMENDATIONS ({len(result.recommendations)} total):")
    for i, rec in enumerate(result.recommendations[:2], 1):
        priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
        print(f"   {i}. {priority_emoji} {rec['recommendation']}")

    print("\n✅ Test completed successfully!")
    return result

if __name__ == "__main__":
    test_content_analyzer()