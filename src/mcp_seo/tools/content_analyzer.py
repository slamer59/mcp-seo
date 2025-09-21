#!/usr/bin/env python3
"""
Content Keyword Density Analyzer
=================================

Advanced content analysis tool for SEO keyword optimization and content strategy.
Provides detailed keyword density analysis, semantic gap detection, and long-tail
keyword opportunity identification.

Features:
- Keyword density analysis with optimal range detection
- Semantic keyword coverage analysis
- Long-tail keyword opportunity identification
- Content cannibalization detection
- Voice search optimization suggestions
- Competitive keyword gap analysis

Author: MCP SEO Team
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class KeywordOpportunity:
    """Represents a keyword optimization opportunity."""

    keyword: str
    current_count: int
    target_count: int
    opportunity_type: str  # 'missing', 'under_optimized', 'over_optimized'
    search_volume: Optional[int] = None
    difficulty: Optional[float] = None
    intent: Optional[str] = None


@dataclass
class ContentAnalysisResult:
    """Complete content analysis results."""

    url: str
    title: str
    word_count: int
    keyword_density: Dict[str, float]
    semantic_coverage: Dict[str, int]
    long_tail_opportunities: List[KeywordOpportunity]
    cannibalization_risk: List[Dict[str, Any]]
    optimization_score: float
    recommendations: List[Dict[str, Any]]


class ContentKeywordAnalyzer:
    """Advanced content keyword analysis and optimization."""

    def __init__(self):
        """Initialize the content analyzer."""
        self.stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "the",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "said",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "would",
            "about",
            "get",
            "all",
            "my",
            "can",
            "said",
        }

    def analyze_content(
        self,
        content: str,
        url: str = "",
        title: str = "",
        target_keywords: Optional[List[str]] = None,
        semantic_groups: Optional[Dict[str, List[str]]] = None,
        competitor_content: Optional[List[str]] = None,
    ) -> ContentAnalysisResult:
        """
        Perform comprehensive content keyword analysis.

        Args:
            content: The content text to analyze
            url: Optional URL of the content
            title: Optional title of the content
            target_keywords: List of target keywords to optimize for
            competitor_content: List of competitor content for gap analysis

        Returns:
            Complete content analysis results
        """
        logger.info(f"Analyzing content: {title or url}")

        # Clean and prepare content
        clean_content = self._clean_content(content)
        word_count = len(clean_content.split())

        # Extract all keywords and calculate density
        keyword_density = self._calculate_keyword_density(clean_content)

        # Analyze semantic coverage
        semantic_coverage = self._analyze_semantic_coverage(clean_content, semantic_groups)

        # Identify long-tail opportunities
        long_tail_opportunities = self._identify_long_tail_opportunities(
            clean_content, target_keywords or []
        )

        # Check for cannibalization risks
        cannibalization_risk = self._analyze_cannibalization_risk(
            clean_content, competitor_content or []
        )

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            keyword_density, semantic_coverage, word_count
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            keyword_density,
            semantic_coverage,
            long_tail_opportunities,
            word_count,
            target_keywords or [],
        )

        return ContentAnalysisResult(
            url=url,
            title=title,
            word_count=word_count,
            keyword_density=keyword_density,
            semantic_coverage=semantic_coverage,
            long_tail_opportunities=long_tail_opportunities,
            cannibalization_risk=cannibalization_risk,
            optimization_score=optimization_score,
            recommendations=recommendations,
        )

    def batch_analyze_content(
        self, content_items: List[Dict[str, str]], detect_cannibalization: bool = True
    ) -> List[ContentAnalysisResult]:
        """
        Analyze multiple content pieces and detect cannibalization.

        Args:
            content_items: List of dicts with 'content', 'url', 'title' keys
            detect_cannibalization: Whether to perform cannibalization analysis

        Returns:
            List of content analysis results
        """
        results = []
        all_content = [item["content"] for item in content_items]

        for i, item in enumerate(content_items):
            competitor_content = (
                all_content[:i] + all_content[i + 1 :] if detect_cannibalization else []
            )

            result = self.analyze_content(
                content=item["content"],
                url=item.get("url", ""),
                title=item.get("title", ""),
                competitor_content=competitor_content,
            )
            results.append(result)

        return results

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for analysis."""
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Remove markdown formatting
        content = re.sub(r"[#*_`\[\]()]", " ", content)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        return content.lower().strip()

    def _calculate_keyword_density(self, content: str) -> Dict[str, float]:
        """Calculate keyword density for all significant keywords."""
        words = content.split()
        total_words = len(words)

        if total_words == 0:
            return {}

        # Count all word combinations (1-5 grams)
        keyword_counts = Counter()

        # 1-grams (single words)
        for word in words:
            clean_word = re.sub(r"[^\w\s-]", "", word)
            if len(clean_word) > 2 and clean_word not in self.stop_words:
                keyword_counts[clean_word] += 1

        # 2-grams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            clean_phrase = re.sub(r"[^\w\s-]", "", phrase)
            if len(clean_phrase.split()) == 2:
                keyword_counts[clean_phrase] += 1

        # 3-grams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            clean_phrase = re.sub(r"[^\w\s-]", "", phrase)
            if len(clean_phrase.split()) == 3:
                keyword_counts[clean_phrase] += 1

        # Calculate density (percentage)
        keyword_density = {}
        for keyword, count in keyword_counts.items():
            density = (count / total_words) * 100
            if density >= 0.1:  # Only include keywords with at least 0.1% density
                keyword_density[keyword] = density

        return dict(sorted(keyword_density.items(), key=lambda x: x[1], reverse=True))

    def _analyze_semantic_coverage(self, content: str, semantic_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, int]:
        """Analyze semantic keyword coverage based on provided semantic groups."""
        if not semantic_groups:
            # If no semantic groups provided, auto-detect from content
            keyword_density = self._calculate_keyword_density(content)
            top_keywords = list(keyword_density.keys())[:20]
            semantic_groups = self._cluster_keywords_automatically(top_keywords)

        coverage = {}
        for group_name, keywords in semantic_groups.items():
            count = 0
            for keyword in keywords:
                # Count exact matches and variations
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, content))
                count += matches

                # Check variations (plurals, etc.)
                variations = [
                    keyword.lower() + 's',
                    keyword.lower() + 'ing',
                    keyword.lower().replace(' ', '-'),
                    keyword.lower().replace('-', ' ')
                ]

                for variation in variations:
                    if variation != keyword.lower():
                        var_pattern = r'\b' + re.escape(variation) + r'\b'
                        count += len(re.findall(var_pattern, content))

            coverage[group_name] = count

        return coverage

    def _cluster_keywords_automatically(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Automatically cluster keywords when no semantic groups are provided."""
        clusters = {}

        # Simple automatic clustering based on word patterns
        for keyword in keywords:
            words = keyword.lower().split()

            # Assign to cluster based on first significant word
            if any(word in ['setup', 'install', 'configure'] for word in words):
                clusters.setdefault('setup_process', []).append(keyword)
            elif any(word in ['best', 'top', 'comparison'] for word in words):
                clusters.setdefault('comparison_terms', []).append(keyword)
            elif any(word in ['how', 'tutorial', 'guide'] for word in words):
                clusters.setdefault('educational_content', []).append(keyword)
            elif len(words) >= 2:
                clusters.setdefault('multi_word_terms', []).append(keyword)
            else:
                clusters.setdefault('single_terms', []).append(keyword)

        return {k: v for k, v in clusters.items() if v}

    def _identify_long_tail_opportunities(
        self, content: str, target_keywords: List[str]
    ) -> List[KeywordOpportunity]:
        """Identify long-tail keyword opportunities."""
        opportunities = []

        # Define long-tail keyword templates based on common patterns
        long_tail_templates = [
            "best {keyword}",
            "how to {keyword}",
            "{keyword} tutorial",
            "{keyword} guide",
            "{keyword} vs {alternative}",
            "{keyword} for beginners",
            "free {keyword}",
            "{keyword} comparison",
            "{keyword} alternative",
            "{keyword} review",
            "top {keyword}",
            "{keyword} tips",
            "{keyword} examples",
            "{keyword} setup",
            "{keyword} configuration",
        ]

        # Check for missing long-tail variations
        for keyword in target_keywords:
            for template in long_tail_templates:
                if "{alternative}" in template:
                    # Skip alternative templates for now
                    continue

                long_tail = template.format(keyword=keyword)

                # Check if this long-tail keyword appears in content
                pattern = r"\b" + re.escape(long_tail.lower()) + r"\b"
                count = len(re.findall(pattern, content))

                if count == 0:
                    opportunities.append(
                        KeywordOpportunity(
                            keyword=long_tail,
                            current_count=0,
                            target_count=1,
                            opportunity_type="missing",
                            intent="informational"
                            if any(x in template for x in ["how", "tutorial", "guide"])
                            else "commercial",
                        )
                    )

        # Identify under-optimized existing keywords
        keyword_density = self._calculate_keyword_density(content)
        for keyword, density in keyword_density.items():
            if len(keyword.split()) >= 2:  # Multi-word keywords
                if density < 0.5:  # Less than 0.5% density
                    opportunities.append(
                        KeywordOpportunity(
                            keyword=keyword,
                            current_count=int(density * len(content.split()) / 100),
                            target_count=int(0.8 * len(content.split()) / 100),
                            opportunity_type="under_optimized",
                        )
                    )

        return opportunities[:20]  # Return top 20 opportunities

    def _analyze_cannibalization_risk(
        self, content: str, competitor_content: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze keyword cannibalization risk against other content."""
        risks = []

        if not competitor_content:
            return risks

        # Get top keywords from current content
        keyword_density = self._calculate_keyword_density(content)
        top_keywords = list(keyword_density.keys())[:10]

        for i, comp_content in enumerate(competitor_content):
            comp_density = self._calculate_keyword_density(comp_content)

            # Check for overlapping keywords
            overlapping_keywords = set(top_keywords) & set(comp_density.keys())

            if overlapping_keywords:
                # Calculate similarity score
                similarity_score = len(overlapping_keywords) / len(
                    set(top_keywords) | set(comp_density.keys())
                )

                if similarity_score > 0.3:  # More than 30% keyword overlap
                    risks.append(
                        {
                            "competitor_index": i,
                            "overlapping_keywords": list(overlapping_keywords),
                            "similarity_score": similarity_score,
                            "risk_level": "high"
                            if similarity_score > 0.6
                            else "medium",
                            "recommendation": "Consider differentiating content or merging pages",
                        }
                    )

        return risks

    def _calculate_optimization_score(
        self,
        keyword_density: Dict[str, float],
        semantic_coverage: Dict[str, int],
        word_count: int,
    ) -> float:
        """Calculate overall content optimization score (0-100)."""
        score = 0.0

        # Word count score (20 points)
        if word_count >= 1500:
            score += 20
        elif word_count >= 1000:
            score += 15
        elif word_count >= 500:
            score += 10
        else:
            score += 5

        # Keyword density score (30 points)
        if keyword_density:
            # Check for optimal density ranges
            optimal_keywords = 0
            for keyword, density in list(keyword_density.items())[:5]:
                if 0.5 <= density <= 3.0:  # Optimal range
                    optimal_keywords += 1
                elif 3.0 < density <= 5.0:  # Acceptable range
                    optimal_keywords += 0.5

            score += min(30, optimal_keywords * 6)

        # Semantic coverage score (25 points)
        total_semantic_coverage = sum(semantic_coverage.values())
        if total_semantic_coverage >= 15:
            score += 25
        elif total_semantic_coverage >= 10:
            score += 20
        elif total_semantic_coverage >= 5:
            score += 15
        else:
            score += 5

        # Keyword diversity score (25 points)
        unique_keywords = len(
            [k for k in keyword_density.keys() if len(k.split()) >= 2]
        )
        if unique_keywords >= 10:
            score += 25
        elif unique_keywords >= 7:
            score += 20
        elif unique_keywords >= 5:
            score += 15
        else:
            score += 10

        return min(100.0, score)

    def _generate_recommendations(
        self,
        keyword_density: Dict[str, float],
        semantic_coverage: Dict[str, int],
        long_tail_opportunities: List[KeywordOpportunity],
        word_count: int,
        target_keywords: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate actionable SEO recommendations."""
        recommendations = []

        # Word count recommendations
        if word_count < 500:
            recommendations.append(
                {
                    "type": "content_length",
                    "priority": "high",
                    "issue": f"Content too short ({word_count} words)",
                    "recommendation": "Expand content to at least 1000 words for better SEO performance",
                    "expected_impact": "high",
                }
            )

        # Keyword density recommendations
        over_optimized = [k for k, d in keyword_density.items() if d > 5.0]
        if over_optimized:
            recommendations.append(
                {
                    "type": "keyword_density",
                    "priority": "high",
                    "issue": f"Over-optimized keywords: {', '.join(over_optimized[:3])}",
                    "recommendation": "Reduce keyword density and use more natural variations",
                    "expected_impact": "medium",
                }
            )

        under_optimized = [
            k for k, d in keyword_density.items() if d < 0.5 and k in target_keywords
        ]
        if under_optimized:
            recommendations.append(
                {
                    "type": "keyword_density",
                    "priority": "medium",
                    "issue": f"Under-optimized target keywords: {', '.join(under_optimized[:3])}",
                    "recommendation": "Increase usage of target keywords naturally throughout content",
                    "expected_impact": "high",
                }
            )

        # Semantic coverage recommendations
        weak_semantic_areas = [
            area for area, count in semantic_coverage.items() if count < 3
        ]
        if weak_semantic_areas:
            recommendations.append(
                {
                    "type": "semantic_coverage",
                    "priority": "medium",
                    "issue": f"Weak semantic coverage in: {', '.join(weak_semantic_areas)}",
                    "recommendation": "Add more related terminology and concepts to improve topical authority",
                    "expected_impact": "medium",
                }
            )

        # Long-tail opportunities
        if len(long_tail_opportunities) > 5:
            high_value_opportunities = [
                op for op in long_tail_opportunities if op.intent == "commercial"
            ][:3]
            if high_value_opportunities:
                recommendations.append(
                    {
                        "type": "long_tail_keywords",
                        "priority": "high",
                        "issue": "Missing high-value long-tail keywords",
                        "recommendation": f"Add these long-tail variations: {', '.join([op.keyword for op in high_value_opportunities])}",
                        "expected_impact": "high",
                    }
                )

        return recommendations

    def generate_content_report(self, result: ContentAnalysisResult) -> str:
        """Generate a formatted content analysis report."""
        report = []

        report.append("=" * 60)
        report.append("CONTENT KEYWORD ANALYSIS REPORT")
        report.append("=" * 60)

        report.append(f"\n📄 CONTENT: {result.title or result.url}")
        report.append(f"   Word Count: {result.word_count:,}")
        report.append(f"   Optimization Score: {result.optimization_score:.1f}/100")

        # Top keywords
        report.append(f"\n🎯 TOP KEYWORDS:")
        for i, (keyword, density) in enumerate(
            list(result.keyword_density.items())[:10], 1
        ):
            status = "✅" if 0.5 <= density <= 3.0 else "⚠️" if density > 3.0 else "🔻"
            report.append(f'   {i:2d}. {status} "{keyword}": {density:.2f}%')

        # Semantic coverage
        report.append(f"\n📊 SEMANTIC COVERAGE:")
        for area, count in result.semantic_coverage.items():
            status = "✅" if count >= 5 else "⚠️" if count >= 3 else "🔻"
            report.append(
                f"   {status} {area.replace('_', ' ').title()}: {count} mentions"
            )

        # Long-tail opportunities
        if result.long_tail_opportunities:
            report.append(f"\n🔍 LONG-TAIL OPPORTUNITIES:")
            for i, opp in enumerate(result.long_tail_opportunities[:5], 1):
                report.append(f'   {i}. "{opp.keyword}" ({opp.opportunity_type})')

        # Cannibalization risks
        if result.cannibalization_risk:
            report.append(f"\n⚠️  CANNIBALIZATION RISKS:")
            for risk in result.cannibalization_risk:
                report.append(f"   Risk Level: {risk['risk_level'].upper()}")
                report.append(
                    f"   Overlapping Keywords: {', '.join(risk['overlapping_keywords'][:3])}"
                )

        # Recommendations
        if result.recommendations:
            report.append(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                priority_emoji = (
                    "🔴"
                    if rec["priority"] == "high"
                    else "🟡"
                    if rec["priority"] == "medium"
                    else "🟢"
                )
                report.append(f"   {i}. {priority_emoji} {rec['recommendation']}")
                report.append(f"      Impact: {rec['expected_impact'].title()}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
