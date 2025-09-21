"""
SERP Competitor Analysis Module

This module provides enhanced SERP analysis and competitor intelligence patterns
for enhanced SEO competitor analysis. It offers generic, reusable functionality
for analyzing competitor positions, identifying domain positions in search results,
and comprehensive keyword ranking analysis.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from mcp_seo.config.settings import get_language_code, get_location_code
from mcp_seo.dataforseo.client import DataForSEOClient


@dataclass
class CompetitorMapping:
    """Configuration for identifying competitors in SERP results."""

    url_patterns: List[str]
    title_patterns: List[str]
    competitor_type: str
    priority: int = 1  # Higher priority competitors are identified first


@dataclass
class KeywordAnalysisConfig:
    """Configuration for comprehensive keyword analysis."""

    location: str = "United States"
    include_serp: bool = True
    include_difficulty: bool = True
    include_volume: bool = True
    rate_limit_delay: float = 1.0
    progress_callback: Optional[Callable[[str, int, int], None]] = None


class SERPCompetitorAnalyzer:
    """
    Enhanced SERP competitor analysis with flexible competitor identification.

    Enhanced competitor analysis engine that provides
    reusable patterns for analyzing competitor positions in search results and
    comprehensive keyword ranking analysis.
    """

    def __init__(
        self,
        client: DataForSEOClient,
        target_domain: Optional[str] = None,
        competitor_mappings: Optional[List[CompetitorMapping]] = None,
    ):
        """
        Initialize SERP competitor analyzer.

        Args:
            client: DataForSEO API client
            target_domain: Optional target domain for analysis
            competitor_mappings: Optional list of competitor identification rules
        """
        self.client = client
        self.target_domain = target_domain
        self.competitor_mappings = competitor_mappings or []

    def get_serp_data(self, keyword: str, location: str = "United States") -> Dict[str, Any]:
        """Get SERP data for a keyword."""
        return self._get_serp_data(keyword, location)

    def analyze_keyword_rankings(self, keywords: List[str], config=None) -> Dict[str, Any]:
        """Analyze keyword rankings for the target domain."""
        try:
            results = {
                "status": "completed",
                "rankings": []
            }

            for keyword in keywords:
                if config and config.progress_callback:
                    config.progress_callback(f"Analyzing {keyword}", len(results["rankings"]) + 1, len(keywords))

                try:
                    serp_data = self.get_serp_data(keyword)
                    target_position = self._find_domain_position(serp_data.get("tasks", [{}])[0].get("result", [{}])[0].get("items", []), self.target_domain)
                    competitors = self._analyze_competitors(serp_data.get("tasks", [{}])[0].get("result", [{}])[0].get("items", []))

                    keyword_result = {
                        "keyword": keyword,
                        "target_position": target_position,
                        "competitors": competitors
                    }
                    results["rankings"].append(keyword_result)

                except Exception as e:
                    # For API exceptions, fail the entire operation
                    from ..dataforseo.client import ApiException
                    if isinstance(e, ApiException):
                        raise e

                    keyword_result = {
                        "keyword": keyword,
                        "target_position": None,
                        "competitors": [],
                        "error": str(e)
                    }
                    results["rankings"].append(keyword_result)

            return results

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "rankings": []
            }

    def _find_domain_position(self, serp_items: List[Dict], domain: str) -> Optional[int]:
        """Find domain position in SERP results."""
        if not domain:
            return None

        for item in serp_items:
            item_domain = item.get("domain", "")
            if domain.lower() in item_domain.lower():
                return item.get("position", item.get("rank_group"))
        return None

    def _analyze_competitors(self, serp_items: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze competitors in SERP results."""
        competitors = []

        for item in serp_items:
            domain = item.get("domain", "")

            # Identify competitor type using mappings
            # Use domain as URL if URL is not available
            url = item.get("url", "") or f"https://{domain}"
            competitor_info = self._identify_competitor(
                url,
                item.get("title", ""),
                item.get("description", ""),
                self.competitor_mappings
            )

            competitor_data = {
                "domain": domain,
                "position": item.get("position", item.get("rank_group")),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "competitor_type": competitor_info["type"] if competitor_info else None
            }
            competitors.append(competitor_data)

        return competitors

    def find_domain_position(
        self,
        serp_data: Dict,
        domain: str,
        exact_match: bool = False,
        subdomain_match: bool = True,
    ) -> Optional[int]:
        """
        Find domain's position in SERP results with flexible matching options.

        Enhanced domain position analysis with additional
        matching options for more precise domain identification.

        Args:
            serp_data: SERP results data from DataForSEO
            domain: Target domain to find
            exact_match: If True, requires exact domain match
            subdomain_match: If True, matches subdomains (e.g., blog.example.com for example.com)

        Returns:
            Position (1-indexed) if found, None otherwise
        """
        if not serp_data.get("tasks") or not serp_data["tasks"][0].get("result"):
            return None

        # Handle both task-based and direct results format
        results = None
        if "tasks" in serp_data and serp_data["tasks"][0].get("result"):
            task_result = serp_data["tasks"][0]["result"][0]
            results = task_result.get("items", [])
        elif "results" in serp_data:
            results = serp_data["results"]

        if not results:
            return None

        # Normalize domain for comparison
        domain_clean = (
            domain.lower()
            .replace("www.", "")
            .replace("http://", "")
            .replace("https://", "")
            .strip("/")
        )

        for i, result in enumerate(results, 1):
            url = result.get("url", "").lower()

            if exact_match:
                # Extract domain from URL for exact matching
                if "://" in url:
                    url_domain = url.split("://")[1].split("/")[0].replace("www.", "")
                    if url_domain == domain_clean:
                        return i
            else:
                # Flexible matching (original behavior)
                if subdomain_match:
                    if domain_clean in url:
                        return i
                else:
                    # More precise matching without subdomain matching
                    url_parts = url.replace("www.", "").split("/")
                    if len(url_parts) > 0 and domain_clean in url_parts[0]:
                        return i

        return None

    def analyze_serp_competitors(
        self,
        serp_data: Dict,
        custom_mappings: Optional[List[CompetitorMapping]] = None,
        top_n: int = 10,
        include_all_results: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Analyze competitor positions in SERP results with flexible identification.

        Enhanced competitor analysis method with configurable
        competitor identification and additional analysis features.

        Args:
            serp_data: SERP results data from DataForSEO
            custom_mappings: Optional competitor mappings for this analysis
            top_n: Number of top results to analyze
            include_all_results: If True, include all results even if not identified as competitors

        Returns:
            List of competitor analysis data
        """
        competitors = []
        mappings = custom_mappings or self.competitor_mappings

        # Handle both task-based and direct results format
        results = None
        if "tasks" in serp_data and serp_data["tasks"][0].get("result"):
            task_result = serp_data["tasks"][0]["result"][0]
            results = task_result.get("items", [])
        elif "results" in serp_data:
            results = serp_data["results"]

        if not results:
            return competitors

        for i, result in enumerate(results[:top_n], 1):
            url = result.get("url", "")
            title = result.get("title", "")
            description = result.get("description", "")
            domain = self._extract_domain(url)

            # Identify competitor type using mappings
            competitor_info = self._identify_competitor(
                url, title, description, mappings
            )

            result_data = {
                "position": i,
                "url": url,
                "domain": domain,
                "title": title,
                "description": description,
                "is_competitor": competitor_info is not None,
                "competitor_type": competitor_info["type"] if competitor_info else None,
                "competitor_priority": competitor_info["priority"]
                if competitor_info
                else None,
                "is_paid": result.get("type") == "paid",
                "breadcrumb": result.get("breadcrumb", ""),
                "highlighted": result.get("highlighted", []),
                "about_this_result": result.get("about_this_result", {}),
                "related_searches": result.get("related_searches", []),
            }

            # Add additional metrics if available
            if "rank_group" in result:
                result_data["rank_group"] = result["rank_group"]
            if "rank_absolute" in result:
                result_data["rank_absolute"] = result["rank_absolute"]

            # Include all results or only identified competitors
            if include_all_results or competitor_info:
                competitors.append(result_data)

        return competitors

    def analyze_keyword_rankings_comprehensive(
        self, keywords: List[str], target_domain: str, config: KeywordAnalysisConfig
    ) -> Dict[str, Any]:
        """
        Comprehensive keyword ranking analysis for target domain and competitors.

        Enhanced keyword ranking analysis method with improved
        configuration options and progress tracking.

        Args:
            keywords: List of keywords to analyze
            target_domain: Primary domain to track
            config: Analysis configuration

        Returns:
            Comprehensive analysis results
        """
        results = {}
        total_keywords = len(keywords)

        for idx, keyword in enumerate(keywords):
            if config.progress_callback:
                config.progress_callback(
                    f"Analyzing: {keyword}", idx + 1, total_keywords
                )

            keyword_result = {
                "keyword": keyword,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "target_domain": target_domain,
            }

            try:
                # SERP Analysis
                if config.include_serp:
                    serp_data = self._get_serp_data(keyword, config.location)
                    keyword_result["serp_results"] = serp_data
                    keyword_result["target_position"] = self.find_domain_position(
                        serp_data, target_domain
                    )
                    keyword_result["competitor_analysis"] = (
                        self.analyze_serp_competitors(serp_data)
                    )
                    keyword_result["serp_features"] = self._analyze_serp_features(
                        serp_data
                    )

                # Keyword Difficulty
                if config.include_difficulty:
                    difficulty_data = self._get_keyword_difficulty(
                        keyword, config.location
                    )
                    keyword_result["difficulty"] = difficulty_data
                    keyword_result["difficulty_score"] = self._extract_difficulty_score(
                        difficulty_data
                    )

                # Search Volume
                if config.include_volume:
                    volume_data = self._get_search_volume(keyword, config.location)
                    keyword_result["search_volume"] = volume_data
                    keyword_result["volume_metrics"] = self._extract_volume_metrics(
                        volume_data
                    )

                # Additional analysis
                keyword_result["opportunity_score"] = (
                    self._calculate_keyword_opportunity(keyword_result)
                )
                keyword_result["competitive_intensity"] = (
                    self._calculate_competitive_intensity(keyword_result)
                )

            except Exception as e:
                keyword_result["error"] = str(e)
                keyword_result["status"] = "failed"
            else:
                keyword_result["status"] = "completed"

            results[keyword] = keyword_result

            # Rate limiting
            if idx < total_keywords - 1:  # Don't sleep after last keyword
                time.sleep(config.rate_limit_delay)

        # Generate summary statistics
        results["_summary"] = self._generate_analysis_summary(results, target_domain)

        return results

    def batch_competitor_analysis(
        self,
        keywords: List[str],
        competitor_domains: List[str],
        config: KeywordAnalysisConfig,
    ) -> Dict[str, Any]:
        """
        Batch analysis of multiple competitors across keywords.

        Args:
            keywords: Keywords to analyze
            competitor_domains: List of competitor domains to track
            config: Analysis configuration

        Returns:
            Comprehensive competitor comparison data
        """
        all_results = {}

        # Analyze each competitor domain
        for domain in competitor_domains:
            domain_results = self.analyze_keyword_rankings_comprehensive(
                keywords, domain, config
            )
            all_results[domain] = domain_results

        # Generate cross-competitor analysis
        comparison_data = self._generate_competitor_comparison(all_results, keywords)

        return {
            "individual_analyses": all_results,
            "competitor_comparison": comparison_data,
            "analysis_config": config,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _identify_competitor(
        self, url: str, title: str, description: str, mappings: List[CompetitorMapping]
    ) -> Optional[Dict[str, Any]]:
        """Identify competitor using configured mappings."""
        url_lower = url.lower()
        title_lower = title.lower()
        description_lower = description.lower()

        # Sort mappings by priority (higher first)
        sorted_mappings = sorted(mappings, key=lambda x: x.priority, reverse=True)

        for mapping in sorted_mappings:
            # Check URL patterns
            for pattern in mapping.url_patterns:
                if pattern.lower() in url_lower:
                    return {
                        "type": mapping.competitor_type,
                        "priority": mapping.priority,
                        "matched_pattern": pattern,
                    }

            # Check title patterns
            for pattern in mapping.title_patterns:
                if pattern.lower() in title_lower:
                    return {
                        "type": mapping.competitor_type,
                        "priority": mapping.priority,
                        "matched_pattern": pattern,
                    }

        return None

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        if "://" in url:
            domain = url.split("://")[1].split("/")[0]
        else:
            domain = url.split("/")[0]

        return domain.replace("www.", "").lower()

    def _get_serp_data(self, keyword: str, location: str) -> Dict:
        """Get SERP data for keyword using DataForSEO client."""
        try:
            # Try to use the get_serp_data method if it exists on the client
            if hasattr(self.client, 'get_serp_data'):
                return self.client.get_serp_data(keyword, location)

            # Fallback to direct API call
            location_code = get_location_code(location)
            language_code = get_language_code("english")  # Default to English

            data = [
                {
                    "keyword": keyword,
                    "location_code": location_code,
                    "language_code": language_code,
                    "device": "desktop",
                    "os": "windows",
                }
            ]

            return self.client._make_request("serp/google/organic/task_post", data)
        except Exception as e:
            # Re-raise API exceptions
            from ..dataforseo.client import ApiException
            if isinstance(e, ApiException):
                raise e

            # Return mock data structure for other exceptions
            return {
                "tasks": [{
                    "result": [{
                        "items": []
                    }]
                }]
            }

    def _get_keyword_difficulty(self, keyword: str, location: str) -> Dict:
        """Get keyword difficulty metrics."""
        location_code = get_location_code(location)
        language_code = get_language_code("english")

        data = [
            {
                "keyword": keyword,
                "location_code": location_code,
                "language_code": language_code,
            }
        ]

        return self.client._make_request(
            "dataforseo_labs/google/keyword_difficulty/task_post", data
        )

    def _get_search_volume(self, keyword: str, location: str) -> Dict:
        """Get search volume data."""
        location_code = get_location_code(location)
        language_code = get_language_code("english")

        data = [
            {
                "keywords": [keyword],
                "location_code": location_code,
                "language_code": language_code,
            }
        ]

        return self.client._make_request(
            "keywords_data/google_ads/search_volume/task_post", data
        )

    def _analyze_serp_features(self, serp_data: Dict) -> Dict[str, Any]:
        """Analyze SERP features present in results."""
        features = {
            "featured_snippet": False,
            "knowledge_panel": False,
            "people_also_ask": False,
            "local_pack": False,
            "shopping_results": False,
            "video_results": False,
            "image_results": False,
            "news_results": False,
            "total_features": 0,
        }

        if not serp_data.get("tasks") or not serp_data["tasks"][0].get("result"):
            return features

        task_result = serp_data["tasks"][0]["result"][0]

        # Check for various SERP features
        if task_result.get("featured_snippet"):
            features["featured_snippet"] = True
            features["total_features"] += 1

        if task_result.get("knowledge_graph"):
            features["knowledge_panel"] = True
            features["total_features"] += 1

        if task_result.get("people_also_ask"):
            features["people_also_ask"] = True
            features["total_features"] += 1

        # Add more feature detection as needed

        return features

    def _extract_difficulty_score(self, difficulty_data: Dict) -> Optional[float]:
        """Extract difficulty score from API response."""
        if not difficulty_data.get("tasks") or not difficulty_data["tasks"][0].get(
            "result"
        ):
            return None

        task_result = difficulty_data["tasks"][0]["result"][0]
        return task_result.get("keyword_difficulty")

    def _extract_volume_metrics(self, volume_data: Dict) -> Dict[str, Any]:
        """Extract volume metrics from API response."""
        metrics = {
            "search_volume": None,
            "cpc": None,
            "competition": None,
            "monthly_searches": [],
        }

        if not volume_data.get("tasks") or not volume_data["tasks"][0].get("result"):
            return metrics

        task_result = volume_data["tasks"][0]["result"][0]
        if task_result.get("items"):
            item = task_result["items"][0]
            metrics["search_volume"] = item.get("search_volume")
            metrics["cpc"] = item.get("cpc")
            metrics["competition"] = item.get("competition")
            metrics["monthly_searches"] = item.get("monthly_searches", [])

        return metrics

    def _calculate_keyword_opportunity(self, keyword_result: Dict) -> float:
        """Calculate opportunity score for a keyword."""
        score = 0.0

        # Base score from search volume
        volume_metrics = keyword_result.get("volume_metrics", {})
        search_volume = volume_metrics.get("search_volume", 0)
        if search_volume:
            score += min(40, search_volume / 1000)  # Max 40 points

        # Position score
        position = keyword_result.get("target_position")
        if position:
            if position <= 3:
                score += 30  # Already ranking well
            elif position <= 10:
                score += 20  # Good opportunity
            elif position <= 20:
                score += 10  # Moderate opportunity
            else:
                score += 5  # Lower opportunity
        else:
            score += 15  # Not ranking - opportunity to start

        # Difficulty score (inverse - easier keywords get higher score)
        difficulty = keyword_result.get("difficulty_score", 50)
        if difficulty:
            score += max(0, 30 - difficulty)  # Max 30 points for easy keywords

        return min(100, score)

    def _calculate_competitive_intensity(self, keyword_result: Dict) -> str:
        """Calculate competitive intensity level."""
        competitor_count = len(keyword_result.get("competitor_analysis", []))
        difficulty = keyword_result.get("difficulty_score", 50)
        volume = keyword_result.get("volume_metrics", {}).get("search_volume", 0)

        # High volume + high difficulty + many competitors = high intensity
        intensity_score = 0

        if volume > 10000:
            intensity_score += 3
        elif volume > 1000:
            intensity_score += 2
        elif volume > 100:
            intensity_score += 1

        if difficulty > 70:
            intensity_score += 3
        elif difficulty > 50:
            intensity_score += 2
        elif difficulty > 30:
            intensity_score += 1

        if competitor_count > 5:
            intensity_score += 2
        elif competitor_count > 2:
            intensity_score += 1

        if intensity_score >= 7:
            return "very_high"
        elif intensity_score >= 5:
            return "high"
        elif intensity_score >= 3:
            return "medium"
        else:
            return "low"

    def _generate_analysis_summary(
        self, results: Dict, target_domain: str
    ) -> Dict[str, Any]:
        """Generate summary statistics for keyword analysis."""
        keyword_results = {k: v for k, v in results.items() if not k.startswith("_")}

        if not keyword_results:
            return {}

        total_keywords = len(keyword_results)
        ranked_keywords = sum(
            1 for r in keyword_results.values() if r.get("target_position")
        )
        top_10_keywords = sum(
            1 for r in keyword_results.values() if r.get("target_position", 999) <= 10
        )
        top_3_keywords = sum(
            1 for r in keyword_results.values() if r.get("target_position", 999) <= 3
        )

        avg_opportunity = (
            sum(r.get("opportunity_score", 0) for r in keyword_results.values())
            / total_keywords
        )

        competitor_counts = [
            len(r.get("competitor_analysis", [])) for r in keyword_results.values()
        ]
        avg_competitors = (
            sum(competitor_counts) / total_keywords if competitor_counts else 0
        )

        return {
            "total_keywords_analyzed": total_keywords,
            "keywords_ranking": ranked_keywords,
            "ranking_percentage": (ranked_keywords / total_keywords) * 100,
            "top_10_rankings": top_10_keywords,
            "top_3_rankings": top_3_keywords,
            "average_opportunity_score": round(avg_opportunity, 2),
            "average_competitors_per_keyword": round(avg_competitors, 2),
            "high_opportunity_keywords": sum(
                1
                for r in keyword_results.values()
                if r.get("opportunity_score", 0) > 70
            ),
            "target_domain": target_domain,
            "analysis_completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_competitor_comparison(
        self, all_results: Dict, keywords: List[str]
    ) -> Dict[str, Any]:
        """Generate cross-competitor comparison analysis."""
        comparison = {
            "domains_analyzed": list(all_results.keys()),
            "keyword_coverage": {},
            "ranking_comparison": {},
            "opportunity_comparison": {},
        }

        # Analyze keyword coverage for each domain
        for domain, results in all_results.items():
            domain_results = {k: v for k, v in results.items() if not k.startswith("_")}

            ranked_count = sum(
                1 for r in domain_results.values() if r.get("target_position")
            )
            top_10_count = sum(
                1
                for r in domain_results.values()
                if r.get("target_position", 999) <= 10
            )

            comparison["keyword_coverage"][domain] = {
                "total_rankings": ranked_count,
                "top_10_rankings": top_10_count,
                "coverage_percentage": (ranked_count / len(keywords)) * 100,
            }

        # Cross-domain ranking comparison for each keyword
        for keyword in keywords:
            keyword_comparison = {}
            for domain in all_results.keys():
                domain_result = all_results[domain].get(keyword, {})
                position = domain_result.get("target_position")
                keyword_comparison[domain] = position

            comparison["ranking_comparison"][keyword] = keyword_comparison

        return comparison
