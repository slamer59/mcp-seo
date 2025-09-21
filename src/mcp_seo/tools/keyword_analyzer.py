"""
Enhanced keyword research and analysis tools using DataForSEO API with advanced reporting.
"""

from typing import Any, Dict, List, Optional

from mcp_seo.config.settings import get_language_code, get_location_code
from mcp_seo.dataforseo.client import ApiException, DataForSEOClient
from mcp_seo.engines import SEORecommendationEngine
from mcp_seo.models.seo_models import (
    KeywordAnalysisRequest,
    KeywordData,
    SERPAnalysisRequest,
)
from mcp_seo.reporting import SEOReporter


class KeywordAnalyzer:
    """Keyword research and analysis tool with advanced reporting and recommendations."""

    def __init__(self, client: DataForSEOClient, use_rich_reporting: bool = True):
        self.client = client
        self.recommendation_engine = SEORecommendationEngine()
        self.reporter = SEOReporter(use_rich=use_rich_reporting)

    def analyze_keywords(self, request: KeywordAnalysisRequest) -> Dict[str, Any]:
        """Analyze keyword search volume and competition data with enhanced reporting."""
        try:
            location_code = get_location_code(request.location)
            language_code = get_language_code(request.language)

            # Get keyword data
            result = self.client.get_keyword_data(
                keywords=request.keywords,
                location_code=location_code,
                language_code=language_code,
            )

            if not result.get("tasks"):
                raise ApiException("No task data returned")

            task_id = result["tasks"][0]["id"]

            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")

            if (not completed_result.get("tasks") or
                not completed_result["tasks"] or
                completed_result["tasks"][0].get("result") is None):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No keyword data available",
                }

            # Process keyword data
            keywords_data = []
            task_result = completed_result["tasks"][0]["result"]
            if not task_result or len(task_result) == 0:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No keyword data available",
                }
            task_result = task_result[0]

            for keyword_info in task_result.get("items", []):
                keyword_data = KeywordData(
                    keyword=keyword_info.get("keyword", ""),
                    search_volume=keyword_info.get("search_volume"),
                    cpc=keyword_info.get("cpc"),
                    competition=keyword_info.get("competition"),
                    competition_level=keyword_info.get("competition_level"),
                    monthly_searches=keyword_info.get("monthly_searches", []),
                )
                keywords_data.append(keyword_data.model_dump())

            # Get keyword suggestions if requested
            suggestions = []
            if request.include_suggestions and keywords_data:
                suggestions = self._get_keyword_suggestions(
                    request.keywords[0],  # Use first keyword for suggestions
                    location_code,
                    language_code,
                    request.suggestion_limit,
                )

            # Create structured result
            result = {
                "task_id": task_id,
                "status": "completed",
                "keywords_data": keywords_data,
                "total_keywords": len(keywords_data),
                "keyword_suggestions": suggestions,
                "location": request.location,
                "language": request.language,
                "analysis_summary": self._create_keyword_summary(keywords_data),
            }

            # Generate SEO recommendations
            keyword_performance_data = {}
            for kw in keywords_data:
                keyword_performance_data[kw["keyword"]] = {
                    "search_volume": {"search_volume": kw.get("search_volume", 0)},
                    "difficulty": {"difficulty": kw.get("competition", 0) * 100},  # Convert competition to difficulty score
                    "position": None,  # Not available in this context
                }

            recommendations = self.recommendation_engine.analyze_keyword_performance(
                keyword_performance_data
            )
            result["seo_recommendations"] = [rec.__dict__ for rec in recommendations]

            # Generate keyword targeting strategy
            result["keyword_targeting_strategy"] = (
                self._generate_keyword_targeting_strategy(keywords_data, suggestions)
            )

            # Add formatted report
            result["formatted_report"] = self.reporter.generate_keyword_report(
                result
            )

            return result

        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to analyze keywords: {str(e)}",
                "keywords": request.keywords,
            }

    def get_keyword_suggestions(
        self,
        seed_keyword: str,
        location: str = "usa",
        language: str = "english",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get keyword suggestions based on seed keyword with enhanced analysis."""
        try:
            location_code = get_location_code(location)
            language_code = get_language_code(language)

            result = self.client.get_keyword_suggestions(
                keyword=seed_keyword,
                location_code=location_code,
                language_code=language_code,
                limit=limit,
            )

            if not result.get("tasks"):
                raise ApiException("No task data returned")

            task_id = result["tasks"][0]["id"]

            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")

            if not completed_result.get("tasks") or not completed_result["tasks"][
                0
            ].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No keyword suggestions available",
                }

            # Process suggestions
            suggestions = []
            task_result = completed_result["tasks"][0]["result"][0]

            for item in task_result.get("items", []):
                suggestion = {
                    "keyword": item.get("keyword", ""),
                    "search_volume": item.get("search_volume"),
                    "cpc": item.get("cpc"),
                    "competition": item.get("competition"),
                    "competition_level": item.get("competition_level"),
                    "relevance_score": item.get("keyword_info", {}).get(
                        "search_volume", 0
                    ),
                }
                suggestions.append(suggestion)

            # Sort by search volume (descending)
            suggestions.sort(key=lambda x: x.get("search_volume", 0), reverse=True)

            # Create structured result
            result = {
                "task_id": task_id,
                "status": "completed",
                "seed_keyword": seed_keyword,
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "location": location,
                "language": language,
                "suggestions_summary": self._create_suggestions_summary(suggestions),
            }

            # Categorize suggestions for strategic targeting
            result["suggestion_categories"] = self._categorize_keyword_suggestions(
                suggestions
            )

            # Add formatted report
            result["formatted_report"] = self.reporter.generate_keyword_report(
                result
            )

            return result

        except Exception as e:
            return {
                "error": f"Failed to get keyword suggestions: {str(e)}",
                "seed_keyword": seed_keyword,
            }

    def analyze_serp_for_keyword(self, request: SERPAnalysisRequest) -> Dict[str, Any]:
        """Analyze SERP results for a specific keyword with content optimization insights."""
        try:
            location_code = get_location_code(request.location)
            language_code = get_language_code(request.language)

            result = self.client.get_serp_results(
                keyword=request.keyword,
                location_code=location_code,
                language_code=language_code,
                device=request.device.value,
                depth=request.depth,
            )

            if not result.get("tasks"):
                raise ApiException("No task data returned")

            task_id = result["tasks"][0]["id"]

            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "serp")

            if not completed_result.get("tasks") or not completed_result["tasks"][
                0
            ].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No SERP data available",
                }

            # Process SERP results
            task_result = completed_result["tasks"][0]["result"][0]
            organic_results = []
            paid_results = []
            featured_snippet = None
            people_also_ask = []
            related_searches = []

            # Extract organic results
            for item in task_result.get("items", []):
                item_type = item.get("type", "")

                if item_type == "organic":
                    organic_results.append(
                        {
                            "position": item.get("rank_group", 0),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "domain": item.get("domain", ""),
                            "breadcrumb": item.get("breadcrumb", ""),
                            "is_featured_snippet": False,
                            "is_paid": False,
                        }
                    )

                elif item_type == "paid" and request.include_paid_results:
                    paid_results.append(
                        {
                            "position": item.get("rank_group", 0),
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "description": item.get("description", ""),
                            "domain": item.get("domain", ""),
                            "is_paid": True,
                        }
                    )

                elif item_type == "featured_snippet":
                    featured_snippet = {
                        "position": 0,
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "domain": item.get("domain", ""),
                        "is_featured_snippet": True,
                        "is_paid": False,
                    }

                elif item_type == "people_also_ask":
                    for paa_item in item.get("items", []):
                        people_also_ask.append(paa_item.get("question", ""))

                elif item_type == "related_searches":
                    for related_item in item.get("items", []):
                        related_searches.append(related_item.get("title", ""))

            # Analyze competitive landscape
            competitive_analysis = self._analyze_serp_competition(organic_results)

            # Create structured result
            result = {
                "task_id": task_id,
                "status": "completed",
                "keyword": request.keyword,
                "location": request.location,
                "language": request.language,
                "device": request.device.value,
                "serp_analysis": {
                    "organic_results": organic_results,
                    "paid_results": paid_results,
                    "featured_snippet": featured_snippet,
                    "people_also_ask": people_also_ask[:10],  # Limit to top 10
                    "related_searches": related_searches[:10],  # Limit to top 10
                    "total_organic_results": len(organic_results),
                    "total_paid_results": len(paid_results),
                    "competitive_analysis": competitive_analysis,
                },
            }

            # Generate content optimization suggestions based on SERP analysis
            if organic_results:
                content_suggestions = self._generate_content_suggestions_from_serp(
                    organic_results, request.keyword
                )
                result["content_optimization_suggestions"] = content_suggestions

            return result

        except Exception as e:
            return {
                "error": f"Failed to analyze SERP for keyword: {str(e)}",
                "keyword": request.keyword,
            }

    def get_keyword_difficulty(
        self, keywords: List[str], location: str = "usa", language: str = "english"
    ) -> Dict[str, Any]:
        """Estimate keyword difficulty with strategic recommendations."""
        try:
            difficulty_scores = []

            for keyword in keywords[:10]:  # Limit to 10 keywords to avoid rate limits
                # Get SERP data for each keyword
                serp_request = SERPAnalysisRequest(
                    keyword=keyword,
                    location=location,
                    language=language,
                    depth=20,  # Top 20 results for difficulty analysis
                )

                serp_result = self.analyze_serp_for_keyword(serp_request)

                if "serp_analysis" in serp_result:
                    difficulty = self._calculate_keyword_difficulty(
                        serp_result["serp_analysis"]["organic_results"]
                    )

                    difficulty_scores.append(
                        {
                            "keyword": keyword,
                            "difficulty_score": difficulty["score"],
                            "difficulty_level": difficulty["level"],
                            "factors": difficulty["factors"],
                        }
                    )

            # Create structured result
            result = {
                "status": "completed",
                "keyword_difficulty": difficulty_scores,
                "total_keywords": len(difficulty_scores),
                "location": location,
                "language": language,
            }

            # Generate keyword targeting recommendations
            if difficulty_scores:
                targeting_recommendations = (
                    self._generate_keyword_targeting_recommendations(difficulty_scores)
                )
                result["targeting_recommendations"] = targeting_recommendations
                result["formatted_report"] = (
                    self.reporter.generate_keyword_analysis_report(result)
                )

            return result

        except Exception as e:
            return {
                "error": f"Failed to calculate keyword difficulty: {str(e)}",
                "keywords": keywords,
            }

    def _generate_keyword_targeting_strategy(
        self, keywords_data: List[Dict[str, Any]], suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate strategic keyword targeting recommendations."""

        # Categorize keywords by difficulty and opportunity
        high_volume_low_competition = []
        long_tail_opportunities = []
        quick_wins = []
        competitive_targets = []

        all_keywords = keywords_data + suggestions[:20]  # Include top suggestions

        for kw in all_keywords:
            volume = kw.get("search_volume", 0)
            competition = kw.get("competition", 0)
            keyword = kw.get("keyword", "")

            if volume > 1000 and competition < 0.3:
                high_volume_low_competition.append(kw)
            elif len(keyword.split()) >= 3 and volume > 100:
                long_tail_opportunities.append(kw)
            elif volume > 500 and competition < 0.5:
                quick_wins.append(kw)
            elif volume > 5000:
                competitive_targets.append(kw)

        return {
            "strategy_overview": {
                "total_opportunities": len(all_keywords),
                "recommended_focus": (
                    "long_tail" if len(long_tail_opportunities) > 5 else "quick_wins"
                ),
            },
            "high_opportunity": high_volume_low_competition[:5],
            "medium_opportunity": long_tail_opportunities[:5] + quick_wins[:5],
            "low_opportunity": competitive_targets[:5],
            "keyword_categories": {
                "high_volume_low_competition": {
                    "count": len(high_volume_low_competition),
                    "keywords": high_volume_low_competition[:5],
                    "priority": "High",
                    "strategy": "Target immediately with comprehensive content",
                },
                "long_tail_opportunities": {
                    "count": len(long_tail_opportunities),
                    "keywords": long_tail_opportunities[:5],
                    "priority": "Medium",
                    "strategy": "Create specific, targeted content for each phrase",
                },
                "quick_wins": {
                    "count": len(quick_wins),
                    "keywords": quick_wins[:5],
                    "priority": "Medium-High",
                    "strategy": "Optimize existing content or create new focused pages",
                },
                "competitive_targets": {
                    "count": len(competitive_targets),
                    "keywords": competitive_targets[:3],
                    "priority": "Long-term",
                    "strategy": "Build authority gradually with supporting content",
                },
            },
            "action_plan": {
                "immediate": "Focus on long-tail and quick-win keywords",
                "short_term": "Build content clusters around high-opportunity keywords",
                "long_term": "Challenge competitive keywords with comprehensive content strategy",
            },
        }

    def _categorize_keyword_suggestions(
        self, suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Categorize keyword suggestions for strategic use."""

        categories = {
            "high_volume": [],
            "low_competition": [],
            "long_tail": [],
            "question_based": [],
            "commercial_intent": [],
        }

        for suggestion in suggestions:
            keyword = suggestion.get("keyword", "").lower()
            volume = suggestion.get("search_volume", 0)
            competition = suggestion.get("competition", 0)

            # High volume
            if volume > 5000:
                categories["high_volume"].append(suggestion)

            # Low competition
            if competition < 0.3:
                categories["low_competition"].append(suggestion)

            # Long tail
            if len(keyword.split()) >= 3:
                categories["long_tail"].append(suggestion)

            # Question based
            if any(
                q in keyword for q in ["how", "what", "why", "when", "where", "who"]
            ):
                categories["question_based"].append(suggestion)

            # Commercial intent
            if any(
                c in keyword
                for c in ["buy", "price", "cost", "review", "best", "compare"]
            ):
                categories["commercial_intent"].append(suggestion)

        return categories

    def _generate_content_suggestions_from_serp(
        self, organic_results: List[Dict[str, Any]], keyword: str
    ) -> Dict[str, Any]:
        """Generate content optimization suggestions based on SERP analysis."""

        # Analyze top-ranking content characteristics
        top_results = organic_results[:5]  # Top 5 results

        content_patterns = {
            "title_patterns": [],
            "content_themes": [],
            "common_words": {},
            "content_gaps": [],
        }

        # Analyze titles for patterns
        for result in top_results:
            title = result.get("title", "").lower()
            content_patterns["title_patterns"].append(title)

            # Extract common words (simple approach)
            words = title.split()
            for word in words:
                if len(word) > 3:  # Ignore short words
                    content_patterns["common_words"][word] = (
                        content_patterns["common_words"].get(word, 0) + 1
                    )

        # Identify frequently used terms
        frequent_terms = sorted(
            content_patterns["common_words"].items(), key=lambda x: x[1], reverse=True
        )[:10]

        suggestions = {
            "content_optimization": {
                "title_suggestions": [
                    f"Include '{keyword}' naturally in your title",
                    f"Consider adding power words like: {', '.join([term[0] for term in frequent_terms[:3]])}",
                ],
                "content_themes": [
                    f"Top-ranking pages often include: {', '.join([term[0] for term in frequent_terms[:5]])}",
                    "Ensure comprehensive coverage of the topic",
                    "Include related subtopics found in competing content",
                ],
                "featured_snippet_opportunity": {
                    "available": not any(
                        result.get("is_featured_snippet") for result in organic_results
                    ),
                    "strategy": (
                        "Create concise, well-formatted answers to common questions"
                        if not any(
                            result.get("is_featured_snippet")
                            for result in organic_results
                        )
                        else "Improve existing content to compete for featured snippet"
                    ),
                },
            },
            "competition_analysis": {
                "average_title_length": (
                    sum(len(r.get("title", "")) for r in top_results) / len(top_results)
                    if top_results
                    else 0
                ),
                "domains_to_study": [
                    result.get("domain", "") for result in top_results[:3]
                ],
                "content_gaps": "Analyze competitor content for unique angles and comprehensive coverage",
            },
        }

        return suggestions

    def _generate_keyword_targeting_recommendations(
        self, difficulty_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate keyword targeting recommendations based on difficulty analysis."""

        easy_keywords = [
            k for k in difficulty_scores if k["difficulty_level"] in ["easy", "medium"]
        ]
        hard_keywords = [
            k
            for k in difficulty_scores
            if k["difficulty_level"] in ["hard", "very_hard"]
        ]

        recommendations = {
            "immediate_targets": {
                "keywords": easy_keywords[:5],
                "strategy": "Target these keywords first with well-optimized content",
                "expected_timeframe": "2-8 weeks for ranking improvements",
            },
            "long_term_targets": {
                "keywords": hard_keywords[:3],
                "strategy": "Build authority gradually with supporting content and backlinks",
                "expected_timeframe": "3-12 months for competitive positioning",
            },
            "content_strategy": {
                "pillar_content": "Create comprehensive guides for competitive keywords",
                "cluster_content": "Build supporting content around easier keywords",
                "link_building": "Focus on earning quality backlinks for authority building",
            },
        }

        return recommendations

    # Include all the existing helper methods from the original class
    def _get_keyword_suggestions(
        self, seed_keyword: str, location_code: int, language_code: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Helper method to get keyword suggestions."""
        try:
            result = self.client.get_keyword_suggestions(
                keyword=seed_keyword,
                location_code=location_code,
                language_code=language_code,
                limit=limit,
            )

            task_id = result["tasks"][0]["id"]
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")

            suggestions = []
            if completed_result.get("tasks") and completed_result["tasks"][0].get(
                "result"
            ):
                task_result = completed_result["tasks"][0]["result"][0]

                for item in task_result.get("items", []):
                    suggestions.append(
                        {
                            "keyword": item.get("keyword", ""),
                            "search_volume": item.get("search_volume"),
                            "cpc": item.get("cpc"),
                            "competition": item.get("competition"),
                        }
                    )

            return suggestions[:limit]

        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []

    def _create_keyword_summary(
        self, keywords_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary statistics for keyword analysis."""
        if not keywords_data:
            return {}

        # Calculate statistics
        search_volumes = [
            kw.get("search_volume", 0)
            for kw in keywords_data
            if kw.get("search_volume")
        ]
        cpcs = [kw.get("cpc", 0) for kw in keywords_data if kw.get("cpc")]
        competitions = [
            kw.get("competition", 0) for kw in keywords_data if kw.get("competition")
        ]

        return {
            "total_keywords": len(keywords_data),
            "keywords_with_volume": len(search_volumes),
            "avg_search_volume": (
                sum(search_volumes) / len(search_volumes) if search_volumes else 0
            ),
            "max_search_volume": max(search_volumes) if search_volumes else 0,
            "min_search_volume": min(search_volumes) if search_volumes else 0,
            "avg_cpc": sum(cpcs) / len(cpcs) if cpcs else 0,
            "avg_competition": (
                sum(competitions) / len(competitions) if competitions else 0
            ),
            "high_volume_keywords": len([v for v in search_volumes if v > 10000]),
            "low_competition_keywords": len([c for c in competitions if c < 0.3]),
        }

    def _create_suggestions_summary(
        self, suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary for keyword suggestions."""
        if not suggestions:
            return {}

        search_volumes = [
            s.get("search_volume", 0) for s in suggestions if s.get("search_volume")
        ]

        return {
            "total_suggestions": len(suggestions),
            "suggestions_with_volume": len(search_volumes),
            "avg_search_volume": (
                sum(search_volumes) / len(search_volumes) if search_volumes else 0
            ),
            "top_volume_suggestion": (
                max(suggestions, key=lambda x: x.get("search_volume", 0))
                if suggestions
                else None
            ),
            "long_tail_keywords": len(
                [s for s in suggestions if len(s.get("keyword", "").split()) >= 3]
            ),
            "high_volume_suggestions": len([v for v in search_volumes if v > 5000]),
        }

    def _analyze_serp_competition(
        self, organic_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze competitive landscape from SERP results."""
        if not organic_results:
            return {}

        domains = [result.get("domain", "") for result in organic_results]
        unique_domains = set(domains)

        # Find domains with multiple rankings
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        multiple_rankings = {
            domain: count for domain, count in domain_counts.items() if count > 1
        }

        return {
            "total_results": len(organic_results),
            "unique_domains": len(unique_domains),
            "domain_diversity": (
                len(unique_domains) / len(organic_results) if organic_results else 0
            ),
            "domains_with_multiple_rankings": len(multiple_rankings),
            "top_domains": list(unique_domains)[:10],
            "multiple_rankings_domains": multiple_rankings,
        }

    def _calculate_keyword_difficulty(
        self, organic_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate keyword difficulty score based on SERP analysis."""
        if not organic_results:
            return {"score": 0, "level": "unknown", "factors": {}}

        # Base difficulty factors
        factors = {
            "serp_diversity": 0,
            "domain_authority_estimate": 0,
            "content_quality_indicators": 0,
        }

        # Analyze domain diversity
        domains = [result.get("domain", "") for result in organic_results[:10]]
        unique_domains = set(domains)
        domain_diversity = len(unique_domains) / len(domains) if domains else 0
        factors["serp_diversity"] = domain_diversity

        # Estimate domain authority based on domain characteristics
        authority_score = 0
        for result in organic_results[:10]:
            domain = result.get("domain", "")
            # Simple heuristics for domain authority estimation
            if any(tld in domain for tld in [".edu", ".gov", ".org"]):
                authority_score += 10
            elif any(
                brand in domain
                for brand in ["wikipedia", "youtube", "facebook", "linkedin", "amazon"]
            ):
                authority_score += 8
            elif len(domain.split(".")[0]) < 6:  # Short, brandable domains
                authority_score += 5
            else:
                authority_score += 3

        factors["domain_authority_estimate"] = authority_score / len(
            organic_results[:10]
        )

        # Content quality indicators (title/description completeness)
        quality_indicators = 0
        for result in organic_results[:10]:
            if result.get("title") and len(result["title"]) > 30:
                quality_indicators += 1
            if result.get("description") and len(result["description"]) > 120:
                quality_indicators += 1

        factors["content_quality_indicators"] = quality_indicators / (
            len(organic_results[:10]) * 2
        )

        # Calculate overall difficulty score (0-100)
        difficulty_score = (
            (1 - domain_diversity) * 30
            + (  # Less diversity = higher difficulty
                factors["domain_authority_estimate"] / 10
            )
            * 40
            + factors["content_quality_indicators"]  # Higher DA = higher difficulty
            * 30  # Better content = higher difficulty
        )

        # Determine difficulty level
        if difficulty_score < 30:
            level = "easy"
        elif difficulty_score < 50:
            level = "medium"
        elif difficulty_score < 70:
            level = "hard"
        else:
            level = "very_hard"

        return {
            "score": min(100, max(0, int(difficulty_score))),
            "level": level,
            "factors": factors,
        }


# Backward compatibility alias for legacy imports
EnhancedKeywordAnalyzer = KeywordAnalyzer
