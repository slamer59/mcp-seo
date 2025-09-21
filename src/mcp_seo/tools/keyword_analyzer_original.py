"""
Keyword research and analysis tools using DataForSEO API.
"""

from typing import Dict, Any, List, Optional
from mcp_seo.dataforseo.client import DataForSEOClient, ApiException
from mcp_seo.models.seo_models import KeywordAnalysisRequest, KeywordData, SERPAnalysisRequest
from mcp_seo.config.settings import get_location_code, get_language_code


class KeywordAnalyzer:
    """Keyword research and analysis tool."""
    
    def __init__(self, client: DataForSEOClient):
        self.client = client
    
    def analyze_keywords(self, request: KeywordAnalysisRequest) -> Dict[str, Any]:
        """Analyze keyword search volume and competition data."""
        try:
            location_code = get_location_code(request.location)
            language_code = get_language_code(request.language)
            
            # Get keyword data
            result = self.client.get_keyword_data(
                keywords=request.keywords,
                location_code=location_code,
                language_code=language_code
            )
            
            if not result.get("tasks"):
                raise ApiException("No task data returned")
            
            task_id = result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No keyword data available"
                }
            
            # Process keyword data
            keywords_data = []
            task_result = completed_result["tasks"][0]["result"][0]
            
            for keyword_info in task_result.get("items", []):
                keyword_data = KeywordData(
                    keyword=keyword_info.get("keyword", ""),
                    search_volume=keyword_info.get("search_volume"),
                    cpc=keyword_info.get("cpc"),
                    competition=keyword_info.get("competition"),
                    competition_level=keyword_info.get("competition_level"),
                    monthly_searches=keyword_info.get("monthly_searches", [])
                )
                keywords_data.append(keyword_data.dict())
            
            # Get keyword suggestions if requested
            suggestions = []
            if request.include_suggestions and keywords_data:
                suggestions = self._get_keyword_suggestions(
                    request.keywords[0],  # Use first keyword for suggestions
                    location_code,
                    language_code,
                    request.suggestion_limit
                )
            
            return {
                "task_id": task_id,
                "status": "completed",
                "keywords_data": keywords_data,
                "total_keywords": len(keywords_data),
                "suggestions": suggestions,
                "location": request.location,
                "language": request.language,
                "analysis_summary": self._create_keyword_summary(keywords_data)
            }
        
        except Exception as e:
            return {
                "error": f"Failed to analyze keywords: {str(e)}",
                "keywords": request.keywords
            }
    
    def get_keyword_suggestions(self, seed_keyword: str, location: str = "usa", 
                               language: str = "english", limit: int = 100) -> Dict[str, Any]:
        """Get keyword suggestions based on seed keyword."""
        try:
            location_code = get_location_code(location)
            language_code = get_language_code(language)
            
            result = self.client.get_keyword_suggestions(
                keyword=seed_keyword,
                location_code=location_code,
                language_code=language_code,
                limit=limit
            )
            
            if not result.get("tasks"):
                raise ApiException("No task data returned")
            
            task_id = result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No keyword suggestions available"
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
                    "relevance_score": item.get("keyword_info", {}).get("search_volume", 0)
                }
                suggestions.append(suggestion)
            
            # Sort by search volume (descending)
            suggestions.sort(key=lambda x: x.get("search_volume", 0), reverse=True)
            
            return {
                "task_id": task_id,
                "status": "completed",
                "seed_keyword": seed_keyword,
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "location": location,
                "language": language,
                "suggestions_summary": self._create_suggestions_summary(suggestions)
            }
        
        except Exception as e:
            return {
                "error": f"Failed to get keyword suggestions: {str(e)}",
                "seed_keyword": seed_keyword
            }
    
    def analyze_serp_for_keyword(self, request: SERPAnalysisRequest) -> Dict[str, Any]:
        """Analyze SERP results for a specific keyword."""
        try:
            location_code = get_location_code(request.location)
            language_code = get_language_code(request.language)
            
            result = self.client.get_serp_results(
                keyword=request.keyword,
                location_code=location_code,
                language_code=language_code,
                device=request.device.value,
                depth=request.depth
            )
            
            if not result.get("tasks"):
                raise ApiException("No task data returned")
            
            task_id = result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "serp")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No SERP data available"
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
                    organic_results.append({
                        "position": item.get("rank_group", 0),
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "domain": item.get("domain", ""),
                        "breadcrumb": item.get("breadcrumb", ""),
                        "is_featured_snippet": False,
                        "is_paid": False
                    })
                
                elif item_type == "paid" and request.include_paid_results:
                    paid_results.append({
                        "position": item.get("rank_group", 0),
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "domain": item.get("domain", ""),
                        "is_paid": True
                    })
                
                elif item_type == "featured_snippet":
                    featured_snippet = {
                        "position": 0,
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "domain": item.get("domain", ""),
                        "is_featured_snippet": True,
                        "is_paid": False
                    }
                
                elif item_type == "people_also_ask":
                    for paa_item in item.get("items", []):
                        people_also_ask.append(paa_item.get("question", ""))
                
                elif item_type == "related_searches":
                    for related_item in item.get("items", []):
                        related_searches.append(related_item.get("title", ""))
            
            # Analyze competitive landscape
            competitive_analysis = self._analyze_serp_competition(organic_results)
            
            return {
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
                    "competitive_analysis": competitive_analysis
                }
            }
        
        except Exception as e:
            return {
                "error": f"Failed to analyze SERP for keyword: {str(e)}",
                "keyword": request.keyword
            }
    
    def get_keyword_difficulty(self, keywords: List[str], location: str = "usa", 
                              language: str = "english") -> Dict[str, Any]:
        """Estimate keyword difficulty based on SERP analysis."""
        try:
            difficulty_scores = []
            
            for keyword in keywords[:10]:  # Limit to 10 keywords to avoid rate limits
                # Get SERP data for each keyword
                serp_request = SERPAnalysisRequest(
                    keyword=keyword,
                    location=location,
                    language=language,
                    depth=20  # Top 20 results for difficulty analysis
                )
                
                serp_result = self.analyze_serp_for_keyword(serp_request)
                
                if "serp_analysis" in serp_result:
                    difficulty = self._calculate_keyword_difficulty(
                        serp_result["serp_analysis"]["organic_results"]
                    )
                    
                    difficulty_scores.append({
                        "keyword": keyword,
                        "difficulty_score": difficulty["score"],
                        "difficulty_level": difficulty["level"],
                        "factors": difficulty["factors"]
                    })
            
            return {
                "status": "completed",
                "keyword_difficulty": difficulty_scores,
                "total_keywords": len(difficulty_scores),
                "location": location,
                "language": language
            }
        
        except Exception as e:
            return {
                "error": f"Failed to calculate keyword difficulty: {str(e)}",
                "keywords": keywords
            }
    
    def _get_keyword_suggestions(self, seed_keyword: str, location_code: int, 
                                language_code: str, limit: int) -> List[Dict[str, Any]]:
        """Helper method to get keyword suggestions."""
        try:
            result = self.client.get_keyword_suggestions(
                keyword=seed_keyword,
                location_code=location_code,
                language_code=language_code,
                limit=limit
            )
            
            task_id = result["tasks"][0]["id"]
            completed_result = self.client.wait_for_task_completion(task_id, "keywords")
            
            suggestions = []
            if completed_result.get("tasks") and completed_result["tasks"][0].get("result"):
                task_result = completed_result["tasks"][0]["result"][0]
                
                for item in task_result.get("items", []):
                    suggestions.append({
                        "keyword": item.get("keyword", ""),
                        "search_volume": item.get("search_volume"),
                        "cpc": item.get("cpc"),
                        "competition": item.get("competition")
                    })
            
            return suggestions[:limit]
        
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []
    
    def _create_keyword_summary(self, keywords_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for keyword analysis."""
        if not keywords_data:
            return {}
        
        # Calculate statistics
        search_volumes = [kw.get("search_volume", 0) for kw in keywords_data if kw.get("search_volume")]
        cpcs = [kw.get("cpc", 0) for kw in keywords_data if kw.get("cpc")]
        competitions = [kw.get("competition", 0) for kw in keywords_data if kw.get("competition")]
        
        return {
            "total_keywords": len(keywords_data),
            "keywords_with_volume": len(search_volumes),
            "avg_search_volume": sum(search_volumes) / len(search_volumes) if search_volumes else 0,
            "max_search_volume": max(search_volumes) if search_volumes else 0,
            "min_search_volume": min(search_volumes) if search_volumes else 0,
            "avg_cpc": sum(cpcs) / len(cpcs) if cpcs else 0,
            "avg_competition": sum(competitions) / len(competitions) if competitions else 0,
            "high_volume_keywords": len([v for v in search_volumes if v > 10000]),
            "low_competition_keywords": len([c for c in competitions if c < 0.3])
        }
    
    def _create_suggestions_summary(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for keyword suggestions."""
        if not suggestions:
            return {}
        
        search_volumes = [s.get("search_volume", 0) for s in suggestions if s.get("search_volume")]
        
        return {
            "total_suggestions": len(suggestions),
            "suggestions_with_volume": len(search_volumes),
            "avg_search_volume": sum(search_volumes) / len(search_volumes) if search_volumes else 0,
            "top_volume_suggestion": max(suggestions, key=lambda x: x.get("search_volume", 0)) if suggestions else None,
            "long_tail_keywords": len([s for s in suggestions if len(s.get("keyword", "").split()) >= 3]),
            "high_volume_suggestions": len([v for v in search_volumes if v > 5000])
        }
    
    def _analyze_serp_competition(self, organic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitive landscape from SERP results."""
        if not organic_results:
            return {}
        
        domains = [result.get("domain", "") for result in organic_results]
        unique_domains = set(domains)
        
        # Find domains with multiple rankings
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        multiple_rankings = {domain: count for domain, count in domain_counts.items() if count > 1}
        
        return {
            "total_results": len(organic_results),
            "unique_domains": len(unique_domains),
            "domain_diversity": len(unique_domains) / len(organic_results) if organic_results else 0,
            "domains_with_multiple_rankings": len(multiple_rankings),
            "top_domains": list(unique_domains)[:10],
            "multiple_rankings_domains": multiple_rankings
        }
    
    def _calculate_keyword_difficulty(self, organic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate keyword difficulty score based on SERP analysis."""
        if not organic_results:
            return {"score": 0, "level": "unknown", "factors": {}}
        
        # Base difficulty factors
        factors = {
            "serp_diversity": 0,
            "domain_authority_estimate": 0,
            "content_quality_indicators": 0
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
            if any(tld in domain for tld in ['.edu', '.gov', '.org']):
                authority_score += 10
            elif any(brand in domain for brand in ['wikipedia', 'youtube', 'facebook', 'linkedin', 'amazon']):
                authority_score += 8
            elif len(domain.split('.')[0]) < 6:  # Short, brandable domains
                authority_score += 5
            else:
                authority_score += 3
        
        factors["domain_authority_estimate"] = authority_score / len(organic_results[:10])
        
        # Content quality indicators (title/description completeness)
        quality_indicators = 0
        for result in organic_results[:10]:
            if result.get("title") and len(result["title"]) > 30:
                quality_indicators += 1
            if result.get("description") and len(result["description"]) > 120:
                quality_indicators += 1
        
        factors["content_quality_indicators"] = quality_indicators / (len(organic_results[:10]) * 2)
        
        # Calculate overall difficulty score (0-100)
        difficulty_score = (
            (1 - domain_diversity) * 30 +  # Less diversity = higher difficulty
            (factors["domain_authority_estimate"] / 10) * 40 +  # Higher DA = higher difficulty
            factors["content_quality_indicators"] * 30  # Better content = higher difficulty
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
            "factors": factors
        }