"""
Competitor analysis tools using DataForSEO API.
"""

from typing import Dict, Any, List, Optional
from mcp_seo.dataforseo.client import DataForSEOClient, ApiException
from mcp_seo.models.seo_models import DomainAnalysisRequest, CompetitorDomain
from mcp_seo.config.settings import get_location_code, get_language_code


class CompetitorAnalyzer:
    """Competitor analysis and domain research tool."""
    
    def __init__(self, client: DataForSEOClient):
        self.client = client
    
    def analyze_domain_overview(self, request: DomainAnalysisRequest) -> Dict[str, Any]:
        """Get comprehensive domain analysis overview."""
        try:
            location_code = get_location_code(request.location)
            language_code = get_language_code(request.language)
            
            # Get domain rank overview
            rank_result = self.client.get_domain_rank_overview(
                target=request.target,
                location_code=location_code,
                language_code=language_code
            )
            
            if not rank_result.get("tasks"):
                raise ApiException("No domain overview task created")
            
            task_id = rank_result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "dataforseo_labs")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": "No domain overview data available"
                }
            
            # Process domain overview
            task_result = completed_result["tasks"][0]["result"][0]
            
            domain_overview = {
                "target": request.target,
                "organic_keywords": task_result.get("organic_keywords", 0),
                "organic_etv": task_result.get("organic_etv", 0),
                "organic_count": task_result.get("organic_count", 0),
                "organic_pos_1": task_result.get("organic_pos_1", 0),
                "organic_pos_2_3": task_result.get("organic_pos_2_3", 0),
                "organic_pos_4_10": task_result.get("organic_pos_4_10", 0),
                "organic_pos_11_20": task_result.get("organic_pos_11_20", 0),
                "organic_pos_21_30": task_result.get("organic_pos_21_30", 0),
                "organic_pos_31_40": task_result.get("organic_pos_31_40", 0),
                "organic_pos_41_50": task_result.get("organic_pos_41_50", 0),
                "organic_pos_51_60": task_result.get("organic_pos_51_60", 0),
                "organic_pos_61_70": task_result.get("organic_pos_61_70", 0),
                "organic_pos_71_80": task_result.get("organic_pos_71_80", 0),
                "organic_pos_81_90": task_result.get("organic_pos_81_90", 0),
                "organic_pos_91_100": task_result.get("organic_pos_91_100", 0)
            }
            
            # Calculate additional metrics
            total_keywords = domain_overview["organic_keywords"]
            if total_keywords > 0:
                domain_overview["top_3_percentage"] = (
                    domain_overview["organic_pos_1"] + domain_overview["organic_pos_2_3"]
                ) / total_keywords * 100
                
                domain_overview["top_10_percentage"] = (
                    domain_overview["organic_pos_1"] + 
                    domain_overview["organic_pos_2_3"] + 
                    domain_overview["organic_pos_4_10"]
                ) / total_keywords * 100
                
                domain_overview["visibility_score"] = self._calculate_visibility_score(domain_overview)
            
            # Get competitors if requested
            competitors = []
            if request.include_competitors:
                competitors = self.get_competitor_domains(
                    request.target,
                    request.location,
                    request.language,
                    request.competitor_limit
                )
            
            # Get ranked keywords if requested
            ranked_keywords = []
            if request.include_keywords:
                ranked_keywords = self.get_top_ranked_keywords(
                    request.target,
                    request.location,
                    request.language,
                    request.keyword_limit
                )
            
            return {
                "task_id": task_id,
                "status": "completed",
                "domain_overview": domain_overview,
                "competitors": competitors,
                "ranked_keywords": ranked_keywords,
                "location": request.location,
                "language": request.language,
                "analysis_summary": self._create_domain_summary(domain_overview, competitors)
            }
        
        except Exception as e:
            return {
                "error": f"Failed to analyze domain overview: {str(e)}",
                "target": request.target
            }
    
    def get_competitor_domains(self, target: str, location: str = "usa", 
                              language: str = "english", limit: int = 50) -> List[Dict[str, Any]]:
        """Get competitor domains for target domain."""
        try:
            location_code = get_location_code(location)
            language_code = get_language_code(language)
            
            result = self.client.get_competitor_domains(
                target=target,
                location_code=location_code,
                language_code=language_code,
                limit=limit
            )
            
            if not result.get("tasks"):
                return []
            
            task_id = result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "dataforseo_labs")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return []
            
            # Process competitors
            competitors = []
            task_result = completed_result["tasks"][0]["result"][0]
            
            for item in task_result.get("items", []):
                competitor = {
                    "domain": item.get("domain", ""),
                    "common_keywords": item.get("intersections", 0),
                    "se_keywords_count": item.get("full_domain_metrics", {}).get("organic_keywords", 0),
                    "etv": item.get("full_domain_metrics", {}).get("organic_etv", 0),
                    "median_position": item.get("avg_position", 0),
                    "visibility": item.get("visibility", 0),
                    "competition_level": self._categorize_competition_level(
                        item.get("intersections", 0),
                        item.get("full_domain_metrics", {}).get("organic_keywords", 0)
                    )
                }
                competitors.append(competitor)
            
            # Sort by common keywords (descending)
            competitors.sort(key=lambda x: x.get("common_keywords", 0), reverse=True)
            
            return competitors
        
        except Exception as e:
            print(f"Error getting competitor domains: {e}")
            return []
    
    def get_top_ranked_keywords(self, target: str, location: str = "usa", 
                               language: str = "english", limit: int = 100) -> List[Dict[str, Any]]:
        """Get top ranked keywords for target domain."""
        try:
            location_code = get_location_code(location)
            language_code = get_language_code(language)
            
            result = self.client.get_ranked_keywords(
                target=target,
                location_code=location_code,
                language_code=language_code,
                limit=limit
            )
            
            if not result.get("tasks"):
                return []
            
            task_id = result["tasks"][0]["id"]
            
            # Wait for completion and get results
            completed_result = self.client.wait_for_task_completion(task_id, "dataforseo_labs")
            
            if not completed_result.get("tasks") or not completed_result["tasks"][0].get("result"):
                return []
            
            # Process ranked keywords
            keywords = []
            task_result = completed_result["tasks"][0]["result"][0]
            
            for item in task_result.get("items", []):
                keyword_data = {
                    "keyword": item.get("keyword", ""),
                    "position": item.get("rank_group", 0),
                    "search_volume": item.get("keyword_info", {}).get("search_volume"),
                    "cpc": item.get("keyword_info", {}).get("cpc"),
                    "competition": item.get("keyword_info", {}).get("competition"),
                    "etv": item.get("etv", 0),
                    "impressions_etv": item.get("impressions_etv", 0),
                    "clicks_etv": item.get("clicks_etv", 0),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "is_featured_snippet": item.get("is_featured_snippet", False),
                    "keyword_difficulty": self._estimate_keyword_difficulty(
                        item.get("rank_group", 0),
                        item.get("keyword_info", {}).get("search_volume", 0),
                        item.get("keyword_info", {}).get("competition", 0)
                    )
                }
                keywords.append(keyword_data)
            
            # Sort by position (ascending) then by search volume (descending)
            keywords.sort(key=lambda x: (x.get("position", 999), -x.get("search_volume", 0)))
            
            return keywords
        
        except Exception as e:
            print(f"Error getting ranked keywords: {e}")
            return []
    
    def compare_domains(self, primary_domain: str, competitor_domains: List[str], 
                       location: str = "usa", language: str = "english") -> Dict[str, Any]:
        """Compare primary domain with competitor domains."""
        try:
            domain_comparisons = []
            
            # Get overview for primary domain
            primary_request = DomainAnalysisRequest(
                target=primary_domain,
                location=location,
                language=language,
                include_competitors=False,
                include_keywords=True,
                keyword_limit=50
            )
            
            primary_analysis = self.analyze_domain_overview(primary_request)
            
            if "error" in primary_analysis:
                return {
                    "error": f"Failed to analyze primary domain: {primary_analysis['error']}",
                    "primary_domain": primary_domain
                }
            
            # Get overview for each competitor
            for competitor_domain in competitor_domains[:10]:  # Limit to 10 competitors
                comp_request = DomainAnalysisRequest(
                    target=competitor_domain,
                    location=location,
                    language=language,
                    include_competitors=False,
                    include_keywords=True,
                    keyword_limit=50
                )
                
                comp_analysis = self.analyze_domain_overview(comp_request)
                
                if "error" not in comp_analysis:
                    domain_comparisons.append({
                        "domain": competitor_domain,
                        "analysis": comp_analysis["domain_overview"]
                    })
            
            # Create comparison metrics
            comparison_result = self._create_domain_comparison(
                primary_analysis["domain_overview"],
                domain_comparisons
            )
            
            return {
                "status": "completed",
                "primary_domain": primary_domain,
                "primary_analysis": primary_analysis["domain_overview"],
                "competitor_analyses": domain_comparisons,
                "comparison_metrics": comparison_result,
                "location": location,
                "language": language,
                "competitive_insights": self._generate_competitive_insights(
                    primary_analysis["domain_overview"],
                    domain_comparisons
                )
            }
        
        except Exception as e:
            return {
                "error": f"Failed to compare domains: {str(e)}",
                "primary_domain": primary_domain
            }
    
    def find_content_gaps(self, primary_domain: str, competitor_domain: str, 
                         location: str = "usa", language: str = "english") -> Dict[str, Any]:
        """Find content gaps between primary and competitor domains."""
        try:
            # Get ranked keywords for both domains
            primary_keywords = self.get_top_ranked_keywords(
                primary_domain, location, language, 500
            )
            
            competitor_keywords = self.get_top_ranked_keywords(
                competitor_domain, location, language, 500
            )
            
            if not primary_keywords or not competitor_keywords:
                return {
                    "error": "Insufficient keyword data for gap analysis",
                    "primary_domain": primary_domain,
                    "competitor_domain": competitor_domain
                }
            
            # Extract keyword lists
            primary_kw_set = {kw["keyword"].lower() for kw in primary_keywords}
            competitor_kw_set = {kw["keyword"].lower() for kw in competitor_keywords}
            
            # Find gaps (keywords competitor ranks for but primary doesn't)
            gap_keywords = competitor_kw_set - primary_kw_set
            
            # Get detailed data for gap keywords
            gap_opportunities = []
            for kw in gap_keywords:
                competitor_kw_data = next(
                    (k for k in competitor_keywords if k["keyword"].lower() == kw), 
                    None
                )
                
                if competitor_kw_data and competitor_kw_data.get("search_volume", 0) > 100:
                    gap_opportunities.append({
                        "keyword": competitor_kw_data["keyword"],
                        "competitor_position": competitor_kw_data["position"],
                        "search_volume": competitor_kw_data["search_volume"],
                        "cpc": competitor_kw_data.get("cpc", 0),
                        "etv": competitor_kw_data.get("etv", 0),
                        "competitor_url": competitor_kw_data.get("url", ""),
                        "opportunity_score": self._calculate_opportunity_score(
                            competitor_kw_data["position"],
                            competitor_kw_data.get("search_volume", 0),
                            competitor_kw_data.get("cpc", 0)
                        )
                    })
            
            # Sort by opportunity score (descending)
            gap_opportunities.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
            
            # Find mutual keywords for position comparison
            mutual_keywords = primary_kw_set & competitor_kw_set
            position_comparison = []
            
            for kw in mutual_keywords:
                primary_data = next((k for k in primary_keywords if k["keyword"].lower() == kw), None)
                competitor_data = next((k for k in competitor_keywords if k["keyword"].lower() == kw), None)
                
                if primary_data and competitor_data:
                    position_comparison.append({
                        "keyword": primary_data["keyword"],
                        "primary_position": primary_data["position"],
                        "competitor_position": competitor_data["position"],
                        "position_difference": primary_data["position"] - competitor_data["position"],
                        "search_volume": primary_data.get("search_volume", 0),
                        "primary_url": primary_data.get("url", ""),
                        "competitor_url": competitor_data.get("url", "")
                    })
            
            # Sort position comparison by search volume (descending)
            position_comparison.sort(key=lambda x: x.get("search_volume", 0), reverse=True)
            
            return {
                "status": "completed",
                "primary_domain": primary_domain,
                "competitor_domain": competitor_domain,
                "content_gaps": {
                    "total_gap_keywords": len(gap_opportunities),
                    "high_opportunity_keywords": len([g for g in gap_opportunities if g["opportunity_score"] > 70]),
                    "gap_keywords": gap_opportunities[:50],  # Top 50 opportunities
                    "total_search_volume_gap": sum(g.get("search_volume", 0) for g in gap_opportunities),
                    "total_etv_gap": sum(g.get("etv", 0) for g in gap_opportunities)
                },
                "position_comparison": {
                    "mutual_keywords_count": len(position_comparison),
                    "primary_wins": len([p for p in position_comparison if p["position_difference"] < 0]),
                    "competitor_wins": len([p for p in position_comparison if p["position_difference"] > 0]),
                    "ties": len([p for p in position_comparison if p["position_difference"] == 0]),
                    "keyword_battles": position_comparison[:30]  # Top 30 by search volume
                },
                "location": location,
                "language": language,
                "gap_analysis_summary": self._create_gap_analysis_summary(gap_opportunities, position_comparison)
            }
        
        except Exception as e:
            return {
                "error": f"Failed to find content gaps: {str(e)}",
                "primary_domain": primary_domain,
                "competitor_domain": competitor_domain
            }
    
    def _calculate_visibility_score(self, domain_overview: Dict[str, Any]) -> float:
        """Calculate domain visibility score based on ranking positions."""
        total_keywords = domain_overview.get("organic_keywords", 0)
        if total_keywords == 0:
            return 0
        
        # Weight different positions
        visibility = (
            domain_overview.get("organic_pos_1", 0) * 100 +
            domain_overview.get("organic_pos_2_3", 0) * 85 +
            domain_overview.get("organic_pos_4_10", 0) * 60 +
            domain_overview.get("organic_pos_11_20", 0) * 30 +
            domain_overview.get("organic_pos_21_30", 0) * 15 +
            domain_overview.get("organic_pos_31_40", 0) * 8 +
            domain_overview.get("organic_pos_41_50", 0) * 5 +
            domain_overview.get("organic_pos_51_60", 0) * 3 +
            domain_overview.get("organic_pos_61_70", 0) * 2 +
            domain_overview.get("organic_pos_71_80", 0) * 1 +
            domain_overview.get("organic_pos_81_90", 0) * 0.5 +
            domain_overview.get("organic_pos_91_100", 0) * 0.2
        ) / total_keywords
        
        return round(visibility, 2)
    
    def _categorize_competition_level(self, common_keywords: int, total_keywords: int) -> str:
        """Categorize competition level based on keyword overlap."""
        if total_keywords == 0:
            return "unknown"
        
        overlap_ratio = common_keywords / total_keywords
        
        if overlap_ratio > 0.5:
            return "direct_competitor"
        elif overlap_ratio > 0.2:
            return "strong_competitor"
        elif overlap_ratio > 0.1:
            return "moderate_competitor"
        else:
            return "indirect_competitor"
    
    def _estimate_keyword_difficulty(self, position: int, search_volume: int, competition: float) -> str:
        """Estimate keyword difficulty based on current ranking and metrics."""
        if position <= 3:
            return "easy"
        elif position <= 10:
            if search_volume > 10000 and competition > 0.7:
                return "hard"
            else:
                return "medium"
        elif position <= 20:
            return "medium"
        else:
            if search_volume > 5000 and competition > 0.5:
                return "hard"
            else:
                return "medium"
    
    def _create_domain_summary(self, domain_overview: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for domain analysis."""
        total_keywords = domain_overview.get("organic_keywords", 0)
        top_10_keywords = (
            domain_overview.get("organic_pos_1", 0) +
            domain_overview.get("organic_pos_2_3", 0) +
            domain_overview.get("organic_pos_4_10", 0)
        )
        
        # Calculate ranking quality
        if total_keywords > 0:
            top_10_ratio = top_10_keywords / total_keywords
            if top_10_ratio > 0.3:
                ranking_quality = "excellent"
            elif top_10_ratio > 0.15:
                ranking_quality = "good"
            else:
                ranking_quality = "needs_improvement"
        else:
            ranking_quality = "unknown"
        
        return {
            "domain_strength": "strong" if total_keywords > 10000 else "moderate" if total_keywords > 1000 else "weak",
            "ranking_quality": ranking_quality,
            "organic_visibility": domain_overview.get("visibility_score", 0),
            "estimated_monthly_traffic": domain_overview.get("organic_etv", 0),
            "top_competitors_count": len([c for c in competitors if c.get("competition_level") == "direct_competitor"]),
            "market_position": self._determine_market_position(domain_overview, competitors)
        }
    
    def _create_domain_comparison(self, primary: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create domain comparison metrics."""
        if not competitors:
            return {}
        
        competitor_data = [comp["analysis"] for comp in competitors]
        
        # Calculate averages
        avg_keywords = sum(c.get("organic_keywords", 0) for c in competitor_data) / len(competitor_data)
        avg_etv = sum(c.get("organic_etv", 0) for c in competitor_data) / len(competitor_data)
        avg_visibility = sum(c.get("visibility_score", 0) for c in competitor_data) / len(competitor_data)
        
        return {
            "keyword_advantage": primary.get("organic_keywords", 0) - avg_keywords,
            "etv_advantage": primary.get("organic_etv", 0) - avg_etv,
            "visibility_advantage": primary.get("visibility_score", 0) - avg_visibility,
            "ranking_comparison": {
                "better_than": len([c for c in competitor_data if primary.get("organic_keywords", 0) > c.get("organic_keywords", 0)]),
                "worse_than": len([c for c in competitor_data if primary.get("organic_keywords", 0) < c.get("organic_keywords", 0)]),
                "similar_to": len([c for c in competitor_data if abs(primary.get("organic_keywords", 0) - c.get("organic_keywords", 0)) < primary.get("organic_keywords", 0) * 0.1])
            }
        }
    
    def _generate_competitive_insights(self, primary: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable competitive insights."""
        insights = []
        
        if not competitors:
            return ["Insufficient competitor data for insights"]
        
        competitor_data = [comp["analysis"] for comp in competitors]
        primary_keywords = primary.get("organic_keywords", 0)
        primary_etv = primary.get("organic_etv", 0)
        
        # Keyword insights
        max_competitor_keywords = max(c.get("organic_keywords", 0) for c in competitor_data)
        if primary_keywords < max_competitor_keywords * 0.5:
            insights.append(f"Significant keyword gap: Top competitor has {max_competitor_keywords} keywords vs your {primary_keywords}")
        
        # Traffic insights
        max_competitor_etv = max(c.get("organic_etv", 0) for c in competitor_data)
        if primary_etv < max_competitor_etv * 0.5:
            insights.append(f"Major traffic opportunity: Top competitor estimates ${max_competitor_etv:.2f} monthly traffic value")
        
        # Position insights
        primary_top10_ratio = (
            primary.get("organic_pos_1", 0) + 
            primary.get("organic_pos_2_3", 0) + 
            primary.get("organic_pos_4_10", 0)
        ) / primary_keywords if primary_keywords > 0 else 0
        
        if primary_top10_ratio < 0.2:
            insights.append("Focus on improving rankings: Less than 20% of keywords in top 10 positions")
        
        return insights
    
    def _determine_market_position(self, domain_overview: Dict[str, Any], competitors: List[Dict[str, Any]]) -> str:
        """Determine market position relative to competitors."""
        if not competitors:
            return "unknown"
        
        primary_keywords = domain_overview.get("organic_keywords", 0)
        competitor_keywords = [c.get("se_keywords_count", 0) for c in competitors]
        
        if not competitor_keywords:
            return "unknown"
        
        avg_competitor_keywords = sum(competitor_keywords) / len(competitor_keywords)
        max_competitor_keywords = max(competitor_keywords)
        
        if primary_keywords > max_competitor_keywords:
            return "market_leader"
        elif primary_keywords > avg_competitor_keywords:
            return "strong_player"
        elif primary_keywords > avg_competitor_keywords * 0.5:
            return "moderate_player"
        else:
            return "underperformer"
    
    def _calculate_opportunity_score(self, position: int, search_volume: int, cpc: float) -> float:
        """Calculate opportunity score for gap keywords."""
        # Base score from search volume (0-40 points)
        volume_score = min(40, search_volume / 1000)
        
        # Position score (closer to top = higher opportunity, 0-30 points)
        position_score = max(0, 30 - position)
        
        # CPC score (higher CPC = higher value, 0-30 points)
        cpc_score = min(30, cpc * 10) if cpc else 0
        
        return min(100, volume_score + position_score + cpc_score)
    
    def _create_gap_analysis_summary(self, gap_opportunities: List[Dict[str, Any]], 
                                   position_comparison: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for gap analysis."""
        if not gap_opportunities and not position_comparison:
            return {}
        
        return {
            "total_opportunities": len(gap_opportunities),
            "high_value_opportunities": len([g for g in gap_opportunities if g.get("search_volume", 0) > 1000]),
            "quick_wins": len([g for g in gap_opportunities if g.get("competitor_position", 999) > 10 and g.get("search_volume", 0) > 500]),
            "content_priority": "high" if len([g for g in gap_opportunities if g.get("opportunity_score", 0) > 70]) > 10 else "medium",
            "competitive_advantage": len([p for p in position_comparison if p.get("position_difference", 0) < 0]),
            "improvement_needed": len([p for p in position_comparison if p.get("position_difference", 0) > 5])
        }