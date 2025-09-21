"""
FastMCP SEO Analysis Server using DataForSEO API.
Provides comprehensive SEO analysis tools through MCP protocol.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl, validator

from mcp_seo.config.settings import (get_language_code, get_location_code,
                                     get_settings)
# Import our modules
from mcp_seo.dataforseo.client import ApiException, DataForSEOClient
from mcp_seo.models.seo_models import (AnalysisStatus, ContentAnalysisRequest,
                                       DeviceType, DomainAnalysisRequest,
                                       KeywordAnalysisRequest,
                                       OnPageAnalysisRequest, SEOAuditRequest,
                                       SERPAnalysisRequest)
from mcp_seo.tools.competitor_analyzer import CompetitorAnalyzer
from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from mcp_seo.tools.onpage_analyzer import OnPageAnalyzer

# Initialize FastMCP server
mcp = FastMCP("FastMCP SEO Analysis Server")

# Global variables for clients (will be initialized on first use)
_dataforseo_client: Optional[DataForSEOClient] = None
_onpage_analyzer: Optional[OnPageAnalyzer] = None
_keyword_analyzer: Optional[KeywordAnalyzer] = None
_competitor_analyzer: Optional[CompetitorAnalyzer] = None


def get_clients():
    """Initialize and return DataForSEO client and analyzers."""
    global _dataforseo_client, _onpage_analyzer, _keyword_analyzer, _competitor_analyzer
    
    if _dataforseo_client is None:
        settings = get_settings()
        _dataforseo_client = DataForSEOClient()
        _onpage_analyzer = OnPageAnalyzer(_dataforseo_client)
        _keyword_analyzer = KeywordAnalyzer(_dataforseo_client)
        _competitor_analyzer = CompetitorAnalyzer(_dataforseo_client)
    
    return _dataforseo_client, _onpage_analyzer, _keyword_analyzer, _competitor_analyzer


# Pydantic models for MCP tool parameters
class OnPageAnalysisParams(BaseModel):
    """Parameters for OnPage SEO analysis."""
    target: str = Field(..., description="Target website URL to analyze")
    max_crawl_pages: int = Field(default=100, ge=1, le=1000, description="Maximum pages to crawl")
    start_url: Optional[str] = Field(None, description="Starting URL (defaults to target)")
    respect_sitemap: bool = Field(default=True, description="Follow XML sitemap")
    custom_sitemap: Optional[str] = Field(None, description="Custom sitemap URL")
    crawl_delay: int = Field(default=1, ge=0, le=10, description="Delay between requests in seconds")
    enable_javascript: bool = Field(default=False, description="Enable JavaScript rendering")


class KeywordAnalysisParams(BaseModel):
    """Parameters for keyword analysis."""
    keywords: List[str] = Field(..., min_length=1, max_length=100, description="List of keywords to analyze")
    location: str = Field(default="usa", description="Geographic location (usa, uk, canada, etc.)")
    language: str = Field(default="english", description="Language for analysis")
    device: str = Field(default="desktop", description="Device type (desktop, mobile, tablet)")
    include_suggestions: bool = Field(default=False, description="Include keyword suggestions")
    suggestion_limit: int = Field(default=50, ge=1, le=200, description="Maximum number of suggestions")


class SERPAnalysisParams(BaseModel):
    """Parameters for SERP analysis."""
    keyword: str = Field(..., description="Keyword to analyze SERP for")
    location: str = Field(default="usa", description="Geographic location")
    language: str = Field(default="english", description="Language for analysis")
    device: str = Field(default="desktop", description="Device type")
    depth: int = Field(default=100, ge=1, le=200, description="Number of results to analyze")
    include_paid_results: bool = Field(default=True, description="Include paid search results")


class DomainAnalysisParams(BaseModel):
    """Parameters for domain analysis."""
    target: str = Field(..., description="Target domain to analyze")
    location: str = Field(default="usa", description="Geographic location")
    language: str = Field(default="english", description="Language for analysis")
    include_competitors: bool = Field(default=True, description="Include competitor analysis")
    competitor_limit: int = Field(default=20, ge=1, le=100, description="Maximum number of competitors")
    include_keywords: bool = Field(default=True, description="Include ranked keywords")
    keyword_limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of keywords")


class CompetitorComparisonParams(BaseModel):
    """Parameters for competitor comparison."""
    primary_domain: str = Field(..., description="Primary domain to compare")
    competitor_domains: List[str] = Field(..., min_length=1, max_length=10, description="Competitor domains")
    location: str = Field(default="usa", description="Geographic location")
    language: str = Field(default="english", description="Language for analysis")


class ContentGapAnalysisParams(BaseModel):
    """Parameters for content gap analysis."""
    primary_domain: str = Field(..., description="Primary domain")
    competitor_domain: str = Field(..., description="Competitor domain to compare against")
    location: str = Field(default="usa", description="Geographic location")
    language: str = Field(default="english", description="Language for analysis")


class TaskStatusParams(BaseModel):
    """Parameters for task status checking."""
    task_id: str = Field(..., description="Task ID to check status for")
    endpoint_type: str = Field(default="onpage", description="API endpoint type (onpage, serp, keywords, etc.)")


# OnPage SEO Analysis Tools
@mcp.tool()
def onpage_analysis_start(params: OnPageAnalysisParams) -> Dict[str, Any]:
    """
    Start comprehensive OnPage SEO analysis for a website.
    
    Analyzes technical SEO factors including:
    - Title tags, meta descriptions, headers
    - Internal/external links
    - Image optimization
    - Page load times
    - Duplicate content
    - Broken pages and redirects
    """
    try:
        client, onpage_analyzer, _, _ = get_clients()
        
        request = OnPageAnalysisRequest(
            target=params.target,
            max_crawl_pages=params.max_crawl_pages,
            start_url=params.start_url,
            respect_sitemap=params.respect_sitemap,
            custom_sitemap=params.custom_sitemap,
            crawl_delay=params.crawl_delay,
            enable_javascript=params.enable_javascript
        )
        
        result = onpage_analyzer.create_analysis_task(request)
        return result
    
    except Exception as e:
        return {"error": f"Failed to start OnPage analysis: {str(e)}"}


@mcp.tool()
def onpage_analysis_results(params: TaskStatusParams) -> Dict[str, Any]:
    """
    Get OnPage SEO analysis results and summary.
    
    Returns comprehensive technical SEO audit including:
    - Overall site health score
    - Critical issues requiring immediate attention
    - Technical SEO recommendations
    - Page-by-page analysis summary
    """
    try:
        client, onpage_analyzer, _, _ = get_clients()
        result = onpage_analyzer.get_analysis_summary(params.task_id)
        return result
    
    except Exception as e:
        return {"error": f"Failed to get OnPage results: {str(e)}"}


@mcp.tool()
def onpage_page_details(params: TaskStatusParams) -> Dict[str, Any]:
    """
    Get detailed OnPage analysis for individual pages.
    
    Provides page-level SEO insights including:
    - Title and meta tag optimization
    - Content quality metrics
    - Internal linking structure
    - Technical issues per page
    """
    try:
        client, onpage_analyzer, _, _ = get_clients()
        result = onpage_analyzer.get_page_details(params.task_id)
        return result
    
    except Exception as e:
        return {"error": f"Failed to get page details: {str(e)}"}


@mcp.tool()
def onpage_duplicate_content(params: TaskStatusParams) -> Dict[str, Any]:
    """
    Get duplicate content analysis from OnPage audit.
    
    Identifies and reports:
    - Duplicate title tags
    - Duplicate meta descriptions
    - Duplicate H1 tags
    - Content cannibalization issues
    """
    try:
        client, onpage_analyzer, _, _ = get_clients()
        result = onpage_analyzer.get_duplicate_content_analysis(params.task_id)
        return result
    
    except Exception as e:
        return {"error": f"Failed to get duplicate content analysis: {str(e)}"}


@mcp.tool()
def onpage_lighthouse_audit(params: TaskStatusParams) -> Dict[str, Any]:
    """
    Get Lighthouse performance and SEO audit results.
    
    Provides Core Web Vitals and performance metrics:
    - Performance, accessibility, best practices, SEO scores
    - First Contentful Paint, Largest Contentful Paint
    - Cumulative Layout Shift, First Input Delay
    - Page speed optimization recommendations
    """
    try:
        client, onpage_analyzer, _, _ = get_clients()
        result = onpage_analyzer.get_lighthouse_analysis(params.task_id)
        return result
    
    except Exception as e:
        return {"error": f"Failed to get Lighthouse audit: {str(e)}"}


# Keyword Research and Analysis Tools
@mcp.tool()
def keyword_analysis(params: KeywordAnalysisParams) -> Dict[str, Any]:
    """
    Comprehensive keyword research and analysis.
    
    Provides keyword metrics including:
    - Search volume and trends
    - Cost-per-click (CPC) data
    - Competition levels
    - Keyword suggestions and variations
    - Monthly search patterns
    """
    try:
        client, _, keyword_analyzer, _ = get_clients()
        
        device_type = DeviceType.DESKTOP
        if params.device.lower() == "mobile":
            device_type = DeviceType.MOBILE
        elif params.device.lower() == "tablet":
            device_type = DeviceType.TABLET
        
        request = KeywordAnalysisRequest(
            keywords=params.keywords,
            location=params.location,
            language=params.language,
            device=device_type,
            include_suggestions=params.include_suggestions,
            suggestion_limit=params.suggestion_limit
        )
        
        result = keyword_analyzer.analyze_keywords(request)
        return result
    
    except Exception as e:
        return {"error": f"Failed to analyze keywords: {str(e)}"}


@mcp.tool()
def keyword_suggestions(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get keyword suggestions based on seed keyword.
    
    Discovers related keywords including:
    - Long-tail keyword variations
    - Related search terms
    - Question-based keywords
    - Search volume and competition data for each suggestion
    """
    try:
        client, _, keyword_analyzer, _ = get_clients()
        
        seed_keyword = params.get("seed_keyword", "")
        location = params.get("location", "usa")
        language = params.get("language", "english")
        limit = params.get("limit", 100)
        
        if not seed_keyword:
            return {"error": "seed_keyword parameter is required"}
        
        result = keyword_analyzer.get_keyword_suggestions(seed_keyword, location, language, limit)
        return result
    
    except Exception as e:
        return {"error": f"Failed to get keyword suggestions: {str(e)}"}


@mcp.tool()
def serp_analysis(params: SERPAnalysisParams) -> Dict[str, Any]:
    """
    Analyze Search Engine Results Page (SERP) for specific keyword.
    
    Provides SERP insights including:
    - Organic and paid search results
    - Featured snippets and rich results
    - People Also Ask questions
    - Related searches
    - Competitive landscape analysis
    """
    try:
        client, _, keyword_analyzer, _ = get_clients()
        
        device_type = DeviceType.DESKTOP
        if params.device.lower() == "mobile":
            device_type = DeviceType.MOBILE
        elif params.device.lower() == "tablet":
            device_type = DeviceType.TABLET
        
        request = SERPAnalysisRequest(
            keyword=params.keyword,
            location=params.location,
            language=params.language,
            device=device_type,
            depth=params.depth,
            include_paid_results=params.include_paid_results
        )
        
        result = keyword_analyzer.analyze_serp_for_keyword(request)
        return result
    
    except Exception as e:
        return {"error": f"Failed to analyze SERP: {str(e)}"}


@mcp.tool()
def keyword_difficulty(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate keyword difficulty scores for target keywords.
    
    Estimates ranking difficulty based on:
    - SERP competition analysis
    - Domain authority of ranking pages
    - Content quality indicators
    - Keyword-specific difficulty metrics
    """
    try:
        client, _, keyword_analyzer, _ = get_clients()
        
        keywords = params.get("keywords", [])
        location = params.get("location", "usa")
        language = params.get("language", "english")
        
        if not keywords:
            return {"error": "keywords parameter is required"}
        
        result = keyword_analyzer.get_keyword_difficulty(keywords, location, language)
        return result
    
    except Exception as e:
        return {"error": f"Failed to calculate keyword difficulty: {str(e)}"}


# Domain and Competitor Analysis Tools
@mcp.tool()
def domain_analysis(params: DomainAnalysisParams) -> Dict[str, Any]:
    """
    Comprehensive domain SEO analysis and competitive intelligence.
    
    Provides domain insights including:
    - Organic keyword rankings and traffic estimates
    - Domain authority and visibility metrics
    - Top performing keywords and pages
    - Competitive landscape overview
    - SEO growth opportunities
    """
    try:
        client, _, _, competitor_analyzer = get_clients()
        
        request = DomainAnalysisRequest(
            target=params.target,
            location=params.location,
            language=params.language,
            include_competitors=params.include_competitors,
            competitor_limit=params.competitor_limit,
            include_keywords=params.include_keywords,
            keyword_limit=params.keyword_limit
        )
        
        result = competitor_analyzer.analyze_domain_overview(request)
        return result
    
    except Exception as e:
        return {"error": f"Failed to analyze domain: {str(e)}"}


@mcp.tool()
def competitor_comparison(params: CompetitorComparisonParams) -> Dict[str, Any]:
    """
    Compare primary domain against competitor domains.
    
    Provides competitive analysis including:
    - Keyword portfolio comparisons
    - Organic traffic estimates
    - Market share analysis
    - Competitive strengths and weaknesses
    - Strategic recommendations
    """
    try:
        client, _, _, competitor_analyzer = get_clients()
        
        result = competitor_analyzer.compare_domains(
            params.primary_domain,
            params.competitor_domains,
            params.location,
            params.language
        )
        return result
    
    except Exception as e:
        return {"error": f"Failed to compare competitors: {str(e)}"}


@mcp.tool()
def content_gap_analysis(params: ContentGapAnalysisParams) -> Dict[str, Any]:
    """
    Identify content gaps and keyword opportunities vs competitors.
    
    Discovers content opportunities including:
    - Keywords competitors rank for but you don't
    - Position improvement opportunities
    - Content topics to target
    - High-value keyword gaps
    - Quick win opportunities
    """
    try:
        client, _, _, competitor_analyzer = get_clients()
        
        result = competitor_analyzer.find_content_gaps(
            params.primary_domain,
            params.competitor_domain,
            params.location,
            params.language
        )
        return result
    
    except Exception as e:
        return {"error": f"Failed to analyze content gaps: {str(e)}"}


# Utility and Management Tools
@mcp.tool()
def account_info() -> Dict[str, Any]:
    """
    Get DataForSEO account information and usage statistics.
    
    Shows account details including:
    - Available API credits
    - Usage limits and remaining quota
    - Account plan information
    - Cost per API call
    """
    try:
        client, _, _, _ = get_clients()
        result = client.get_account_info()
        return result
    
    except Exception as e:
        return {"error": f"Failed to get account info: {str(e)}"}


@mcp.tool()
def available_locations() -> Dict[str, Any]:
    """
    Get list of available geographic locations for SEO analysis.
    
    Returns supported locations including:
    - Country and region codes
    - Location names and identifiers
    - Supported languages per location
    """
    try:
        client, _, _, _ = get_clients()
        result = client.get_serp_locations()
        return result
    
    except Exception as e:
        return {"error": f"Failed to get available locations: {str(e)}"}


@mcp.tool()
def available_languages() -> Dict[str, Any]:
    """
    Get list of supported languages for SEO analysis.
    
    Returns language options including:
    - Language codes and names
    - Regional language variants
    - Character set information
    """
    try:
        client, _, _, _ = get_clients()
        result = client.get_serp_languages()
        return result
    
    except Exception as e:
        return {"error": f"Failed to get available languages: {str(e)}"}


@mcp.tool()
def task_status(params: TaskStatusParams) -> Dict[str, Any]:
    """
    Check status of running SEO analysis task.
    
    Provides task information including:
    - Current task status (pending, in_progress, completed, failed)
    - Estimated completion time
    - Progress information
    - Error details if applicable
    """
    try:
        client, _, _, _ = get_clients()
        result = client.get_task_results(params.task_id, params.endpoint_type)
        
        # Process result to provide status information
        if result.get("tasks") and result["tasks"][0].get("result"):
            return {
                "task_id": params.task_id,
                "status": "completed",
                "result_available": True,
                "message": "Task completed successfully"
            }
        else:
            return {
                "task_id": params.task_id,
                "status": "in_progress",
                "result_available": False,
                "message": "Task is still processing"
            }
    
    except Exception as e:
        return {
            "task_id": params.task_id,
            "status": "error",
            "error": f"Failed to check task status: {str(e)}"
        }


# Comprehensive SEO Audit Tool
@mcp.tool()
def comprehensive_seo_audit(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive SEO audit combining multiple analysis types.
    
    Provides complete SEO overview including:
    - Technical SEO audit (OnPage analysis)
    - Keyword performance analysis
    - Competitive positioning
    - Content optimization opportunities
    - Actionable recommendations with priority scoring
    """
    try:
        target = params.get("target", "")
        location = params.get("location", "usa")
        language = params.get("language", "english")
        max_crawl_pages = params.get("max_crawl_pages", 50)
        
        if not target:
            return {"error": "target parameter is required"}
        
        client, onpage_analyzer, keyword_analyzer, competitor_analyzer = get_clients()
        
        audit_results = {
            "target": target,
            "location": location,
            "language": language,
            "audit_components": [],
            "overall_scores": {},
            "priority_recommendations": []
        }
        
        # 1. OnPage Technical SEO Analysis
        try:
            onpage_request = OnPageAnalysisRequest(
                target=target,
                max_crawl_pages=max_crawl_pages,
                respect_sitemap=True,
                crawl_delay=1
            )
            
            onpage_task = onpage_analyzer.create_analysis_task(onpage_request)
            if "task_id" in onpage_task:
                # Wait for completion (simplified - in production you'd check status periodically)
                import time
                time.sleep(30)  # Wait 30 seconds for basic analysis
                
                onpage_results = onpage_analyzer.get_analysis_summary(onpage_task["task_id"])
                audit_results["onpage_analysis"] = onpage_results
                audit_results["audit_components"].append("onpage_analysis")
        except Exception as e:
            audit_results["onpage_analysis"] = {"error": str(e)}
        
        # 2. Domain Performance Analysis
        try:
            domain_request = DomainAnalysisRequest(
                target=target.replace("https://", "").replace("http://", "").split("/")[0],
                location=location,
                language=language,
                include_competitors=True,
                competitor_limit=10,
                include_keywords=True,
                keyword_limit=100
            )
            
            domain_results = competitor_analyzer.analyze_domain_overview(domain_request)
            audit_results["domain_analysis"] = domain_results
            audit_results["audit_components"].append("domain_analysis")
        except Exception as e:
            audit_results["domain_analysis"] = {"error": str(e)}
        
        # 3. Generate Overall Recommendations
        recommendations = []
        
        # Technical SEO recommendations
        if "onpage_analysis" in audit_results and "summary" in audit_results["onpage_analysis"]:
            onpage_summary = audit_results["onpage_analysis"]["summary"]
            critical_issues = onpage_summary.get("critical_issues", 0)
            if critical_issues > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "technical_seo",
                    "issue": f"{critical_issues} critical technical SEO issues found",
                    "recommendation": "Address critical issues immediately: broken pages, duplicate content, missing title tags",
                    "impact": "high"
                })
        
        # Domain performance recommendations
        if "domain_analysis" in audit_results and "domain_overview" in audit_results["domain_analysis"]:
            domain_overview = audit_results["domain_analysis"]["domain_overview"]
            organic_keywords = domain_overview.get("organic_keywords", 0)
            if organic_keywords < 100:
                recommendations.append({
                    "priority": "high",
                    "category": "content_seo",
                    "issue": f"Limited organic keyword presence ({organic_keywords} keywords)",
                    "recommendation": "Develop content strategy targeting high-value keywords in your niche",
                    "impact": "high"
                })
        
        audit_results["priority_recommendations"] = recommendations
        audit_results["audit_status"] = "completed"
        
        return audit_results
    
    except Exception as e:
        return {"error": f"Failed to run comprehensive SEO audit: {str(e)}"}


# Register PageRank and graph analysis tools
try:
    from mcp_seo.tools.graph.pagerank_tools import register_pagerank_tools
    register_pagerank_tools(mcp)
except (ImportError, NameError) as e:
    print(f"Warning: PageRank tools not available: {e}")
    print("Install additional dependencies: kuzu, aiohttp, beautifulsoup4, numpy")

# Register NetworkX graph analysis tools
try:
    from mcp_seo.tools.graph.networkx_tools import register_networkx_tools
    register_networkx_tools(mcp)
except ImportError as e:
    print(f"Warning: NetworkX tools not available: {e}")
    print("Install additional dependencies: networkx")
except Exception as e:
    print(f"Warning: Failed to register NetworkX tools: {e}")


if __name__ == "__main__":
    mcp.run()