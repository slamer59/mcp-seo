"""
OnPage SEO analysis tools using DataForSEO API.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from mcp_seo.dataforseo.client import DataForSEOClient, ApiException
from mcp_seo.models.seo_models import OnPageAnalysisRequest, OnPageSummary, OnPageIssue


class OnPageAnalyzer:
    """OnPage SEO analysis tool."""
    
    def __init__(self, client: DataForSEOClient):
        self.client = client
    
    def create_analysis_task(self, request: OnPageAnalysisRequest) -> Dict[str, Any]:
        """Create OnPage analysis task."""
        try:
            task_data = {
                "max_crawl_pages": request.max_crawl_pages,
                "start_url": str(request.start_url) if request.start_url else str(request.target),
                "respect_sitemap": request.respect_sitemap,
                "crawl_delay": request.crawl_delay
            }
            
            if request.custom_sitemap:
                task_data["custom_sitemap"] = str(request.custom_sitemap)
            
            if request.user_agent:
                task_data["user_agent"] = request.user_agent
                
            if request.enable_javascript:
                task_data["enable_javascript"] = True
            
            result = self.client.create_onpage_task(str(request.target), **task_data)
            
            if result.get("tasks") and result["tasks"][0].get("id"):
                task_id = result["tasks"][0]["id"]
                return {
                    "task_id": task_id,
                    "status": "created",
                    "message": "OnPage analysis task created successfully",
                    "target": str(request.target),
                    "estimated_completion_time": request.max_crawl_pages * request.crawl_delay + 300
                }
            else:
                raise ApiException("Failed to create OnPage task")
        
        except Exception as e:
            return {
                "error": f"Failed to create OnPage analysis task: {str(e)}",
                "target": str(request.target)
            }
    
    def get_analysis_summary(self, task_id: str) -> Dict[str, Any]:
        """Get OnPage analysis summary."""
        try:
            result = self.client.get_onpage_summary(task_id)
            
            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "in_progress",
                    "message": "Analysis still in progress"
                }
            
            task_result = result["tasks"][0]["result"][0]
            
            # Extract summary information
            summary_data = {
                "task_id": task_id,
                "target": task_result.get("target", ""),
                "crawled_pages": task_result.get("crawled_pages", 0),
                "total_issues": 0,
                "critical_issues": 0,
                "high_priority_issues": 0,
                "medium_priority_issues": 0,
                "low_priority_issues": 0,
                "issues": [],
                "pages_by_status_code": task_result.get("pages_by_status_code", {}),
                "broken_pages": task_result.get("broken_pages", 0),
                "duplicate_title_tags": task_result.get("duplicate_title_tags", 0),
                "duplicate_meta_descriptions": task_result.get("duplicate_meta_descriptions", 0),
                "duplicate_h1_tags": task_result.get("duplicate_h1_tags", 0)
            }
            
            # Analyze issues
            issues = self._analyze_technical_issues(task_result)
            summary_data["issues"] = issues
            summary_data["total_issues"] = len(issues)
            
            # Count issues by severity
            for issue in issues:
                severity = issue.get("severity", "low")
                if severity == "critical":
                    summary_data["critical_issues"] += 1
                elif severity == "high":
                    summary_data["high_priority_issues"] += 1
                elif severity == "medium":
                    summary_data["medium_priority_issues"] += 1
                else:
                    summary_data["low_priority_issues"] += 1
            
            return {
                "task_id": task_id,
                "status": "completed",
                "summary": summary_data
            }
        
        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get OnPage summary: {str(e)}"
            }
    
    def get_page_details(self, task_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get detailed page analysis."""
        try:
            result = self.client.get_onpage_pages(task_id, limit=limit, offset=offset)
            
            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "error": "No page data available"
                }
            
            pages = result["tasks"][0]["result"]
            analyzed_pages = []
            
            for page in pages:
                page_analysis = {
                    "url": page.get("url", ""),
                    "status_code": page.get("status_code", 0),
                    "title": page.get("title", ""),
                    "meta_description": page.get("meta_description", ""),
                    "h1": page.get("h1", ""),
                    "word_count": page.get("plain_text_word_count", 0),
                    "internal_links_count": page.get("internal_links_count", 0),
                    "external_links_count": page.get("external_links_count", 0),
                    "images_count": page.get("images_count", 0),
                    "images_without_alt": page.get("images_without_alt", 0),
                    "load_time": page.get("load_time", 0),
                    "page_size": page.get("page_size", 0),
                    "issues": self._analyze_page_issues(page)
                }
                analyzed_pages.append(page_analysis)
            
            return {
                "task_id": task_id,
                "pages": analyzed_pages,
                "total_pages": len(analyzed_pages),
                "has_more": len(pages) >= limit
            }
        
        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get page details: {str(e)}"
            }
    
    def get_duplicate_content_analysis(self, task_id: str) -> Dict[str, Any]:
        """Get duplicate content analysis."""
        try:
            result = self.client.get_onpage_duplicate_tags(task_id)
            
            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "error": "No duplicate content data available"
                }
            
            duplicates = result["tasks"][0]["result"][0]
            
            duplicate_analysis = {
                "task_id": task_id,
                "duplicate_title_tags": {
                    "count": len(duplicates.get("duplicate_title", [])),
                    "duplicates": duplicates.get("duplicate_title", [])
                },
                "duplicate_meta_descriptions": {
                    "count": len(duplicates.get("duplicate_description", [])),
                    "duplicates": duplicates.get("duplicate_description", [])
                },
                "duplicate_h1_tags": {
                    "count": len(duplicates.get("duplicate_h1", [])),
                    "duplicates": duplicates.get("duplicate_h1", [])
                }
            }
            
            return {
                "task_id": task_id,
                "duplicate_analysis": duplicate_analysis
            }
        
        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get duplicate content analysis: {str(e)}"
            }
    
    def get_lighthouse_analysis(self, task_id: str) -> Dict[str, Any]:
        """Get Lighthouse performance analysis."""
        try:
            result = self.client.get_onpage_lighthouse(task_id)
            
            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "message": "Lighthouse data not available for this task"
                }
            
            lighthouse_data = result["tasks"][0]["result"][0]
            
            return {
                "task_id": task_id,
                "lighthouse_analysis": {
                    "performance_score": lighthouse_data.get("performance", 0),
                    "accessibility_score": lighthouse_data.get("accessibility", 0),
                    "best_practices_score": lighthouse_data.get("best_practices", 0),
                    "seo_score": lighthouse_data.get("seo", 0),
                    "first_contentful_paint": lighthouse_data.get("first_contentful_paint", {}),
                    "largest_contentful_paint": lighthouse_data.get("largest_contentful_paint", {}),
                    "cumulative_layout_shift": lighthouse_data.get("cumulative_layout_shift", {}),
                    "first_input_delay": lighthouse_data.get("first_input_delay", {}),
                    "speed_index": lighthouse_data.get("speed_index", {}),
                    "total_blocking_time": lighthouse_data.get("total_blocking_time", {})
                }
            }
        
        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get Lighthouse analysis: {str(e)}"
            }
    
    def _analyze_technical_issues(self, task_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze technical SEO issues from task result."""
        issues = []
        
        # Check for broken pages
        broken_pages = task_result.get("broken_pages", 0)
        if broken_pages > 0:
            issues.append({
                "issue_type": "broken_pages",
                "severity": "high" if broken_pages > 10 else "medium",
                "affected_pages": broken_pages,
                "description": f"Found {broken_pages} broken pages (4xx/5xx status codes)",
                "recommendation": "Fix or remove broken pages, implement proper redirects for moved content"
            })
        
        # Check for duplicate title tags
        duplicate_titles = task_result.get("duplicate_title_tags", 0)
        if duplicate_titles > 0:
            issues.append({
                "issue_type": "duplicate_title_tags",
                "severity": "high",
                "affected_pages": duplicate_titles,
                "description": f"Found {duplicate_titles} pages with duplicate title tags",
                "recommendation": "Create unique, descriptive title tags for each page (50-60 characters)"
            })
        
        # Check for duplicate meta descriptions
        duplicate_descriptions = task_result.get("duplicate_meta_descriptions", 0)
        if duplicate_descriptions > 0:
            issues.append({
                "issue_type": "duplicate_meta_descriptions",
                "severity": "medium",
                "affected_pages": duplicate_descriptions,
                "description": f"Found {duplicate_descriptions} pages with duplicate meta descriptions",
                "recommendation": "Write unique meta descriptions for each page (150-160 characters)"
            })
        
        # Check for duplicate H1 tags
        duplicate_h1 = task_result.get("duplicate_h1_tags", 0)
        if duplicate_h1 > 0:
            issues.append({
                "issue_type": "duplicate_h1_tags",
                "severity": "medium",
                "affected_pages": duplicate_h1,
                "description": f"Found {duplicate_h1} pages with duplicate H1 tags",
                "recommendation": "Ensure each page has a unique, descriptive H1 tag"
            })
        
        # Check status code distribution
        status_codes = task_result.get("pages_by_status_code", {})
        if status_codes:
            total_pages = sum(status_codes.values())
            redirect_pages = status_codes.get("3xx", 0)
            
            if redirect_pages > 0:
                redirect_ratio = redirect_pages / total_pages
                if redirect_ratio > 0.1:  # More than 10% redirects
                    issues.append({
                        "issue_type": "excessive_redirects",
                        "severity": "medium",
                        "affected_pages": redirect_pages,
                        "description": f"High number of redirects detected ({redirect_pages} pages, {redirect_ratio:.1%})",
                        "recommendation": "Review redirect chains and minimize unnecessary redirects"
                    })
        
        return issues
    
    def _analyze_page_issues(self, page_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Analyze individual page issues."""
        issues = []
        
        # Check title tag
        title = page_data.get("title", "")
        if not title:
            issues.append({
                "type": "missing_title",
                "severity": "high",
                "message": "Missing title tag"
            })
        elif len(title) < 30:
            issues.append({
                "type": "short_title",
                "severity": "medium",
                "message": f"Title tag too short ({len(title)} characters)"
            })
        elif len(title) > 60:
            issues.append({
                "type": "long_title",
                "severity": "medium",
                "message": f"Title tag too long ({len(title)} characters)"
            })
        
        # Check meta description
        meta_desc = page_data.get("meta_description", "")
        if not meta_desc:
            issues.append({
                "type": "missing_meta_description",
                "severity": "medium",
                "message": "Missing meta description"
            })
        elif len(meta_desc) < 120:
            issues.append({
                "type": "short_meta_description",
                "severity": "low",
                "message": f"Meta description too short ({len(meta_desc)} characters)"
            })
        elif len(meta_desc) > 160:
            issues.append({
                "type": "long_meta_description",
                "severity": "low",
                "message": f"Meta description too long ({len(meta_desc)} characters)"
            })
        
        # Check H1 tag
        h1 = page_data.get("h1", "")
        if not h1:
            issues.append({
                "type": "missing_h1",
                "severity": "high",
                "message": "Missing H1 tag"
            })
        
        # Check images without alt text
        images_without_alt = page_data.get("images_without_alt", 0)
        total_images = page_data.get("images_count", 0)
        if images_without_alt > 0 and total_images > 0:
            issues.append({
                "type": "images_missing_alt",
                "severity": "medium",
                "message": f"{images_without_alt}/{total_images} images missing alt text"
            })
        
        # Check page load time
        load_time = page_data.get("load_time", 0)
        if load_time > 3000:  # 3 seconds
            issues.append({
                "type": "slow_load_time",
                "severity": "high" if load_time > 5000 else "medium",
                "message": f"Slow page load time ({load_time/1000:.1f}s)"
            })
        
        # Check content length
        word_count = page_data.get("plain_text_word_count", 0)
        if word_count < 300:
            issues.append({
                "type": "thin_content",
                "severity": "medium",
                "message": f"Thin content ({word_count} words)"
            })
        
        return issues