"""
DataForSEO API client implementation.
Based on DataForSEO Python client: https://github.com/dataforseo/PythonClient
"""

import json
import requests
from typing import Dict, Any, List, Optional, Union
import base64
import os
from ..config.settings import get_settings


class ApiException(Exception):
    """Exception raised for API errors."""
    pass


class DataForSEOClient:
    """DataForSEO API client with comprehensive SEO analysis capabilities."""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize DataForSEO client with credentials."""
        settings = get_settings()
        self.username = username or settings.dataforseo_login
        self.password = password or settings.dataforseo_password
        
        if not self.username or not self.password:
            raise ValueError(
                "DataForSEO credentials required. Set DATAFORSEO_LOGIN and "
                "DATAFORSEO_PASSWORD environment variables or pass directly."
            )
        
        self.base_url = "https://api.dataforseo.com/v3"
        self.session = requests.Session()
        
        # Set up basic authentication
        credentials = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("utf-8")
        self.session.headers.update({
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make authenticated request to DataForSEO API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                response = self.session.get(url, params=data)
            
            response.raise_for_status()
            result = response.json()
            
            # Check API-specific errors
            if result.get("status_code") != 20000:
                raise ApiException(f"API Error: {result.get('status_message', 'Unknown error')}")
            
            return result
        
        except requests.exceptions.RequestException as e:
            raise ApiException(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ApiException(f"Invalid JSON response: {str(e)}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information and usage statistics."""
        return self._make_request("/appendix/user_data")
    
    def get_serp_locations(self, country: Optional[str] = None) -> Dict[str, Any]:
        """Get available SERP API locations."""
        endpoint = "/serp/google/locations"
        params = {"country": country} if country else None
        return self._make_request(endpoint, data=params)
    
    def get_serp_languages(self) -> Dict[str, Any]:
        """Get available SERP API languages."""
        return self._make_request("/serp/google/languages")
    
    # OnPage API Methods
    def create_onpage_task(self, target: str, **kwargs) -> Dict[str, Any]:
        """Create OnPage API task for technical SEO analysis."""
        data = [{
            "target": target,
            "max_crawl_pages": kwargs.get("max_crawl_pages", 100),
            "start_url": kwargs.get("start_url", target),
            "respect_sitemap": kwargs.get("respect_sitemap", True),
            "custom_sitemap": kwargs.get("custom_sitemap"),
            "crawl_delay": kwargs.get("crawl_delay", 1),
            "tag": kwargs.get("tag", "fastmcp-seo")
        }]
        return self._make_request("/on_page/task_post", "POST", data)
    
    def get_onpage_tasks(self) -> Dict[str, Any]:
        """Get list of OnPage tasks."""
        return self._make_request("/on_page/tasks_ready")
    
    def get_onpage_summary(self, task_id: str) -> Dict[str, Any]:
        """Get OnPage task summary."""
        return self._make_request(f"/on_page/summary/{task_id}")
    
    def get_onpage_pages(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Get OnPage crawled pages data."""
        data = [{
            "id": task_id,
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0),
            "filters": kwargs.get("filters", [])
        }]
        return self._make_request("/on_page/pages", "POST", data)
    
    def get_onpage_resources(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Get OnPage resources (CSS, JS, images)."""
        data = [{
            "id": task_id,
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0)
        }]
        return self._make_request("/on_page/resources", "POST", data)
    
    def get_onpage_duplicate_tags(self, task_id: str) -> Dict[str, Any]:
        """Get OnPage duplicate title/meta tags."""
        data = [{"id": task_id}]
        return self._make_request("/on_page/duplicate_tags", "POST", data)
    
    def get_onpage_lighthouse(self, task_id: str) -> Dict[str, Any]:
        """Get OnPage Lighthouse performance data."""
        data = [{"id": task_id}]
        return self._make_request("/on_page/lighthouse", "POST", data)
    
    # SERP API Methods
    def get_serp_results(self, keyword: str, location_code: int = 2840,
                        language_code: str = "en", **kwargs) -> Dict[str, Any]:
        """
        Get Google SERP results for keyword using live synchronous API.

        Returns results immediately without task polling (6 seconds max).
        Note: Default depth is 10. Specify depth parameter for more results.
        """
        data = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": kwargs.get("device", "desktop"),
            "os": kwargs.get("os", "windows"),
            "depth": kwargs.get("depth", 100),
            "tag": kwargs.get("tag", "fastmcp-seo")
        }]
        return self._make_request("/serp/google/organic/live/advanced", "POST", data)

    def get_serp_data(self, keyword: str, location: str = "United States") -> Dict[str, Any]:
        """Get SERP data for a keyword (alias for get_serp_results)."""
        from ..config.settings import get_location_code, get_language_code
        location_code = get_location_code(location)
        language_code = get_language_code("english")
        return self.get_serp_results(keyword, location_code, language_code)

    def get_serp_tasks_ready(self) -> Dict[str, Any]:
        """Get completed SERP tasks."""
        return self._make_request("/serp/google/organic/tasks_ready")
    
    def get_serp_task_results(self, task_id: str) -> Dict[str, Any]:
        """Get SERP task results."""
        return self._make_request(f"/serp/google/organic/task_get/{task_id}")
    
    # Keywords Data API Methods
    def get_keyword_data(self, keywords: List[str], location_code: int = 2840,
                        language_code: str = "en") -> Dict[str, Any]:
        """
        Get keyword search volume and competition data using live synchronous API.

        Returns results immediately without task polling (much faster than task-based API).
        """
        data = [{
            "keywords": keywords,
            "location_code": location_code,
            "language_code": language_code,
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/keywords_data/google_ads/search_volume/live", "POST", data)
    
    def get_keyword_suggestions(self, keyword: str, location_code: int = 2840,
                              language_code: str = "en", **kwargs) -> Dict[str, Any]:
        """
        Get keyword suggestions using live synchronous API.

        Returns results immediately without task polling (much faster than task-based API).
        """
        data = [{
            "keywords": [keyword],  # Live API expects 'keywords' array
            "location_code": location_code,
            "language_code": language_code,
            "limit": kwargs.get("limit", 100),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/keywords_data/google_ads/keywords_for_keywords/live", "POST", data)
    
    # DataForSEO Labs API Methods
    def get_domain_rank_overview(self, target: str, location_code: int = 2840, 
                                language_code: str = "en") -> Dict[str, Any]:
        """Get domain ranking overview."""
        data = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code,
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/dataforseo_labs/google/domain_rank_overview/task_post", "POST", data)
    
    def get_competitor_domains(self, target: str, location_code: int = 2840, 
                              language_code: str = "en", **kwargs) -> Dict[str, Any]:
        """Get competitor domains analysis."""
        data = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code,
            "limit": kwargs.get("limit", 50),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/dataforseo_labs/google/competitors_domain/task_post", "POST", data)
    
    def get_ranked_keywords(self, target: str, location_code: int = 2840, 
                           language_code: str = "en", **kwargs) -> Dict[str, Any]:
        """Get domain's ranked keywords."""
        data = [{
            "target": target,
            "location_code": location_code,
            "language_code": language_code,
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0),
            "filters": kwargs.get("filters", []),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/dataforseo_labs/google/ranked_keywords/task_post", "POST", data)
    
    # Domain Analytics API Methods
    def get_domain_technologies(self, target: str) -> Dict[str, Any]:
        """Get domain technology stack."""
        data = [{
            "target": target,
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/domain_analytics/technologies/domain_technologies/task_post", "POST", data)
    
    def get_domain_whois(self, target: str) -> Dict[str, Any]:
        """Get domain WHOIS information."""
        data = [{
            "target": target,
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/domain_analytics/whois/overview/task_post", "POST", data)
    
    # Backlinks API Methods
    def get_backlinks_summary(self, target: str, **kwargs) -> Dict[str, Any]:
        """Get backlinks summary for domain."""
        data = [{
            "target": target,
            "mode": kwargs.get("mode", "domain"),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/backlinks/summary/task_post", "POST", data)
    
    def get_backlinks_history(self, target: str, **kwargs) -> Dict[str, Any]:
        """Get backlinks history data."""
        data = [{
            "target": target,
            "mode": kwargs.get("mode", "domain"),
            "date_from": kwargs.get("date_from"),
            "date_to": kwargs.get("date_to"),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/backlinks/history/task_post", "POST", data)
    
    def get_referring_domains(self, target: str, **kwargs) -> Dict[str, Any]:
        """Get referring domains list."""
        data = [{
            "target": target,
            "mode": kwargs.get("mode", "domain"),
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0),
            "filters": kwargs.get("filters", []),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/backlinks/referring_domains/task_post", "POST", data)
    
    # Content Analysis API Methods
    def analyze_content_sentiment(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Analyze content sentiment."""
        data = [{
            "text": text,
            "language": language,
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/content_analysis/sentiment_analysis/task_post", "POST", data)
    
    def get_content_summary(self, text: str, **kwargs) -> Dict[str, Any]:
        """Get content summary and analysis."""
        data = [{
            "text": text,
            "summary_length": kwargs.get("summary_length", "short"),
            "tag": "fastmcp-seo"
        }]
        return self._make_request("/content_analysis/summary/task_post", "POST", data)
    
    # Utility Methods
    def get_task_results(self, task_id: str, endpoint_type: str = "serp") -> Dict[str, Any]:
        """Generic method to get task results by ID."""
        endpoint_map = {
            "serp": f"/serp/google/organic/task_get/{task_id}",
            "keywords": f"/keywords_data/google_ads/search_volume/task_get/{task_id}",
            "onpage": f"/on_page/summary/{task_id}",
            "backlinks": f"/backlinks/summary/task_get/{task_id}",
            "domain_analytics": f"/domain_analytics/technologies/domain_technologies/task_get/{task_id}",
            "content_analysis": f"/content_analysis/sentiment_analysis/task_get/{task_id}",
            "dataforseo_labs": f"/dataforseo_labs/google/domain_rank_overview/task_get/{task_id}"
        }
        
        endpoint = endpoint_map.get(endpoint_type)
        if not endpoint:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
        
        return self._make_request(endpoint)
    
    def wait_for_task_completion(self, task_id: str, endpoint_type: str = "serp",
                                max_attempts: int = 30, delay: int = 2) -> Dict[str, Any]:
        """
        Wait for task completion and return results with progressive delay.

        Uses progressive delay: starts at 2s, increases to max 10s for efficient polling.
        Most tasks complete within 1-3 seconds, so we check frequently at first.
        """
        import time

        for attempt in range(max_attempts):
            try:
                result = self.get_task_results(task_id, endpoint_type)
                if result.get("tasks") and result["tasks"][0].get("result"):
                    return result

                # Progressive delay: start fast, then slow down
                # 2s, 2s, 3s, 4s, 5s, ..., max 10s
                current_delay = min(delay + attempt, 10)
                time.sleep(current_delay)
            except ApiException as e:
                if "not found" in str(e).lower():
                    current_delay = min(delay + attempt, 10)
                    time.sleep(current_delay)
                    continue
                raise

        raise ApiException(f"Task {task_id} did not complete within timeout")