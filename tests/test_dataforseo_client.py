"""
Comprehensive tests for DataForSEO client implementation.

Tests cover:
- API request/response mocking for all endpoints
- Authentication and API key handling
- Error handling and retry logic
- Rate limiting scenarios
- Response parsing and data transformation
- Network failure scenarios
- Task management and polling functionality
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import base64
import time

from mcp_seo.dataforseo.client import DataForSEOClient, ApiException


class TestDataForSEOClientInitialization:
    """Test client initialization and authentication setup."""

    def test_initialization_with_credentials(self):
        """Test client initialization with provided credentials."""
        client = DataForSEOClient(username="test_user", password="test_pass")

        assert client.username == "test_user"
        assert client.password == "test_pass"
        assert client.base_url == "https://api.dataforseo.com/v3"
        assert isinstance(client.session, requests.Session)

        # Check authentication header
        expected_credentials = base64.b64encode(b"test_user:test_pass").decode("utf-8")
        assert client.session.headers["Authorization"] == f"Basic {expected_credentials}"
        assert client.session.headers["Content-Type"] == "application/json"

    @patch("mcp_seo.dataforseo.client.get_settings")
    def test_initialization_with_env_credentials(self, mock_get_settings):
        """Test client initialization with environment variables."""
        mock_settings = Mock()
        mock_settings.dataforseo_login = "env_user"
        mock_settings.dataforseo_password = "env_pass"
        mock_get_settings.return_value = mock_settings

        client = DataForSEOClient()

        assert client.username == "env_user"
        assert client.password == "env_pass"

    @patch("mcp_seo.dataforseo.client.get_settings")
    def test_initialization_missing_credentials(self, mock_get_settings):
        """Test client initialization fails without credentials."""
        mock_settings = Mock()
        mock_settings.dataforseo_login = None
        mock_settings.dataforseo_password = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="DataForSEO credentials required"):
            DataForSEOClient()

    @patch("mcp_seo.dataforseo.client.get_settings")
    def test_initialization_partial_credentials(self, mock_get_settings):
        """Test client initialization fails with partial credentials."""
        mock_settings = Mock()
        mock_settings.dataforseo_login = "user"
        mock_settings.dataforseo_password = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="DataForSEO credentials required"):
            DataForSEOClient()


class TestDataForSEOClientRequestHandling:
    """Test HTTP request handling and error scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_make_request_get_success(self, client):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status_code": 20000, "tasks": []}
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            result = client._make_request("/test", "GET", {"param": "value"})

        assert result == {"status_code": 20000, "tasks": []}
        mock_get.assert_called_once_with(
            "https://api.dataforseo.com/v3/test",
            params={"param": "value"}
        )

    def test_make_request_post_success(self, client):
        """Test successful POST request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status_code": 20000, "tasks": []}
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'post', return_value=mock_response) as mock_post:
            result = client._make_request("/test", "POST", {"data": "value"})

        assert result == {"status_code": 20000, "tasks": []}
        mock_post.assert_called_once_with(
            "https://api.dataforseo.com/v3/test",
            json={"data": "value"}
        )

    def test_make_request_api_error(self, client):
        """Test API-specific error handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status_code": 40000,
            "status_message": "Bad Request"
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="API Error: Bad Request"):
                client._make_request("/test")

    def test_make_request_http_error(self, client):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="Request failed: 404 Not Found"):
                client._make_request("/test")

    def test_make_request_connection_error(self, client):
        """Test connection error handling."""
        with patch.object(client.session, 'get', side_effect=requests.exceptions.ConnectionError("Connection failed")):
            with pytest.raises(ApiException, match="Request failed: Connection failed"):
                client._make_request("/test")

    def test_make_request_timeout_error(self, client):
        """Test timeout error handling."""
        with patch.object(client.session, 'get', side_effect=requests.exceptions.Timeout("Request timeout")):
            with pytest.raises(ApiException, match="Request failed: Request timeout"):
                client._make_request("/test")

    def test_make_request_json_decode_error(self, client):
        """Test JSON decode error handling."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="Invalid JSON response"):
                client._make_request("/test")

    def test_make_request_endpoint_formatting(self, client):
        """Test endpoint URL formatting."""
        mock_response = Mock()
        mock_response.json.return_value = {"status_code": 20000}
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            # Test with leading slash
            client._make_request("/test")
            # Test without leading slash
            client._make_request("test2")

        # Check both calls were made correctly
        assert mock_get.call_count == 2
        mock_get.assert_any_call("https://api.dataforseo.com/v3/test", params=None)
        mock_get.assert_any_call("https://api.dataforseo.com/v3/test2", params=None)


class TestDataForSEOClientAccountMethods:
    """Test account and metadata API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_account_info(self, client):
        """Test account info retrieval."""
        expected_response = {
            "status_code": 20000,
            "tasks": [{
                "id": "12345",
                "result": [{
                    "login": "test_user",
                    "type": "standard",
                    "money": {"remaining": 100.0}
                }]
            }]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_account_info()

        assert result == expected_response
        mock_request.assert_called_once_with("/appendix/user_data")

    def test_get_serp_locations(self, client):
        """Test SERP locations retrieval."""
        expected_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [
                    {"location_code": 2840, "location_name": "United States"},
                    {"location_code": 2826, "location_name": "United Kingdom"}
                ]
            }]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_serp_locations("US")

        assert result == expected_response
        mock_request.assert_called_once_with("/serp/google/locations", data={"country": "US"})

    def test_get_serp_locations_no_country(self, client):
        """Test SERP locations retrieval without country filter."""
        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_serp_locations()

        mock_request.assert_called_once_with("/serp/google/locations", data=None)

    def test_get_serp_languages(self, client):
        """Test SERP languages retrieval."""
        expected_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [
                    {"language_code": "en", "language_name": "English"},
                    {"language_code": "es", "language_name": "Spanish"}
                ]
            }]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_serp_languages()

        assert result == expected_response
        mock_request.assert_called_once_with("/serp/google/languages")


class TestDataForSEOClientOnPageMethods:
    """Test OnPage API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_create_onpage_task_default_params(self, client):
        """Test OnPage task creation with default parameters."""
        expected_data = [{
            "target": "https://example.com",
            "max_crawl_pages": 100,
            "start_url": "https://example.com",
            "respect_sitemap": True,
            "custom_sitemap": None,
            "crawl_delay": 1,
            "tag": "fastmcp-seo"
        }]
        expected_response = {
            "status_code": 20000,
            "tasks": [{"id": "task_123", "status_code": 20100}]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.create_onpage_task("https://example.com")

        assert result == expected_response
        mock_request.assert_called_once_with("/on_page/task_post", "POST", expected_data)

    def test_create_onpage_task_custom_params(self, client):
        """Test OnPage task creation with custom parameters."""
        expected_data = [{
            "target": "https://example.com",
            "max_crawl_pages": 50,
            "start_url": "https://example.com/start",
            "respect_sitemap": False,
            "custom_sitemap": "https://example.com/sitemap.xml",
            "crawl_delay": 2,
            "tag": "custom-tag"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.create_onpage_task(
                "https://example.com",
                max_crawl_pages=50,
                start_url="https://example.com/start",
                respect_sitemap=False,
                custom_sitemap="https://example.com/sitemap.xml",
                crawl_delay=2,
                tag="custom-tag"
            )

        mock_request.assert_called_once_with("/on_page/task_post", "POST", expected_data)

    def test_get_onpage_tasks(self, client):
        """Test OnPage tasks retrieval."""
        expected_response = {
            "status_code": 20000,
            "tasks": [{"id": "task_123", "status_code": 20000}]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_onpage_tasks()

        assert result == expected_response
        mock_request.assert_called_once_with("/on_page/tasks_ready")

    def test_get_onpage_summary(self, client):
        """Test OnPage task summary retrieval."""
        task_id = "task_123"
        expected_response = {
            "status_code": 20000,
            "tasks": [{"result": [{"crawl_progress": "finished"}]}]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_onpage_summary(task_id)

        assert result == expected_response
        mock_request.assert_called_once_with(f"/on_page/summary/{task_id}")

    def test_get_onpage_pages_default_params(self, client):
        """Test OnPage pages retrieval with default parameters."""
        task_id = "task_123"
        expected_data = [{
            "id": task_id,
            "limit": 100,
            "offset": 0,
            "filters": []
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_onpage_pages(task_id)

        mock_request.assert_called_once_with("/on_page/pages", "POST", expected_data)

    def test_get_onpage_pages_custom_params(self, client):
        """Test OnPage pages retrieval with custom parameters."""
        task_id = "task_123"
        expected_data = [{
            "id": task_id,
            "limit": 50,
            "offset": 10,
            "filters": [["status_code", "=", 200]]
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_onpage_pages(
                task_id,
                limit=50,
                offset=10,
                filters=[["status_code", "=", 200]]
            )

        mock_request.assert_called_once_with("/on_page/pages", "POST", expected_data)

    def test_get_onpage_resources(self, client):
        """Test OnPage resources retrieval."""
        task_id = "task_123"
        expected_data = [{
            "id": task_id,
            "limit": 100,
            "offset": 0
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_onpage_resources(task_id)

        mock_request.assert_called_once_with("/on_page/resources", "POST", expected_data)

    def test_get_onpage_duplicate_tags(self, client):
        """Test OnPage duplicate tags retrieval."""
        task_id = "task_123"
        expected_data = [{"id": task_id}]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_onpage_duplicate_tags(task_id)

        mock_request.assert_called_once_with("/on_page/duplicate_tags", "POST", expected_data)

    def test_get_onpage_lighthouse(self, client):
        """Test OnPage Lighthouse data retrieval."""
        task_id = "task_123"
        expected_data = [{"id": task_id}]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_onpage_lighthouse(task_id)

        mock_request.assert_called_once_with("/on_page/lighthouse", "POST", expected_data)


class TestDataForSEOClientSERPMethods:
    """Test SERP API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_serp_results_default_params(self, client):
        """Test SERP results with default parameters."""
        expected_data = [{
            "keyword": "test keyword",
            "location_code": 2840,
            "language_code": "en",
            "device": "desktop",
            "os": "windows",
            "depth": 100,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_serp_results("test keyword")

        mock_request.assert_called_once_with("/serp/google/organic/task_post", "POST", expected_data)

    def test_get_serp_results_custom_params(self, client):
        """Test SERP results with custom parameters."""
        expected_data = [{
            "keyword": "mobile keyword",
            "location_code": 2826,
            "language_code": "es",
            "device": "mobile",
            "os": "android",
            "depth": 50,
            "tag": "custom-tag"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_serp_results(
                "mobile keyword",
                location_code=2826,
                language_code="es",
                device="mobile",
                os="android",
                depth=50,
                tag="custom-tag"
            )

        mock_request.assert_called_once_with("/serp/google/organic/task_post", "POST", expected_data)

    def test_get_serp_tasks_ready(self, client):
        """Test SERP tasks ready retrieval."""
        expected_response = {
            "status_code": 20000,
            "tasks": [{"id": "serp_task_123", "status_code": 20000}]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_serp_tasks_ready()

        assert result == expected_response
        mock_request.assert_called_once_with("/serp/google/organic/tasks_ready")

    def test_get_serp_task_results(self, client):
        """Test SERP task results retrieval."""
        task_id = "serp_task_123"
        expected_response = {
            "status_code": 20000,
            "tasks": [{"result": [{"items": []}]}]
        }

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_serp_task_results(task_id)

        assert result == expected_response
        mock_request.assert_called_once_with(f"/serp/google/organic/task_get/{task_id}")


class TestDataForSEOClientKeywordMethods:
    """Test Keywords Data API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_keyword_data(self, client):
        """Test keyword data retrieval."""
        keywords = ["seo tools", "keyword research"]
        expected_data = [{
            "keywords": keywords,
            "location_code": 2840,
            "language_code": "en",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_keyword_data(keywords)

        mock_request.assert_called_once_with("/keywords_data/google_ads/search_volume/task_post", "POST", expected_data)

    def test_get_keyword_data_custom_location(self, client):
        """Test keyword data with custom location and language."""
        keywords = ["herramientas seo"]
        expected_data = [{
            "keywords": keywords,
            "location_code": 2724,  # Spain
            "language_code": "es",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_keyword_data(keywords, location_code=2724, language_code="es")

        mock_request.assert_called_once_with("/keywords_data/google_ads/search_volume/task_post", "POST", expected_data)

    def test_get_keyword_suggestions_default_params(self, client):
        """Test keyword suggestions with default parameters."""
        expected_data = [{
            "keyword": "seo",
            "location_code": 2840,
            "language_code": "en",
            "limit": 100,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_keyword_suggestions("seo")

        mock_request.assert_called_once_with("/keywords_data/google_ads/keywords_for_keywords/task_post", "POST", expected_data)

    def test_get_keyword_suggestions_custom_params(self, client):
        """Test keyword suggestions with custom parameters."""
        expected_data = [{
            "keyword": "digital marketing",
            "location_code": 2826,
            "language_code": "en",
            "limit": 50,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_keyword_suggestions(
                "digital marketing",
                location_code=2826,
                language_code="en",
                limit=50
            )

        mock_request.assert_called_once_with("/keywords_data/google_ads/keywords_for_keywords/task_post", "POST", expected_data)


class TestDataForSEOClientDataForSEOLabsMethods:
    """Test DataForSEO Labs API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_domain_rank_overview(self, client):
        """Test domain rank overview retrieval."""
        expected_data = [{
            "target": "example.com",
            "location_code": 2840,
            "language_code": "en",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_domain_rank_overview("example.com")

        mock_request.assert_called_once_with("/dataforseo_labs/google/domain_rank_overview/task_post", "POST", expected_data)

    def test_get_competitor_domains_default_params(self, client):
        """Test competitor domains with default parameters."""
        expected_data = [{
            "target": "example.com",
            "location_code": 2840,
            "language_code": "en",
            "limit": 50,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_competitor_domains("example.com")

        mock_request.assert_called_once_with("/dataforseo_labs/google/competitors_domain/task_post", "POST", expected_data)

    def test_get_competitor_domains_custom_limit(self, client):
        """Test competitor domains with custom limit."""
        expected_data = [{
            "target": "example.com",
            "location_code": 2840,
            "language_code": "en",
            "limit": 25,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_competitor_domains("example.com", limit=25)

        mock_request.assert_called_once_with("/dataforseo_labs/google/competitors_domain/task_post", "POST", expected_data)

    def test_get_ranked_keywords_default_params(self, client):
        """Test ranked keywords with default parameters."""
        expected_data = [{
            "target": "example.com",
            "location_code": 2840,
            "language_code": "en",
            "limit": 100,
            "offset": 0,
            "filters": [],
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_ranked_keywords("example.com")

        mock_request.assert_called_once_with("/dataforseo_labs/google/ranked_keywords/task_post", "POST", expected_data)

    def test_get_ranked_keywords_with_filters(self, client):
        """Test ranked keywords with filters."""
        filters = [["keyword_data.keyword_info.search_volume", ">", 1000]]
        expected_data = [{
            "target": "example.com",
            "location_code": 2840,
            "language_code": "en",
            "limit": 50,
            "offset": 20,
            "filters": filters,
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_ranked_keywords(
                "example.com",
                limit=50,
                offset=20,
                filters=filters
            )

        mock_request.assert_called_once_with("/dataforseo_labs/google/ranked_keywords/task_post", "POST", expected_data)


class TestDataForSEOClientDomainAnalyticsMethods:
    """Test Domain Analytics API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_domain_technologies(self, client):
        """Test domain technologies retrieval."""
        expected_data = [{
            "target": "example.com",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_domain_technologies("example.com")

        mock_request.assert_called_once_with("/domain_analytics/technologies/domain_technologies/task_post", "POST", expected_data)

    def test_get_domain_whois(self, client):
        """Test domain WHOIS retrieval."""
        expected_data = [{
            "target": "example.com",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_domain_whois("example.com")

        mock_request.assert_called_once_with("/domain_analytics/whois/overview/task_post", "POST", expected_data)


class TestDataForSEOClientBacklinksMethods:
    """Test Backlinks API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_backlinks_summary_default_mode(self, client):
        """Test backlinks summary with default mode."""
        expected_data = [{
            "target": "example.com",
            "mode": "domain",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_backlinks_summary("example.com")

        mock_request.assert_called_once_with("/backlinks/summary/task_post", "POST", expected_data)

    def test_get_backlinks_summary_custom_mode(self, client):
        """Test backlinks summary with custom mode."""
        expected_data = [{
            "target": "example.com/page",
            "mode": "exact",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_backlinks_summary("example.com/page", mode="exact")

        mock_request.assert_called_once_with("/backlinks/summary/task_post", "POST", expected_data)

    def test_get_backlinks_history(self, client):
        """Test backlinks history retrieval."""
        expected_data = [{
            "target": "example.com",
            "mode": "domain",
            "date_from": "2023-01-01",
            "date_to": "2023-12-31",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_backlinks_history(
                "example.com",
                date_from="2023-01-01",
                date_to="2023-12-31"
            )

        mock_request.assert_called_once_with("/backlinks/history/task_post", "POST", expected_data)

    def test_get_referring_domains(self, client):
        """Test referring domains retrieval."""
        expected_data = [{
            "target": "example.com",
            "mode": "domain",
            "limit": 100,
            "offset": 0,
            "filters": [],
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_referring_domains("example.com")

        mock_request.assert_called_once_with("/backlinks/referring_domains/task_post", "POST", expected_data)


class TestDataForSEOClientContentAnalysisMethods:
    """Test Content Analysis API methods."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_analyze_content_sentiment(self, client):
        """Test content sentiment analysis."""
        text = "This is a great product that I really love!"
        expected_data = [{
            "text": text,
            "language": "en",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.analyze_content_sentiment(text)

        mock_request.assert_called_once_with("/content_analysis/sentiment_analysis/task_post", "POST", expected_data)

    def test_analyze_content_sentiment_custom_language(self, client):
        """Test content sentiment analysis with custom language."""
        text = "Este es un producto excelente que me encanta!"
        expected_data = [{
            "text": text,
            "language": "es",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.analyze_content_sentiment(text, language="es")

        mock_request.assert_called_once_with("/content_analysis/sentiment_analysis/task_post", "POST", expected_data)

    def test_get_content_summary_default_length(self, client):
        """Test content summary with default length."""
        text = "This is a long article about SEO optimization techniques..."
        expected_data = [{
            "text": text,
            "summary_length": "short",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_content_summary(text)

        mock_request.assert_called_once_with("/content_analysis/summary/task_post", "POST", expected_data)

    def test_get_content_summary_custom_length(self, client):
        """Test content summary with custom length."""
        text = "This is a long article about SEO optimization techniques..."
        expected_data = [{
            "text": text,
            "summary_length": "long",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_content_summary(text, summary_length="long")

        mock_request.assert_called_once_with("/content_analysis/summary/task_post", "POST", expected_data)


class TestDataForSEOClientUtilityMethods:
    """Test utility methods for task management."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_get_task_results_serp(self, client):
        """Test generic task results for SERP."""
        task_id = "task_123"
        expected_response = {"status_code": 20000, "tasks": []}

        with patch.object(client, '_make_request', return_value=expected_response) as mock_request:
            result = client.get_task_results(task_id, "serp")

        assert result == expected_response
        mock_request.assert_called_once_with(f"/serp/google/organic/task_get/{task_id}")

    def test_get_task_results_keywords(self, client):
        """Test generic task results for keywords."""
        task_id = "task_123"

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_task_results(task_id, "keywords")

        mock_request.assert_called_once_with(f"/keywords_data/google_ads/search_volume/task_get/{task_id}")

    def test_get_task_results_onpage(self, client):
        """Test generic task results for onpage."""
        task_id = "task_123"

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_task_results(task_id, "onpage")

        mock_request.assert_called_once_with(f"/on_page/summary/{task_id}")

    def test_get_task_results_invalid_endpoint(self, client):
        """Test generic task results with invalid endpoint type."""
        with pytest.raises(ValueError, match="Unknown endpoint type: invalid"):
            client.get_task_results("task_123", "invalid")

    @patch('time.sleep')
    def test_wait_for_task_completion_success(self, mock_sleep, client):
        """Test successful task completion waiting."""
        task_id = "task_123"
        expected_response = {
            "tasks": [{
                "result": [{"keyword": "test", "results": []}]
            }]
        }

        with patch.object(client, 'get_task_results', return_value=expected_response) as mock_get:
            result = client.wait_for_task_completion(task_id, "serp")

        assert result == expected_response
        mock_get.assert_called_once_with(task_id, "serp")
        mock_sleep.assert_not_called()  # Should complete on first try

    @patch('time.sleep')
    def test_wait_for_task_completion_multiple_attempts(self, mock_sleep, client):
        """Test task completion waiting with multiple attempts."""
        task_id = "task_123"
        incomplete_response = {"tasks": [{}]}  # No result
        complete_response = {
            "tasks": [{
                "result": [{"keyword": "test", "results": []}]
            }]
        }

        with patch.object(client, 'get_task_results', side_effect=[incomplete_response, complete_response]) as mock_get:
            result = client.wait_for_task_completion(task_id, "serp", delay=1)

        assert result == complete_response
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    def test_wait_for_task_completion_timeout(self, mock_sleep, client):
        """Test task completion timeout."""
        task_id = "task_123"
        incomplete_response = {"tasks": [{}]}  # No result

        with patch.object(client, 'get_task_results', return_value=incomplete_response) as mock_get:
            with pytest.raises(ApiException, match="did not complete within"):
                client.wait_for_task_completion(task_id, "serp", max_attempts=2, delay=1)

        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 2

    @patch('time.sleep')
    def test_wait_for_task_completion_not_found_error(self, mock_sleep, client):
        """Test task completion with 'not found' error handling."""
        task_id = "task_123"
        complete_response = {
            "tasks": [{
                "result": [{"keyword": "test", "results": []}]
            }]
        }

        with patch.object(client, 'get_task_results', side_effect=[
            ApiException("Task not found"),
            complete_response
        ]) as mock_get:
            result = client.wait_for_task_completion(task_id, "serp", delay=1)

        assert result == complete_response
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch('time.sleep')
    def test_wait_for_task_completion_other_error(self, mock_sleep, client):
        """Test task completion with other API errors."""
        task_id = "task_123"

        with patch.object(client, 'get_task_results', side_effect=ApiException("Authentication failed")) as mock_get:
            with pytest.raises(ApiException, match="Authentication failed"):
                client.wait_for_task_completion(task_id, "serp")

        mock_get.assert_called_once()
        mock_sleep.assert_not_called()


class TestDataForSEOClientRateLimitingAndRetries:
    """Test rate limiting and retry scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_rate_limit_error_handling(self, client):
        """Test rate limit error (429) handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="Request failed: 429 Too Many Requests"):
                client._make_request("/test")

    def test_server_error_handling(self, client):
        """Test server error (5xx) handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="Request failed: 500 Internal Server Error"):
                client._make_request("/test")

    def test_network_unreachable_error(self, client):
        """Test network unreachable scenarios."""
        with patch.object(client.session, 'get', side_effect=requests.exceptions.ConnectionError("Network is unreachable")):
            with pytest.raises(ApiException, match="Request failed: Network is unreachable"):
                client._make_request("/test")

    def test_dns_resolution_error(self, client):
        """Test DNS resolution failures."""
        with patch.object(client.session, 'get', side_effect=requests.exceptions.ConnectionError("Name or service not known")):
            with pytest.raises(ApiException, match="Request failed: Name or service not known"):
                client._make_request("/test")


class TestDataForSEOClientResponseParsing:
    """Test response parsing and data transformation."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_successful_response_parsing(self, client):
        """Test parsing of successful API response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "version": "0.1.20231117",
            "status_code": 20000,
            "status_message": "Ok.",
            "time": "1.2345 sec.",
            "cost": 0.01,
            "tasks_count": 1,
            "tasks_error": 0,
            "tasks": [{
                "id": "11111111-1111-1111-1111-111111111111",
                "status_code": 20000,
                "status_message": "Ok.",
                "time": "1.2345 sec.",
                "cost": 0.01,
                "result_count": 1,
                "path": ["v3", "serp", "google", "organic", "task_post"],
                "data": {
                    "se_type": "google",
                    "api": "serp",
                    "function": "task_post"
                },
                "result": [{
                    "keyword": "test keyword",
                    "type": "organic",
                    "se_domain": "google.com",
                    "location_code": 2840,
                    "language_code": "en",
                    "items": []
                }]
            }]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client._make_request("/test")

        assert result["status_code"] == 20000
        assert result["status_message"] == "Ok."
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["result"][0]["keyword"] == "test keyword"

    def test_empty_response_parsing(self, client):
        """Test parsing of empty but valid response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status_code": 20000,
            "status_message": "Ok.",
            "tasks": []
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client._make_request("/test")

        assert result["status_code"] == 20000
        assert result["tasks"] == []

    def test_malformed_json_response(self, client):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            with pytest.raises(ApiException, match="Invalid JSON response"):
                client._make_request("/test")

    def test_partial_response_data(self, client):
        """Test handling of partial response data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status_code": 20000,
            "tasks": [{
                "id": "task_123",
                "status_code": 20100,  # Task created but not completed
                "status_message": "Task Created.",
                "result": None
            }]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client._make_request("/test")

        assert result["tasks"][0]["status_code"] == 20100
        assert result["tasks"][0]["result"] is None


class TestDataForSEOClientRealWorldScenarios:
    """Test real-world usage scenarios and edge cases."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return DataForSEOClient(username="test_user", password="test_pass")

    def test_full_onpage_analysis_workflow(self, client):
        """Test complete OnPage analysis workflow."""
        # Step 1: Create task
        create_response = {
            "status_code": 20000,
            "tasks": [{"id": "onpage_task_123", "status_code": 20100}]
        }

        # Step 2: Check task completion
        summary_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [{
                    "crawl_progress": "finished",
                    "pages_in_queue": 0,
                    "pages_crawled": 25,
                    "total_pages": 25
                }]
            }]
        }

        with patch.object(client, '_make_request', side_effect=[create_response, summary_response]) as mock_request:
            # Create task
            create_result = client.create_onpage_task("https://example.com")
            task_id = create_result["tasks"][0]["id"]

            # Get summary
            summary_result = client.get_onpage_summary(task_id)

        assert create_result["tasks"][0]["status_code"] == 20100
        assert summary_result["tasks"][0]["result"][0]["crawl_progress"] == "finished"

    def test_keyword_research_workflow(self, client):
        """Test keyword research workflow."""
        # Get keyword data
        keyword_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [{
                    "keyword": "seo tools",
                    "search_volume": 12100,
                    "competition": 0.8
                }]
            }]
        }

        # Get keyword suggestions
        suggestions_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [{
                    "keywords": [
                        {"keyword": "best seo tools", "search_volume": 8100},
                        {"keyword": "free seo tools", "search_volume": 6600}
                    ]
                }]
            }]
        }

        with patch.object(client, '_make_request', side_effect=[keyword_response, suggestions_response]) as mock_request:
            keyword_data = client.get_keyword_data(["seo tools"])
            suggestions = client.get_keyword_suggestions("seo tools")

        assert keyword_data["tasks"][0]["result"][0]["keyword"] == "seo tools"
        assert len(suggestions["tasks"][0]["result"][0]["keywords"]) == 2

    def test_competitor_analysis_workflow(self, client):
        """Test competitor analysis workflow."""
        # Get domain rank overview
        rank_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [{
                    "target": "example.com",
                    "rank": 1234567,
                    "etv": 12345.67,
                    "impressions_etv": 98765.43
                }]
            }]
        }

        # Get competitor domains
        competitors_response = {
            "status_code": 20000,
            "tasks": [{
                "result": [{
                    "competitors": [
                        {"domain": "competitor1.com", "intersections": 567},
                        {"domain": "competitor2.com", "intersections": 432}
                    ]
                }]
            }]
        }

        with patch.object(client, '_make_request', side_effect=[rank_response, competitors_response]) as mock_request:
            rank_data = client.get_domain_rank_overview("example.com")
            competitors = client.get_competitor_domains("example.com")

        assert rank_data["tasks"][0]["result"][0]["target"] == "example.com"
        assert len(competitors["tasks"][0]["result"][0]["competitors"]) == 2

    def test_error_recovery_scenario(self, client):
        """Test error recovery in real workflow."""
        # First request fails
        error_response = Mock()
        error_response.raise_for_status.side_effect = requests.exceptions.HTTPError("503 Service Unavailable")

        # Second request succeeds
        success_response = Mock()
        success_response.json.return_value = {"status_code": 20000, "tasks": []}
        success_response.raise_for_status = Mock()

        with patch.object(client.session, 'get', side_effect=[error_response, success_response]):
            # First call should fail
            with pytest.raises(ApiException):
                client.get_account_info()

            # Second call should succeed
            result = client.get_account_info()
            assert result["status_code"] == 20000

    def test_large_keyword_batch_processing(self, client):
        """Test processing large batches of keywords."""
        large_keyword_list = [f"keyword_{i}" for i in range(1000)]

        expected_data = [{
            "keywords": large_keyword_list,
            "location_code": 2840,
            "language_code": "en",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.get_keyword_data(large_keyword_list)

        mock_request.assert_called_once_with("/keywords_data/google_ads/search_volume/task_post", "POST", expected_data)

    def test_unicode_content_handling(self, client):
        """Test handling of Unicode content in requests."""
        unicode_text = "SEO工具测试 émojis 🚀 русский"

        expected_data = [{
            "text": unicode_text,
            "language": "en",
            "tag": "fastmcp-seo"
        }]

        with patch.object(client, '_make_request', return_value={"status_code": 20000}) as mock_request:
            client.analyze_content_sentiment(unicode_text)

        mock_request.assert_called_once_with("/content_analysis/sentiment_analysis/task_post", "POST", expected_data)


@pytest.mark.integration
class TestDataForSEOClientIntegration:
    """Integration tests that can be run with real API credentials (if available)."""

    def test_client_creation_without_real_credentials(self):
        """Test that client creation works without throwing if no real credentials."""
        # This test ensures our mocking doesn't break actual usage
        try:
            client = DataForSEOClient(username="fake_user", password="fake_pass")
            assert client.username == "fake_user"
            assert client.password == "fake_pass"
        except Exception as e:
            pytest.fail(f"Client creation should not fail with fake credentials: {e}")

    @pytest.mark.skip(reason="Requires real API credentials and network access")
    def test_real_api_account_info(self):
        """Test real API call (skipped by default)."""
        # This test would be enabled only when running integration tests
        # with real credentials in environment variables
        client = DataForSEOClient()
        result = client.get_account_info()
        assert result["status_code"] == 20000

    @pytest.mark.skip(reason="Requires real API credentials and network access")
    def test_real_api_locations(self):
        """Test real API locations call (skipped by default)."""
        client = DataForSEOClient()
        result = client.get_serp_locations()
        assert result["status_code"] == 20000
        assert "tasks" in result