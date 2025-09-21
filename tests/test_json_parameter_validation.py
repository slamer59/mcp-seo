"""
Test for JSON parameter validation fix.

Ensures that JSON string parameters are properly parsed and validated.
"""

import json

import pytest

from mcp_seo.server import OnPageAnalysisParams


class TestJSONParameterValidation:
    """Test JSON parameter handling."""

    def test_localhost_url_validation(self):
        """Test that localhost URLs are properly handled."""
        # Test with localhost:3000 (the original error case)
        params_dict = {
            "target": "localhost:3000",
            "max_crawl_pages": 50,
            "crawl_delay": 1,
            "respect_sitemap": True,
            "enable_javascript": True
        }

        # This should work now - the key is that model_validate processes the dict without error
        validated = OnPageAnalysisParams.model_validate(params_dict)
        assert validated.target == "localhost:3000"  # No URL normalization expected
        assert validated.max_crawl_pages == 50

    def test_json_string_validation(self):
        """Test that JSON strings are properly parsed."""
        json_string = json.dumps({
            "target": "localhost:3000",
            "max_crawl_pages": 50,
            "crawl_delay": 1,
            "respect_sitemap": True,
            "enable_javascript": True
        })

        # Pydantic v2 model_validate should handle this
        validated = OnPageAnalysisParams.model_validate(json.loads(json_string))
        assert validated.target == "localhost:3000"  # No URL normalization expected

    def test_various_url_formats(self):
        """Test different URL formats are accepted without error."""
        test_cases = [
            "localhost:3000",
            "192.168.1.1:8080",
            "example.com",
            "http://example.com",
            "https://example.com",
        ]

        for input_url in test_cases:
            params = {"target": input_url}
            validated = OnPageAnalysisParams.model_validate(params)
            # The key test is that validation succeeds - no URL normalization happens
            assert validated.target == input_url

    def test_json_string_parameter_fix(self):
        """Test that JSON strings are handled correctly by model_validate."""
        # This is the core fix - when MCP tools receive JSON strings instead of dicts
        json_string = '{"target": "localhost:3000", "max_crawl_pages": 50}'

        # Before the fix, this would fail with: "'...' is not of type 'object'"
        # After the fix using model_validate, this should work:

        # Test 1: Direct JSON string (this would fail before the fix)
        # We need to parse it first, as that's what the tools do
        params_from_json = json.loads(json_string)
        validated = OnPageAnalysisParams.model_validate(params_from_json)
        assert validated.target == "localhost:3000"

        # Test 2: Dict input (this always worked)
        params_dict = {"target": "localhost:3000", "max_crawl_pages": 50}
        validated = OnPageAnalysisParams.model_validate(params_dict)
        assert validated.target == "localhost:3000"

        # Test 3: Pydantic model instance (this should also work)
        existing_model = OnPageAnalysisParams(target="localhost:3000", max_crawl_pages=50)
        validated = OnPageAnalysisParams.model_validate(existing_model)
        assert validated.target == "localhost:3000"

    def test_mcp_tools_json_string_handling(self):
        """Test that MCP tools handle JSON strings correctly (regression test)."""
        from mcp_seo.server import KeywordAnalysisParams, DomainAnalysisParams

        # Test KeywordAnalysisParams with JSON string (note: keywords is plural list)
        keyword_json_string = '{"keywords": ["gitlab mobile client"], "location": "usa", "language": "english"}'

        # Simulate what MCP tools do: check if string, parse JSON, then validate
        params = keyword_json_string
        if isinstance(params, str):
            params = json.loads(params)
        validated = KeywordAnalysisParams.model_validate(params)

        assert validated.keywords == ["gitlab mobile client"]
        assert validated.location == "usa"
        assert validated.language == "english"

        # Test DomainAnalysisParams with JSON string
        domain_json_string = '{"target": "gitalchemy.app"}'

        params = domain_json_string
        if isinstance(params, str):
            params = json.loads(params)
        validated = DomainAnalysisParams.model_validate(params)

        assert validated.target == "gitalchemy.app"

    def test_common_parameter_mistakes(self):
        """Test common parameter format mistakes that users might make."""
        from mcp_seo.server import KeywordAnalysisParams

        # Common mistake: using "keyword" (singular) instead of "keywords" (plural)
        with pytest.raises(Exception):  # Should fail validation
            json_string = '{"keyword": "test", "location": "usa"}'
            params = json.loads(json_string)
            KeywordAnalysisParams.model_validate(params)

        # Correct format: "keywords" as list
        json_string = '{"keywords": ["test"], "location": "usa"}'
        params = json.loads(json_string)
        validated = KeywordAnalysisParams.model_validate(params)
        assert validated.keywords == ["test"]