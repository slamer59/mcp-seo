"""
Comprehensive tests for the configuration system.
"""

import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError

from mcp_seo.config.settings import (
    Settings,
    get_settings,
    get_location_code,
    get_language_code,
    LOCATION_MAPPINGS,
    LANGUAGE_MAPPINGS,
    DEVICE_MAPPINGS
)


class TestSettings:
    """Test the Settings configuration class."""

    def test_default_values(self):
        """Test that default values are properly set."""
        settings = Settings()

        # DataForSEO credentials should be None by default
        assert settings.dataforseo_login is None
        assert settings.dataforseo_password is None

        # Server settings defaults
        assert settings.server_name == "mcp-seo"
        assert settings.server_version == "1.0.0"

        # API settings defaults
        assert settings.api_timeout == 30
        assert settings.max_retries == 3
        assert settings.retry_delay == 1

        # SEO analysis defaults
        assert settings.default_location_code == 2840  # USA
        assert settings.default_language_code == "en"
        assert settings.default_device == "desktop"

        # OnPage analysis defaults
        assert settings.onpage_max_crawl_pages == 100
        assert settings.onpage_crawl_delay == 1

        # Keyword analysis defaults
        assert settings.keyword_limit == 100
        assert settings.competitor_limit == 50

        # Task management defaults
        assert settings.task_completion_timeout == 300
        assert settings.task_check_interval == 10

        # Logging defaults
        assert settings.log_level == "INFO"
        assert settings.enable_request_logging is False

    def test_settings_from_environment_variables(self, monkeypatch):
        """Test loading settings from environment variables."""
        # Set environment variables
        monkeypatch.setenv("DATAFORSEO_LOGIN", "test_login")
        monkeypatch.setenv("DATAFORSEO_PASSWORD", "test_password")
        monkeypatch.setenv("SERVER_NAME", "custom-seo-server")
        monkeypatch.setenv("API_TIMEOUT", "60")
        monkeypatch.setenv("DEFAULT_LOCATION_CODE", "2826")  # UK
        monkeypatch.setenv("DEFAULT_LANGUAGE_CODE", "fr")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENABLE_REQUEST_LOGGING", "true")

        settings = Settings()

        assert settings.dataforseo_login == "test_login"
        assert settings.dataforseo_password == "test_password"
        assert settings.server_name == "custom-seo-server"
        assert settings.api_timeout == 60
        assert settings.default_location_code == 2826
        assert settings.default_language_code == "fr"
        assert settings.log_level == "DEBUG"
        assert settings.enable_request_logging is True

    def test_case_insensitive_environment_variables(self, monkeypatch):
        """Test that environment variables are case insensitive."""
        monkeypatch.setenv("dataforseo_login", "lowercase_login")
        monkeypatch.setenv("DATAFORSEO_PASSWORD", "uppercase_password")

        settings = Settings()

        # Both should work regardless of case
        assert settings.dataforseo_login == "lowercase_login"
        assert settings.dataforseo_password == "uppercase_password"

    def test_invalid_integer_values(self, monkeypatch):
        """Test validation of integer fields with invalid values."""
        monkeypatch.setenv("API_TIMEOUT", "not_a_number")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "api_timeout" in str(exc_info.value)

    def test_invalid_boolean_values(self, monkeypatch):
        """Test validation of boolean fields with invalid values."""
        monkeypatch.setenv("ENABLE_REQUEST_LOGGING", "maybe")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "enable_request_logging" in str(exc_info.value)

    def test_negative_integer_values(self, monkeypatch):
        """Test handling of negative integer values."""
        monkeypatch.setenv("API_TIMEOUT", "-10")
        monkeypatch.setenv("MAX_RETRIES", "-5")

        # Pydantic should accept negative integers, but they might not make sense
        settings = Settings()
        assert settings.api_timeout == -10
        assert settings.max_retries == -5

    def test_zero_values(self, monkeypatch):
        """Test handling of zero values for integer fields."""
        monkeypatch.setenv("API_TIMEOUT", "0")
        monkeypatch.setenv("RETRY_DELAY", "0")

        settings = Settings()
        assert settings.api_timeout == 0
        assert settings.retry_delay == 0

    def test_boolean_string_variations(self, monkeypatch):
        """Test various string representations of boolean values."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("ENABLE_REQUEST_LOGGING", env_value)
            settings = Settings()
            assert settings.enable_request_logging == expected, f"Failed for {env_value}"

    def test_optional_fields_remain_none(self):
        """Test that optional fields remain None when not set."""
        settings = Settings()

        assert settings.dataforseo_login is None
        assert settings.dataforseo_password is None

    def test_settings_with_custom_defaults(self, monkeypatch):
        """Test settings with all custom values to ensure no defaults interfere."""
        env_vars = {
            "DATAFORSEO_LOGIN": "custom_login",
            "DATAFORSEO_PASSWORD": "custom_password",
            "SERVER_NAME": "custom-server",
            "SERVER_VERSION": "2.0.0",
            "API_TIMEOUT": "45",
            "MAX_RETRIES": "5",
            "RETRY_DELAY": "2",
            "DEFAULT_LOCATION_CODE": "2276",  # Germany
            "DEFAULT_LANGUAGE_CODE": "de",
            "DEFAULT_DEVICE": "mobile",
            "ONPAGE_MAX_CRAWL_PAGES": "200",
            "ONPAGE_CRAWL_DELAY": "2",
            "KEYWORD_LIMIT": "150",
            "COMPETITOR_LIMIT": "75",
            "TASK_COMPLETION_TIMEOUT": "600",
            "TASK_CHECK_INTERVAL": "15",
            "LOG_LEVEL": "WARNING",
            "ENABLE_REQUEST_LOGGING": "true"
        }

        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        settings = Settings()

        assert settings.dataforseo_login == "custom_login"
        assert settings.dataforseo_password == "custom_password"
        assert settings.server_name == "custom-server"
        assert settings.server_version == "2.0.0"
        assert settings.api_timeout == 45
        assert settings.max_retries == 5
        assert settings.retry_delay == 2
        assert settings.default_location_code == 2276
        assert settings.default_language_code == "de"
        assert settings.default_device == "mobile"
        assert settings.onpage_max_crawl_pages == 200
        assert settings.onpage_crawl_delay == 2
        assert settings.keyword_limit == 150
        assert settings.competitor_limit == 75
        assert settings.task_completion_timeout == 600
        assert settings.task_check_interval == 15
        assert settings.log_level == "WARNING"
        assert settings.enable_request_logging is True


class TestGetSettings:
    """Test the get_settings() function caching behavior."""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings() returns the same instance (caching)."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_caching_with_environment_changes(self, monkeypatch):
        """Test that get_settings() caching behavior with environment changes."""
        # First call
        settings1 = get_settings()
        initial_login = settings1.dataforseo_login

        # Change environment variable
        monkeypatch.setenv("DATAFORSEO_LOGIN", "new_login")

        # Second call should return cached instance
        settings2 = get_settings()
        assert settings2 is settings1
        assert settings2.dataforseo_login == initial_login  # Should not reflect the change

    def test_get_settings_cache_clear(self, monkeypatch):
        """Test clearing the cache to pick up new environment variables."""
        # First call
        settings1 = get_settings()
        initial_login = settings1.dataforseo_login

        # Change environment variable
        monkeypatch.setenv("DATAFORSEO_LOGIN", "new_login")

        # Clear cache
        get_settings.cache_clear()

        # New call should pick up the environment change
        settings2 = get_settings()
        assert settings2 is not settings1
        assert settings2.dataforseo_login == "new_login"


class TestLocationCodeMapping:
    """Test location code mapping functionality."""

    def test_get_location_code_valid_mappings(self):
        """Test get_location_code with valid location names."""
        test_cases = [
            ("usa", 2840),
            ("USA", 2840),
            ("uk", 2826),
            ("UK", 2826),
            ("canada", 2124),
            ("germany", 2276),
            ("france", 2250),
            ("japan", 2392),
        ]

        for location, expected_code in test_cases:
            assert get_location_code(location) == expected_code

    def test_get_location_code_case_insensitive(self):
        """Test that location code mapping is case insensitive."""
        assert get_location_code("USA") == 2840
        assert get_location_code("usa") == 2840
        assert get_location_code("Usa") == 2840
        assert get_location_code("uSa") == 2840

    def test_get_location_code_with_spaces(self):
        """Test location code mapping with spaces converted to underscores."""
        assert get_location_code("south korea") == 2410
        assert get_location_code("South Korea") == 2410
        assert get_location_code("SOUTH KOREA") == 2410

    def test_get_location_code_invalid_location(self):
        """Test get_location_code with invalid location returns default."""
        invalid_locations = [
            "invalid_country",
            "mars",
            "",
            "123",
            "unknown location"
        ]

        for location in invalid_locations:
            assert get_location_code(location) == 2840  # Default USA

    def test_get_location_code_all_mappings(self):
        """Test all defined location mappings."""
        for location_name, expected_code in LOCATION_MAPPINGS.items():
            assert get_location_code(location_name) == expected_code
            # Test uppercase version
            assert get_location_code(location_name.upper()) == expected_code


class TestLanguageCodeMapping:
    """Test language code mapping functionality."""

    def test_get_language_code_valid_mappings(self):
        """Test get_language_code with valid language names."""
        test_cases = [
            ("english", "en"),
            ("Spanish", "es"),
            ("FRENCH", "fr"),
            ("german", "de"),
            ("italian", "it"),
            ("portuguese", "pt"),
            ("japanese", "ja"),
            ("korean", "ko"),
            ("chinese", "zh"),
            ("russian", "ru"),
        ]

        for language, expected_code in test_cases:
            assert get_language_code(language) == expected_code

    def test_get_language_code_case_insensitive(self):
        """Test that language code mapping is case insensitive."""
        assert get_language_code("ENGLISH") == "en"
        assert get_language_code("english") == "en"
        assert get_language_code("English") == "en"
        assert get_language_code("eNgLiSh") == "en"

    def test_get_language_code_invalid_language(self):
        """Test get_language_code with invalid language returns default."""
        invalid_languages = [
            "invalid_language",
            "klingon",
            "",
            "123",
            "unknown language"
        ]

        for language in invalid_languages:
            assert get_language_code(language) == "en"  # Default English

    def test_get_language_code_all_mappings(self):
        """Test all defined language mappings."""
        for language_name, expected_code in LANGUAGE_MAPPINGS.items():
            assert get_language_code(language_name) == expected_code
            # Test uppercase version
            assert get_language_code(language_name.upper()) == expected_code


class TestMappingConstants:
    """Test the mapping constants are properly defined."""

    def test_location_mappings_structure(self):
        """Test that LOCATION_MAPPINGS has the expected structure."""
        assert isinstance(LOCATION_MAPPINGS, dict)
        assert len(LOCATION_MAPPINGS) > 0

        for key, value in LOCATION_MAPPINGS.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert value > 0  # Location codes should be positive

    def test_language_mappings_structure(self):
        """Test that LANGUAGE_MAPPINGS has the expected structure."""
        assert isinstance(LANGUAGE_MAPPINGS, dict)
        assert len(LANGUAGE_MAPPINGS) > 0

        for key, value in LANGUAGE_MAPPINGS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) == 2  # Language codes should be 2 characters

    def test_device_mappings_structure(self):
        """Test that DEVICE_MAPPINGS has the expected structure."""
        assert isinstance(DEVICE_MAPPINGS, dict)
        assert len(DEVICE_MAPPINGS) > 0

        expected_devices = {"desktop", "mobile", "tablet"}
        assert set(DEVICE_MAPPINGS.keys()) == expected_devices
        assert set(DEVICE_MAPPINGS.values()) == expected_devices

    def test_location_mappings_contains_defaults(self):
        """Test that location mappings contain expected default countries."""
        expected_countries = {
            "usa", "uk", "canada", "australia", "germany",
            "france", "spain", "italy", "japan", "brazil"
        }

        for country in expected_countries:
            assert country in LOCATION_MAPPINGS

    def test_language_mappings_contains_defaults(self):
        """Test that language mappings contain expected default languages."""
        expected_languages = {
            "english", "spanish", "french", "german", "italian",
            "portuguese", "japanese", "korean", "chinese", "russian"
        }

        for language in expected_languages:
            assert language in LANGUAGE_MAPPINGS


class TestSettingsValidation:
    """Test settings validation scenarios."""

    def test_settings_validation_with_missing_credentials(self):
        """Test that settings work without credentials (optional fields)."""
        settings = Settings()

        # These should be None and not raise validation errors
        assert settings.dataforseo_login is None
        assert settings.dataforseo_password is None

    def test_settings_validation_with_empty_string_credentials(self, monkeypatch):
        """Test settings with empty string credentials."""
        monkeypatch.setenv("DATAFORSEO_LOGIN", "")
        monkeypatch.setenv("DATAFORSEO_PASSWORD", "")

        settings = Settings()

        # Empty strings should be preserved
        assert settings.dataforseo_login == ""
        assert settings.dataforseo_password == ""

    def test_settings_validation_with_whitespace_strings(self, monkeypatch):
        """Test settings with whitespace-only strings."""
        monkeypatch.setenv("DATAFORSEO_LOGIN", "   ")
        monkeypatch.setenv("SERVER_NAME", "\t\n")

        settings = Settings()

        # Whitespace should be preserved (Pydantic doesn't strip by default)
        assert settings.dataforseo_login == "   "
        assert settings.server_name == "\t\n"

    def test_settings_model_config(self):
        """Test that model configuration is properly set."""
        settings = Settings()

        # Check that model_config is accessible
        config = settings.model_config
        assert config["env_file"] == ".env"
        assert config["env_file_encoding"] == "utf-8"
        assert config["case_sensitive"] is False


class TestEnvironmentVariablePrecedence:
    """Test environment variable precedence and priority."""

    def test_environment_over_defaults(self, monkeypatch):
        """Test that environment variables take precedence over defaults."""
        # Set environment variable that overrides default
        monkeypatch.setenv("SERVER_NAME", "env-server")

        settings = Settings()

        # Should use environment value, not default
        assert settings.server_name == "env-server"
        assert settings.server_name != "mcp-seo"  # Not the default

    def test_explicit_none_in_environment(self, monkeypatch):
        """Test handling of explicit 'None' string in environment."""
        monkeypatch.setenv("DATAFORSEO_LOGIN", "None")

        settings = Settings()

        # Should treat "None" as a string, not None
        assert settings.dataforseo_login == "None"
        assert settings.dataforseo_login is not None

    def test_numeric_strings_in_environment(self, monkeypatch):
        """Test that numeric strings are properly converted."""
        monkeypatch.setenv("API_TIMEOUT", "120")
        monkeypatch.setenv("DEFAULT_LOCATION_CODE", "2276")

        settings = Settings()

        assert settings.api_timeout == 120
        assert isinstance(settings.api_timeout, int)
        assert settings.default_location_code == 2276
        assert isinstance(settings.default_location_code, int)


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_invalid_integer_conversion_error_message(self, monkeypatch):
        """Test that invalid integer conversion provides clear error message."""
        monkeypatch.setenv("API_TIMEOUT", "not_an_integer")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_message = str(exc_info.value)
        assert "api_timeout" in error_message
        assert "Input should be a valid integer" in error_message

    def test_multiple_validation_errors(self, monkeypatch):
        """Test handling of multiple validation errors at once."""
        monkeypatch.setenv("API_TIMEOUT", "invalid")
        monkeypatch.setenv("MAX_RETRIES", "also_invalid")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_message = str(exc_info.value)
        # Should contain both field errors
        assert "api_timeout" in error_message
        assert "max_retries" in error_message

    def test_extremely_large_integer_values(self, monkeypatch):
        """Test handling of extremely large integer values."""
        monkeypatch.setenv("API_TIMEOUT", "999999999999999999999")

        # Pydantic should handle this gracefully
        settings = Settings()
        assert settings.api_timeout == 999999999999999999999

    def test_float_values_for_integer_fields(self, monkeypatch):
        """Test that float values in string format raise validation errors for integer fields."""
        monkeypatch.setenv("API_TIMEOUT", "30.7")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_message = str(exc_info.value)
        assert "api_timeout" in error_message
        assert "Input should be a valid integer" in error_message


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_environment_variable(self, monkeypatch):
        """Test behavior with empty environment variables."""
        monkeypatch.setenv("SERVER_NAME", "")

        settings = Settings()

        # Empty string should be accepted
        assert settings.server_name == ""

    def test_unicode_in_environment_variables(self, monkeypatch):
        """Test handling of unicode characters in environment variables."""
        monkeypatch.setenv("SERVER_NAME", "seo-мой-сервер")
        monkeypatch.setenv("DATAFORSEO_LOGIN", "ユーザー名")

        settings = Settings()

        assert settings.server_name == "seo-мой-сервер"
        assert settings.dataforseo_login == "ユーザー名"

    def test_very_long_string_values(self, monkeypatch):
        """Test handling of very long string values."""
        long_string = "x" * 10000
        monkeypatch.setenv("DATAFORSEO_LOGIN", long_string)

        settings = Settings()

        assert settings.dataforseo_login == long_string
        assert len(settings.dataforseo_login) == 10000

    def test_special_characters_in_strings(self, monkeypatch):
        """Test handling of special characters in string fields."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        monkeypatch.setenv("DATAFORSEO_PASSWORD", special_chars)

        settings = Settings()

        assert settings.dataforseo_password == special_chars

    def test_newlines_and_tabs_in_strings(self, monkeypatch):
        """Test handling of newlines and tabs in string fields."""
        string_with_whitespace = "line1\nline2\tword"
        monkeypatch.setenv("DATAFORSEO_LOGIN", string_with_whitespace)

        settings = Settings()

        assert settings.dataforseo_login == string_with_whitespace