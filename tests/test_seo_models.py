"""
Comprehensive test suite for SEO Pydantic models.

Tests model validation, field constraints, data transformation,
serialization/deserialization, enum validation, edge cases,
and model inheritance.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import ValidationError, HttpUrl

from mcp_seo.models.seo_models import (
    AnalysisStatus,
    DeviceType,
    SEOTask,
    OnPageAnalysisRequest,
    OnPageIssue,
    OnPageSummary,
    KeywordData,
    KeywordAnalysisRequest,
    SERPResult,
    SERPAnalysisRequest,
    SERPAnalysisResult,
    CompetitorDomain,
    DomainAnalysisRequest,
    DomainRankOverview,
    TechnologyStack,
    DomainTechnologies,
    BacklinkData,
    BacklinksSummary,
    ContentAnalysisRequest,
    SentimentAnalysis,
    ContentSummary,
    SEOAuditRequest,
    SEOAuditResult,
    TaskResponse,
)


class TestEnumValidation:
    """Test enum field validation."""

    def test_analysis_status_valid_values(self):
        """Test valid AnalysisStatus enum values."""
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        for status in valid_statuses:
            task = SEOTask(task_id="test", target="example.com", status=status)
            assert task.status == status

    def test_analysis_status_invalid_value(self):
        """Test invalid AnalysisStatus enum value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SEOTask(task_id="test", target="example.com", status="invalid_status")

        assert "Input should be 'pending', 'in_progress', 'completed' or 'failed'" in str(exc_info.value)

    def test_device_type_valid_values(self):
        """Test valid DeviceType enum values."""
        valid_devices = ["desktop", "mobile", "tablet"]
        for device in valid_devices:
            request = KeywordAnalysisRequest(keywords=["test"], device=device)
            assert request.device == device

    def test_device_type_invalid_value(self):
        """Test invalid DeviceType enum value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KeywordAnalysisRequest(keywords=["test"], device="smartwatch")

        assert "Input should be 'desktop', 'mobile' or 'tablet'" in str(exc_info.value)

    @pytest.mark.parametrize("enum_class,valid_values", [
        (AnalysisStatus, ["pending", "in_progress", "completed", "failed"]),
        (DeviceType, ["desktop", "mobile", "tablet"]),
    ])
    def test_enum_case_sensitivity(self, enum_class, valid_values):
        """Test that enum values are case-sensitive."""
        for value in valid_values:
            # Test uppercase should fail
            with pytest.raises(ValidationError):
                if enum_class == AnalysisStatus:
                    SEOTask(task_id="test", target="example.com", status=value.upper())
                elif enum_class == DeviceType:
                    KeywordAnalysisRequest(keywords=["test"], device=value.upper())


class TestSEOTask:
    """Test SEOTask model validation."""

    def test_seo_task_valid_minimal(self):
        """Test valid minimal SEOTask creation."""
        task = SEOTask(task_id="test-123", target="example.com")
        assert task.task_id == "test-123"
        assert task.target == "example.com"
        assert task.status == AnalysisStatus.PENDING
        assert task.created_at is not None
        assert task.completed_at is None
        assert task.error_message is None

    def test_seo_task_valid_complete(self):
        """Test valid complete SEOTask creation."""
        now = datetime.now(timezone.utc)
        task = SEOTask(
            task_id="test-456",
            target="https://example.com",
            status=AnalysisStatus.COMPLETED,
            created_at=now,
            completed_at=now,
            error_message="All good"
        )
        assert task.task_id == "test-456"
        assert task.target == "https://example.com"
        assert task.status == AnalysisStatus.COMPLETED
        assert task.created_at == now
        assert task.completed_at == now
        assert task.error_message == "All good"

    def test_seo_task_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError) as exc_info:
            SEOTask()

        errors = exc_info.value.errors()
        required_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
        assert "task_id" in required_fields
        assert "target" in required_fields

    def test_seo_task_datetime_auto_generation(self):
        """Test that created_at is auto-generated if not provided."""
        before = datetime.now(timezone.utc)
        task = SEOTask(task_id="test", target="example.com")
        after = datetime.now(timezone.utc)

        assert before <= task.created_at.replace(tzinfo=timezone.utc) <= after


class TestOnPageAnalysisRequest:
    """Test OnPageAnalysisRequest model validation."""

    def test_onpage_request_valid_minimal(self):
        """Test valid minimal OnPageAnalysisRequest."""
        request = OnPageAnalysisRequest(target="https://example.com")
        assert str(request.target) == "https://example.com/"
        assert request.max_crawl_pages == 100
        assert request.respect_sitemap is True
        assert request.crawl_delay == 1
        assert request.enable_javascript is False

    def test_onpage_request_valid_complete(self):
        """Test valid complete OnPageAnalysisRequest."""
        request = OnPageAnalysisRequest(
            target="https://example.com",
            max_crawl_pages=500,
            start_url="https://example.com/start",
            respect_sitemap=False,
            custom_sitemap="https://example.com/sitemap.xml",
            crawl_delay=5,
            user_agent="Custom Bot 1.0",
            enable_javascript=True
        )
        assert str(request.target) == "https://example.com/"
        assert request.max_crawl_pages == 500
        assert str(request.start_url) == "https://example.com/start"
        assert request.respect_sitemap is False
        assert str(request.custom_sitemap) == "https://example.com/sitemap.xml"
        assert request.crawl_delay == 5
        assert request.user_agent == "Custom Bot 1.0"
        assert request.enable_javascript is True

    @pytest.mark.parametrize("invalid_url", [
        "not-a-url",
        "ftp://example.com",
        "mailto:test@example.com",
        "",
        "javascript:alert('xss')",
    ])
    def test_onpage_request_invalid_urls(self, invalid_url):
        """Test that invalid URLs raise ValidationError."""
        with pytest.raises(ValidationError):
            OnPageAnalysisRequest(target=invalid_url)

    @pytest.mark.parametrize("field,value,expected_error", [
        ("max_crawl_pages", 0, "Input should be greater than or equal to 1"),
        ("max_crawl_pages", 10001, "Input should be less than or equal to 10000"),
        ("crawl_delay", -1, "Input should be greater than or equal to 0"),
        ("crawl_delay", 61, "Input should be less than or equal to 60"),
    ])
    def test_onpage_request_field_constraints(self, field, value, expected_error):
        """Test field constraint validation."""
        with pytest.raises(ValidationError) as exc_info:
            OnPageAnalysisRequest(target="https://example.com", **{field: value})

        assert expected_error in str(exc_info.value)

    def test_onpage_request_start_url_validator(self):
        """Test that start_url defaults to None if not provided."""
        request = OnPageAnalysisRequest(target="https://example.com")
        # The start_url should be None if not explicitly provided
        assert request.start_url is None

        # Test explicit start_url
        request_with_start = OnPageAnalysisRequest(
            target="https://example.com",
            start_url="https://example.com/start"
        )
        assert str(request_with_start.start_url) == "https://example.com/start"


class TestKeywordAnalysisRequest:
    """Test KeywordAnalysisRequest model validation."""

    def test_keyword_request_valid_minimal(self):
        """Test valid minimal KeywordAnalysisRequest."""
        request = KeywordAnalysisRequest(keywords=["seo", "marketing"])
        assert request.keywords == ["seo", "marketing"]
        assert request.location == "usa"
        assert request.language == "english"
        assert request.device == DeviceType.DESKTOP
        assert request.include_suggestions is False
        assert request.suggestion_limit == 100

    def test_keyword_request_valid_complete(self):
        """Test valid complete KeywordAnalysisRequest."""
        request = KeywordAnalysisRequest(
            keywords=["seo tools", "keyword research", "backlink analysis"],
            location="canada",
            language="french",
            device=DeviceType.MOBILE,
            include_suggestions=True,
            suggestion_limit=500
        )
        assert request.keywords == ["seo tools", "keyword research", "backlink analysis"]
        assert request.location == "canada"
        assert request.language == "french"
        assert request.device == DeviceType.MOBILE
        assert request.include_suggestions is True
        assert request.suggestion_limit == 500

    @pytest.mark.parametrize("keywords,should_fail", [
        ([], True),  # Empty list should fail (min_length=1)
        (["valid"], False),  # Single keyword should pass
        (["a"] * 1000, False),  # Exactly 1000 keywords should pass
        (["a"] * 1001, True),  # 1001 keywords should fail (max_length=1000)
    ])
    def test_keyword_request_keywords_length_constraints(self, keywords, should_fail):
        """Test keywords list length constraints."""
        if should_fail:
            with pytest.raises(ValidationError):
                KeywordAnalysisRequest(keywords=keywords)
        else:
            request = KeywordAnalysisRequest(keywords=keywords)
            assert request.keywords == keywords

    @pytest.mark.parametrize("suggestion_limit,should_fail", [
        (0, True),  # Below minimum
        (1, False),  # Minimum valid value
        (500, False),  # Valid value
        (1000, False),  # Maximum valid value
        (1001, True),  # Above maximum
    ])
    def test_keyword_request_suggestion_limit_constraints(self, suggestion_limit, should_fail):
        """Test suggestion_limit field constraints."""
        if should_fail:
            with pytest.raises(ValidationError):
                KeywordAnalysisRequest(keywords=["test"], suggestion_limit=suggestion_limit)
        else:
            request = KeywordAnalysisRequest(keywords=["test"], suggestion_limit=suggestion_limit)
            assert request.suggestion_limit == suggestion_limit


class TestSERPAnalysisRequest:
    """Test SERPAnalysisRequest model validation."""

    def test_serp_request_valid_minimal(self):
        """Test valid minimal SERPAnalysisRequest."""
        request = SERPAnalysisRequest(keyword="seo tools")
        assert request.keyword == "seo tools"
        assert request.location == "usa"
        assert request.language == "english"
        assert request.device == DeviceType.DESKTOP
        assert request.depth == 100
        assert request.include_paid_results is True

    def test_serp_request_valid_complete(self):
        """Test valid complete SERPAnalysisRequest."""
        request = SERPAnalysisRequest(
            keyword="best seo tools 2024",
            location="uk",
            language="english",
            device=DeviceType.TABLET,
            depth=500,
            include_paid_results=False
        )
        assert request.keyword == "best seo tools 2024"
        assert request.location == "uk"
        assert request.language == "english"
        assert request.device == DeviceType.TABLET
        assert request.depth == 500
        assert request.include_paid_results is False

    @pytest.mark.parametrize("depth,should_fail", [
        (0, True),  # Below minimum
        (1, False),  # Minimum valid value
        (350, False),  # Valid value
        (700, False),  # Maximum valid value
        (701, True),  # Above maximum
    ])
    def test_serp_request_depth_constraints(self, depth, should_fail):
        """Test depth field constraints."""
        if should_fail:
            with pytest.raises(ValidationError):
                SERPAnalysisRequest(keyword="test", depth=depth)
        else:
            request = SERPAnalysisRequest(keyword="test", depth=depth)
            assert request.depth == depth


class TestSERPResult:
    """Test SERPResult model validation."""

    def test_serp_result_valid_minimal(self):
        """Test valid minimal SERPResult."""
        result = SERPResult(
            position=1,
            url="https://example.com",
            title="Example Page",
            domain="example.com"
        )
        assert result.position == 1
        assert str(result.url) == "https://example.com/"
        assert result.title == "Example Page"
        assert result.domain == "example.com"
        assert result.description is None
        assert result.breadcrumb is None
        assert result.is_featured_snippet is False
        assert result.is_paid is False

    def test_serp_result_valid_complete(self):
        """Test valid complete SERPResult."""
        result = SERPResult(
            position=3,
            url="https://example.com/page",
            title="Example Page Title",
            description="This is a description",
            domain="example.com",
            breadcrumb="Home > Category > Page",
            is_featured_snippet=True,
            is_paid=True
        )
        assert result.position == 3
        assert str(result.url) == "https://example.com/page"
        assert result.title == "Example Page Title"
        assert result.description == "This is a description"
        assert result.domain == "example.com"
        assert result.breadcrumb == "Home > Category > Page"
        assert result.is_featured_snippet is True
        assert result.is_paid is True

    def test_serp_result_required_fields(self):
        """Test that required fields are validated."""
        with pytest.raises(ValidationError) as exc_info:
            SERPResult()

        errors = exc_info.value.errors()
        required_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
        expected_required = {"position", "url", "title", "domain"}
        assert expected_required.issubset(required_fields)


class TestContentAnalysisRequest:
    """Test ContentAnalysisRequest model validation."""

    def test_content_request_valid_minimal(self):
        """Test valid minimal ContentAnalysisRequest."""
        request = ContentAnalysisRequest(text="This is a test content for analysis.")
        assert request.text == "This is a test content for analysis."
        assert request.language == "english"
        assert request.analysis_type == ["sentiment", "summary", "readability"]
        assert request.summary_length == "medium"

    def test_content_request_valid_complete(self):
        """Test valid complete ContentAnalysisRequest."""
        request = ContentAnalysisRequest(
            text="This is a comprehensive test content for detailed analysis.",
            language="spanish",
            analysis_type=["sentiment", "readability"],
            summary_length="long"
        )
        assert request.text == "This is a comprehensive test content for detailed analysis."
        assert request.language == "spanish"
        assert request.analysis_type == ["sentiment", "readability"]
        assert request.summary_length == "long"

    @pytest.mark.parametrize("text_length,should_fail", [
        (9, True),  # Below minimum (10 chars)
        (10, False),  # Minimum valid length
        (5000, False),  # Valid length
        (100000, False),  # Maximum valid length
        (100001, True),  # Above maximum
    ])
    def test_content_request_text_length_constraints(self, text_length, should_fail):
        """Test text field length constraints."""
        text = "a" * text_length
        if should_fail:
            with pytest.raises(ValidationError):
                ContentAnalysisRequest(text=text)
        else:
            request = ContentAnalysisRequest(text=text)
            assert len(request.text) == text_length

    @pytest.mark.parametrize("summary_length,should_fail", [
        ("short", False),
        ("medium", False),
        ("long", False),
        ("extra_long", True),  # Invalid value
        ("SHORT", True),  # Case sensitive
        ("", True),  # Empty string
    ])
    def test_content_request_summary_length_pattern(self, summary_length, should_fail):
        """Test summary_length field pattern validation."""
        if should_fail:
            with pytest.raises(ValidationError):
                ContentAnalysisRequest(text="Test content", summary_length=summary_length)
        else:
            request = ContentAnalysisRequest(text="Test content", summary_length=summary_length)
            assert request.summary_length == summary_length


class TestSentimentAnalysis:
    """Test SentimentAnalysis model validation."""

    def test_sentiment_analysis_valid(self):
        """Test valid SentimentAnalysis creation."""
        sentiment = SentimentAnalysis(
            sentiment="positive",
            confidence=0.85,
            positive_probability=0.8,
            negative_probability=0.1,
            neutral_probability=0.1
        )
        assert sentiment.sentiment == "positive"
        assert sentiment.confidence == 0.85
        assert sentiment.positive_probability == 0.8
        assert sentiment.negative_probability == 0.1
        assert sentiment.neutral_probability == 0.1

    @pytest.mark.parametrize("field,value,should_fail", [
        ("confidence", -0.1, True),  # Below minimum
        ("confidence", 0.0, False),  # Minimum valid
        ("confidence", 0.5, False),  # Valid value
        ("confidence", 1.0, False),  # Maximum valid
        ("confidence", 1.1, True),  # Above maximum
        ("positive_probability", -0.1, True),
        ("positive_probability", 1.1, True),
        ("negative_probability", -0.1, True),
        ("negative_probability", 1.1, True),
        ("neutral_probability", -0.1, True),
        ("neutral_probability", 1.1, True),
    ])
    def test_sentiment_analysis_probability_constraints(self, field, value, should_fail):
        """Test probability field constraints (0.0 to 1.0)."""
        base_data = {
            "sentiment": "positive",
            "confidence": 0.8,
            "positive_probability": 0.7,
            "negative_probability": 0.2,
            "neutral_probability": 0.1,
        }
        base_data[field] = value

        if should_fail:
            with pytest.raises(ValidationError):
                SentimentAnalysis(**base_data)
        else:
            sentiment = SentimentAnalysis(**base_data)
            assert getattr(sentiment, field) == value


class TestSEOAuditRequest:
    """Test SEOAuditRequest model validation."""

    def test_seo_audit_request_valid_minimal(self):
        """Test valid minimal SEOAuditRequest."""
        request = SEOAuditRequest(target="example.com")
        assert request.target == "example.com"
        assert request.location == "usa"
        assert request.language == "english"
        assert request.include_onpage is True
        assert request.include_keywords is True
        assert request.include_competitors is True
        assert request.include_backlinks is True
        assert request.include_content_analysis is False
        assert request.max_crawl_pages == 100
        assert request.keyword_list is None
        assert request.competitor_limit == 20
        assert request.content_urls is None

    def test_seo_audit_request_valid_complete(self):
        """Test valid complete SEOAuditRequest."""
        request = SEOAuditRequest(
            target="https://example.com",
            location="canada",
            language="french",
            include_onpage=False,
            include_keywords=True,
            include_competitors=False,
            include_backlinks=True,
            include_content_analysis=True,
            max_crawl_pages=500,
            keyword_list=["seo", "marketing", "tools"],
            competitor_limit=50,
            content_urls=["https://example.com/blog1", "https://example.com/blog2"]
        )
        assert request.target == "https://example.com"
        assert request.location == "canada"
        assert request.language == "french"
        assert request.include_onpage is False
        assert request.include_keywords is True
        assert request.include_competitors is False
        assert request.include_backlinks is True
        assert request.include_content_analysis is True
        assert request.max_crawl_pages == 500
        assert request.keyword_list == ["seo", "marketing", "tools"]
        assert request.competitor_limit == 50
        assert len(request.content_urls) == 2

    @pytest.mark.parametrize("field,value,should_fail", [
        ("max_crawl_pages", 0, True),  # Below minimum
        ("max_crawl_pages", 1, False),  # Minimum valid
        ("max_crawl_pages", 1000, False),  # Maximum valid
        ("max_crawl_pages", 1001, True),  # Above maximum
        ("competitor_limit", 0, True),  # Below minimum
        ("competitor_limit", 1, False),  # Minimum valid
        ("competitor_limit", 100, False),  # Maximum valid
        ("competitor_limit", 101, True),  # Above maximum
    ])
    def test_seo_audit_request_constraints(self, field, value, should_fail):
        """Test field constraint validation."""
        if should_fail:
            with pytest.raises(ValidationError):
                SEOAuditRequest(target="example.com", **{field: value})
        else:
            request = SEOAuditRequest(target="example.com", **{field: value})
            assert getattr(request, field) == value


class TestSEOAuditResult:
    """Test SEOAuditResult model validation."""

    def test_seo_audit_result_valid_minimal(self):
        """Test valid minimal SEOAuditResult."""
        result = SEOAuditResult(target="example.com")
        assert result.target == "example.com"
        assert result.audit_date is not None
        assert result.onpage_summary is None
        assert result.technical_issues is None
        assert result.target_keywords is None
        assert result.overall_seo_score is None

    def test_seo_audit_result_valid_complete(self):
        """Test valid complete SEOAuditResult."""
        audit_date = datetime.now(timezone.utc)
        onpage_summary = OnPageSummary(
            crawled_pages=100,
            total_issues=25,
            critical_issues=5,
            high_priority_issues=10,
            medium_priority_issues=8,
            low_priority_issues=2,
            issues=[]
        )

        result = SEOAuditResult(
            target="example.com",
            audit_date=audit_date,
            onpage_summary=onpage_summary,
            overall_seo_score=85,
            technical_seo_score=90,
            content_seo_score=80,
            off_page_seo_score=75
        )

        assert result.target == "example.com"
        assert result.audit_date == audit_date
        assert result.onpage_summary == onpage_summary
        assert result.overall_seo_score == 85
        assert result.technical_seo_score == 90
        assert result.content_seo_score == 80
        assert result.off_page_seo_score == 75

    @pytest.mark.parametrize("score_field,value,should_fail", [
        ("overall_seo_score", -1, True),  # Below minimum
        ("overall_seo_score", 0, False),  # Minimum valid
        ("overall_seo_score", 50, False),  # Valid value
        ("overall_seo_score", 100, False),  # Maximum valid
        ("overall_seo_score", 101, True),  # Above maximum
        ("technical_seo_score", -1, True),
        ("technical_seo_score", 101, True),
        ("content_seo_score", -1, True),
        ("content_seo_score", 101, True),
        ("off_page_seo_score", -1, True),
        ("off_page_seo_score", 101, True),
    ])
    def test_seo_audit_result_score_constraints(self, score_field, value, should_fail):
        """Test SEO score field constraints (0 to 100)."""
        if should_fail:
            with pytest.raises(ValidationError):
                SEOAuditResult(target="example.com", **{score_field: value})
        else:
            result = SEOAuditResult(target="example.com", **{score_field: value})
            assert getattr(result, score_field) == value


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_seo_task_serialization(self):
        """Test SEOTask serialization to dict."""
        task = SEOTask(
            task_id="test-123",
            target="example.com",
            status=AnalysisStatus.COMPLETED
        )

        data = task.model_dump()
        assert data["task_id"] == "test-123"
        assert data["target"] == "example.com"
        assert data["status"] == "completed"
        assert "created_at" in data

        # Test round-trip
        reconstructed = SEOTask.model_validate(data)
        assert reconstructed.task_id == task.task_id
        assert reconstructed.target == task.target
        assert reconstructed.status == task.status

    def test_keyword_analysis_request_serialization(self):
        """Test KeywordAnalysisRequest serialization."""
        request = KeywordAnalysisRequest(
            keywords=["seo", "marketing"],
            device=DeviceType.MOBILE,
            include_suggestions=True
        )

        data = request.model_dump()
        assert data["keywords"] == ["seo", "marketing"]
        assert data["device"] == "mobile"
        assert data["include_suggestions"] is True

        # Test round-trip
        reconstructed = KeywordAnalysisRequest.model_validate(data)
        assert reconstructed.keywords == request.keywords
        assert reconstructed.device == request.device
        assert reconstructed.include_suggestions == request.include_suggestions

    def test_nested_model_serialization(self):
        """Test nested model serialization with SEOAuditResult."""
        onpage_issue = OnPageIssue(
            issue_type="missing_meta_description",
            severity="high",
            affected_pages=15,
            description="15 pages are missing meta descriptions",
            recommendation="Add unique meta descriptions to all pages"
        )

        onpage_summary = OnPageSummary(
            crawled_pages=100,
            total_issues=1,
            critical_issues=0,
            high_priority_issues=1,
            medium_priority_issues=0,
            low_priority_issues=0,
            issues=[onpage_issue]
        )

        audit_result = SEOAuditResult(
            target="example.com",
            onpage_summary=onpage_summary,
            overall_seo_score=85
        )

        data = audit_result.model_dump()
        assert data["target"] == "example.com"
        assert data["overall_seo_score"] == 85
        assert data["onpage_summary"]["crawled_pages"] == 100
        assert len(data["onpage_summary"]["issues"]) == 1
        assert data["onpage_summary"]["issues"][0]["issue_type"] == "missing_meta_description"

        # Test round-trip
        reconstructed = SEOAuditResult.model_validate(data)
        assert reconstructed.target == audit_result.target
        assert reconstructed.overall_seo_score == audit_result.overall_seo_score
        assert reconstructed.onpage_summary.crawled_pages == onpage_summary.crawled_pages
        assert len(reconstructed.onpage_summary.issues) == 1
        assert reconstructed.onpage_summary.issues[0].issue_type == onpage_issue.issue_type


class TestModelInheritance:
    """Test model inheritance and composition."""

    def test_base_model_inheritance(self):
        """Test that all models inherit from BaseModel correctly."""
        models_to_test = [
            SEOTask,
            OnPageAnalysisRequest,
            KeywordAnalysisRequest,
            SERPAnalysisRequest,
            ContentAnalysisRequest,
            SEOAuditRequest,
        ]

        for model_class in models_to_test:
            # Check that the model is a subclass of BaseModel
            from pydantic import BaseModel
            assert issubclass(model_class, BaseModel)

            # Check that the model has expected BaseModel methods
            assert hasattr(model_class, "model_validate")
            assert hasattr(model_class, "model_dump")
            assert hasattr(model_class, "model_json_schema")

    def test_model_composition(self):
        """Test models that contain other models as fields."""
        # Test OnPageSummary containing OnPageIssue
        issue = OnPageIssue(
            issue_type="test",
            severity="low",
            affected_pages=1,
            description="Test issue",
            recommendation="Fix it"
        )

        summary = OnPageSummary(
            crawled_pages=10,
            total_issues=1,
            critical_issues=0,
            high_priority_issues=0,
            medium_priority_issues=0,
            low_priority_issues=1,
            issues=[issue]
        )

        assert len(summary.issues) == 1
        assert summary.issues[0].issue_type == "test"

        # Test SEOAuditResult containing multiple nested models
        audit = SEOAuditResult(
            target="example.com",
            onpage_summary=summary
        )

        assert audit.onpage_summary == summary
        assert audit.onpage_summary.issues[0].issue_type == "test"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_fields(self):
        """Test handling of empty string fields."""
        # Test that empty strings are accepted (Pydantic allows empty strings by default)
        task = SEOTask(task_id="", target="example.com")
        assert task.task_id == ""
        assert task.target == "example.com"

        task2 = SEOTask(task_id="test", target="")
        assert task2.task_id == "test"
        assert task2.target == ""

    def test_none_values_for_optional_fields(self):
        """Test that None values are accepted for optional fields."""
        task = SEOTask(
            task_id="test",
            target="example.com",
            completed_at=None,
            error_message=None
        )
        assert task.completed_at is None
        assert task.error_message is None

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        # Test Unicode in text fields
        task = SEOTask(
            task_id="test-üñíçødé",
            target="example.com",
            error_message="Error with émojis: 🚨 and special chars: «»"
        )
        assert "üñíçødé" in task.task_id
        assert "🚨" in task.error_message

        # Test Unicode in content analysis
        content_request = ContentAnalysisRequest(
            text="Análisis de contenido con caracteres especiales: ñ, ü, é, 中文, 🎯"
        )
        assert "中文" in content_request.text
        assert "🎯" in content_request.text

    def test_very_long_strings(self):
        """Test handling of very long strings within limits."""
        # Test maximum length content
        max_content = "a" * 100000  # Maximum allowed length
        content_request = ContentAnalysisRequest(text=max_content)
        assert len(content_request.text) == 100000

        # Test that exceeding maximum fails
        with pytest.raises(ValidationError):
            ContentAnalysisRequest(text="a" * 100001)

    def test_datetime_edge_cases(self):
        """Test datetime field edge cases."""
        from datetime import datetime, timezone

        # Test very old date
        old_date = datetime(1900, 1, 1, tzinfo=timezone.utc)
        task = SEOTask(
            task_id="test",
            target="example.com",
            created_at=old_date
        )
        assert task.created_at == old_date

        # Test far future date
        future_date = datetime(2100, 12, 31, tzinfo=timezone.utc)
        task = SEOTask(
            task_id="test",
            target="example.com",
            created_at=future_date
        )
        assert task.created_at == future_date

    def test_url_edge_cases(self):
        """Test URL field edge cases."""
        # Test various valid URL formats
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://example.com:8080",
            "https://subdomain.example.com",
            "https://example.com/path/to/page",
            "https://example.com/path?query=value",
            "https://example.com/path#fragment",
            "https://example.com/path?query=value&other=param#fragment",
        ]

        for url in valid_urls:
            request = OnPageAnalysisRequest(target=url)
            assert str(request.target).startswith(("http://", "https://"))

        # Test that invalid URLs fail
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "file:///local/path",
            "javascript:alert('xss')",
            "mailto:test@example.com",
            "https://xn--example-9qa.com",  # IDN domain (not supported by Pydantic HttpUrl)
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                OnPageAnalysisRequest(target=url)


class TestModelValidationMessages:
    """Test that validation error messages are informative."""

    def test_missing_required_field_messages(self):
        """Test that missing required field messages are clear."""
        with pytest.raises(ValidationError) as exc_info:
            SEOTask()

        error_messages = str(exc_info.value)
        assert "Field required" in error_messages
        assert any(field in error_messages for field in ["task_id", "target"])

    def test_constraint_violation_messages(self):
        """Test that constraint violation messages are informative."""
        with pytest.raises(ValidationError) as exc_info:
            OnPageAnalysisRequest(
                target="https://example.com",
                max_crawl_pages=0
            )

        error_message = str(exc_info.value)
        assert "Input should be greater than or equal to 1" in error_message

    def test_enum_validation_messages(self):
        """Test that enum validation messages list valid options."""
        with pytest.raises(ValidationError) as exc_info:
            KeywordAnalysisRequest(
                keywords=["test"],
                device="invalid_device"
            )

        error_message = str(exc_info.value)
        assert "Input should be 'desktop', 'mobile' or 'tablet'" in error_message

    def test_url_validation_messages(self):
        """Test that URL validation messages are helpful."""
        with pytest.raises(ValidationError) as exc_info:
            OnPageAnalysisRequest(target="not-a-valid-url")

        error_message = str(exc_info.value)
        assert "URL" in error_message or "url" in error_message


if __name__ == "__main__":
    pytest.main([__file__])