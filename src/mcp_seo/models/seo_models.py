"""
Pydantic models for SEO analysis data structures.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationInfo
from datetime import datetime
from enum import Enum
import re


class AnalysisStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DeviceType(str, Enum):
    """Device type enumeration."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class SEOTask(BaseModel):
    """Base SEO analysis task model."""
    task_id: str
    target: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class OnPageAnalysisRequest(BaseModel):
    """OnPage analysis request model."""
    target: str
    max_crawl_pages: int = Field(default=100, ge=1, le=10000)
    start_url: Optional[str] = None
    respect_sitemap: bool = True
    custom_sitemap: Optional[str] = None
    crawl_delay: int = Field(default=1, ge=0, le=60)
    user_agent: Optional[str] = None
    enable_javascript: bool = False

    @field_validator('target', mode='before')
    @classmethod
    def validate_target_url(cls, v):
        """Validate and normalize target URL, handling localhost cases."""
        if not v:
            raise ValueError("Target URL is required")

        # Convert to string if it's not already
        url = str(v)

        # Handle localhost without scheme
        if re.match(r'^localhost(:\d+)?(/.*)?$', url):
            url = f'http://{url}'
        elif re.match(r'^\d+\.\d+\.\d+\.\d+(:\d+)?(/.*)?$', url):  # IP address
            url = f'http://{url}'
        elif not url.startswith(('http://', 'https://')):
            # Default to https for domain names
            url = f'https://{url}'

        # Basic URL validation
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
            raise ValueError(f"Invalid URL format: {url}")

        return url

    @field_validator('start_url', mode='before')
    @classmethod
    def validate_start_url(cls, v, info: ValidationInfo):
        """Validate start_url or use target if not provided."""
        if not v and info.data:
            return info.data.get('target')

        if v:
            url = str(v)
            # Handle localhost without scheme
            if re.match(r'^localhost(:\d+)?(/.*)?$', url):
                url = f'http://{url}'
            elif re.match(r'^\d+\.\d+\.\d+\.\d+(:\d+)?(/.*)?$', url):  # IP address
                url = f'http://{url}'
            elif not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # Basic URL validation
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
                raise ValueError(f"Invalid start_url format: {url}")

            return url

        return v

    @field_validator('custom_sitemap', mode='before')
    @classmethod
    def validate_custom_sitemap(cls, v):
        """Validate custom sitemap URL."""
        if not v:
            return v

        url = str(v)
        # Handle localhost without scheme
        if re.match(r'^localhost(:\d+)?(/.*)?$', url):
            url = f'http://{url}'
        elif re.match(r'^\d+\.\d+\.\d+\.\d+(:\d+)?(/.*)?$', url):  # IP address
            url = f'http://{url}'
        elif not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        # Basic URL validation
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
            raise ValueError(f"Invalid custom_sitemap format: {url}")

        return url


class OnPageIssue(BaseModel):
    """OnPage SEO issue model."""
    issue_type: str
    severity: str  # critical, high, medium, low
    affected_pages: int
    description: str
    recommendation: str


class OnPageSummary(BaseModel):
    """OnPage analysis summary model."""
    crawled_pages: int
    total_issues: int
    critical_issues: int
    high_priority_issues: int
    medium_priority_issues: int
    low_priority_issues: int
    issues: List[OnPageIssue]
    lighthouse_score: Optional[int] = None
    core_web_vitals: Optional[Dict[str, Any]] = None


class KeywordData(BaseModel):
    """Keyword analysis data model."""
    keyword: str
    search_volume: Optional[int] = None
    cpc: Optional[float] = None
    competition: Optional[float] = None
    competition_level: Optional[str] = None
    monthly_searches: Optional[List[Dict[str, Any]]] = None


class KeywordAnalysisRequest(BaseModel):
    """Keyword analysis request model."""
    keywords: List[str] = Field(..., min_length=1, max_length=1000)
    location: str = "usa"
    language: str = "english" 
    device: DeviceType = DeviceType.DESKTOP
    include_suggestions: bool = False
    suggestion_limit: int = Field(default=100, ge=1, le=1000)


class SERPResult(BaseModel):
    """SERP result model."""
    position: int
    url: HttpUrl
    title: str
    description: Optional[str] = None
    domain: str
    breadcrumb: Optional[str] = None
    is_featured_snippet: bool = False
    is_paid: bool = False


class SERPAnalysisRequest(BaseModel):
    """SERP analysis request model."""
    keyword: str
    location: str = "usa"
    language: str = "english"
    device: DeviceType = DeviceType.DESKTOP
    depth: int = Field(default=100, ge=1, le=700)
    include_paid_results: bool = True


class SERPAnalysisResult(BaseModel):
    """SERP analysis result model."""
    keyword: str
    total_results: int
    results: List[SERPResult]
    featured_snippet: Optional[SERPResult] = None
    people_also_ask: Optional[List[str]] = None
    related_searches: Optional[List[str]] = None


class CompetitorDomain(BaseModel):
    """Competitor domain model."""
    domain: str
    common_keywords: int
    se_keywords_count: int
    etv: float  # Estimated Traffic Value
    median_position: float
    visibility: float


class DomainAnalysisRequest(BaseModel):
    """Domain analysis request model."""
    target: str
    location: str = "usa"
    language: str = "english"
    include_competitors: bool = True
    competitor_limit: int = Field(default=50, ge=1, le=1000)
    include_keywords: bool = True
    keyword_limit: int = Field(default=100, ge=1, le=10000)


class DomainRankOverview(BaseModel):
    """Domain ranking overview model."""
    target: str
    organic_keywords: int
    organic_etv: float
    organic_count: int
    organic_pos_1: int
    organic_pos_2_3: int
    organic_pos_4_10: int
    organic_pos_11_20: int
    organic_pos_21_30: int
    organic_pos_31_40: int
    organic_pos_41_50: int
    organic_pos_51_60: int
    organic_pos_61_70: int
    organic_pos_71_80: int
    organic_pos_81_90: int
    organic_pos_91_100: int


class TechnologyStack(BaseModel):
    """Website technology stack model."""
    category: str
    technology: str
    version: Optional[str] = None


class DomainTechnologies(BaseModel):
    """Domain technologies model."""
    target: str
    technologies: List[TechnologyStack]
    cms: Optional[str] = None
    server: Optional[str] = None
    ssl_info: Optional[Dict[str, Any]] = None


class BacklinkData(BaseModel):
    """Backlink data model."""
    referring_domain: str
    backlinks_count: int
    first_seen: Optional[datetime] = None
    lost_date: Optional[datetime] = None
    rank: Optional[int] = None
    domain_rank: Optional[int] = None
    page_rank: Optional[int] = None
    links_external: Optional[int] = None
    links_internal: Optional[int] = None


class BacklinksSummary(BaseModel):
    """Backlinks summary model."""
    target: str
    referring_domains: int
    referring_main_domains: int
    referring_ips: int
    backlinks: int
    first_seen: Optional[datetime] = None
    lost_date: Optional[datetime] = None
    broken_backlinks: int
    broken_pages: int
    referring_domains_nofollow: int
    backlinks_nofollow: int


class ContentAnalysisRequest(BaseModel):
    """Content analysis request model."""
    text: str = Field(..., min_length=10, max_length=100000)
    language: str = "english"
    analysis_type: List[str] = Field(default=["sentiment", "summary", "readability"])
    summary_length: str = Field(default="medium", pattern="^(short|medium|long)$")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result model."""
    sentiment: str  # positive, negative, neutral
    confidence: float = Field(..., ge=0.0, le=1.0)
    positive_probability: float = Field(..., ge=0.0, le=1.0)
    negative_probability: float = Field(..., ge=0.0, le=1.0)
    neutral_probability: float = Field(..., ge=0.0, le=1.0)


class ContentSummary(BaseModel):
    """Content summary model."""
    summary: str
    key_phrases: List[str]
    readability_score: Optional[float] = None
    word_count: int
    sentence_count: int
    paragraph_count: int
    sentiment: Optional[SentimentAnalysis] = None


class SEOAuditRequest(BaseModel):
    """Comprehensive SEO audit request model."""
    target: str
    location: str = "usa"
    language: str = "english"
    include_onpage: bool = True
    include_keywords: bool = True
    include_competitors: bool = True
    include_backlinks: bool = True
    include_content_analysis: bool = False
    max_crawl_pages: int = Field(default=100, ge=1, le=1000)
    
    # Specific analysis options
    keyword_list: Optional[List[str]] = None
    competitor_limit: int = Field(default=20, ge=1, le=100)
    content_urls: Optional[List[HttpUrl]] = None


class SEOAuditResult(BaseModel):
    """Comprehensive SEO audit result model."""
    target: str
    audit_date: datetime = Field(default_factory=datetime.utcnow)
    
    # OnPage Analysis
    onpage_summary: Optional[OnPageSummary] = None
    technical_issues: Optional[List[OnPageIssue]] = None
    
    # Keyword Analysis
    target_keywords: Optional[List[KeywordData]] = None
    ranked_keywords_count: Optional[int] = None
    organic_traffic_estimate: Optional[float] = None
    
    # Competitor Analysis
    main_competitors: Optional[List[CompetitorDomain]] = None
    competitive_keywords: Optional[List[str]] = None
    
    # Domain Analysis
    domain_rank_overview: Optional[DomainRankOverview] = None
    domain_technologies: Optional[DomainTechnologies] = None
    
    # Backlinks Analysis
    backlinks_summary: Optional[BacklinksSummary] = None
    top_referring_domains: Optional[List[BacklinkData]] = None
    
    # Content Analysis
    content_summaries: Optional[List[ContentSummary]] = None
    
    # Overall Scores
    overall_seo_score: Optional[int] = Field(None, ge=0, le=100)
    technical_seo_score: Optional[int] = Field(None, ge=0, le=100)
    content_seo_score: Optional[int] = Field(None, ge=0, le=100)
    off_page_seo_score: Optional[int] = Field(None, ge=0, le=100)


class TaskResponse(BaseModel):
    """Generic task response model."""
    task_id: str
    status: str
    message: Optional[str] = None
    estimated_completion_time: Optional[int] = None  # seconds
    result_url: Optional[str] = None