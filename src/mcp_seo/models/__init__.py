"""
SEO data models and contracts.

This module contains data models, request/response objects, and domain contracts
for the SEO analysis system.
"""

from .seo_models import (
    OnPageAnalysisRequest,
    OnPageSummary,
    OnPageIssue,
    KeywordAnalysisRequest,
    SERPAnalysisRequest,
    DomainAnalysisRequest,
    ContentAnalysisRequest,
    SEOAuditRequest,
    DeviceType,
    AnalysisStatus
)

__all__ = [
    "OnPageAnalysisRequest",
    "OnPageSummary",
    "OnPageIssue",
    "KeywordAnalysisRequest",
    "SERPAnalysisRequest",
    "DomainAnalysisRequest",
    "ContentAnalysisRequest",
    "SEOAuditRequest",
    "DeviceType",
    "AnalysisStatus"
]