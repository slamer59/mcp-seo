"""
Enhanced OnPage SEO analysis tools using DataForSEO API with advanced reporting.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp_seo.content_analysis import BlogContentOptimizer
from mcp_seo.dataforseo.client import ApiException, DataForSEOClient
from mcp_seo.engines import SEORecommendationEngine
from mcp_seo.models.seo_models import OnPageAnalysisRequest, OnPageIssue, OnPageSummary
from mcp_seo.reporting import SEOReporter


class OnPageAnalyzer:
    """OnPage SEO analysis tool with advanced reporting and recommendations."""

    def __init__(self, client: DataForSEOClient, use_rich_reporting: bool = True):
        self.client = client
        self.recommendation_engine = SEORecommendationEngine()
        self.reporter = SEOReporter(use_rich=use_rich_reporting)
        self.content_optimizer = BlogContentOptimizer()

    def create_analysis_task(self, request: OnPageAnalysisRequest) -> Dict[str, Any]:
        """Create OnPage analysis task."""
        try:
            task_data = {
                "max_crawl_pages": request.max_crawl_pages,
                "start_url": (
                    str(request.start_url) if request.start_url else str(request.target)
                ),
                "respect_sitemap": request.respect_sitemap,
                "crawl_delay": request.crawl_delay,
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
                    "estimated_completion_time": request.max_crawl_pages
                    * request.crawl_delay
                    + 300,
                }
            else:
                raise ApiException("Failed to create OnPage task")

        except Exception as e:
            return {
                "error": f"Failed to create OnPage analysis task: {str(e)}",
                "target": str(request.target),
            }

    def get_analysis_summary(self, task_id: str) -> Dict[str, Any]:
        """Get OnPage analysis summary with enhanced recommendations."""
        try:
            result = self.client.get_onpage_summary(task_id)

            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "status": "in_progress",
                    "message": "Analysis still in progress",
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
                "duplicate_meta_descriptions": task_result.get(
                    "duplicate_meta_descriptions", 0
                ),
                "duplicate_h1_tags": task_result.get("duplicate_h1_tags", 0),
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

            # Generate comprehensive recommendations
            recommendations = self.recommendation_engine.analyze_technical_issues(
                {"summary": summary_data}
            )

            # Calculate SEO health score
            seo_health_score = self._calculate_seo_health_score(summary_data)

            result = {
                "task_id": task_id,
                "status": "completed",
                "summary": summary_data,
                "seo_recommendations": [rec.__dict__ for rec in recommendations],
                "seo_health_score": seo_health_score,
                "formatted_report": self.reporter.generate_onpage_analysis_report(
                    {"target": summary_data.get("target", ""), "summary": summary_data}
                ),
                "optimization_priorities": self._generate_optimization_priorities(
                    summary_data, recommendations
                ),
            }

            return result

        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get OnPage summary: {str(e)}",
            }

    def get_page_details(
        self, task_id: str, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """Get detailed page analysis with content optimization suggestions."""
        try:
            result = self.client.get_onpage_pages(task_id, limit=limit, offset=offset)

            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {"task_id": task_id, "error": "No page data available"}

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
                    "issues": self._analyze_page_issues(page),
                }

                # Add content quality analysis
                if page_analysis.get("word_count", 0) > 100:
                    content_suggestions = self._analyze_page_content_quality(
                        page_analysis
                    )
                    page_analysis["content_suggestions"] = content_suggestions

                analyzed_pages.append(page_analysis)

            result = {
                "task_id": task_id,
                "pages": analyzed_pages,
                "total_pages": len(analyzed_pages),
                "has_more": len(pages) >= limit,
                "content_optimization_summary": self._generate_content_optimization_summary(
                    analyzed_pages
                ),
                "page_performance_insights": self._generate_page_performance_insights(
                    analyzed_pages
                ),
            }

            return result

        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get page details: {str(e)}",
            }

    def get_duplicate_content_analysis(self, task_id: str) -> Dict[str, Any]:
        """Get duplicate content analysis with recommendations."""
        try:
            result = self.client.get_onpage_duplicate_tags(task_id)

            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "error": "No duplicate content data available",
                }

            duplicates = result["tasks"][0]["result"][0]

            duplicate_analysis = {
                "task_id": task_id,
                "duplicate_title_tags": {
                    "count": len(duplicates.get("duplicate_title", [])),
                    "duplicates": duplicates.get("duplicate_title", []),
                },
                "duplicate_meta_descriptions": {
                    "count": len(duplicates.get("duplicate_description", [])),
                    "duplicates": duplicates.get("duplicate_description", []),
                },
                "duplicate_h1_tags": {
                    "count": len(duplicates.get("duplicate_h1", [])),
                    "duplicates": duplicates.get("duplicate_h1", []),
                },
            }

            # Generate recommendations for duplicate content issues
            duplicate_recommendations = (
                self._generate_duplicate_content_recommendations(duplicate_analysis)
            )

            result = {
                "task_id": task_id,
                "duplicate_analysis": duplicate_analysis,
                "duplicate_content_recommendations": duplicate_recommendations,
                "prioritized_fixes": self._prioritize_duplicate_content_fixes(
                    duplicate_analysis
                ),
            }

            return result

        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get duplicate content analysis: {str(e)}",
            }

    def get_lighthouse_analysis(self, task_id: str) -> Dict[str, Any]:
        """Get Lighthouse performance analysis with actionable recommendations."""
        try:
            result = self.client.get_onpage_lighthouse(task_id)

            if not result.get("tasks") or not result["tasks"][0].get("result"):
                return {
                    "task_id": task_id,
                    "message": "Lighthouse data not available for this task",
                }

            lighthouse_data = result["tasks"][0]["result"][0]

            lighthouse_analysis = {
                "performance_score": lighthouse_data.get("performance", 0),
                "accessibility_score": lighthouse_data.get("accessibility", 0),
                "best_practices_score": lighthouse_data.get("best_practices", 0),
                "seo_score": lighthouse_data.get("seo", 0),
                "first_contentful_paint": lighthouse_data.get(
                    "first_contentful_paint", {}
                ),
                "largest_contentful_paint": lighthouse_data.get(
                    "largest_contentful_paint", {}
                ),
                "cumulative_layout_shift": lighthouse_data.get(
                    "cumulative_layout_shift", {}
                ),
                "first_input_delay": lighthouse_data.get("first_input_delay", {}),
                "speed_index": lighthouse_data.get("speed_index", {}),
                "total_blocking_time": lighthouse_data.get("total_blocking_time", {}),
            }

            # Generate performance recommendations
            performance_recommendations = self._generate_performance_recommendations(
                lighthouse_analysis
            )

            result = {
                "task_id": task_id,
                "lighthouse_analysis": lighthouse_analysis,
                "performance_recommendations": performance_recommendations,
                "core_web_vitals_assessment": self._assess_core_web_vitals(
                    lighthouse_analysis
                ),
                "performance_optimization_plan": self._create_performance_optimization_plan(
                    lighthouse_analysis
                ),
            }

            return result

        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to get Lighthouse analysis: {str(e)}",
            }

    def generate_comprehensive_seo_audit(self, task_id: str) -> Dict[str, Any]:
        """Generate a comprehensive SEO audit combining all analysis types."""
        try:
            # Get all analysis data
            summary_result = self.get_analysis_summary(task_id)
            page_details = self.get_page_details(task_id, limit=50)  # Sample of pages
            duplicate_analysis = self.get_duplicate_content_analysis(task_id)
            lighthouse_analysis = self.get_lighthouse_analysis(task_id)

            # Generate comprehensive recommendations
            comprehensive_recommendations = (
                self.recommendation_engine.generate_comprehensive_recommendations(
                    onpage_data=summary_result,
                    content_data=page_details,
                    performance_data=lighthouse_analysis,
                )
            )

            audit_result = {
                "task_id": task_id,
                "audit_timestamp": summary_result.get("summary", {}).get("target", ""),
                "overall_seo_score": comprehensive_recommendations.get("seo_score", {}),
                "comprehensive_recommendations": comprehensive_recommendations,
                "audit_sections": {
                    "technical_seo": summary_result,
                    "content_analysis": page_details,
                    "duplicate_content": duplicate_analysis,
                    "performance": lighthouse_analysis,
                },
                "formatted_audit_report": self.reporter.generate_comprehensive_seo_report(
                    comprehensive_recommendations
                ),
                "action_plan": comprehensive_recommendations.get("action_plan", {}),
                "estimated_impact": self._estimate_optimization_impact(
                    comprehensive_recommendations
                ),
            }

            return audit_result

        except Exception as e:
            return {
                "task_id": task_id,
                "error": f"Failed to generate comprehensive audit: {str(e)}",
            }

    # Enhanced helper methods
    def _calculate_seo_health_score(
        self, summary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall SEO health score."""
        score = 100
        factors = {}

        # Technical issues impact
        critical_issues = summary_data.get("critical_issues", 0)
        high_issues = summary_data.get("high_priority_issues", 0)
        medium_issues = summary_data.get("medium_priority_issues", 0)

        technical_deduction = min(
            critical_issues * 15 + high_issues * 8 + medium_issues * 3, 50
        )
        score -= technical_deduction
        factors["technical_issues"] = {
            "score": 100 - technical_deduction,
            "impact": technical_deduction,
        }

        # Duplicate content impact
        duplicate_titles = summary_data.get("duplicate_title_tags", 0)
        duplicate_descriptions = summary_data.get("duplicate_meta_descriptions", 0)

        crawled_pages = summary_data.get("crawled_pages", 1)
        duplicate_ratio = (
            (duplicate_titles + duplicate_descriptions) / (crawled_pages * 2)
            if crawled_pages > 0
            else 0
        )
        duplicate_deduction = min(duplicate_ratio * 30, 20)
        score -= duplicate_deduction
        factors["duplicate_content"] = {
            "score": 100 - duplicate_deduction,
            "impact": duplicate_deduction,
        }

        # Broken pages impact
        broken_pages = summary_data.get("broken_pages", 0)
        broken_ratio = broken_pages / crawled_pages if crawled_pages > 0 else 0
        broken_deduction = min(broken_ratio * 40, 25)
        score -= broken_deduction
        factors["broken_pages"] = {
            "score": 100 - broken_deduction,
            "impact": broken_deduction,
        }

        # Determine grade
        if score >= 90:
            grade = "A"
            status = "Excellent"
        elif score >= 80:
            grade = "B"
            status = "Good"
        elif score >= 70:
            grade = "C"
            status = "Fair"
        elif score >= 60:
            grade = "D"
            status = "Poor"
        else:
            grade = "F"
            status = "Critical"

        return {
            "overall_score": max(0, int(score)),
            "grade": grade,
            "status": status,
            "factors": factors,
            "recommendations": self._get_score_improvement_recommendations(factors),
        }

    def _analyze_page_content_quality(
        self, page_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze individual page content quality."""
        suggestions = {
            "title_optimization": [],
            "meta_description_optimization": [],
            "content_structure": [],
            "performance": [],
        }

        # Title analysis
        title = page_data.get("title", "")
        if not title:
            suggestions["title_optimization"].append("Add a title tag")
        elif len(title) < 30:
            suggestions["title_optimization"].append(
                f"Expand title (currently {len(title)} chars, aim for 30-60)"
            )
        elif len(title) > 60:
            suggestions["title_optimization"].append(
                f"Shorten title (currently {len(title)} chars, aim for 30-60)"
            )

        # Meta description analysis
        meta_desc = page_data.get("meta_description", "")
        if not meta_desc:
            suggestions["meta_description_optimization"].append(
                "Add a meta description"
            )
        elif len(meta_desc) < 120:
            suggestions["meta_description_optimization"].append(
                f"Expand meta description (currently {len(meta_desc)} chars)"
            )
        elif len(meta_desc) > 160:
            suggestions["meta_description_optimization"].append(
                f"Shorten meta description (currently {len(meta_desc)} chars)"
            )

        # Content structure
        word_count = page_data.get("word_count", 0)
        if word_count < 300:
            suggestions["content_structure"].append(
                f"Increase content length (currently {word_count} words)"
            )

        images_without_alt = page_data.get("images_without_alt", 0)
        if images_without_alt > 0:
            suggestions["content_structure"].append(
                f"Add alt text to {images_without_alt} images"
            )

        # Performance
        load_time = page_data.get("load_time", 0)
        if load_time > 3000:
            suggestions["performance"].append(
                f"Improve page load time (currently {load_time / 1000:.1f}s)"
            )

        return suggestions

    def _generate_content_optimization_summary(
        self, analyzed_pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of content optimization opportunities."""
        total_pages = len(analyzed_pages)

        thin_content_pages = len(
            [p for p in analyzed_pages if p.get("word_count", 0) < 300]
        )
        missing_titles = len([p for p in analyzed_pages if not p.get("title")])
        missing_descriptions = len(
            [p for p in analyzed_pages if not p.get("meta_description")]
        )
        slow_pages = len([p for p in analyzed_pages if p.get("load_time", 0) > 3000])

        return {
            "total_pages_analyzed": total_pages,
            "optimization_opportunities": {
                "thin_content_pages": {
                    "count": thin_content_pages,
                    "percentage": (
                        (thin_content_pages / total_pages * 100)
                        if total_pages > 0
                        else 0
                    ),
                    "priority": "Medium",
                },
                "missing_titles": {
                    "count": missing_titles,
                    "percentage": (
                        (missing_titles / total_pages * 100) if total_pages > 0 else 0
                    ),
                    "priority": "High",
                },
                "missing_meta_descriptions": {
                    "count": missing_descriptions,
                    "percentage": (
                        (missing_descriptions / total_pages * 100)
                        if total_pages > 0
                        else 0
                    ),
                    "priority": "Medium",
                },
                "slow_loading_pages": {
                    "count": slow_pages,
                    "percentage": (
                        (slow_pages / total_pages * 100) if total_pages > 0 else 0
                    ),
                    "priority": "High",
                },
            },
            "recommendations": [
                "Focus on improving thin content pages with comprehensive information",
                "Add missing title tags and meta descriptions",
                "Optimize page load times for better user experience",
                "Ensure all images have descriptive alt text",
            ],
        }

    def _generate_page_performance_insights(
        self, analyzed_pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate performance insights for analyzed pages."""
        if not analyzed_pages:
            return {}

        load_times = [
            p.get("load_time", 0) for p in analyzed_pages if p.get("load_time", 0) > 0
        ]
        page_sizes = [
            p.get("page_size", 0) for p in analyzed_pages if p.get("page_size", 0) > 0
        ]

        insights = {
            "performance_metrics": {
                "average_load_time": (
                    sum(load_times) / len(load_times) if load_times else 0
                ),
                "slowest_page_load_time": max(load_times) if load_times else 0,
                "pages_over_3_seconds": len([t for t in load_times if t > 3000]),
                "average_page_size": (
                    sum(page_sizes) / len(page_sizes) if page_sizes else 0
                ),
                "largest_page_size": max(page_sizes) if page_sizes else 0,
            },
            "performance_recommendations": [],
        }

        # Generate specific recommendations based on metrics
        avg_load_time = insights["performance_metrics"]["average_load_time"]
        if avg_load_time > 3000:
            insights["performance_recommendations"].append(
                "Overall site speed needs improvement - average load time is over 3 seconds"
            )

        slow_pages_count = insights["performance_metrics"]["pages_over_3_seconds"]
        if slow_pages_count > 0:
            insights["performance_recommendations"].append(
                f"{slow_pages_count} pages load slower than 3 seconds and need optimization"
            )

        return insights

    def _generate_duplicate_content_recommendations(
        self, duplicate_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate specific recommendations for duplicate content issues."""
        recommendations = []

        duplicate_titles_count = duplicate_analysis.get("duplicate_title_tags", {}).get(
            "count", 0
        )
        if duplicate_titles_count > 0:
            recommendations.append(
                {
                    "issue": "Duplicate Title Tags",
                    "count": duplicate_titles_count,
                    "priority": "High",
                    "recommendation": "Create unique, descriptive title tags for each page. Focus on primary keywords and page-specific content.",
                    "action_items": [
                        "Review all pages with duplicate titles",
                        "Create unique titles that reflect each page's content",
                        "Include target keywords naturally",
                        "Keep titles between 30-60 characters",
                    ],
                }
            )

        duplicate_descriptions_count = duplicate_analysis.get(
            "duplicate_meta_descriptions", {}
        ).get("count", 0)
        if duplicate_descriptions_count > 0:
            recommendations.append(
                {
                    "issue": "Duplicate Meta Descriptions",
                    "count": duplicate_descriptions_count,
                    "priority": "Medium",
                    "recommendation": "Write unique meta descriptions that summarize each page's content and encourage clicks.",
                    "action_items": [
                        "Review all pages with duplicate meta descriptions",
                        "Write compelling, unique descriptions",
                        "Include call-to-action phrases",
                        "Keep descriptions between 120-160 characters",
                    ],
                }
            )

        duplicate_h1_count = duplicate_analysis.get("duplicate_h1_tags", {}).get(
            "count", 0
        )
        if duplicate_h1_count > 0:
            recommendations.append(
                {
                    "issue": "Duplicate H1 Tags",
                    "count": duplicate_h1_count,
                    "priority": "Medium",
                    "recommendation": "Ensure each page has a unique, descriptive H1 tag that clearly indicates the page topic.",
                    "action_items": [
                        "Review all pages with duplicate H1 tags",
                        "Create unique H1s that reflect page content",
                        "Use only one H1 per page",
                        "Make H1s descriptive and keyword-rich",
                    ],
                }
            )

        return recommendations

    def _prioritize_duplicate_content_fixes(
        self, duplicate_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prioritize duplicate content fixes based on impact."""
        priorities = {"immediate": [], "short_term": [], "long_term": []}

        # Title tags - highest priority
        title_count = duplicate_analysis.get("duplicate_title_tags", {}).get("count", 0)
        if title_count > 0:
            if title_count > 10:
                priorities["immediate"].append(
                    {
                        "task": "Fix duplicate title tags",
                        "count": title_count,
                        "reason": "Critical for search rankings",
                    }
                )
            else:
                priorities["short_term"].append(
                    {
                        "task": "Fix duplicate title tags",
                        "count": title_count,
                        "reason": "Important for search rankings",
                    }
                )

        # H1 tags - medium priority
        h1_count = duplicate_analysis.get("duplicate_h1_tags", {}).get("count", 0)
        if h1_count > 0:
            priorities["short_term"].append(
                {
                    "task": "Fix duplicate H1 tags",
                    "count": h1_count,
                    "reason": "Improves content structure",
                }
            )

        # Meta descriptions - lower priority
        desc_count = duplicate_analysis.get("duplicate_meta_descriptions", {}).get(
            "count", 0
        )
        if desc_count > 0:
            priorities["long_term"].append(
                {
                    "task": "Fix duplicate meta descriptions",
                    "count": desc_count,
                    "reason": "Improves click-through rates",
                }
            )

        return priorities

    def _generate_performance_recommendations(
        self, lighthouse_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate specific performance optimization recommendations."""
        recommendations = []

        performance_score = lighthouse_analysis.get("performance_score", 0)
        if performance_score < 50:
            recommendations.append(
                {
                    "category": "Performance",
                    "priority": "Critical",
                    "recommendation": "Overall performance needs immediate attention",
                    "details": "Performance score is below 50, indicating serious performance issues",
                    "actions": [
                        "Optimize images and use modern formats (WebP)",
                        "Minimize and compress CSS/JavaScript",
                        "Implement proper caching strategies",
                        "Consider using a Content Delivery Network (CDN)",
                    ],
                }
            )
        elif performance_score < 75:
            recommendations.append(
                {
                    "category": "Performance",
                    "priority": "High",
                    "recommendation": "Performance could be significantly improved",
                    "details": "Performance score is below 75, indicating room for optimization",
                    "actions": [
                        "Optimize largest contentful paint",
                        "Reduce cumulative layout shift",
                        "Minimize total blocking time",
                    ],
                }
            )

        # Specific Core Web Vitals recommendations
        lcp = lighthouse_analysis.get("largest_contentful_paint", {}).get("value", 0)
        if lcp > 2500:  # 2.5 seconds threshold
            recommendations.append(
                {
                    "category": "Core Web Vitals",
                    "priority": "High",
                    "recommendation": "Improve Largest Contentful Paint (LCP)",
                    "details": f"LCP is {lcp}ms, should be under 2500ms",
                    "actions": [
                        "Optimize server response times",
                        "Preload important resources",
                        "Optimize images and text rendering",
                    ],
                }
            )

        return recommendations

    def _assess_core_web_vitals(
        self, lighthouse_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess Core Web Vitals performance."""
        assessment = {"overall_status": "good", "metrics": {}, "recommendations": []}

        # LCP assessment
        lcp = lighthouse_analysis.get("largest_contentful_paint", {}).get("value", 0)
        if lcp <= 2500:
            assessment["metrics"]["lcp"] = {
                "status": "good",
                "value": lcp,
                "threshold": "≤2.5s",
            }
        elif lcp <= 4000:
            assessment["metrics"]["lcp"] = {
                "status": "needs_improvement",
                "value": lcp,
                "threshold": "≤2.5s",
            }
            assessment["overall_status"] = "needs_improvement"
        else:
            assessment["metrics"]["lcp"] = {
                "status": "poor",
                "value": lcp,
                "threshold": "≤2.5s",
            }
            assessment["overall_status"] = "poor"

        # CLS assessment
        cls = lighthouse_analysis.get("cumulative_layout_shift", {}).get("value", 0)
        if cls <= 0.1:
            assessment["metrics"]["cls"] = {
                "status": "good",
                "value": cls,
                "threshold": "≤0.1",
            }
        elif cls <= 0.25:
            assessment["metrics"]["cls"] = {
                "status": "needs_improvement",
                "value": cls,
                "threshold": "≤0.1",
            }
            if assessment["overall_status"] == "good":
                assessment["overall_status"] = "needs_improvement"
        else:
            assessment["metrics"]["cls"] = {
                "status": "poor",
                "value": cls,
                "threshold": "≤0.1",
            }
            assessment["overall_status"] = "poor"

        return assessment

    def _create_performance_optimization_plan(
        self, lighthouse_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a structured performance optimization plan."""
        plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "estimated_impact": {},
        }

        performance_score = lighthouse_analysis.get("performance_score", 0)

        if performance_score < 50:
            plan["immediate_actions"].extend(
                [
                    "Audit and optimize largest images",
                    "Implement basic caching",
                    "Minify CSS and JavaScript",
                ]
            )
            plan["estimated_impact"]["performance_gain"] = "30-50 points"
        elif performance_score < 75:
            plan["short_term_actions"].extend(
                [
                    "Optimize Core Web Vitals",
                    "Implement advanced caching strategies",
                    "Optimize critical rendering path",
                ]
            )
            plan["estimated_impact"]["performance_gain"] = "15-25 points"

        plan["long_term_actions"].extend(
            [
                "Monitor performance regularly",
                "Implement performance budgets",
                "Consider Progressive Web App features",
            ]
        )

        return plan

    def _generate_optimization_priorities(
        self, summary_data: Dict[str, Any], recommendations: List
    ) -> Dict[str, Any]:
        """Generate prioritized optimization tasks."""
        priorities = {"week_1": [], "week_2_4": [], "month_2_plus": []}

        # Critical issues first
        if summary_data.get("critical_issues", 0) > 0:
            priorities["week_1"].append(
                {
                    "task": "Fix critical technical issues",
                    "impact": "High",
                    "effort": "High",
                    "count": summary_data["critical_issues"],
                }
            )

        # High priority issues
        if summary_data.get("high_priority_issues", 0) > 0:
            priorities["week_1"].append(
                {
                    "task": "Address high priority technical issues",
                    "impact": "High",
                    "effort": "Medium",
                    "count": summary_data["high_priority_issues"],
                }
            )

        # Duplicate content
        if summary_data.get("duplicate_title_tags", 0) > 5:
            priorities["week_2_4"].append(
                {
                    "task": "Fix duplicate title tags",
                    "impact": "Medium",
                    "effort": "Medium",
                    "count": summary_data["duplicate_title_tags"],
                }
            )

        return priorities

    def _estimate_optimization_impact(
        self, comprehensive_recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate the impact of implementing all recommendations."""

        current_score = comprehensive_recommendations.get("seo_score", {}).get(
            "overall_score", 0
        )

        # Estimate potential improvement based on recommendations
        critical_fixes = comprehensive_recommendations.get("summary", {}).get(
            "critical_issues", 0
        )
        high_priority_fixes = comprehensive_recommendations.get("summary", {}).get(
            "high_priority", 0
        )

        potential_improvement = min(critical_fixes * 15 + high_priority_fixes * 8, 40)
        estimated_new_score = min(100, current_score + potential_improvement)

        return {
            "current_seo_score": current_score,
            "estimated_new_score": estimated_new_score,
            "potential_improvement": potential_improvement,
            "expected_timeframe": "2-6 months",
            "confidence_level": "High" if potential_improvement > 20 else "Medium",
            "key_impact_areas": [
                "Improved search engine crawling and indexing",
                "Better user experience and page load times",
                "Higher rankings for target keywords",
                "Increased organic traffic potential",
            ],
        }

    def _get_score_improvement_recommendations(
        self, factors: Dict[str, Any]
    ) -> List[str]:
        """Get specific recommendations to improve SEO health score."""
        recommendations = []

        for factor, data in factors.items():
            impact = data.get("impact", 0)
            if impact > 10:
                if factor == "technical_issues":
                    recommendations.append(
                        "Fix critical technical issues to improve crawlability"
                    )
                elif factor == "duplicate_content":
                    recommendations.append(
                        "Eliminate duplicate content to avoid keyword cannibalization"
                    )
                elif factor == "broken_pages":
                    recommendations.append(
                        "Fix broken pages and implement proper redirects"
                    )

        return recommendations

    # Include all original helper methods for backward compatibility
    def _analyze_technical_issues(
        self, task_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze technical SEO issues from task result."""
        issues = []

        # Check for broken pages
        broken_pages = task_result.get("broken_pages", 0)
        if broken_pages > 0:
            issues.append(
                {
                    "issue_type": "broken_pages",
                    "severity": "high" if broken_pages > 10 else "medium",
                    "affected_pages": broken_pages,
                    "description": f"Found {broken_pages} broken pages (4xx/5xx status codes)",
                    "recommendation": "Fix or remove broken pages, implement proper redirects for moved content",
                }
            )

        # Check for duplicate title tags
        duplicate_titles = task_result.get("duplicate_title_tags", 0)
        if duplicate_titles > 0:
            issues.append(
                {
                    "issue_type": "duplicate_title_tags",
                    "severity": "high",
                    "affected_pages": duplicate_titles,
                    "description": f"Found {duplicate_titles} pages with duplicate title tags",
                    "recommendation": "Create unique, descriptive title tags for each page (50-60 characters)",
                }
            )

        # Check for duplicate meta descriptions
        duplicate_descriptions = task_result.get("duplicate_meta_descriptions", 0)
        if duplicate_descriptions > 0:
            issues.append(
                {
                    "issue_type": "duplicate_meta_descriptions",
                    "severity": "medium",
                    "affected_pages": duplicate_descriptions,
                    "description": f"Found {duplicate_descriptions} pages with duplicate meta descriptions",
                    "recommendation": "Write unique meta descriptions for each page (150-160 characters)",
                }
            )

        # Check for duplicate H1 tags
        duplicate_h1 = task_result.get("duplicate_h1_tags", 0)
        if duplicate_h1 > 0:
            issues.append(
                {
                    "issue_type": "duplicate_h1_tags",
                    "severity": "medium",
                    "affected_pages": duplicate_h1,
                    "description": f"Found {duplicate_h1} pages with duplicate H1 tags",
                    "recommendation": "Ensure each page has a unique, descriptive H1 tag",
                }
            )

        # Check status code distribution
        status_codes = task_result.get("pages_by_status_code", {})
        if status_codes:
            total_pages = sum(status_codes.values())
            redirect_pages = status_codes.get("3xx", 0)

            if redirect_pages > 0:
                redirect_ratio = redirect_pages / total_pages
                if redirect_ratio > 0.1:  # More than 10% redirects
                    issues.append(
                        {
                            "issue_type": "excessive_redirects",
                            "severity": "medium",
                            "affected_pages": redirect_pages,
                            "description": f"High number of redirects detected ({redirect_pages} pages, {redirect_ratio:.1%})",
                            "recommendation": "Review redirect chains and minimize unnecessary redirects",
                        }
                    )

        return issues

    def _analyze_page_issues(self, page_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Analyze individual page issues."""
        issues = []

        # Check title tag
        title = page_data.get("title", "")
        if not title:
            issues.append(
                {
                    "type": "missing_title",
                    "severity": "high",
                    "message": "Missing title tag",
                }
            )
        elif len(title) < 30:
            issues.append(
                {
                    "type": "short_title",
                    "severity": "medium",
                    "message": f"Title tag too short ({len(title)} characters)",
                }
            )
        elif len(title) > 60:
            issues.append(
                {
                    "type": "long_title",
                    "severity": "medium",
                    "message": f"Title tag too long ({len(title)} characters)",
                }
            )

        # Check meta description
        meta_desc = page_data.get("meta_description", "")
        if not meta_desc:
            issues.append(
                {
                    "type": "missing_meta_description",
                    "severity": "medium",
                    "message": "Missing meta description",
                }
            )
        elif len(meta_desc) < 120:
            issues.append(
                {
                    "type": "short_meta_description",
                    "severity": "low",
                    "message": f"Meta description too short ({len(meta_desc)} characters)",
                }
            )
        elif len(meta_desc) > 160:
            issues.append(
                {
                    "type": "long_meta_description",
                    "severity": "low",
                    "message": f"Meta description too long ({len(meta_desc)} characters)",
                }
            )

        # Check H1 tag
        h1 = page_data.get("h1", "")
        if not h1:
            issues.append(
                {"type": "missing_h1", "severity": "high", "message": "Missing H1 tag"}
            )

        # Check images without alt text
        images_without_alt = page_data.get("images_without_alt", 0)
        total_images = page_data.get("images_count", 0)
        if images_without_alt > 0 and total_images > 0:
            issues.append(
                {
                    "type": "images_missing_alt",
                    "severity": "medium",
                    "message": f"{images_without_alt}/{total_images} images missing alt text",
                }
            )

        # Check page load time
        load_time = page_data.get("load_time", 0)
        if load_time > 3000:  # 3 seconds
            issues.append(
                {
                    "type": "slow_load_time",
                    "severity": "high" if load_time > 5000 else "medium",
                    "message": f"Slow page load time ({load_time / 1000:.1f}s)",
                }
            )

        # Check content length
        word_count = page_data.get("plain_text_word_count", 0)
        if word_count < 300:
            issues.append(
                {
                    "type": "thin_content",
                    "severity": "medium",
                    "message": f"Thin content ({word_count} words)",
                }
            )

        return issues


# Backward compatibility alias for legacy imports
EnhancedOnPageAnalyzer = OnPageAnalyzer
EnhancedOnPageAnalyzer = OnPageAnalyzer
