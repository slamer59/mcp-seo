"""
Enhanced PageRank analysis tools for MCP Data4SEO server with content analysis integration.

Provides MCP tools for PageRank calculation, pillar page identification,
internal link structure optimization, and content strategy recommendations.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.engines import SEORecommendationEngine
from mcp_seo.reporting import SEOReporter
from mcp_seo.content_analysis import BlogContentOptimizer

logger = logging.getLogger(__name__)


class PageRankRequest(BaseModel):
    """Request model for PageRank analysis."""

    domain: HttpUrl = Field(description="Domain to analyze (e.g., https://example.com)")
    max_pages: int = Field(
        default=100, ge=1, le=1000, description="Maximum pages to analyze"
    )
    damping_factor: float = Field(
        default=0.85, ge=0.0, le=1.0, description="PageRank damping factor"
    )
    max_iterations: int = Field(
        default=100, ge=1, le=1000, description="Maximum PageRank iterations"
    )
    use_sitemap: bool = Field(
        default=True, description="Use sitemap.xml for page discovery"
    )
    include_content_analysis: bool = Field(
        default=True, description="Include content optimization recommendations"
    )


class LinkGraphRequest(BaseModel):
    """Request model for link graph building."""

    domain: HttpUrl = Field(description="Domain to analyze")
    max_pages: int = Field(
        default=100, ge=1, le=1000, description="Maximum pages to crawl"
    )
    use_sitemap: bool = Field(
        default=True, description="Use sitemap.xml for page discovery"
    )
    urls: Optional[List[HttpUrl]] = Field(
        default=None, description="Specific URLs to analyze (if not using sitemap)"
    )
    include_content_strategy: bool = Field(
        default=True, description="Include content strategy recommendations"
    )


class PillarPagesRequest(BaseModel):
    """Request model for pillar pages identification."""

    domain: HttpUrl = Field(description="Domain to analyze")
    percentile: float = Field(
        default=90.0,
        ge=50.0,
        le=99.0,
        description="Percentile threshold for pillar pages",
    )
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum number of pillar pages to return"
    )
    include_content_recommendations: bool = Field(
        default=True,
        description="Include content optimization recommendations for pillar pages",
    )


class OrphanedPagesRequest(BaseModel):
    """Request model for orphaned pages detection."""

    domain: HttpUrl = Field(description="Domain to analyze")
    include_content_audit: bool = Field(
        default=True,
        description="Include content audit recommendations for orphaned pages",
    )


def register_enhanced_pagerank_tools(mcp: FastMCP):
    """Register enhanced PageRank analysis tools with the MCP server."""

    @mcp.tool()
    async def analyze_pagerank_with_content(
        domain: str,
        max_pages: int = 100,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        use_sitemap: bool = True,
        include_content_analysis: bool = True,
    ) -> Dict:
        """
        Analyze PageRank for a website's internal link structure with content optimization recommendations.

        This enhanced tool crawls a website, builds an internal link graph using Kuzu database,
        calculates PageRank scores, and provides comprehensive content strategy recommendations.

        Returns analysis including pillar pages, orphaned pages, content optimization opportunities,
        and strategic recommendations for improving both link structure and content quality.
        """
        try:
            # Create PageRankRequest from individual parameters for validation
            request = PageRankRequest(
                domain=domain,
                max_pages=max_pages,
                damping_factor=damping_factor,
                max_iterations=max_iterations,
                use_sitemap=use_sitemap,
                include_content_analysis=include_content_analysis,
            )

            domain_str = str(request.domain)
            logger.info(f"Starting enhanced PageRank analysis for {domain_str}")

            # Initialize components
            recommendation_engine = SEORecommendationEngine()
            reporter = SEOReporter(use_rich=True)
            content_optimizer = BlogContentOptimizer()

            # Initialize Kuzu manager with temporary database
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()

                # Build link graph
                graph_builder = LinkGraphBuilder(
                    base_url=domain_str,
                    kuzu_manager=kuzu_manager,
                    max_pages=request.max_pages,
                )

                if request.use_sitemap:
                    graph_stats = await graph_builder.build_link_graph_from_sitemap()
                else:
                    # If no sitemap, start from homepage
                    graph_stats = await graph_builder.build_link_graph_from_urls(
                        [domain_str]
                    )

                if "error" in graph_stats:
                    return {
                        "error": f"Failed to build link graph: {graph_stats['error']}"
                    }

                # Calculate PageRank
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                pagerank_scores = pagerank_analyzer.calculate_pagerank(
                    damping_factor=request.damping_factor,
                    max_iterations=request.max_iterations,
                )

                if not pagerank_scores:
                    return {"error": "Failed to calculate PageRank scores"}

                # Generate comprehensive analysis
                analysis = pagerank_analyzer.generate_analysis_summary()
                analysis["graph_statistics"] = graph_stats
                analysis["domain"] = domain_str
                analysis["parameters"] = {
                    "max_pages": request.max_pages,
                    "damping_factor": request.damping_factor,
                    "max_iterations": request.max_iterations,
                    "used_sitemap": request.use_sitemap,
                }

                # Add enhanced SEO recommendations using the new engine
                link_recommendations = recommendation_engine.analyze_link_opportunities(
                    analysis
                )
                analysis["seo_recommendations"] = [
                    rec.__dict__ for rec in link_recommendations
                ]

                # Add formatted report using the new reporter
                analysis["formatted_report"] = (
                    reporter.generate_pagerank_analysis_report(analysis)
                )

                # Add content optimization suggestions for high-authority pages
                if include_content_analysis and "pillar_pages" in analysis:
                    content_suggestions = []
                    for page in analysis["pillar_pages"][:5]:  # Top 5 pillar pages
                        page_url = page.get("url", "")
                        # Generate content optimization suggestions for pillar pages
                        suggestions = {
                            "url": page_url,
                            "pagerank_score": page.get("pagerank", 0),
                            "optimization_opportunities": [
                                "Expand content with comprehensive information on key topics",
                                "Add internal links to related lower-authority pages",
                                "Optimize for featured snippets with structured content",
                                "Create topic clusters around this pillar page",
                                "Ensure mobile-friendly content formatting",
                                "Add multimedia elements to enhance user engagement",
                            ],
                            "content_strategy": {
                                "keyword_targeting": "Focus on broad, high-volume keywords",
                                "content_depth": "Create comprehensive, authoritative content",
                                "internal_linking": "Hub for related topic clusters",
                                "update_frequency": "Regular updates to maintain freshness",
                            },
                        }
                        content_suggestions.append(suggestions)
                    analysis["content_optimization_suggestions"] = content_suggestions

                # Add content performance metrics
                if include_content_analysis:
                    analysis["content_performance_insights"] = (
                        _generate_content_performance_insights(analysis)
                    )

                logger.info(f"Enhanced PageRank analysis completed for {domain_str}")
                return analysis

        except Exception as e:
            logger.error(f"Error in enhanced PageRank analysis: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def build_link_graph_with_strategy(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
        include_content_strategy: bool = True,
    ) -> Dict:
        """
        Build internal link graph for a website with comprehensive content strategy recommendations.

        Crawls website pages and creates a graph database of internal links, then provides
        detailed content strategy recommendations for improving both link structure and content quality.

        Returns graph statistics, link structure metrics, and strategic content recommendations.
        """
        try:
            # Create LinkGraphRequest from individual parameters for validation
            request = LinkGraphRequest(
                domain=domain,
                max_pages=max_pages,
                use_sitemap=use_sitemap,
                urls=urls,
                include_content_strategy=include_content_strategy,
            )

            domain_str = str(request.domain)
            logger.info(f"Building enhanced link graph for {domain_str}")

            # Initialize components
            recommendation_engine = SEORecommendationEngine()
            content_optimizer = BlogContentOptimizer()

            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()

                graph_builder = LinkGraphBuilder(
                    base_url=domain_str,
                    kuzu_manager=kuzu_manager,
                    max_pages=request.max_pages,
                )

                if request.use_sitemap:
                    graph_stats = await graph_builder.build_link_graph_from_sitemap()
                elif request.urls:
                    urls_list = [str(url) for url in request.urls]
                    graph_stats = await graph_builder.build_link_graph_from_urls(
                        urls_list
                    )
                else:
                    # Default to homepage
                    graph_stats = await graph_builder.build_link_graph_from_urls(
                        [domain_str]
                    )

                if "error" in graph_stats:
                    return {
                        "error": f"Failed to build link graph: {graph_stats['error']}"
                    }

                # Add basic link analysis
                pages_data = kuzu_manager.get_page_data()
                links_data = kuzu_manager.get_links_data()

                basic_metrics = {
                    "total_pages": len(pages_data),
                    "total_links": len(links_data),
                    "average_in_degree": (
                        sum(p["in_degree"] for p in pages_data) / len(pages_data)
                        if pages_data
                        else 0
                    ),
                    "average_out_degree": (
                        sum(p["out_degree"] for p in pages_data) / len(pages_data)
                        if pages_data
                        else 0
                    ),
                    "orphaned_pages_count": len(
                        [p for p in pages_data if p["in_degree"] == 0]
                    ),
                    "hub_pages_count": len(
                        [p for p in pages_data if p["out_degree"] > 10]
                    ),
                }

                result = {
                    "domain": domain_str,
                    "graph_statistics": graph_stats,
                    "basic_metrics": basic_metrics,
                    "top_pages_by_links": sorted(
                        pages_data, key=lambda x: x["in_degree"], reverse=True
                    )[:10],
                }

                # Add enhanced content strategy recommendations
                if include_content_strategy:
                    # Add link structure analysis and recommendations
                    link_structure_data = {
                        "orphaned_pages": [
                            p for p in pages_data if p["in_degree"] == 0
                        ],
                        "basic_metrics": basic_metrics,
                    }
                    link_recommendations = (
                        recommendation_engine.analyze_link_opportunities(
                            link_structure_data
                        )
                    )
                    result["link_structure_recommendations"] = [
                        rec.__dict__ for rec in link_recommendations
                    ]

                    # Add content strategy recommendations for hub pages
                    hub_pages = [p for p in pages_data if p["out_degree"] > 10][:5]
                    if hub_pages:
                        content_strategy = {
                            "hub_page_optimization": {
                                "count": len(hub_pages),
                                "recommendations": [
                                    "Use hub pages to boost authority of target content",
                                    "Ensure hub pages link to comprehensive, valuable content",
                                    "Optimize hub page content for broad, high-value keywords",
                                    "Create clear navigation paths from hub pages to topic clusters",
                                    "Implement strategic anchor text for outgoing links",
                                    "Monitor hub page performance and user engagement",
                                ],
                                "top_hub_pages": hub_pages,
                                "optimization_framework": {
                                    "content_development": [
                                        "Create comprehensive, authoritative content on hub pages",
                                        "Develop supporting content that naturally links to hub pages",
                                        "Establish clear topic hierarchies and content clusters",
                                    ],
                                    "link_strategy": [
                                        "Use descriptive, keyword-rich anchor text",
                                        "Balance outgoing links to avoid over-optimization",
                                        "Create natural linking patterns that serve users",
                                    ],
                                    "performance_monitoring": [
                                        "Track hub page rankings and traffic",
                                        "Monitor click-through rates on outgoing links",
                                        "Analyze user engagement and time on page",
                                    ],
                                },
                            }
                        }
                        result["content_strategy"] = content_strategy

                return result

        except Exception as e:
            logger.error(f"Error building enhanced link graph: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def find_pillar_pages_with_content_strategy(
        domain: str,
        percentile: float = 90.0,
        limit: int = 10,
        include_content_recommendations: bool = True,
    ) -> Dict:
        """
        Identify pillar pages with high PageRank authority and comprehensive content strategy.

        Requires an existing PageRank analysis. Identifies the most authoritative pages and provides
        detailed content optimization strategies for maximizing their SEO potential.
        """
        try:
            # Create PillarPagesRequest from individual parameters for validation
            request = PillarPagesRequest(
                domain=domain,
                percentile=percentile,
                limit=limit,
                include_content_recommendations=include_content_recommendations,
            )

            domain_str = str(request.domain)
            logger.info(f"Finding pillar pages with content strategy for {domain_str}")

            # Initialize components
            content_optimizer = BlogContentOptimizer()

            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()

                # Check if we have existing PageRank data
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {
                        "error": "No page data found. Run 'analyze_pagerank_with_content' first."
                    }

                # Check if PageRank has been calculated
                if all(p["pagerank"] == 0.0 for p in pages_data):
                    return {
                        "error": "No PageRank data found. Run 'analyze_pagerank_with_content' first."
                    }

                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                pillar_pages = pagerank_analyzer.get_pillar_pages(
                    percentile=request.percentile, limit=request.limit
                )

                if not pillar_pages:
                    return {
                        "message": "No pillar pages found with the specified criteria"
                    }

                # Generate enhanced content strategy for pillar pages
                pillar_content_strategy = {
                    "content_expansion": [
                        "Create comprehensive, authoritative content on pillar pages",
                        "Add FAQ sections to address related questions",
                        "Include multimedia content (images, videos, infographics)",
                        "Update content regularly to maintain freshness",
                        "Develop in-depth guides and tutorials",
                        "Add user-generated content where appropriate",
                    ],
                    "keyword_strategy": [
                        "Target high-volume, broad keywords on pillar pages",
                        "Create supporting content for long-tail variations",
                        "Optimize for featured snippets with structured content",
                        "Use semantic keyword variations throughout content",
                        "Implement topic modeling for comprehensive coverage",
                        "Monitor keyword rankings and adjust strategy",
                    ],
                    "internal_linking": [
                        "Link from pillar pages to related cluster content",
                        "Ensure all cluster content links back to pillar pages",
                        "Use descriptive, keyword-rich anchor text",
                        "Create clear topic hierarchies with internal links",
                        "Implement strategic link placement for maximum impact",
                        "Balance link equity distribution across content clusters",
                    ],
                    "user_experience": [
                        "Optimize page load speeds for pillar pages",
                        "Ensure mobile-responsive design",
                        "Implement clear navigation and site structure",
                        "Add interactive elements to increase engagement",
                        "Optimize for Core Web Vitals",
                        "Create accessible content for all users",
                    ],
                }

                result = {
                    "domain": domain_str,
                    "pillar_pages": pillar_pages,
                    "criteria": {
                        "percentile_threshold": request.percentile,
                        "limit": request.limit,
                    },
                    "recommendations": [
                        "Feature these pillar pages prominently in your main navigation",
                        "Link to pillar pages from other high-traffic content",
                        "Optimize pillar page content for target keywords",
                        "Create internal links from lower-authority pages to pillar pages",
                    ],
                    "content_strategy": pillar_content_strategy,
                    "seo_optimization_plan": {
                        "immediate_actions": [
                            "Audit existing pillar page content quality",
                            "Optimize title tags and meta descriptions",
                            "Add internal links to pillar pages from relevant content",
                            "Ensure proper heading structure (H1, H2, H3)",
                            "Optimize images with descriptive alt text",
                        ],
                        "content_development": [
                            "Expand thin pillar page content",
                            "Create supporting cluster content",
                            "Develop content calendars around pillar topics",
                            "Add multimedia elements to enhance engagement",
                            "Create downloadable resources and tools",
                        ],
                        "performance_monitoring": [
                            "Track pillar page rankings for target keywords",
                            "Monitor internal link click-through rates",
                            "Analyze user engagement metrics on pillar pages",
                            "Measure conversion rates and goal completions",
                            "Track social media shares and backlink acquisition",
                        ],
                    },
                }

                # Add individual pillar page optimization recommendations
                if include_content_recommendations:
                    detailed_recommendations = []
                    for page in pillar_pages:
                        page_recommendations = {
                            "url": page.get("url", ""),
                            "pagerank_score": page.get("pagerank", 0),
                            "in_degree": page.get("in_degree", 0),
                            "out_degree": page.get("out_degree", 0),
                            "optimization_checklist": [
                                "Conduct keyword research for page-specific optimization",
                                "Analyze competitor content for this topic",
                                "Create comprehensive content outline",
                                "Optimize for search intent and user journey",
                                "Implement structured data markup",
                                "Plan content update schedule",
                            ],
                            "content_enhancement": {
                                "depth_improvement": "Add comprehensive sections covering all aspects of the topic",
                                "multimedia_integration": "Include relevant images, videos, and interactive elements",
                                "user_engagement": "Add calls-to-action and conversion elements",
                                "freshness_strategy": "Plan regular updates and content refreshes",
                            },
                        }
                        detailed_recommendations.append(page_recommendations)

                    result["detailed_page_recommendations"] = detailed_recommendations

                return result

        except Exception as e:
            logger.error(f"Error finding pillar pages: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def find_orphaned_pages_with_content_audit(
        domain: str, include_content_audit: bool = True
    ) -> Dict:
        """
        Find orphaned pages with no incoming internal links and comprehensive content audit.

        Identifies pages that have no incoming links and provides detailed content audit
        recommendations for integrating them into the site's link structure effectively.
        """
        try:
            # Create OrphanedPagesRequest from individual parameters for validation
            request = OrphanedPagesRequest(
                domain=domain, include_content_audit=include_content_audit
            )

            domain_str = str(request.domain)
            logger.info(f"Finding orphaned pages with content audit for {domain_str}")

            # Initialize components
            content_optimizer = BlogContentOptimizer()

            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()

                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {
                        "error": "No page data found. Run 'build_link_graph_with_strategy' or 'analyze_pagerank_with_content' first."
                    }

                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                orphaned_pages = pagerank_analyzer.get_orphaned_pages()

                # Enhanced orphaned page analysis with content recommendations
                orphaned_analysis = {
                    "impact_assessment": {
                        "seo_impact": (
                            "High"
                            if len(orphaned_pages) > len(pages_data) * 0.1
                            else "Medium"
                        ),
                        "content_discovery_issues": len(orphaned_pages),
                        "potential_traffic_loss": (
                            "Significant" if len(orphaned_pages) > 20 else "Moderate"
                        ),
                        "indexing_challenges": "Pages may not be properly indexed by search engines",
                        "user_experience_impact": "Users cannot discover content through site navigation",
                    },
                    "content_optimization_strategy": {
                        "content_audit": [
                            "Review orphaned pages for content quality and relevance",
                            "Identify valuable content that should be preserved",
                            "Consider consolidating thin orphaned pages",
                            "Update outdated content before re-linking",
                            "Assess content uniqueness and value proposition",
                            "Evaluate content alignment with business objectives",
                        ],
                        "linking_strategy": [
                            "Identify thematically related pages for contextual linking",
                            "Create topic cluster maps to guide internal linking",
                            "Use descriptive anchor text for new internal links",
                            "Prioritize linking from high-authority pages",
                            "Implement breadcrumb navigation where appropriate",
                            "Add orphaned pages to relevant category listings",
                        ],
                        "content_integration": [
                            "Include orphaned pages in content calendars",
                            "Create hub pages that can link to related orphaned content",
                            "Develop content series that naturally link to orphaned pages",
                            "Add orphaned pages to relevant category and tag pages",
                            "Create cross-promotional opportunities within existing content",
                            "Implement related content recommendations",
                        ],
                    },
                    "technical_optimization": [
                        "Ensure orphaned pages are included in XML sitemaps",
                        "Verify proper URL structure and redirects",
                        "Implement structured data where appropriate",
                        "Optimize page load speeds for orphaned content",
                        "Ensure mobile-friendly design",
                        "Add proper meta tags and descriptions",
                    ],
                }

                result = {
                    "domain": domain_str,
                    "orphaned_pages": orphaned_pages,
                    "total_orphaned": len(orphaned_pages),
                    "percentage_orphaned": (
                        (len(orphaned_pages) / len(pages_data)) * 100
                        if pages_data
                        else 0
                    ),
                    "recommendations": [
                        "Add internal links to orphaned pages from relevant content",
                        "Include orphaned pages in navigation menus or sitemaps",
                        "Create topic clusters that link to orphaned pages",
                        "Review orphaned pages for content quality and relevance",
                    ],
                    "enhanced_analysis": orphaned_analysis,
                }

                if orphaned_pages:
                    # Group by domain path for better organization
                    path_groups = {}
                    for page in orphaned_pages:
                        path_parts = page["path"].split("/")
                        if len(path_parts) > 1:
                            category = path_parts[1] or "root"
                        else:
                            category = "root"

                        if category not in path_groups:
                            path_groups[category] = []
                        path_groups[category].append(page)

                    result["orphaned_by_category"] = path_groups

                    # Add prioritization framework
                    if include_content_audit:
                        prioritization = {
                            "high_priority": [],
                            "medium_priority": [],
                            "low_priority": [],
                        }

                        for page in orphaned_pages:
                            # Simple prioritization based on URL structure and potential value
                            path = page.get("path", "")

                            # High priority: Main content pages
                            if any(
                                indicator in path
                                for indicator in [
                                    "/blog/",
                                    "/article/",
                                    "/guide/",
                                    "/tutorial/",
                                ]
                            ):
                                prioritization["high_priority"].append(page)
                            # Medium priority: Product or service pages
                            elif any(
                                indicator in path
                                for indicator in [
                                    "/product/",
                                    "/service/",
                                    "/solution/",
                                ]
                            ):
                                prioritization["medium_priority"].append(page)
                            # Low priority: Other pages
                            else:
                                prioritization["low_priority"].append(page)

                        result["prioritization"] = prioritization

                        # Add action plan
                        result["action_plan"] = {
                            "week_1": "Audit and prioritize high-value orphaned content",
                            "week_2_4": "Create internal links from relevant existing content",
                            "month_2_3": "Develop content clusters and hub pages for orphaned content",
                            "ongoing": "Monitor orphaned page discovery and implement systematic linking strategy",
                        }

                return result

        except Exception as e:
            logger.error(f"Error finding orphaned pages: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def optimize_internal_links_with_content_strategy(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate comprehensive internal link optimization recommendations with content strategy.

        Analyzes current link structure and provides specific recommendations for improving
        internal linking, link equity distribution, page discovery, and content optimization.
        """
        try:
            # Create LinkGraphRequest from individual parameters for validation
            request = LinkGraphRequest(
                domain=domain, max_pages=max_pages, use_sitemap=use_sitemap, urls=urls
            )

            domain_str = str(request.domain)
            logger.info(
                f"Generating comprehensive link optimization recommendations for {domain_str}"
            )

            # Initialize components
            recommendation_engine = SEORecommendationEngine()
            content_optimizer = BlogContentOptimizer()

            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()

                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {
                        "error": "No page data found. Run 'build_link_graph_with_strategy' or 'analyze_pagerank_with_content' first."
                    }

                # Get link opportunities
                graph_builder = LinkGraphBuilder(domain_str, kuzu_manager)
                opportunities = graph_builder.get_link_opportunities()

                if "error" in opportunities:
                    return {"error": opportunities["error"]}

                # Generate detailed recommendations
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                analysis_summary = pagerank_analyzer.generate_analysis_summary()

                # Enhanced optimization plan with content strategy
                enhanced_optimization_plan = {
                    "priority_1_actions": [
                        {
                            "action": "Fix orphaned pages",
                            "pages_affected": len(
                                opportunities.get("orphaned_pages", [])
                            ),
                            "description": "Add internal links to pages with no incoming links",
                            "impact": "High - Improves page discovery and indexing",
                            "content_strategy": {
                                "audit_orphaned_content": "Review content quality before linking",
                                "create_hub_pages": "Develop topic hubs that can link to orphaned content",
                                "contextual_linking": "Add links from thematically related pages",
                                "navigation_integration": "Include valuable orphaned pages in site navigation",
                            },
                        }
                    ],
                    "priority_2_actions": [
                        {
                            "action": "Enhance low-outlink pages",
                            "pages_affected": len(
                                opportunities.get("low_outlink_pages", [])
                            ),
                            "description": "Add more outgoing links from pages with few links",
                            "impact": "Medium - Better link equity distribution",
                            "content_strategy": {
                                "content_expansion": "Add related resources and references",
                                "topic_clustering": "Link to related content within topic clusters",
                                "user_value": "Ensure all added links provide user value",
                                "anchor_optimization": "Use descriptive, keyword-rich anchor text",
                            },
                        }
                    ],
                    "priority_3_actions": [
                        {
                            "action": "Leverage high-authority pages",
                            "pages_affected": len(
                                opportunities.get("high_authority_pages", [])
                            ),
                            "description": "Use high PageRank pages to boost other content",
                            "impact": "Medium - Strategic link equity transfer",
                            "content_strategy": {
                                "strategic_linking": "Link from authority pages to target content",
                                "content_optimization": "Ensure linked-to pages are optimized",
                                "monitoring": "Track ranking improvements from authority links",
                                "topic_relevance": "Maintain topical relevance in linking strategy",
                            },
                        }
                    ],
                    "content_optimization_framework": {
                        "topic_cluster_development": [
                            "Create pillar pages for main topics",
                            "Develop supporting cluster content",
                            "Establish clear topic hierarchies",
                            "Use consistent internal linking patterns",
                            "Implement semantic keyword strategies",
                            "Create content calendars around topic clusters",
                        ],
                        "content_quality_enhancement": [
                            "Audit and improve thin content pages",
                            "Add multimedia elements to key pages",
                            "Optimize for featured snippets",
                            "Ensure mobile-friendly content formatting",
                            "Implement structured data markup",
                            "Create comprehensive resource pages",
                        ],
                        "technical_optimization": [
                            "Optimize page load speeds",
                            "Implement structured data markup",
                            "Ensure proper URL structure",
                            "Monitor crawl budget allocation",
                            "Optimize images and media files",
                            "Implement proper redirect strategies",
                        ],
                        "user_experience_focus": [
                            "Create intuitive navigation pathways",
                            "Implement breadcrumb navigation",
                            "Add related content recommendations",
                            "Optimize for mobile user experience",
                            "Include clear calls-to-action",
                            "Monitor user engagement metrics",
                        ],
                    },
                }

                result = {
                    "domain": domain_str,
                    "link_opportunities": opportunities,
                    "optimization_plan": enhanced_optimization_plan,
                    "general_recommendations": analysis_summary.get(
                        "recommendations", []
                    ),
                    "metrics": analysis_summary.get("metrics", {}),
                    "content_seo_integration": {
                        "content_audit_checklist": [
                            "Review all pages for content quality and relevance",
                            "Identify keyword optimization opportunities",
                            "Assess internal linking structure",
                            "Evaluate user engagement metrics",
                            "Analyze content performance data",
                            "Check for content gaps and opportunities",
                        ],
                        "seo_content_calendar": [
                            "Plan content updates around link building opportunities",
                            "Schedule regular content audits and optimizations",
                            "Coordinate new content creation with link strategy",
                            "Monitor and adjust based on performance metrics",
                            "Align content creation with seasonal trends",
                            "Plan content promotion and distribution strategies",
                        ],
                        "measurement_framework": [
                            "Track internal link click-through rates",
                            "Monitor page authority improvements",
                            "Measure organic traffic changes",
                            "Analyze user engagement improvements",
                            "Track keyword ranking improvements",
                            "Monitor conversion rate impacts",
                        ],
                    },
                }

                return result

        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return {"error": str(e)}


def _generate_content_performance_insights(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate content performance insights from PageRank analysis."""
    insights = {
        "authority_distribution": {},
        "content_gaps": [],
        "optimization_opportunities": [],
        "strategic_recommendations": [],
    }

    # Analyze authority distribution
    if "basic_metrics" in analysis:
        metrics = analysis["basic_metrics"]
        total_pages = metrics.get("total_pages", 0)
        orphaned_count = metrics.get("orphaned_pages_count", 0)
        hub_count = metrics.get("hub_pages_count", 0)

        insights["authority_distribution"] = {
            "total_pages": total_pages,
            "orphaned_ratio": (orphaned_count / total_pages) if total_pages > 0 else 0,
            "hub_ratio": (hub_count / total_pages) if total_pages > 0 else 0,
            "average_connectivity": metrics.get("average_in_degree", 0),
        }

        # Identify content gaps
        if orphaned_count > total_pages * 0.1:
            insights["content_gaps"].append(
                "High number of orphaned pages indicates poor content integration"
            )

        if hub_count < total_pages * 0.05:
            insights["content_gaps"].append(
                "Limited hub pages - opportunity to create authoritative content"
            )

        # Generate optimization opportunities
        insights["optimization_opportunities"] = [
            "Develop content clusters around high-authority pages",
            "Create topic hubs to connect related content",
            "Optimize internal linking for better content discovery",
            "Implement strategic content planning based on PageRank insights",
        ]

        # Strategic recommendations
        insights["strategic_recommendations"] = [
            "Focus content creation on topics with high internal linking potential",
            "Develop comprehensive guides that can serve as hub pages",
            "Create content series that naturally link to each other",
            "Implement topic modeling for better content organization",
        ]

    return insights


# Backward compatibility - register original function names
def register_pagerank_tools(mcp: FastMCP):
    """Register PageRank analysis tools with backward compatibility."""
    register_enhanced_pagerank_tools(mcp)

    # Keep the original tool names for backward compatibility
    @mcp.tool()
    async def analyze_pagerank(
        domain: str,
        max_pages: int = 100,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        use_sitemap: bool = True,
    ) -> Dict:
        """
        Analyze PageRank for a website's internal link structure.
        (Backward compatible version)
        """
        return await analyze_pagerank_with_content(
            domain=domain,
            max_pages=max_pages,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            use_sitemap=use_sitemap,
            include_content_analysis=False,  # Disable by default for backward compatibility
        )

    @mcp.tool()
    async def build_link_graph(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build internal link graph for a website.
        (Backward compatible version)
        """
        return await build_link_graph_with_strategy(
            domain=domain,
            max_pages=max_pages,
            use_sitemap=use_sitemap,
            urls=urls,
            include_content_strategy=False,  # Disable by default for backward compatibility
        )

    @mcp.tool()
    async def find_pillar_pages(
        domain: str, percentile: float = 90.0, limit: int = 10
    ) -> Dict:
        """
        Identify pillar pages with high PageRank authority.
        (Backward compatible version)
        """
        return await find_pillar_pages_with_content_strategy(
            domain=domain,
            percentile=percentile,
            limit=limit,
            include_content_recommendations=False,  # Disable by default for backward compatibility
        )

    @mcp.tool()
    async def find_orphaned_pages(domain: str) -> Dict:
        """
        Find orphaned pages with no incoming internal links.
        (Backward compatible version)
        """
        return await find_orphaned_pages_with_content_audit(
            domain=domain,
            include_content_audit=False,  # Disable by default for backward compatibility
        )

    @mcp.tool()
    async def optimize_internal_links(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate internal link optimization recommendations.
        (Backward compatible version)
        """
        return await optimize_internal_links_with_content_strategy(
            domain=domain, max_pages=max_pages, use_sitemap=use_sitemap, urls=urls
        )
