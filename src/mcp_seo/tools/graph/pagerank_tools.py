"""
PageRank analysis tools for MCP Data4SEO server.

This module provides both standalone functions and MCP tool registration
for PageRank analysis functionality.
"""

import logging
import re
from typing import Dict, List, Optional

# Import FastMCP and Pydantic for request models
from fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator, HttpUrl

# Import the core components
from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer

logger = logging.getLogger(__name__)


class PageRankRequest(BaseModel):
    """Request model for PageRank analysis."""

    domain: str = Field(description="Domain to analyze (e.g., https://example.com)")
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

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain_url(cls, v):
        """Validate and normalize domain URL, handling cases without scheme."""
        if not v:
            raise ValueError("Domain is required")

        url = str(v)

        # Handle domain without scheme
        if re.match(r"^localhost(:\d+)?(/.*)?$", url):
            url = f"http://{url}"
        elif re.match(r"^\d+\.\d+\.\d+\.\d+(:\d+)?(/.*)?$", url):  # IP address
            url = f"http://{url}"
        elif not url.startswith(("http://", "https://")):
            # Default to https for domain names
            url = f"https://{url}"

        # Basic URL validation
        if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url):
            raise ValueError(f"Invalid domain format: {url}")

        return url


class LinkGraphRequest(BaseModel):
    """Request model for link graph building."""

    domain: str = Field(description="Domain to analyze")
    max_pages: int = Field(
        default=100, ge=1, le=1000, description="Maximum pages to crawl"
    )
    use_sitemap: bool = Field(
        default=True, description="Use sitemap.xml for page discovery"
    )

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain_url(cls, v):
        """Validate and normalize domain URL, handling cases without scheme."""
        if not v:
            raise ValueError("Domain is required")

        url = str(v)

        # Handle domain without scheme
        if re.match(r"^localhost(:\d+)?(/.*)?$", url):
            url = f"http://{url}"
        elif re.match(r"^\d+\.\d+\.\d+\.\d+(:\d+)?(/.*)?$", url):  # IP address
            url = f"http://{url}"
        elif not url.startswith(("http://", "https://")):
            # Default to https for domain names
            url = f"https://{url}"

        # Basic URL validation
        if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url):
            raise ValueError(f"Invalid domain format: {url}")

        return url

    urls: Optional[List[HttpUrl]] = Field(
        default=None, description="Specific URLs to analyze (if not using sitemap)"
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


class OrphanedPagesRequest(BaseModel):
    """Request model for orphaned pages detection."""

    domain: HttpUrl = Field(description="Domain to analyze")


def register_pagerank_tools(mcp: FastMCP):
    """Register PageRank analysis tools with the MCP server."""

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

        This tool crawls a website, builds an internal link graph using Kuzu database,
        and calculates PageRank scores to identify authoritative pages.

        Returns comprehensive analysis including pillar pages, orphaned pages,
        and optimization recommendations.
        """
        # Create PageRankRequest from individual parameters for validation
        request = PageRankRequest(
            domain=domain,
            max_pages=max_pages,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            use_sitemap=use_sitemap,
        )
        return await analyze_pagerank_standalone(request)

    @mcp.tool()
    async def build_link_graph(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build internal link graph for a website.

        Crawls website pages and creates a graph database of internal links
        for subsequent PageRank analysis and link optimization.

        Returns graph statistics and basic link structure metrics.
        """
        # Create LinkGraphRequest from individual parameters for validation
        request = LinkGraphRequest(
            domain=domain, max_pages=max_pages, use_sitemap=use_sitemap, urls=urls
        )
        return await build_link_graph_standalone(request)

    @mcp.tool()
    async def find_pillar_pages(
        domain: str, percentile: float = 90.0, limit: int = 10
    ) -> Dict:
        """
        Identify pillar pages with high PageRank authority.

        Requires an existing PageRank analysis. Identifies the most authoritative
        pages based on PageRank scores for content strategy and navigation optimization.
        """
        # Create PillarPagesRequest from individual parameters for validation
        request = PillarPagesRequest(domain=domain, percentile=percentile, limit=limit)
        return await find_pillar_pages_standalone(request)

    @mcp.tool()
    async def find_orphaned_pages(domain: str) -> Dict:
        """
        Find orphaned pages with no incoming internal links.

        Identifies pages that have no incoming links and are therefore
        difficult to discover and missing potential link equity.
        """
        # Create OrphanedPagesRequest from individual parameters for validation
        request = OrphanedPagesRequest(domain=domain)
        return await find_orphaned_pages_standalone(request)

    @mcp.tool()
    async def optimize_internal_links(
        domain: str,
        max_pages: int = 100,
        use_sitemap: bool = True,
        urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate internal link optimization recommendations.

        Analyzes current link structure and provides specific recommendations
        for improving internal linking, link equity distribution, and page discovery.
        """
        # Create LinkGraphRequest from individual parameters for validation
        request = LinkGraphRequest(
            domain=domain, max_pages=max_pages, use_sitemap=use_sitemap, urls=urls
        )
        return await optimize_internal_links_standalone(request)


# Standalone function versions (renamed to avoid conflicts with MCP tools)
async def analyze_pagerank_standalone(request: PageRankRequest) -> Dict:
    """
    Analyze PageRank for a website's internal link structure.

    This is a standalone function version of the MCP tool.
    """
    try:
        domain_str = str(request.domain)
        logger.info(f"Starting PageRank analysis for {domain_str}")

        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()

            # Build the link graph
            graph_builder = LinkGraphBuilder(
                domain_str, kuzu_manager, request.max_pages
            )

            if request.use_sitemap:
                graph_result = await graph_builder.build_link_graph_from_sitemap()
            else:
                graph_result = await graph_builder.build_link_graph_from_urls(
                    [domain_str]
                )

            if "error" in graph_result:
                return {"error": f"Failed to build link graph: {graph_result['error']}"}

            # Calculate PageRank
            pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
            scores = pagerank_analyzer.calculate_pagerank(
                damping_factor=request.damping_factor,
                max_iterations=request.max_iterations,
            )

            if not scores:
                return {"error": "PageRank calculation failed or returned no results"}

            # Generate comprehensive analysis
            analysis_summary = pagerank_analyzer.generate_analysis_summary()

            result = {
                "domain": domain_str,
                "graph_statistics": graph_result,
                "metrics": analysis_summary.get("metrics", {}),
                "insights": analysis_summary.get("insights", {}),
                "recommendations": analysis_summary.get("recommendations", []),
                "pagerank_scores": scores,
                "parameters": {
                    "max_pages": request.max_pages,
                    "damping_factor": request.damping_factor,
                    "max_iterations": request.max_iterations,
                    "used_sitemap": request.use_sitemap,
                },
            }

            return result

    except Exception as e:
        logger.error(f"Error in PageRank analysis: {e}")
        return {"error": str(e)}


async def build_link_graph_standalone(request: LinkGraphRequest) -> Dict:
    """
    Build internal link graph for a website.

    This is a standalone function version of the MCP tool.
    """
    try:
        domain_str = str(request.domain)
        logger.info(f"Building link graph for {domain_str}")

        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()

            graph_builder = LinkGraphBuilder(
                domain_str, kuzu_manager, request.max_pages
            )

            if request.use_sitemap:
                graph_result = await graph_builder.build_link_graph_from_sitemap()
            elif request.urls:
                urls = [str(url) for url in request.urls]
                graph_result = await graph_builder.build_link_graph_from_urls(urls)
            else:
                graph_result = await graph_builder.build_link_graph_from_urls(
                    [domain_str]
                )

            if "error" in graph_result:
                return {"error": graph_result["error"]}

            # Get graph statistics
            stats = kuzu_manager.get_graph_stats()
            pages_data = kuzu_manager.get_page_data()
            links_data = kuzu_manager.get_links_data()

            # Calculate basic metrics
            basic_metrics = {
                "total_pages": len(pages_data),
                "total_links": len(links_data),
                "orphaned_pages_count": len(
                    [p for p in pages_data if p.get("in_degree", 0) == 0]
                ),
                "hub_pages_count": len(
                    [p for p in pages_data if p.get("out_degree", 0) > 5]
                ),
                "average_in_degree": sum(p.get("in_degree", 0) for p in pages_data)
                / len(pages_data)
                if pages_data
                else 0,
                "average_out_degree": sum(p.get("out_degree", 0) for p in pages_data)
                / len(pages_data)
                if pages_data
                else 0,
            }

            result = {
                "domain": domain_str,
                "graph_statistics": graph_result,
                "basic_metrics": basic_metrics,
                "top_pages_by_links": sorted(
                    pages_data, key=lambda x: x.get("in_degree", 0), reverse=True
                )[:10],
                "parameters": {
                    "max_pages": request.max_pages,
                    "used_sitemap": request.use_sitemap,
                    "custom_urls": bool(request.urls),
                },
            }

            return result

    except Exception as e:
        logger.error(f"Error building link graph: {e}")
        return {"error": str(e)}


async def find_pillar_pages_standalone(request: PillarPagesRequest) -> Dict:
    """
    Identify pillar pages with high PageRank authority.

    This is a standalone function version of the MCP tool.
    """
    try:
        domain_str = str(request.domain)
        logger.info(f"Finding pillar pages for {domain_str}")

        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()

            pages_data = kuzu_manager.get_page_data()
            if not pages_data:
                return {"error": "No page data found. Run 'analyze_pagerank' first."}

            pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
            pillar_pages = pagerank_analyzer.get_pillar_pages(
                percentile=request.percentile, limit=request.limit
            )

            result = {
                "domain": domain_str,
                "pillar_pages": pillar_pages,
                "criteria": {
                    "percentile_threshold": request.percentile,
                    "limit": request.limit,
                },
                "recommendations": [
                    "Use pillar pages as hub content for topic clusters",
                    "Link from pillar pages to related supporting content",
                    "Optimize pillar pages for competitive keywords",
                    "Ensure pillar pages have comprehensive, authoritative content",
                ],
            }

            return result

    except Exception as e:
        logger.error(f"Error finding pillar pages: {e}")
        return {"error": str(e)}


async def find_orphaned_pages_standalone(request: OrphanedPagesRequest) -> Dict:
    """
    Find orphaned pages with no incoming internal links.

    This is a standalone function version of the MCP tool.
    """
    try:
        domain_str = str(request.domain)
        logger.info(f"Finding orphaned pages for {domain_str}")

        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()

            pages_data = kuzu_manager.get_page_data()
            if not pages_data:
                return {"error": "No page data found. Run 'build_link_graph' first."}

            pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
            orphaned_pages = pagerank_analyzer.get_orphaned_pages()

            # Categorize orphaned pages
            orphaned_by_category = {}
            for page in orphaned_pages:
                path = page.get("path", page.get("url", ""))
                if "/blog/" in path:
                    category = "blog"
                elif "/product/" in path:
                    category = "product"
                elif "/category/" in path:
                    category = "category"
                else:
                    category = "other"

                if category not in orphaned_by_category:
                    orphaned_by_category[category] = []
                orphaned_by_category[category].append(page)

            total_pages = len(pages_data)
            total_orphaned = len(orphaned_pages)
            percentage_orphaned = (
                (total_orphaned / total_pages * 100) if total_pages > 0 else 0
            )

            result = {
                "domain": domain_str,
                "orphaned_pages": orphaned_pages,
                "total_orphaned": total_orphaned,
                "percentage_orphaned": round(percentage_orphaned, 2),
                "orphaned_by_category": orphaned_by_category,
                "recommendations": [
                    "Add internal links to orphaned pages from relevant content",
                    "Include orphaned pages in navigation menus or category pages",
                    "Create topic cluster pages that link to related orphaned content",
                    "Review if orphaned pages should be consolidated or removed",
                ],
            }

            return result

    except Exception as e:
        logger.error(f"Error finding orphaned pages: {e}")
        return {"error": str(e)}


async def optimize_internal_links_standalone(request: LinkGraphRequest) -> Dict:
    """
    Generate internal link optimization recommendations.

    This is a standalone function version of the MCP tool.
    """
    try:
        domain_str = str(request.domain)
        logger.info(f"Generating link optimization recommendations for {domain_str}")

        with KuzuManager() as kuzu_manager:
            kuzu_manager.initialize_schema()

            pages_data = kuzu_manager.get_page_data()
            if not pages_data:
                return {
                    "error": "No page data found. Run 'build_link_graph' or 'analyze_pagerank' first."
                }

            # Get link opportunities
            graph_builder = LinkGraphBuilder(domain_str, kuzu_manager)
            opportunities = graph_builder.get_link_opportunities()

            if "error" in opportunities:
                return {"error": opportunities["error"]}

            # Generate detailed recommendations
            pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
            analysis_summary = pagerank_analyzer.generate_analysis_summary()

            result = {
                "domain": domain_str,
                "link_opportunities": opportunities,
                "optimization_plan": {
                    "priority_1_actions": [
                        {
                            "action": "Fix orphaned pages",
                            "pages_affected": len(
                                opportunities.get("orphaned_pages", [])
                            ),
                            "description": "Add internal links to pages with no incoming links",
                            "impact": "High - Improves page discovery and indexing",
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
                        }
                    ],
                },
                "general_recommendations": analysis_summary.get("recommendations", []),
                "metrics": analysis_summary.get("metrics", {}),
            }

            return result

    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {e}")
        return {"error": str(e)}


# Backward compatibility aliases for standalone functions
analyze_pagerank = analyze_pagerank_standalone
build_link_graph = build_link_graph_standalone
find_pillar_pages = find_pillar_pages_standalone
find_orphaned_pages = find_orphaned_pages_standalone
optimize_internal_links = optimize_internal_links_standalone

# Export all the functions and classes
__all__ = [
    "PageRankRequest",
    "LinkGraphRequest",
    "PillarPagesRequest",
    "OrphanedPagesRequest",
    "analyze_pagerank",
    "build_link_graph",
    "find_pillar_pages",
    "find_orphaned_pages",
    "optimize_internal_links",
    "register_pagerank_tools",
]
