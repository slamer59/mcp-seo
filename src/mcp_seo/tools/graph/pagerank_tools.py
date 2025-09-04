"""
PageRank analysis tools for MCP Data4SEO server.

Provides MCP tools for PageRank calculation, pillar page identification,
and internal link structure optimization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder

logger = logging.getLogger(__name__)


class PageRankRequest(BaseModel):
    """Request model for PageRank analysis."""
    domain: HttpUrl = Field(description="Domain to analyze (e.g., https://example.com)")
    max_pages: int = Field(default=100, ge=1, le=1000, description="Maximum pages to analyze")
    damping_factor: float = Field(default=0.85, ge=0.0, le=1.0, description="PageRank damping factor")
    max_iterations: int = Field(default=100, ge=1, le=1000, description="Maximum PageRank iterations")
    use_sitemap: bool = Field(default=True, description="Use sitemap.xml for page discovery")


class LinkGraphRequest(BaseModel):
    """Request model for link graph building."""
    domain: HttpUrl = Field(description="Domain to analyze")
    max_pages: int = Field(default=100, ge=1, le=1000, description="Maximum pages to crawl")
    use_sitemap: bool = Field(default=True, description="Use sitemap.xml for page discovery")
    urls: Optional[List[HttpUrl]] = Field(default=None, description="Specific URLs to analyze (if not using sitemap)")


class PillarPagesRequest(BaseModel):
    """Request model for pillar pages identification."""
    domain: HttpUrl = Field(description="Domain to analyze")
    percentile: float = Field(default=90.0, ge=50.0, le=99.0, description="Percentile threshold for pillar pages")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of pillar pages to return")


class OrphanedPagesRequest(BaseModel):
    """Request model for orphaned pages detection."""
    domain: HttpUrl = Field(description="Domain to analyze")


def register_pagerank_tools(mcp: FastMCP):
    """Register PageRank analysis tools with the MCP server."""

    @mcp.tool()
    async def analyze_pagerank(request: PageRankRequest) -> Dict:
        """
        Analyze PageRank for a website's internal link structure.
        
        This tool crawls a website, builds an internal link graph using Kuzu database,
        and calculates PageRank scores to identify authoritative pages.
        
        Returns comprehensive analysis including pillar pages, orphaned pages,
        and optimization recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting PageRank analysis for {domain_str}")
            
            # Initialize Kuzu manager with temporary database
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                # Build link graph
                graph_builder = LinkGraphBuilder(
                    base_url=domain_str,
                    kuzu_manager=kuzu_manager,
                    max_pages=request.max_pages
                )
                
                if request.use_sitemap:
                    graph_stats = await graph_builder.build_link_graph_from_sitemap()
                else:
                    # If no sitemap, start from homepage
                    graph_stats = await graph_builder.build_link_graph_from_urls([domain_str])
                
                if "error" in graph_stats:
                    return {"error": f"Failed to build link graph: {graph_stats['error']}"}
                
                # Calculate PageRank
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                pagerank_scores = pagerank_analyzer.calculate_pagerank(
                    damping_factor=request.damping_factor,
                    max_iterations=request.max_iterations
                )
                
                if not pagerank_scores:
                    return {"error": "Failed to calculate PageRank scores"}
                
                # Generate comprehensive analysis
                analysis = pagerank_analyzer.generate_analysis_summary()
                analysis['graph_statistics'] = graph_stats
                analysis['domain'] = domain_str
                analysis['parameters'] = {
                    'max_pages': request.max_pages,
                    'damping_factor': request.damping_factor,
                    'max_iterations': request.max_iterations,
                    'used_sitemap': request.use_sitemap
                }
                
                logger.info(f"PageRank analysis completed for {domain_str}")
                return analysis
                
        except Exception as e:
            logger.error(f"Error in PageRank analysis: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def build_link_graph(request: LinkGraphRequest) -> Dict:
        """
        Build internal link graph for a website.
        
        Crawls website pages and creates a graph database of internal links
        for subsequent PageRank analysis and link optimization.
        
        Returns graph statistics and basic link structure metrics.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Building link graph for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                graph_builder = LinkGraphBuilder(
                    base_url=domain_str,
                    kuzu_manager=kuzu_manager,
                    max_pages=request.max_pages
                )
                
                if request.use_sitemap:
                    graph_stats = await graph_builder.build_link_graph_from_sitemap()
                elif request.urls:
                    urls = [str(url) for url in request.urls]
                    graph_stats = await graph_builder.build_link_graph_from_urls(urls)
                else:
                    # Default to homepage
                    graph_stats = await graph_builder.build_link_graph_from_urls([domain_str])
                
                if "error" in graph_stats:
                    return {"error": f"Failed to build link graph: {graph_stats['error']}"}
                
                # Add basic link analysis
                pages_data = kuzu_manager.get_page_data()
                links_data = kuzu_manager.get_links_data()
                
                result = {
                    'domain': domain_str,
                    'graph_statistics': graph_stats,
                    'basic_metrics': {
                        'total_pages': len(pages_data),
                        'total_links': len(links_data),
                        'average_in_degree': sum(p['in_degree'] for p in pages_data) / len(pages_data) if pages_data else 0,
                        'average_out_degree': sum(p['out_degree'] for p in pages_data) / len(pages_data) if pages_data else 0,
                        'orphaned_pages_count': len([p for p in pages_data if p['in_degree'] == 0]),
                        'hub_pages_count': len([p for p in pages_data if p['out_degree'] > 10])
                    },
                    'top_pages_by_links': sorted(pages_data, key=lambda x: x['in_degree'], reverse=True)[:10]
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error building link graph: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def find_pillar_pages(request: PillarPagesRequest) -> Dict:
        """
        Identify pillar pages with high PageRank authority.
        
        Requires an existing PageRank analysis. Identifies the most authoritative
        pages based on PageRank scores for content strategy and navigation optimization.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Finding pillar pages for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                # Check if we have existing PageRank data
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No page data found. Run 'analyze_pagerank' first."}
                
                # Check if PageRank has been calculated
                if all(p['pagerank'] == 0.0 for p in pages_data):
                    return {"error": "No PageRank data found. Run 'analyze_pagerank' first."}
                
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                pillar_pages = pagerank_analyzer.get_pillar_pages(
                    percentile=request.percentile,
                    limit=request.limit
                )
                
                if not pillar_pages:
                    return {"message": "No pillar pages found with the specified criteria"}
                
                result = {
                    'domain': domain_str,
                    'pillar_pages': pillar_pages,
                    'criteria': {
                        'percentile_threshold': request.percentile,
                        'limit': request.limit
                    },
                    'recommendations': [
                        "Feature these pillar pages prominently in your main navigation",
                        "Link to pillar pages from other high-traffic content",
                        "Optimize pillar page content for target keywords",
                        "Create internal links from lower-authority pages to pillar pages"
                    ]
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error finding pillar pages: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def find_orphaned_pages(request: OrphanedPagesRequest) -> Dict:
        """
        Find orphaned pages with no incoming internal links.
        
        Identifies pages that have no incoming links and are therefore
        difficult to discover and missing potential link equity.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Finding orphaned pages for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No page data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                orphaned_pages = pagerank_analyzer.get_orphaned_pages()
                
                result = {
                    'domain': domain_str,
                    'orphaned_pages': orphaned_pages,
                    'total_orphaned': len(orphaned_pages),
                    'percentage_orphaned': (len(orphaned_pages) / len(pages_data)) * 100 if pages_data else 0,
                    'recommendations': [
                        "Add internal links to orphaned pages from relevant content",
                        "Include orphaned pages in navigation menus or sitemaps",
                        "Create topic clusters that link to orphaned pages",
                        "Review orphaned pages for content quality and relevance"
                    ]
                }
                
                if orphaned_pages:
                    # Group by domain path for better organization
                    path_groups = {}
                    for page in orphaned_pages:
                        path_parts = page['path'].split('/')
                        if len(path_parts) > 1:
                            category = path_parts[1] or 'root'
                        else:
                            category = 'root'
                        
                        if category not in path_groups:
                            path_groups[category] = []
                        path_groups[category].append(page)
                    
                    result['orphaned_by_category'] = path_groups
                
                return result
                
        except Exception as e:
            logger.error(f"Error finding orphaned pages: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def optimize_internal_links(request: LinkGraphRequest) -> Dict:
        """
        Generate internal link optimization recommendations.
        
        Analyzes current link structure and provides specific recommendations
        for improving internal linking, link equity distribution, and page discovery.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Generating link optimization recommendations for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No page data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Get link opportunities
                graph_builder = LinkGraphBuilder(domain_str, kuzu_manager)
                opportunities = graph_builder.get_link_opportunities()
                
                if "error" in opportunities:
                    return {"error": opportunities["error"]}
                
                # Generate detailed recommendations
                pagerank_analyzer = PageRankAnalyzer(kuzu_manager)
                analysis_summary = pagerank_analyzer.generate_analysis_summary()
                
                result = {
                    'domain': domain_str,
                    'link_opportunities': opportunities,
                    'optimization_plan': {
                        'priority_1_actions': [
                            {
                                'action': 'Fix orphaned pages',
                                'pages_affected': len(opportunities.get('orphaned_pages', [])),
                                'description': 'Add internal links to pages with no incoming links',
                                'impact': 'High - Improves page discovery and indexing'
                            }
                        ],
                        'priority_2_actions': [
                            {
                                'action': 'Enhance low-outlink pages',
                                'pages_affected': len(opportunities.get('low_outlink_pages', [])),
                                'description': 'Add more outgoing links from pages with few links',
                                'impact': 'Medium - Better link equity distribution'
                            }
                        ],
                        'priority_3_actions': [
                            {
                                'action': 'Leverage high-authority pages',
                                'pages_affected': len(opportunities.get('high_authority_pages', [])),
                                'description': 'Use high PageRank pages to boost other content',
                                'impact': 'Medium - Strategic link equity transfer'
                            }
                        ]
                    },
                    'general_recommendations': analysis_summary.get('recommendations', []),
                    'metrics': analysis_summary.get('metrics', {})
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return {"error": str(e)}