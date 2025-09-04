"""
NetworkX analysis tools for MCP Data4SEO server.

Provides MCP tools for comprehensive graph analysis using NetworkX algorithms
including centrality analysis, community detection, path optimization,
structural analysis, and connector page identification.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field, HttpUrl

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.networkx_analyzer import NetworkXAnalyzer

logger = logging.getLogger(__name__)


class NetworkXAnalysisRequest(BaseModel):
    """Request model for NetworkX analysis."""
    domain: HttpUrl = Field(description="Domain to analyze (must have existing link graph)")


class CentralityAnalysisRequest(BaseModel):
    """Request model for centrality analysis."""
    domain: HttpUrl = Field(description="Domain to analyze")
    metrics: List[str] = Field(
        default=["betweenness", "closeness", "eigenvector"], 
        description="Centrality metrics to calculate"
    )
    top_k: int = Field(default=20, ge=1, le=100, description="Number of top pages to return per metric")


class CommunityDetectionRequest(BaseModel):
    """Request model for community detection."""
    domain: HttpUrl = Field(description="Domain to analyze")
    algorithm: str = Field(default="louvain", description="Algorithm to use (louvain, greedy_modularity)")
    min_community_size: int = Field(default=3, ge=1, description="Minimum community size to report")


class ConnectorAnalysisRequest(BaseModel):
    """Request model for connector pages analysis."""
    domain: HttpUrl = Field(description="Domain to analyze")
    min_betweenness: float = Field(default=0.0, ge=0.0, description="Minimum betweenness centrality threshold")


def register_networkx_tools(mcp: FastMCP):
    """Register NetworkX analysis tools with the MCP server."""

    @mcp.tool()
    async def analyze_centrality(request: CentralityAnalysisRequest) -> Dict:
        """
        Comprehensive centrality analysis for authority identification.
        
        Analyzes multiple centrality measures to identify:
        - Authority pages (high eigenvector/Katz centrality)
        - Bridge pages (high betweenness centrality)  
        - Hub pages (high degree centrality)
        - Accessible pages (high closeness centrality)
        
        Returns rankings, insights, and strategic recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting centrality analysis for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                # Check if we have link graph data
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Perform centrality analysis
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                analysis_results = networkx_analyzer.analyze_centrality()
                
                if "error" in analysis_results:
                    return analysis_results
                
                # Process results to get top pages per metric
                centrality_data = analysis_results['centrality_analysis']
                top_pages = {}
                
                # Get top pages for each requested metric
                for metric in request.metrics:
                    metric_key = f"{metric}_centrality"
                    if metric_key in list(centrality_data.values())[0]:  # Check if metric exists
                        sorted_pages = sorted(
                            centrality_data.values(), 
                            key=lambda x: x.get(metric_key, 0), 
                            reverse=True
                        )
                        top_pages[metric] = sorted_pages[:request.top_k]
                
                # Generate SEO recommendations
                recommendations = _generate_centrality_recommendations(top_pages, centrality_data)
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'centrality',
                    'metrics_analyzed': request.metrics,
                    'top_pages_by_metric': top_pages,
                    'total_pages_analyzed': analysis_results['total_pages'],
                    'seo_insights': analysis_results['insights'],
                    'recommendations': recommendations
                }
                
                logger.info(f"Centrality analysis completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error in centrality analysis: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def detect_communities(request: CommunityDetectionRequest) -> Dict:
        """
        Content community detection for topic clustering.
        
        Identifies content clusters and topic groups using community detection:
        - Louvain algorithm for high-quality communities
        - Greedy modularity for interpretable clusters
        - Content category analysis
        - Internal vs external linking patterns
        
        Returns community structure, topic insights, and content strategy recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting community detection for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Perform community detection
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                community_results = networkx_analyzer.detect_communities(request.algorithm)
                
                if "error" in community_results:
                    return community_results
                
                # Filter communities by minimum size
                filtered_communities = [
                    community for community in community_results['communities']
                    if community['size'] >= request.min_community_size
                ]
                
                # Generate content strategy recommendations
                content_recommendations = _generate_community_recommendations(
                    filtered_communities, 
                    community_results['modularity_score']
                )
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'community_detection',
                    'algorithm_used': request.algorithm,
                    'modularity_score': community_results['modularity_score'],
                    'total_communities': community_results['num_communities'],
                    'communities': filtered_communities,
                    'content_insights': community_results['insights'],
                    'content_strategy': content_recommendations
                }
                
                logger.info(f"Community detection completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def analyze_site_structure(request: NetworkXAnalysisRequest) -> Dict:
        """
        Comprehensive site architecture analysis.
        
        Analyzes structural properties for architecture optimization:
        - K-core decomposition (site hierarchy strength)
        - Clustering coefficient (topic coherence)
        - Critical connectivity points
        - Link density analysis
        
        Returns structural insights and architecture recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting structural analysis for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Perform structural analysis
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                structural_results = networkx_analyzer.analyze_structure()
                
                if "error" in structural_results:
                    return structural_results
                
                # Generate architecture recommendations
                architecture_recommendations = _generate_structure_recommendations(structural_results)
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'site_structure',
                    'k_core_analysis': structural_results['k_core_decomposition'],
                    'clustering_analysis': structural_results['clustering_analysis'],
                    'critical_pages': structural_results['critical_pages'],
                    'graph_density': structural_results['graph_density'],
                    'structural_insights': structural_results['structural_insights'],
                    'architecture_recommendations': architecture_recommendations
                }
                
                logger.info(f"Structural analysis completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error in structural analysis: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def analyze_navigation_paths(request: NetworkXAnalysisRequest) -> Dict:
        """
        Navigation path analysis for UX optimization.
        
        Analyzes path efficiency and navigation structure:
        - Average path lengths between pages
        - Site diameter and radius
        - Hard-to-reach valuable content
        - Navigation bottlenecks
        
        Returns path metrics and navigation improvement recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting path analysis for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Perform path analysis
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                path_results = networkx_analyzer.analyze_paths()
                
                if "error" in path_results:
                    return path_results
                
                # Generate navigation recommendations
                navigation_recommendations = _generate_navigation_recommendations(path_results)
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'navigation_paths',
                    'path_metrics': path_results['path_metrics'],
                    'connectivity_analysis': {
                        'strongly_connected_components': path_results['strongly_connected_components'],
                        'largest_scc_size': path_results['largest_scc_size'],
                        'connectivity_ratio': path_results['connectivity_ratio']
                    },
                    'navigation_issues': {
                        'hard_to_reach_pages': path_results['hard_to_reach_pages'],
                        'easy_to_reach_pages': path_results['easy_to_reach_pages'][:10]  # Limit for readability
                    },
                    'path_insights': path_results['insights'],
                    'navigation_recommendations': navigation_recommendations
                }
                
                logger.info(f"Path analysis completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error in path analysis: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def find_connector_pages(request: ConnectorAnalysisRequest) -> Dict:
        """
        Identify connector pages and bridge opportunities.
        
        Finds pages that connect different site sections:
        - High betweenness centrality pages (natural bridges)
        - Underutilized high-authority pages
        - Cross-section connection opportunities
        - Navigation hub optimization
        
        Returns connector pages and internal linking opportunities.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Finding connector pages for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Find connector pages
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                connector_results = networkx_analyzer.find_connector_pages()
                
                if "error" in connector_results:
                    return connector_results
                
                # Filter by minimum betweenness
                filtered_connectors = [
                    page for page in connector_results['connector_pages']
                    if page['betweenness_centrality'] >= request.min_betweenness
                ]
                
                # Generate linking strategy
                linking_strategy = _generate_linking_strategy(
                    filtered_connectors,
                    connector_results['bridge_opportunities'],
                    connector_results['cross_section_connections']
                )
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'connector_analysis',
                    'connector_pages': filtered_connectors,
                    'bridge_opportunities': connector_results['bridge_opportunities'],
                    'cross_section_analysis': {
                        'connections': connector_results['cross_section_connections'],
                        'connection_matrix': _build_connection_matrix(
                            connector_results['cross_section_connections']
                        )
                    },
                    'connector_insights': connector_results['insights'],
                    'linking_strategy': linking_strategy
                }
                
                logger.info(f"Connector analysis completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error finding connector pages: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def comprehensive_graph_analysis(request: NetworkXAnalysisRequest) -> Dict:
        """
        Complete NetworkX graph analysis combining all algorithms.
        
        Provides comprehensive SEO graph analysis including:
        - Authority identification (centrality analysis)
        - Content clustering (community detection)
        - Navigation optimization (path analysis)
        - Architecture insights (structural analysis)
        - Connector opportunities (bridge analysis)
        
        Returns unified insights and prioritized recommendations.
        """
        try:
            domain_str = str(request.domain)
            logger.info(f"Starting comprehensive graph analysis for {domain_str}")
            
            with KuzuManager() as kuzu_manager:
                kuzu_manager.initialize_schema()
                
                pages_data = kuzu_manager.get_page_data()
                if not pages_data:
                    return {"error": "No link graph data found. Run 'build_link_graph' or 'analyze_pagerank' first."}
                
                # Initialize analyzer
                networkx_analyzer = NetworkXAnalyzer(kuzu_manager)
                
                # Run all analyses
                results = {}
                
                # 1. Centrality Analysis
                try:
                    results['centrality'] = networkx_analyzer.analyze_centrality()
                except Exception as e:
                    results['centrality'] = {"error": str(e)}
                    logger.warning(f"Centrality analysis failed: {e}")
                
                # 2. Community Detection
                try:
                    results['communities'] = networkx_analyzer.detect_communities()
                except Exception as e:
                    results['communities'] = {"error": str(e)}
                    logger.warning(f"Community detection failed: {e}")
                
                # 3. Path Analysis
                try:
                    results['paths'] = networkx_analyzer.analyze_paths()
                except Exception as e:
                    results['paths'] = {"error": str(e)}
                    logger.warning(f"Path analysis failed: {e}")
                
                # 4. Structural Analysis
                try:
                    results['structure'] = networkx_analyzer.analyze_structure()
                except Exception as e:
                    results['structure'] = {"error": str(e)}
                    logger.warning(f"Structural analysis failed: {e}")
                
                # 5. Connector Analysis
                try:
                    results['connectors'] = networkx_analyzer.find_connector_pages()
                except Exception as e:
                    results['connectors'] = {"error": str(e)}
                    logger.warning(f"Connector analysis failed: {e}")
                
                # Generate unified recommendations
                unified_recommendations = _generate_unified_recommendations(results)
                
                # Calculate analysis summary
                analysis_summary = _generate_analysis_summary(results)
                
                result = {
                    'domain': domain_str,
                    'analysis_type': 'comprehensive_graph_analysis',
                    'total_pages_analyzed': len(pages_data),
                    'analysis_summary': analysis_summary,
                    'detailed_results': results,
                    'unified_insights': _extract_unified_insights(results),
                    'prioritized_recommendations': unified_recommendations
                }
                
                logger.info(f"Comprehensive graph analysis completed for {domain_str}")
                return result
                
        except Exception as e:
            logger.error(f"Error in comprehensive graph analysis: {e}")
            return {"error": str(e)}


def _generate_centrality_recommendations(top_pages: Dict, centrality_data: Dict) -> List[Dict]:
    """Generate specific SEO recommendations from centrality analysis."""
    recommendations = []
    
    if 'betweenness' in top_pages and top_pages['betweenness']:
        top_bridge = top_pages['betweenness'][0]
        recommendations.append({
            'priority': 'high',
            'type': 'navigation_optimization',
            'action': 'Leverage top bridge page for cross-linking',
            'target_page': top_bridge['title'],
            'description': f"Use '{top_bridge['title']}' as a hub to connect different site sections",
            'expected_impact': 'Improved link equity distribution and user navigation'
        })
    
    if 'eigenvector' in top_pages and top_pages['eigenvector']:
        top_authority = top_pages['eigenvector'][0]
        recommendations.append({
            'priority': 'high',
            'type': 'authority_leverage',
            'action': 'Strategic internal linking from authority page',
            'target_page': top_authority['title'],
            'description': f"Add internal links from high-authority page '{top_authority['title']}' to boost other content",
            'expected_impact': 'Increased PageRank for linked pages'
        })
    
    # Find underutilized authorities
    authorities_low_bridges = []
    for url, data in centrality_data.items():
        if (data['eigenvector_centrality'] > 0.01 and 
            data['betweenness_centrality'] < 0.01):
            authorities_low_bridges.append(data)
    
    if authorities_low_bridges:
        recommendations.append({
            'priority': 'medium',
            'type': 'opportunity_identification',
            'action': 'Utilize underused authority pages as connectors',
            'description': f"Found {len(authorities_low_bridges)} high-authority pages that could serve as better navigation bridges",
            'expected_impact': 'Better link equity distribution across site sections'
        })
    
    return recommendations


def _generate_community_recommendations(communities: List[Dict], modularity: float) -> List[Dict]:
    """Generate content strategy recommendations from community analysis."""
    recommendations = []
    
    if modularity < 0.3:
        recommendations.append({
            'priority': 'high',
            'type': 'content_organization',
            'action': 'Improve content clustering',
            'description': f"Low modularity score ({modularity:.3f}) indicates weak content organization",
            'expected_impact': 'Better topic authority and user experience'
        })
    
    # Large communities analysis
    large_communities = [c for c in communities if c['size'] > 10]
    if large_communities:
        for community in large_communities[:3]:
            recommendations.append({
                'priority': 'medium',
                'type': 'topic_optimization',
                'action': f"Optimize {community['size']}-page content cluster",
                'description': f"Strengthen internal linking within {', '.join(community['dominant_paths'])} topic area",
                'expected_impact': 'Enhanced topical authority and content discoverability'
            })
    
    # Small isolated communities
    small_communities = [c for c in communities if c['size'] <= 3 and c['external_edges'] == 0]
    if small_communities:
        recommendations.append({
            'priority': 'medium',
            'type': 'content_integration',
            'action': 'Connect isolated content groups',
            'description': f"Integrate {len(small_communities)} isolated content groups with main site areas",
            'expected_impact': 'Improved content discoverability and link equity flow'
        })
    
    return recommendations


def _generate_structure_recommendations(structural_results: Dict) -> List[Dict]:
    """Generate architecture recommendations from structural analysis."""
    recommendations = []
    
    max_core = structural_results['k_core_decomposition']['max_core_number']
    
    if max_core < 3:
        recommendations.append({
            'priority': 'high',
            'type': 'architecture_strengthening',
            'action': 'Build stronger content core',
            'description': f"Current {max_core}-core is weak. Add more interconnected hub pages",
            'expected_impact': 'Stronger site architecture and better link equity distribution'
        })
    
    density = structural_results['graph_density']
    if density < 0.01:
        recommendations.append({
            'priority': 'medium',
            'type': 'internal_linking',
            'action': 'Increase internal link density',
            'description': f"Low link density ({density:.4f}) indicates missed internal linking opportunities",
            'expected_impact': 'Improved navigation and SEO value distribution'
        })
    
    critical_pages = structural_results['critical_pages']
    if critical_pages:
        recommendations.append({
            'priority': 'high',
            'type': 'critical_page_maintenance',
            'action': 'Monitor critical connectivity pages',
            'description': f"Ensure {len(critical_pages)} critical pages remain accessible and well-maintained",
            'expected_impact': 'Maintained site connectivity and navigation flow'
        })
    
    return recommendations


def _generate_navigation_recommendations(path_results: Dict) -> List[Dict]:
    """Generate navigation recommendations from path analysis."""
    recommendations = []
    
    if 'average_path_length' in path_results['path_metrics']:
        avg_length = path_results['path_metrics']['average_path_length']
        if avg_length > 4:
            recommendations.append({
                'priority': 'high',
                'type': 'navigation_efficiency',
                'action': 'Reduce navigation path lengths',
                'description': f"Average path length of {avg_length:.2f} is too high for optimal UX",
                'expected_impact': 'Improved user experience and content discoverability'
            })
    
    hard_to_reach = path_results['hard_to_reach_pages']
    valuable_hard_to_reach = [p for p in hard_to_reach if p['pagerank'] > 0.001]
    
    if valuable_hard_to_reach:
        recommendations.append({
            'priority': 'high',
            'type': 'content_accessibility',
            'action': 'Improve access to valuable content',
            'description': f"Add internal links to {len(valuable_hard_to_reach)} valuable but hard-to-reach pages",
            'expected_impact': 'Better content utilization and improved user journey'
        })
    
    connectivity_ratio = path_results['connectivity_ratio']
    if connectivity_ratio < 0.8:
        recommendations.append({
            'priority': 'medium',
            'type': 'site_connectivity',
            'action': 'Improve overall site connectivity',
            'description': f"Only {connectivity_ratio:.1%} of pages are well-connected",
            'expected_impact': 'Better internal link structure and crawlability'
        })
    
    return recommendations


def _generate_linking_strategy(connectors: List[Dict], opportunities: List[Dict], 
                              cross_sections: Dict) -> List[Dict]:
    """Generate internal linking strategy recommendations."""
    strategy = []
    
    if connectors:
        top_connector = connectors[0]
        strategy.append({
            'strategy_type': 'leverage_existing_bridges',
            'action': f"Use '{top_connector['title']}' as primary navigation hub",
            'details': f"This page already serves as a bridge (score: {top_connector['betweenness_centrality']:.4f})",
            'implementation': 'Add more strategic outbound links from this page to important content'
        })
    
    if opportunities:
        strategy.append({
            'strategy_type': 'utilize_underused_authorities',
            'action': f"Convert {len(opportunities)} high-authority pages into bridges",
            'details': 'These pages have high PageRank but low connector function',
            'implementation': 'Add strategic internal links from these pages to connect different site areas'
        })
    
    # Cross-section connectivity
    isolated_sections = [section for section, connections in cross_sections.items() 
                        if len(connections) <= 1]
    if isolated_sections:
        strategy.append({
            'strategy_type': 'connect_isolated_sections',
            'action': f"Integrate {len(isolated_sections)} isolated site sections",
            'details': f"Sections: {', '.join(isolated_sections[:5])}",
            'implementation': 'Create navigation links and contextual links between these sections and main content areas'
        })
    
    return strategy


def _build_connection_matrix(cross_sections: Dict) -> Dict:
    """Build a connection matrix showing which sections link to which."""
    all_sections = set(cross_sections.keys())
    for connections in cross_sections.values():
        all_sections.update(connections)
    
    matrix = {}
    for section in all_sections:
        matrix[section] = {
            'outgoing_connections': list(cross_sections.get(section, [])),
            'incoming_connections': []
        }
    
    # Fill incoming connections
    for source, targets in cross_sections.items():
        for target in targets:
            if target in matrix:
                matrix[target]['incoming_connections'].append(source)
    
    return matrix


def _generate_unified_recommendations(results: Dict) -> List[Dict]:
    """Generate prioritized recommendations from all analyses."""
    unified_recommendations = []
    
    # High priority: Critical structural issues
    if 'structure' in results and 'error' not in results['structure']:
        critical_pages = results['structure'].get('critical_pages', [])
        if critical_pages:
            unified_recommendations.append({
                'priority': 1,
                'category': 'critical_infrastructure',
                'title': 'Protect critical connectivity pages',
                'description': f"Monitor and maintain {len(critical_pages)} critical pages that are essential for site connectivity",
                'impact': 'high',
                'effort': 'low'
            })
    
    # High priority: Navigation efficiency
    if 'paths' in results and 'error' not in results['paths']:
        path_metrics = results['paths'].get('path_metrics', {})
        if path_metrics.get('average_path_length', 0) > 4:
            unified_recommendations.append({
                'priority': 1,
                'category': 'navigation_optimization',
                'title': 'Improve navigation efficiency',
                'description': 'Average path length is too high - add strategic internal links to reduce clicks to content',
                'impact': 'high',
                'effort': 'medium'
            })
    
    # Medium priority: Authority utilization
    if 'centrality' in results and 'error' not in results['centrality']:
        unified_recommendations.append({
            'priority': 2,
            'category': 'authority_optimization',
            'title': 'Leverage high-authority pages',
            'description': 'Use pages with high centrality scores to boost other content through strategic internal linking',
            'impact': 'high',
            'effort': 'medium'
        })
    
    # Medium priority: Content organization
    if 'communities' in results and 'error' not in results['communities']:
        modularity = results['communities'].get('modularity_score', 0)
        if modularity < 0.3:
            unified_recommendations.append({
                'priority': 2,
                'category': 'content_organization',
                'title': 'Improve content clustering',
                'description': 'Low modularity indicates weak content organization - create clearer topic clusters',
                'impact': 'medium',
                'effort': 'high'
            })
    
    # Lower priority: Bridge opportunities
    if 'connectors' in results and 'error' not in results['connectors']:
        opportunities = results['connectors'].get('bridge_opportunities', [])
        if opportunities:
            unified_recommendations.append({
                'priority': 3,
                'category': 'link_optimization',
                'title': 'Utilize underused bridge opportunities',
                'description': f"Convert {len(opportunities)} high-authority pages into better connectors between site sections",
                'impact': 'medium',
                'effort': 'medium'
            })
    
    return sorted(unified_recommendations, key=lambda x: x['priority'])


def _generate_analysis_summary(results: Dict) -> Dict:
    """Generate overall analysis summary."""
    summary = {
        'analyses_completed': [],
        'analyses_failed': [],
        'key_metrics': {},
        'overall_health': 'unknown'
    }
    
    for analysis_type, result in results.items():
        if 'error' in result:
            summary['analyses_failed'].append(analysis_type)
        else:
            summary['analyses_completed'].append(analysis_type)
    
    # Extract key metrics
    if 'centrality' in results and 'error' not in results['centrality']:
        summary['key_metrics']['total_pages'] = results['centrality'].get('total_pages', 0)
    
    if 'communities' in results and 'error' not in results['communities']:
        summary['key_metrics']['modularity_score'] = results['communities'].get('modularity_score', 0)
        summary['key_metrics']['num_communities'] = results['communities'].get('num_communities', 0)
    
    if 'paths' in results and 'error' not in results['paths']:
        path_metrics = results['paths'].get('path_metrics', {})
        summary['key_metrics']['avg_path_length'] = path_metrics.get('average_path_length', 0)
        summary['key_metrics']['connectivity_ratio'] = results['paths'].get('connectivity_ratio', 0)
    
    if 'structure' in results and 'error' not in results['structure']:
        summary['key_metrics']['max_core_number'] = results['structure']['k_core_decomposition']['max_core_number']
        summary['key_metrics']['graph_density'] = results['structure']['graph_density']
    
    # Determine overall health
    health_score = 0
    total_checks = 0
    
    # Navigation health
    if 'avg_path_length' in summary['key_metrics']:
        total_checks += 1
        if summary['key_metrics']['avg_path_length'] <= 3:
            health_score += 1
    
    # Content organization health
    if 'modularity_score' in summary['key_metrics']:
        total_checks += 1
        if summary['key_metrics']['modularity_score'] >= 0.3:
            health_score += 1
    
    # Connectivity health
    if 'connectivity_ratio' in summary['key_metrics']:
        total_checks += 1
        if summary['key_metrics']['connectivity_ratio'] >= 0.8:
            health_score += 1
    
    # Structure health
    if 'max_core_number' in summary['key_metrics']:
        total_checks += 1
        if summary['key_metrics']['max_core_number'] >= 3:
            health_score += 1
    
    if total_checks > 0:
        health_ratio = health_score / total_checks
        if health_ratio >= 0.8:
            summary['overall_health'] = 'excellent'
        elif health_ratio >= 0.6:
            summary['overall_health'] = 'good'
        elif health_ratio >= 0.4:
            summary['overall_health'] = 'fair'
        else:
            summary['overall_health'] = 'poor'
    
    return summary


def _extract_unified_insights(results: Dict) -> List[str]:
    """Extract key insights from all analyses."""
    insights = []
    
    for analysis_type, result in results.items():
        if 'insights' in result:
            insights.extend(result['insights'])
        elif analysis_type == 'structure' and 'structural_insights' in result:
            insights.extend(result['structural_insights'])
        elif analysis_type == 'connectors' and 'connector_insights' in result:
            insights.extend(result['connector_insights'])
        elif analysis_type == 'paths' and 'path_insights' in result:
            insights.extend(result['path_insights'])
        elif analysis_type == 'communities' and 'content_insights' in result:
            insights.extend(result['content_insights'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_insights = []
    for insight in insights:
        if insight not in seen:
            seen.add(insight)
            unique_insights.append(insight)
    
    return unique_insights