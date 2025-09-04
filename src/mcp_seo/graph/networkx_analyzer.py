"""
NetworkX-based Graph Analysis for SEO Link Structure
                                                          
Provides comprehensive graph algorithms for SEO analysis including:
- Centrality analysis (authority, hub identification)
- Community detection (content clustering) 
- Path analysis (navigation optimization)
- Structural analysis (site architecture insights)
- Connectivity analysis (link opportunity identification)
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import networkx as nx
from collections import defaultdict

from .kuzu_manager import KuzuManager

logger = logging.getLogger(__name__)


class NetworkXAnalyzer:
    """Analyzes website link structure using NetworkX graph algorithms."""

    def __init__(self, kuzu_manager: KuzuManager):
        """
        Initialize NetworkX analyzer.
        
        Args:
            kuzu_manager: Initialized KuzuManager instance
        """
        self.kuzu_manager = kuzu_manager
        self.graph: Optional[nx.DiGraph] = None
        self.undirected_graph: Optional[nx.Graph] = None
        
    def build_networkx_graph(self) -> bool:
        """
        Build NetworkX graph from Kuzu database.
        
        Returns:
            True if graph built successfully, False otherwise
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            links_data = self.kuzu_manager.get_links_data()
            
            if not pages_data or not links_data:
                logger.error("No page or link data found in database")
                return False
            
            # Create directed graph
            self.graph = nx.DiGraph()
            
            # Add nodes with page attributes
            for page in pages_data:
                self.graph.add_node(
                    page['url'],
                    title=page.get('title', ''),
                    path=page.get('path', ''),
                    status_code=page.get('status_code', 200),
                    in_degree=page.get('in_degree', 0),
                    out_degree=page.get('out_degree', 0),
                    pagerank=page.get('pagerank', 0.0)
                )
            
            # Add edges
            for link in links_data:
                if link['source_url'] in self.graph and link['target_url'] in self.graph:
                    self.graph.add_edge(link['source_url'], link['target_url'])
            
            # Create undirected version for some algorithms
            self.undirected_graph = self.graph.to_undirected()
            
            logger.info(f"Built NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error building NetworkX graph: {e}")
            return False
    
    def analyze_centrality(self) -> Dict:
        """
        Comprehensive centrality analysis for authority identification.
        
        Returns:
            Dictionary with centrality metrics and SEO insights
        """
        try:
            if not self.graph:
                if not self.build_networkx_graph():
                    return {"error": "Failed to build graph"}
            
            logger.info("Starting comprehensive centrality analysis")
            
            # Calculate different centrality measures
            centrality_metrics = {}
            
            # 1. Degree Centrality (basic link popularity)
            in_degree_centrality = nx.in_degree_centrality(self.graph)
            out_degree_centrality = nx.out_degree_centrality(self.graph)
            
            # 2. Betweenness Centrality (bridge pages)
            betweenness_centrality = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            
            # 3. Closeness Centrality (accessibility)
            try:
                # Use strongly connected components for directed graph
                largest_scc = max(nx.strongly_connected_components(self.graph), key=len)
                scc_subgraph = self.graph.subgraph(largest_scc)
                closeness_centrality = nx.closeness_centrality(scc_subgraph)
            except:
                closeness_centrality = {}
                logger.warning("Could not calculate closeness centrality for full graph")
            
            # 4. Eigenvector Centrality (authority from authoritative sources)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                eigenvector_centrality = {}
                logger.warning("Could not calculate eigenvector centrality")
            
            # 5. Katz Centrality (alternative to eigenvector)
            try:
                katz_centrality = nx.katz_centrality(self.graph, max_iter=1000)
            except:
                katz_centrality = {}
                logger.warning("Could not calculate Katz centrality")
            
            # Combine results
            all_urls = set(self.graph.nodes())
            results = {}
            
            for url in all_urls:
                node_data = self.graph.nodes[url]
                results[url] = {
                    'url': url,
                    'title': node_data.get('title', ''),
                    'path': node_data.get('path', ''),
                    'pagerank': node_data.get('pagerank', 0.0),
                    'in_degree_centrality': in_degree_centrality.get(url, 0.0),
                    'out_degree_centrality': out_degree_centrality.get(url, 0.0),
                    'betweenness_centrality': betweenness_centrality.get(url, 0.0),
                    'closeness_centrality': closeness_centrality.get(url, 0.0),
                    'eigenvector_centrality': eigenvector_centrality.get(url, 0.0),
                    'katz_centrality': katz_centrality.get(url, 0.0)
                }
            
            # Generate SEO insights
            insights = self._generate_centrality_insights(results)
            
            return {
                'centrality_analysis': results,
                'insights': insights,
                'total_pages': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in centrality analysis: {e}")
            return {"error": str(e)}
    
    def detect_communities(self, algorithm: str = 'louvain') -> Dict:
        """
        Detect content communities/clusters in the link graph.
        
        Args:
            algorithm: Community detection algorithm ('louvain', 'greedy_modularity')
            
        Returns:
            Dictionary with community structure and SEO insights
        """
        try:
            if not self.undirected_graph:
                if not self.build_networkx_graph():
                    return {"error": "Failed to build graph"}
            
            logger.info(f"Starting community detection using {algorithm}")
            
            # Apply community detection algorithm
            if algorithm == 'louvain':
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = nx_comm.louvain_communities(self.undirected_graph, seed=42)
                except ImportError:
                    # Fallback to greedy modularity if louvain not available
                    logger.warning("Louvain algorithm not available, using greedy modularity")
                    communities = nx.algorithms.community.greedy_modularity_communities(self.undirected_graph)
            else:
                communities = nx.algorithms.community.greedy_modularity_communities(self.undirected_graph)
            
            # Calculate modularity score
            modularity = nx.algorithms.community.modularity(self.undirected_graph, communities)
            
            # Process communities
            community_data = []
            for i, community in enumerate(communities):
                community_pages = []
                total_pagerank = 0.0
                paths = set()
                
                for url in community:
                    node_data = self.graph.nodes[url]
                    page_info = {
                        'url': url,
                        'title': node_data.get('title', ''),
                        'path': node_data.get('path', ''),
                        'pagerank': node_data.get('pagerank', 0.0),
                        'in_degree': node_data.get('in_degree', 0),
                        'out_degree': node_data.get('out_degree', 0)
                    }
                    community_pages.append(page_info)
                    total_pagerank += page_info['pagerank']
                    
                    # Extract path category
                    path_parts = page_info['path'].strip('/').split('/')
                    if path_parts and path_parts[0]:
                        paths.add(path_parts[0])
                
                community_data.append({
                    'community_id': i,
                    'size': len(community),
                    'pages': community_pages,
                    'total_authority': total_pagerank,
                    'avg_authority': total_pagerank / len(community) if community else 0,
                    'dominant_paths': list(paths),
                    'internal_edges': self._count_internal_edges(community),
                    'external_edges': self._count_external_edges(community)
                })
            
            # Sort by authority and size
            community_data.sort(key=lambda x: (x['total_authority'], x['size']), reverse=True)
            
            # Generate SEO insights
            insights = self._generate_community_insights(community_data, modularity)
            
            return {
                'communities': community_data,
                'modularity_score': modularity,
                'num_communities': len(communities),
                'algorithm_used': algorithm,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return {"error": str(e)}
    
    def analyze_paths(self) -> Dict:
        """
        Analyze path structure and navigation efficiency.
        
        Returns:
            Dictionary with path analysis and navigation insights
        """
        try:
            if not self.graph:
                if not self.build_networkx_graph():
                    return {"error": "Failed to build graph"}
            
            logger.info("Starting path analysis")
            
            # Find strongly connected components
            scc_components = list(nx.strongly_connected_components(self.graph))
            largest_scc = max(scc_components, key=len) if scc_components else set()
            
            # Calculate shortest paths within largest SCC (or attempt full graph)
            try:
                if len(largest_scc) > 1:
                    scc_subgraph = self.graph.subgraph(largest_scc)
                    # Sample nodes for efficiency
                    sample_nodes = list(largest_scc)[:50] if len(largest_scc) > 50 else list(largest_scc)
                    path_lengths = dict(nx.all_pairs_shortest_path_length(scc_subgraph, cutoff=6))
                else:
                    path_lengths = {}
            except Exception as e:
                logger.warning(f"Could not calculate all shortest paths: {e}")
                path_lengths = {}
            
            # Calculate eccentricity and other metrics
            metrics = {}
            if largest_scc and len(largest_scc) > 1:
                scc_subgraph = self.graph.subgraph(largest_scc)
                try:
                    # Calculate various path metrics
                    diameter = nx.diameter(scc_subgraph)
                    radius = nx.radius(scc_subgraph)
                    center_nodes = nx.center(scc_subgraph)
                    periphery_nodes = nx.periphery(scc_subgraph)
                    
                    metrics.update({
                        'diameter': diameter,
                        'radius': radius,
                        'center_nodes': list(center_nodes),
                        'periphery_nodes': list(periphery_nodes)
                    })
                except:
                    logger.warning("Could not calculate graph diameter/radius")
            
            # Calculate average path length
            if path_lengths:
                all_lengths = []
                for source_paths in path_lengths.values():
                    all_lengths.extend([length for target, length in source_paths.items() if length > 0])
                
                avg_path_length = np.mean(all_lengths) if all_lengths else 0
                max_path_length = max(all_lengths) if all_lengths else 0
                
                metrics.update({
                    'average_path_length': avg_path_length,
                    'max_path_length': max_path_length,
                    'total_path_calculations': len(all_lengths)
                })
            
            # Find pages that are difficult to reach (high eccentricity)
            hard_to_reach = []
            easy_to_reach = []
            
            for url in self.graph.nodes():
                node_data = self.graph.nodes[url]
                in_degree = node_data.get('in_degree', 0)
                
                if in_degree <= 1:
                    hard_to_reach.append({
                        'url': url,
                        'title': node_data.get('title', ''),
                        'path': node_data.get('path', ''),
                        'in_degree': in_degree,
                        'pagerank': node_data.get('pagerank', 0.0)
                    })
                elif in_degree >= 5:
                    easy_to_reach.append({
                        'url': url,
                        'title': node_data.get('title', ''),
                        'path': node_data.get('path', ''),
                        'in_degree': in_degree,
                        'pagerank': node_data.get('pagerank', 0.0)
                    })
            
            # Sort by PageRank
            hard_to_reach.sort(key=lambda x: x['pagerank'], reverse=True)
            easy_to_reach.sort(key=lambda x: x['pagerank'], reverse=True)
            
            # Generate navigation insights
            insights = self._generate_path_insights(metrics, hard_to_reach, easy_to_reach)
            
            return {
                'path_metrics': metrics,
                'strongly_connected_components': len(scc_components),
                'largest_scc_size': len(largest_scc),
                'connectivity_ratio': len(largest_scc) / len(self.graph) if self.graph else 0,
                'hard_to_reach_pages': hard_to_reach[:20],
                'easy_to_reach_pages': easy_to_reach[:20],
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in path analysis: {e}")
            return {"error": str(e)}
    
    def analyze_structure(self) -> Dict:
        """
        Analyze structural properties of the link graph.
        
        Returns:
            Dictionary with structural analysis and architecture insights
        """
        try:
            if not self.graph:
                if not self.build_networkx_graph():
                    return {"error": "Failed to build graph"}
            
            logger.info("Starting structural analysis")
            
            # K-core decomposition
            core_numbers = nx.core_number(self.undirected_graph)
            max_core = max(core_numbers.values()) if core_numbers else 0
            
            # Group pages by core number
            cores = defaultdict(list)
            for url, core_num in core_numbers.items():
                node_data = self.graph.nodes[url]
                cores[core_num].append({
                    'url': url,
                    'title': node_data.get('title', ''),
                    'path': node_data.get('path', ''),
                    'pagerank': node_data.get('pagerank', 0.0),
                    'in_degree': node_data.get('in_degree', 0),
                    'out_degree': node_data.get('out_degree', 0)
                })
            
            # Sort pages within each core by PageRank
            for core_num in cores:
                cores[core_num].sort(key=lambda x: x['pagerank'], reverse=True)
            
            # Calculate clustering coefficient
            clustering_coeffs = nx.clustering(self.undirected_graph)
            avg_clustering = np.mean(list(clustering_coeffs.values())) if clustering_coeffs else 0
            
            # Find high clustering pages (good for topic clusters)
            high_clustering_pages = []
            for url, coeff in clustering_coeffs.items():
                if coeff > avg_clustering * 1.5:  # Above average clustering
                    node_data = self.graph.nodes[url]
                    high_clustering_pages.append({
                        'url': url,
                        'title': node_data.get('title', ''),
                        'path': node_data.get('path', ''),
                        'clustering_coefficient': coeff,
                        'pagerank': node_data.get('pagerank', 0.0),
                        'degree': node_data.get('in_degree', 0) + node_data.get('out_degree', 0)
                    })
            
            high_clustering_pages.sort(key=lambda x: x['clustering_coefficient'], reverse=True)
            
            # Calculate transitivity (global clustering coefficient)
            transitivity = nx.transitivity(self.undirected_graph)
            
            # Density analysis
            density = nx.density(self.graph)
            
            # Find articulation points (critical pages for connectivity)
            articulation_points = set(nx.articulation_points(self.undirected_graph))
            critical_pages = []
            
            for url in articulation_points:
                node_data = self.graph.nodes[url]
                critical_pages.append({
                    'url': url,
                    'title': node_data.get('title', ''),
                    'path': node_data.get('path', ''),
                    'pagerank': node_data.get('pagerank', 0.0),
                    'role': 'articulation_point'
                })
            
            critical_pages.sort(key=lambda x: x['pagerank'], reverse=True)
            
            # Generate structural insights
            insights = self._generate_structural_insights(
                max_core, cores, avg_clustering, transitivity, density, 
                high_clustering_pages, critical_pages
            )
            
            return {
                'k_core_decomposition': {
                    'max_core_number': max_core,
                    'cores': dict(cores)
                },
                'clustering_analysis': {
                    'average_clustering': avg_clustering,
                    'transitivity': transitivity,
                    'high_clustering_pages': high_clustering_pages[:15]
                },
                'critical_pages': critical_pages,
                'graph_density': density,
                'structural_insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in structural analysis: {e}")
            return {"error": str(e)}
    
    def find_connector_pages(self) -> Dict:
        """
        Find pages that connect different sections of the site.
        
        Returns:
            Dictionary with connector pages and bridge opportunities
        """
        try:
            if not self.graph:
                if not self.build_networkx_graph():
                    return {"error": "Failed to build graph"}
            
            logger.info("Finding connector pages")
            
            # Calculate betweenness centrality (bridge detection)
            betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            
            # Find pages with high betweenness (good connectors)
            connector_pages = []
            for url, score in betweenness.items():
                if score > 0:  # Has some bridging function
                    node_data = self.graph.nodes[url]
                    connector_pages.append({
                        'url': url,
                        'title': node_data.get('title', ''),
                        'path': node_data.get('path', ''),
                        'betweenness_centrality': score,
                        'pagerank': node_data.get('pagerank', 0.0),
                        'in_degree': node_data.get('in_degree', 0),
                        'out_degree': node_data.get('out_degree', 0)
                    })
            
            # Sort by betweenness centrality
            connector_pages.sort(key=lambda x: x['betweenness_centrality'], reverse=True)
            
            # Identify potential bridge opportunities
            bridge_opportunities = []
            
            # Find pages with high PageRank but low betweenness (underutilized connectors)
            high_pr_pages = [p for p in connector_pages if p['pagerank'] > 0]
            high_pr_pages.sort(key=lambda x: x['pagerank'], reverse=True)
            
            for page in high_pr_pages[:20]:
                if page['betweenness_centrality'] < np.mean([p['betweenness_centrality'] for p in connector_pages]):
                    bridge_opportunities.append({
                        **page,
                        'opportunity_type': 'underutilized_authority',
                        'recommendation': 'Use this high-authority page to link between different site sections'
                    })
            
            # Find path-based categories that are poorly connected
            path_connections = defaultdict(set)
            for edge in self.graph.edges():
                source_path = self.graph.nodes[edge[0]].get('path', '').split('/')[1] if '/' in self.graph.nodes[edge[0]].get('path', '') else 'root'
                target_path = self.graph.nodes[edge[1]].get('path', '').split('/')[1] if '/' in self.graph.nodes[edge[1]].get('path', '') else 'root'
                
                if source_path != target_path:
                    path_connections[source_path].add(target_path)
            
            # Generate insights
            insights = self._generate_connector_insights(
                connector_pages, bridge_opportunities, path_connections
            )
            
            return {
                'connector_pages': connector_pages[:25],
                'bridge_opportunities': bridge_opportunities[:15],
                'cross_section_connections': dict(path_connections),
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error finding connector pages: {e}")
            return {"error": str(e)}
    
    def _count_internal_edges(self, community: Set[str]) -> int:
        """Count edges within a community."""
        count = 0
        for node in community:
            for neighbor in self.graph.neighbors(node):
                if neighbor in community:
                    count += 1
        return count
    
    def _count_external_edges(self, community: Set[str]) -> int:
        """Count edges from community to outside."""
        count = 0
        for node in community:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community:
                    count += 1
        return count
    
    def _generate_centrality_insights(self, results: Dict) -> List[str]:
        """Generate SEO insights from centrality analysis."""
        insights = []
        
        # Sort by different metrics
        by_betweenness = sorted(results.values(), key=lambda x: x['betweenness_centrality'], reverse=True)
        by_eigenvector = sorted(results.values(), key=lambda x: x['eigenvector_centrality'], reverse=True)
        by_in_degree = sorted(results.values(), key=lambda x: x['in_degree_centrality'], reverse=True)
        
        # Top bridge pages
        if by_betweenness and by_betweenness[0]['betweenness_centrality'] > 0:
            top_bridge = by_betweenness[0]
            insights.append(f"🌉 Top bridge page: '{top_bridge['title']}' - Use this page to connect different site sections")
        
        # Top authority pages
        if by_eigenvector and by_eigenvector[0]['eigenvector_centrality'] > 0:
            top_authority = by_eigenvector[0]
            insights.append(f"⭐ Highest authority page: '{top_authority['title']}' - Leverage for strategic internal linking")
        
        # Most linked pages
        if by_in_degree:
            most_linked = by_in_degree[0]
            insights.append(f"🔗 Most linked page: '{most_linked['title']}' - Already well-integrated in site structure")
        
        # Find pages with high authority but low linking
        authority_opportunity = []
        for url, data in results.items():
            if data['eigenvector_centrality'] > 0.01 and data['in_degree_centrality'] < 0.05:
                authority_opportunity.append(data)
        
        if authority_opportunity:
            insights.append(f"💡 Found {len(authority_opportunity)} high-authority pages with low internal linking - linking opportunities")
        
        return insights
    
    def _generate_community_insights(self, communities: List[Dict], modularity: float) -> List[str]:
        """Generate SEO insights from community detection."""
        insights = []
        
        insights.append(f"📊 Detected {len(communities)} content communities with modularity score {modularity:.3f}")
        
        if modularity > 0.3:
            insights.append("✅ Good content clustering - clear topic separation")
        else:
            insights.append("⚠️ Weak content clustering - consider organizing content into clearer topic clusters")
        
        # Largest community analysis
        if communities:
            largest = communities[0]
            insights.append(f"🏆 Largest community: {largest['size']} pages with {largest['total_authority']:.3f} total authority")
            
            if largest['dominant_paths']:
                paths_str = ', '.join(largest['dominant_paths'])
                insights.append(f"📁 Dominant content areas in main cluster: {paths_str}")
        
        # Find isolated communities
        small_communities = [c for c in communities if c['size'] <= 3]
        if small_communities:
            insights.append(f"🔍 Found {len(small_communities)} small communities - potential content silos to integrate")
        
        # Authority distribution
        total_auth = sum(c['total_authority'] for c in communities)
        if communities:
            auth_concentration = communities[0]['total_authority'] / total_auth if total_auth > 0 else 0
            if auth_concentration > 0.5:
                insights.append("⚖️ Authority concentrated in one community - consider distributing link equity")
            else:
                insights.append("✅ Authority well-distributed across content communities")
        
        return insights
    
    def _generate_path_insights(self, metrics: Dict, hard_to_reach: List[Dict], easy_to_reach: List[Dict]) -> List[str]:
        """Generate SEO insights from path analysis."""
        insights = []
        
        if 'average_path_length' in metrics:
            avg_length = metrics['average_path_length']
            insights.append(f"🚀 Average navigation distance: {avg_length:.2f} clicks")
            
            if avg_length > 4:
                insights.append("⚠️ High average path length - users need too many clicks to reach content")
            elif avg_length < 2.5:
                insights.append("✅ Excellent navigation efficiency - content easily accessible")
        
        if 'diameter' in metrics:
            diameter = metrics['diameter']
            insights.append(f"📏 Site diameter: {diameter} clicks maximum")
            
            if diameter > 6:
                insights.append("🔄 Consider adding cross-links to reduce maximum navigation distance")
        
        # Hard to reach pages
        if hard_to_reach:
            valuable_orphans = [p for p in hard_to_reach if p['pagerank'] > 0.001]
            if valuable_orphans:
                insights.append(f"🎯 {len(valuable_orphans)} valuable pages are hard to reach - add internal links")
                insights.append(f"   → Priority: '{valuable_orphans[0]['title']}'")
        
        # Navigation hubs
        if easy_to_reach:
            top_hub = easy_to_reach[0]
            insights.append(f"🏢 Main navigation hub: '{top_hub['title']}' ({top_hub['in_degree']} incoming links)")
        
        connectivity_ratio = metrics.get('connectivity_ratio', 0)
        if connectivity_ratio < 0.8:
            insights.append(f"🔗 Only {connectivity_ratio:.1%} of pages are well-connected - improve internal linking")
        else:
            insights.append("✅ Good site connectivity - most pages are reachable")
        
        return insights
    
    def _generate_structural_insights(self, max_core: int, cores: Dict, avg_clustering: float, 
                                    transitivity: float, density: float, high_clustering_pages: List[Dict],
                                    critical_pages: List[Dict]) -> List[str]:
        """Generate SEO insights from structural analysis."""
        insights = []
        
        insights.append(f"🏗️ Site structure: {max_core}-core maximum, {len(cores)} different core levels")
        
        if max_core >= 3:
            insights.append("✅ Strong core structure - good foundation for link equity distribution")
        else:
            insights.append("⚠️ Weak core structure - consider adding more interconnected content hubs")
        
        # Clustering analysis
        insights.append(f"🔗 Average clustering coefficient: {avg_clustering:.3f}")
        if avg_clustering > 0.3:
            insights.append("✅ Good local connectivity - pages form coherent topic clusters")
        else:
            insights.append("💡 Low clustering - add more links between related pages")
        
        # Topic cluster opportunities
        if high_clustering_pages:
            top_cluster = high_clustering_pages[0]
            insights.append(f"🎯 Best topic cluster hub: '{top_cluster['title']}' (clustering: {top_cluster['clustering_coefficient']:.3f})")
        
        # Critical pages
        if critical_pages:
            critical_count = len(critical_pages)
            insights.append(f"🔑 {critical_count} critical connectivity pages - ensure these are well-maintained")
            if critical_count > 0:
                top_critical = critical_pages[0]
                insights.append(f"   → Most critical: '{top_critical['title']}'")
        
        # Density analysis
        insights.append(f"📊 Graph density: {density:.4f}")
        if density < 0.01:
            insights.append("📈 Low link density - many opportunities for internal linking")
        elif density > 0.05:
            insights.append("⚠️ High link density - might be over-linking, focus on quality links")
        
        return insights
    
    def _generate_connector_insights(self, connector_pages: List[Dict], 
                                   bridge_opportunities: List[Dict], 
                                   path_connections: Dict) -> List[str]:
        """Generate SEO insights for connector pages."""
        insights = []
        
        if connector_pages:
            top_connector = connector_pages[0]
            insights.append(f"🌉 Top connector page: '{top_connector['title']}' (bridging score: {top_connector['betweenness_centrality']:.4f})")
            insights.append("   → This page is crucial for site navigation flow")
        
        # Bridge opportunities
        if bridge_opportunities:
            insights.append(f"💡 {len(bridge_opportunities)} high-authority pages could serve as better connectors")
            top_opportunity = bridge_opportunities[0]
            insights.append(f"   → Priority: '{top_opportunity['title']}' - high PageRank but underused as connector")
        
        # Cross-section analysis
        well_connected_sections = sum(1 for connections in path_connections.values() if len(connections) >= 3)
        total_sections = len(path_connections)
        
        if total_sections > 0:
            connection_ratio = well_connected_sections / total_sections
            insights.append(f"🔗 {well_connected_sections}/{total_sections} site sections are well-connected")
            
            if connection_ratio < 0.5:
                insights.append("⚠️ Many site sections are isolated - add cross-section navigation links")
        
        # Poorly connected sections
        isolated_sections = [section for section, connections in path_connections.items() 
                           if len(connections) <= 1]
        if isolated_sections:
            insights.append(f"🏝️ Isolated sections: {', '.join(isolated_sections[:5])}")
            insights.append("   → Add navigation links between these and main site areas")
        
        return insights