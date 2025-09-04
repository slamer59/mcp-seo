"""
NetworkX-based network visualization for link graphs and PageRank analysis.
"""

import logging
import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse
import base64

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from PIL import Image

from mcp_seo.graph.kuzu_manager import KuzuManager

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """
    Visualizes link graphs and PageRank scores using NetworkX and Matplotlib.
    """
    
    def __init__(self, kuzu_manager: KuzuManager):
        """Initialize the network visualizer."""
        self.kuzu_manager = kuzu_manager
        
        # Set up matplotlib for better rendering
        plt.style.use('seaborn-v0_8')
        
        # Color palettes
        self.color_palette = sns.color_palette("viridis", as_cmap=True)
        self.node_colors = {
            'pillar': '#FF6B6B',      # Red for high PageRank
            'content': '#4ECDC4',     # Teal for normal pages
            'orphaned': '#95A5A6',    # Gray for orphaned pages
            'entry': '#F39C12'        # Orange for entry pages
        }
        
    def create_networkx_graph(self, include_pagerank: bool = True) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from Kuzu data.
        
        Args:
            include_pagerank: Whether to include PageRank scores as node attributes
            
        Returns:
            NetworkX directed graph
        """
        try:
            # Create directed graph
            G = nx.DiGraph()
            
            # Get all pages
            pages_data = self.kuzu_manager.get_page_data()
            
            # Add nodes with attributes
            for page in pages_data:
                node_id = page['url']
                attributes = {
                    'url': page['url'],
                    'title': page.get('title', ''),
                    'domain': urlparse(page['url']).netloc,
                    'path': urlparse(page['url']).path,
                    'depth': page.get('depth', 0)
                }
                
                if include_pagerank and 'pagerank_score' in page:
                    attributes['pagerank'] = page['pagerank_score']
                
                G.add_node(node_id, **attributes)
            
            # Get all links
            links_data = self.kuzu_manager.get_links_data()
            
            # Add edges
            for link in links_data:
                source = link['source_url']
                target = link['target_url']
                
                # Only add edge if both nodes exist
                if source in G.nodes() and target in G.nodes():
                    edge_attributes = {
                        'anchor_text': link.get('anchor_text', ''),
                        'link_type': link.get('link_type', 'internal')
                    }
                    G.add_edge(source, target, **edge_attributes)
            
            logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error creating NetworkX graph: {e}")
            raise
    
    def calculate_layout_positions(self, G: nx.DiGraph, layout: str = "spring") -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions for visualization layout.
        
        Args:
            G: NetworkX graph
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai')
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        try:
            if layout == "spring":
                # Spring layout with PageRank influence if available
                if any('pagerank' in G.nodes[node] for node in G.nodes()):
                    # Weight positions by PageRank
                    pagerank_weights = {node: G.nodes[node].get('pagerank', 0.1) for node in G.nodes()}
                    pos = nx.spring_layout(G, k=3, iterations=50, weight=None)
                else:
                    pos = nx.spring_layout(G, k=3, iterations=50)
                    
            elif layout == "circular":
                pos = nx.circular_layout(G)
                
            elif layout == "hierarchical":
                pos = nx.multipartite_layout(G, subset_key='depth')
                
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
                
            else:
                # Default to spring layout
                pos = nx.spring_layout(G)
            
            return pos
            
        except Exception as e:
            logger.error(f"Error calculating layout positions: {e}")
            # Fallback to random positions
            return nx.random_layout(G)
    
    def categorize_nodes(self, G: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Categorize nodes based on their characteristics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping categories to lists of node IDs
        """
        categories = {
            'pillar': [],      # High PageRank or many incoming links
            'content': [],     # Regular content pages
            'orphaned': [],    # No incoming internal links
            'entry': []        # High depth or entry points
        }
        
        # Calculate basic metrics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        for node in G.nodes():
            node_data = G.nodes[node]
            pagerank = node_data.get('pagerank', 0)
            in_degree = in_degrees.get(node, 0)
            depth = node_data.get('depth', 0)
            
            # Categorize based on characteristics
            if pagerank > 0.1 or in_degree > 10:  # High authority
                categories['pillar'].append(node)
            elif in_degree == 0:  # Orphaned
                categories['orphaned'].append(node)
            elif depth <= 1:  # Entry points
                categories['entry'].append(node)
            else:  # Regular content
                categories['content'].append(node)
        
        return categories
    
    def create_pagerank_visualization(
        self,
        output_path: str,
        layout: str = "spring",
        figsize: Tuple[int, int] = (16, 12),
        show_labels: bool = False,
        highlight_pillars: bool = True,
        node_size_scale: float = 1000
    ) -> str:
        """
        Create a PageRank visualization and save as PNG.
        
        Args:
            output_path: Path to save the PNG file
            layout: Layout algorithm
            figsize: Figure size (width, height)
            show_labels: Whether to show node labels
            highlight_pillars: Whether to highlight pillar pages
            node_size_scale: Scale factor for node sizes
            
        Returns:
            Path to the created PNG file
        """
        try:
            # Create the graph
            G = self.create_networkx_graph(include_pagerank=True)
            
            if G.number_of_nodes() == 0:
                raise ValueError("No nodes in graph to visualize")
            
            # Calculate positions
            pos = self.calculate_layout_positions(G, layout)
            
            # Categorize nodes
            node_categories = self.categorize_nodes(G)
            
            # Set up the plot
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle('Website Link Graph with PageRank Analysis', fontsize=16, fontweight='bold')
            
            # Get PageRank values for node sizing and coloring
            pagerank_values = [G.nodes[node].get('pagerank', 0) for node in G.nodes()]
            pagerank_dict = {node: G.nodes[node].get('pagerank', 0) for node in G.nodes()}
            
            if not pagerank_values or max(pagerank_values) == 0:
                # No PageRank data, use degree centrality
                pagerank_dict = nx.degree_centrality(G)
                pagerank_values = list(pagerank_dict.values())
            
            # Normalize PageRank for visualization
            if pagerank_values:
                min_pr, max_pr = min(pagerank_values), max(pagerank_values)
                if max_pr > min_pr:
                    normalized_pr = [(pr - min_pr) / (max_pr - min_pr) for pr in pagerank_values]
                else:
                    normalized_pr = [0.5] * len(pagerank_values)
            else:
                normalized_pr = [0.5] * G.number_of_nodes()
            
            # Draw edges first (so they appear behind nodes)
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color='#CCCCCC',
                alpha=0.5,
                width=0.5,
                arrows=True,
                arrowsize=10,
                arrowstyle='->',
                connectionstyle="arc3,rad=0.1"
            )
            
            # Draw nodes by category
            for category, nodes in node_categories.items():
                if not nodes:
                    continue
                
                # Get PageRank values for these nodes
                cat_pagerank = [pagerank_dict.get(node, 0) for node in nodes]
                
                # Calculate node sizes (proportional to PageRank)
                if cat_pagerank and max(cat_pagerank) > 0:
                    node_sizes = [max(100, pr * node_size_scale) for pr in cat_pagerank]
                else:
                    node_sizes = [200] * len(nodes)
                
                # Draw nodes with category colors
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes, ax=ax,
                    node_color=self.node_colors[category],
                    node_size=node_sizes,
                    alpha=0.8 if category == 'pillar' else 0.6,
                    edgecolors='white',
                    linewidths=2 if category == 'pillar' else 1
                )
            
            # Add labels for high PageRank nodes if requested
            if show_labels or highlight_pillars:
                # Show labels for top PageRank nodes
                top_nodes = sorted(G.nodes(), key=lambda x: pagerank_dict.get(x, 0), reverse=True)[:10]
                labels = {}
                for node in top_nodes:
                    node_data = G.nodes[node]
                    domain = node_data.get('domain', '')
                    path = node_data.get('path', '')
                    # Create short label
                    if len(path) > 20:
                        path = path[:17] + "..."
                    labels[node] = f"{domain}{path}"
                
                nx.draw_networkx_labels(
                    G, pos, labels=labels, ax=ax,
                    font_size=8, font_weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color=self.node_colors['pillar'], label='Pillar Pages (High PageRank)'),
                mpatches.Patch(color=self.node_colors['content'], label='Content Pages'),
                mpatches.Patch(color=self.node_colors['entry'], label='Entry Pages'),
                mpatches.Patch(color=self.node_colors['orphaned'], label='Orphaned Pages')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Add statistics text
            stats_text = f"""Graph Statistics:
• Nodes: {G.number_of_nodes()}
• Edges: {G.number_of_edges()}
• Pillar Pages: {len(node_categories['pillar'])}
• Orphaned Pages: {len(node_categories['orphaned'])}
• Avg PageRank: {np.mean(pagerank_values):.4f}
• Max PageRank: {np.max(pagerank_values):.4f}"""
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                   verticalalignment='bottom')
            
            # Remove axis
            ax.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"PageRank visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating PageRank visualization: {e}")
            raise
    
    def create_network_summary_image(
        self,
        output_path: str,
        figsize: Tuple[int, int] = (20, 16)
    ) -> str:
        """
        Create a comprehensive network summary with multiple views.
        
        Args:
            output_path: Path to save the PNG file
            figsize: Figure size (width, height)
            
        Returns:
            Path to the created PNG file
        """
        try:
            # Create the graph
            G = self.create_networkx_graph(include_pagerank=True)
            
            if G.number_of_nodes() == 0:
                raise ValueError("No nodes in graph to visualize")
            
            # Set up subplots
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Main network view (top left, spans 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            self._draw_main_network(G, ax1)
            
            # PageRank distribution (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            self._draw_pagerank_distribution(G, ax2)
            
            # In-degree distribution (bottom left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._draw_degree_distribution(G, ax3, 'in')
            
            # Out-degree distribution (bottom center)
            ax4 = fig.add_subplot(gs[1, 1])
            self._draw_degree_distribution(G, ax4, 'out')
            
            # Network metrics (bottom right)
            ax5 = fig.add_subplot(gs[1, 2])
            self._draw_network_metrics(G, ax5)
            
            # Main title
            fig.suptitle('Website Link Graph Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Network summary visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating network summary: {e}")
            raise
    
    def _draw_main_network(self, G: nx.DiGraph, ax):
        """Draw the main network visualization."""
        pos = self.calculate_layout_positions(G, "spring")
        node_categories = self.categorize_nodes(G)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#CCCCCC',
            alpha=0.3,
            width=0.5,
            arrows=True,
            arrowsize=5
        )
        
        # Draw nodes by category
        pagerank_dict = {node: G.nodes[node].get('pagerank', 0) for node in G.nodes()}
        
        for category, nodes in node_categories.items():
            if not nodes:
                continue
            
            cat_pagerank = [pagerank_dict.get(node, 0) for node in nodes]
            node_sizes = [max(50, pr * 500) if pr > 0 else 50 for pr in cat_pagerank]
            
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, ax=ax,
                node_color=self.node_colors[category],
                node_size=node_sizes,
                alpha=0.8,
                edgecolors='white',
                linewidths=1
            )
        
        ax.set_title('Link Graph Structure', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _draw_pagerank_distribution(self, G: nx.DiGraph, ax):
        """Draw PageRank distribution histogram."""
        pagerank_values = [G.nodes[node].get('pagerank', 0) for node in G.nodes()]
        
        if pagerank_values and max(pagerank_values) > 0:
            ax.hist(pagerank_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('PageRank Score')
            ax.set_ylabel('Number of Pages')
            ax.set_title('PageRank Distribution', fontweight='bold')
            
            # Add statistics
            mean_pr = np.mean(pagerank_values)
            max_pr = np.max(pagerank_values)
            ax.axvline(mean_pr, color='red', linestyle='--', label=f'Mean: {mean_pr:.4f}')
            ax.axvline(max_pr, color='orange', linestyle='--', label=f'Max: {max_pr:.4f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No PageRank Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PageRank Distribution', fontweight='bold')
    
    def _draw_degree_distribution(self, G: nx.DiGraph, ax, degree_type: str):
        """Draw degree distribution histogram."""
        if degree_type == 'in':
            degrees = [d for n, d in G.in_degree()]
            title = 'In-Degree Distribution'
            xlabel = 'Incoming Links'
        else:
            degrees = [d for n, d in G.out_degree()]
            title = 'Out-Degree Distribution'
            xlabel = 'Outgoing Links'
        
        if degrees:
            ax.hist(degrees, bins=min(20, max(degrees) + 1), alpha=0.7, 
                   color='lightgreen', edgecolor='black')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number of Pages')
            ax.set_title(title, fontweight='bold')
            
            # Add mean line
            mean_degree = np.mean(degrees)
            ax.axvline(mean_degree, color='red', linestyle='--', label=f'Mean: {mean_degree:.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    def _draw_network_metrics(self, G: nx.DiGraph, ax):
        """Draw network metrics summary."""
        # Calculate metrics
        try:
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G)
            if nx.is_weakly_connected(G):
                diameter = nx.diameter(G.to_undirected())
            else:
                diameter = "N/A (Disconnected)"
            
            # Get largest component size
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            largest_cc_size = len(largest_cc)
            
        except Exception:
            density = nx.density(G)
            avg_clustering = 0
            diameter = "N/A"
            largest_cc_size = G.number_of_nodes()
        
        # Create metrics text
        metrics_text = f"""Network Metrics:

Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}

Density: {density:.4f}
Avg Clustering: {avg_clustering:.4f}
Diameter: {diameter}

Largest Component:
{largest_cc_size} nodes
({100 * largest_cc_size / G.number_of_nodes():.1f}%)

Connectivity:
{nx.number_weakly_connected_components(G)} weak components
{nx.number_strongly_connected_components(G)} strong components"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Network Metrics', fontweight='bold')
        ax.axis('off')
    
    def export_to_base64(self, image_path: str) -> str:
        """
        Convert PNG image to base64 string for embedding.
        
        Args:
            image_path: Path to the PNG file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_string = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{base64_string}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise