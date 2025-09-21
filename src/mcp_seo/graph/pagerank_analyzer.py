"""
PageRank Analyzer for SEO Link Structure Analysis

Enhanced PageRank analyzer for comprehensive SEO link structure analysis with MCP Data4SEO server.
Calculates PageRank scores using Kuzu graph database for efficient analysis.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from .kuzu_manager import KuzuManager

logger = logging.getLogger(__name__)


class PageRankAnalyzer:
    """Analyzes website PageRank using Kuzu graph database."""

    def __init__(self, kuzu_manager: KuzuManager):
        """
        Initialize PageRank analyzer.
        
        Args:
            kuzu_manager: Initialized KuzuManager instance
        """
        self.kuzu_manager = kuzu_manager
        self.pagerank_scores: Dict[str, float] = {}

    def calculate_pagerank(self, damping_factor: float = 0.85, 
                          max_iterations: int = 100, 
                          tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Calculate PageRank using power iteration method.
        
        Args:
            damping_factor: PageRank damping factor (typically 0.85)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping URLs to PageRank scores
        """
        try:
            # Get all pages
            pages_data = self.kuzu_manager.get_page_data()
            if not pages_data:
                logger.error("No pages found in database")
                return {}

            pages = [page['url'] for page in pages_data]
            n_pages = len(pages)

            # Initialize PageRank scores
            initial_score = 1.0 / n_pages
            for page in pages:
                self.pagerank_scores[page] = initial_score

            logger.info(f"Starting PageRank calculation for {n_pages} pages")

            # Power iteration
            for iteration in range(max_iterations):
                new_scores = {}
                total_change = 0.0

                for page in pages:
                    # Get incoming links
                    incoming_links = self.kuzu_manager.get_incoming_links(page)

                    # Calculate new PageRank score
                    rank_sum = 0.0
                    for link in incoming_links:
                        source_url = link['source_url']
                        out_degree = link['source_out_degree']
                        if out_degree > 0 and source_url in self.pagerank_scores:
                            rank_sum += self.pagerank_scores[source_url] / out_degree

                    new_score = (1 - damping_factor) / n_pages + damping_factor * rank_sum
                    new_scores[page] = new_score
                    total_change += abs(new_score - self.pagerank_scores[page])

                # Update scores
                self.pagerank_scores.update(new_scores)

                logger.debug(f"PageRank iteration {iteration + 1}, change: {total_change:.6f}")

                # Check convergence
                if total_change < tolerance:
                    logger.info(f"PageRank converged after {iteration + 1} iterations")
                    break

            # Update database with PageRank scores
            self.kuzu_manager.update_pagerank_scores(self.pagerank_scores)

            logger.info(f"PageRank calculation completed for {n_pages} pages")
            return self.pagerank_scores

        except Exception as e:
            logger.error(f"Error calculating PageRank: {e}")
            return {}

    def get_pillar_pages(self, percentile: float = 90, limit: int = 10) -> List[Dict]:
        """
        Get pillar pages (highest authority pages).
        
        Args:
            percentile: Percentile threshold for pillar pages
            limit: Maximum number of pages to return
            
        Returns:
            List of pillar page dictionaries
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            if not pages_data:
                return []

            # Calculate percentile threshold
            pagerank_scores = [p['pagerank'] for p in pages_data if p['pagerank'] > 0]
            if not pagerank_scores:
                return []

            threshold = np.percentile(pagerank_scores, percentile)
            pillar_pages = [p for p in pages_data if p['pagerank'] >= threshold]

            # Sort by PageRank and limit results
            pillar_pages.sort(key=lambda x: x['pagerank'], reverse=True)
            return pillar_pages[:limit]

        except Exception as e:
            logger.error(f"Error getting pillar pages: {e}")
            return []

    def get_orphaned_pages(self) -> List[Dict]:
        """
        Get orphaned pages (no incoming links).
        
        Returns:
            List of orphaned page dictionaries
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            orphaned = [p for p in pages_data if p['in_degree'] == 0]
            return orphaned

        except Exception as e:
            logger.error(f"Error getting orphaned pages: {e}")
            return []

    def get_low_outlink_pages(self, percentile: float = 25, limit: int = 10) -> List[Dict]:
        """
        Get pages with few outgoing links.
        
        Args:
            percentile: Percentile threshold for low outlink pages
            limit: Maximum number of pages to return
            
        Returns:
            List of low outlink page dictionaries
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            if not pages_data:
                return []

            # Calculate percentile threshold
            out_degrees = [p['out_degree'] for p in pages_data]
            if not out_degrees:
                return []

            threshold = np.percentile(out_degrees, percentile)
            low_outlink_pages = [p for p in pages_data if p['out_degree'] <= threshold]

            # Sort by out_degree and limit results
            low_outlink_pages.sort(key=lambda x: x['out_degree'])
            return low_outlink_pages[:limit]

        except Exception as e:
            logger.error(f"Error getting low outlink pages: {e}")
            return []

    def get_hub_pages(self, limit: int = 10) -> List[Dict]:
        """
        Get hub pages (highest number of outgoing links).
        
        Args:
            limit: Maximum number of pages to return
            
        Returns:
            List of hub page dictionaries
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            if not pages_data:
                return []

            # Sort by out_degree
            hub_pages = sorted(pages_data, key=lambda x: x['out_degree'], reverse=True)
            return hub_pages[:limit]

        except Exception as e:
            logger.error(f"Error getting hub pages: {e}")
            return []

    def generate_analysis_summary(self) -> Dict:
        """
        Generate comprehensive analysis summary.
        
        Returns:
            Dictionary containing analysis results and insights
        """
        try:
            # Get basic data
            pages_data = self.kuzu_manager.get_page_data()
            graph_stats = self.kuzu_manager.get_graph_stats()

            if not pages_data:
                return {"error": "No page data available"}

            # Calculate statistics
            pagerank_scores = [p['pagerank'] for p in pages_data if p['pagerank'] > 0]
            in_degrees = [p['in_degree'] for p in pages_data]
            out_degrees = [p['out_degree'] for p in pages_data]

            # Get insights
            pillar_pages = self.get_pillar_pages()
            orphaned_pages = self.get_orphaned_pages()
            low_outlink_pages = self.get_low_outlink_pages()
            hub_pages = self.get_hub_pages()

            # Generate recommendations
            recommendations = self._generate_recommendations(
                pages_data, pillar_pages, orphaned_pages, low_outlink_pages
            )

            summary = {
                'graph_stats': graph_stats,
                'metrics': {
                    'total_pages': len(pages_data),
                    'avg_pagerank': float(np.mean(pagerank_scores)) if pagerank_scores else 0.0,
                    'max_pagerank': float(max(pagerank_scores)) if pagerank_scores else 0.0,
                    'min_pagerank': float(min(pagerank_scores)) if pagerank_scores else 0.0,
                    'avg_in_degree': float(np.mean(in_degrees)),
                    'avg_out_degree': float(np.mean(out_degrees)),
                    'max_in_degree': max(in_degrees),
                    'max_out_degree': max(out_degrees)
                },
                'insights': {
                    'pillar_pages': pillar_pages,
                    'orphaned_pages': orphaned_pages,
                    'low_outlink_pages': low_outlink_pages,
                    'hub_pages': hub_pages
                },
                'recommendations': recommendations,
                'pagerank_scores': self.pagerank_scores
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, all_pages: List[Dict], pillar_pages: List[Dict],
                                orphaned_pages: List[Dict], low_outlink_pages: List[Dict]) -> List[str]:
        """
        Generate specific recommendations for link structure optimization.
        
        Args:
            all_pages: All page data
            pillar_pages: High authority pages
            orphaned_pages: Pages with no incoming links
            low_outlink_pages: Pages with few outgoing links
            
        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Pillar page recommendations
        if pillar_pages:
            recommendations.append(
                f"✅ Identified {len(pillar_pages)} pillar pages with high authority. "
                f"Ensure these pages are easily accessible from your main navigation."
            )

            top_pillar = pillar_pages[0]
            recommendations.append(
                f"🎯 Top authority page: '{top_pillar['title']}' (PageRank: {top_pillar['pagerank']:.4f}). "
                f"Consider featuring this prominently in your site structure."
            )

        # Orphaned pages recommendations
        if orphaned_pages:
            recommendations.append(
                f"⚠️ Found {len(orphaned_pages)} orphaned pages with no incoming links. "
                f"These pages are missing valuable link equity and may be hard to discover."
            )

            for orphan in orphaned_pages[:3]:
                recommendations.append(
                    f"   → Add internal links to: '{orphan['title']}' ({orphan['path']})"
                )

        # Low outlink recommendations
        if low_outlink_pages:
            recommendations.append(
                f"🔗 {len(low_outlink_pages)} pages have very few outgoing links. "
                f"Consider adding relevant internal links to distribute link equity better."
            )

        # Hub page opportunities
        if all_pages:
            high_inlink_pages = sorted(all_pages, key=lambda x: x['in_degree'], reverse=True)[:5]
            if high_inlink_pages:
                recommendations.append(
                    f"📊 Create content hubs around high-authority pages: "
                    f"{', '.join([p['title'] for p in high_inlink_pages if p['title']])}"
                )

        # General recommendations
        if all_pages:
            avg_pagerank = np.mean([p['pagerank'] for p in all_pages if p['pagerank'] > 0])
            underperforming = [p for p in all_pages if p['pagerank'] < avg_pagerank * 0.5]
            
            if underperforming:
                recommendations.append(
                    f"📈 {len(underperforming)} pages have significantly below-average PageRank. "
                    f"Consider improving content quality and internal linking for these pages."
                )

        return recommendations

    def export_pagerank_data(self) -> Dict:
        """
        Export PageRank data for external use.
        
        Returns:
            Dictionary with exportable PageRank data
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            links_data = self.kuzu_manager.get_links_data()

            return {
                'pages': pages_data,
                'links': links_data,
                'pagerank_scores': self.pagerank_scores,
                'analysis_summary': self.generate_analysis_summary()
            }

        except Exception as e:
            logger.error(f"Error exporting PageRank data: {e}")
            return {"error": str(e)}