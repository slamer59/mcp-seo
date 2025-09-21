#!/usr/bin/env python3
"""
Blog Content SEO Analyzer
=========================

Comprehensive SEO analysis for blog content using graph metrics and content analysis.
Analyzes content quality, identifies pillar pages, detects content clusters, and provides
actionable SEO recommendations.

Features:
- Content quality scoring based on multiple metrics
- Pillar page identification using PageRank and authority scores
- Content clustering by keywords and topics
- Performance analysis for underperforming content
- SEO recommendations with priority scoring

Author: MCP SEO Team
"""

import logging
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class BlogAnalyzer:
    """Comprehensive SEO analysis for blog content."""

    def __init__(self, posts_data, metrics: Optional[Dict[str, Dict]] = None):
        """
        Initialize the blog analyzer.

        Args:
            posts_data: Either a dictionary of post data keyed by slug, or a KuzuManager instance
            metrics: Optional dictionary of pre-calculated metrics keyed by slug
        """
        # Handle different initialization patterns
        if hasattr(posts_data, "get_all_pages"):
            # posts_data is a KuzuManager instance
            self.kuzu_manager = posts_data
            self.posts_data = {}
            self.metrics = {}
        else:
            # posts_data is a dictionary of posts
            self.kuzu_manager = None
            self.posts_data = posts_data
            self.metrics = metrics or {}

        self.analysis_cache = {}

    def calculate_networkx_metrics(
        self, nodes: List[Dict], edges: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Calculate comprehensive graph metrics using NetworkX.

        Args:
            nodes: List of node dictionaries with slug and metadata
            edges: List of edge dictionaries with source, target, and metadata

        Returns:
            Dictionary of metrics for each node
        """
        logger.info("Calculating graph metrics using NetworkX")

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in nodes:
            G.add_node(node["slug"], **node)

        # Add edges with attributes
        for edge in edges:
            if edge["source"] in G.nodes and edge["target"] in G.nodes:
                G.add_edge(edge["source"], edge["target"], **edge)

        metrics = {}

        try:
            # Calculate various centrality measures
            pagerank = self._safe_pagerank(G)
            betweenness = self._safe_betweenness_centrality(G)
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            closeness = self._safe_closeness_centrality(G)
            hubs, authorities = self._safe_hits(G)

            # Additional metrics
            clustering = self._safe_clustering(G)
            katz_centrality = self._safe_katz_centrality(G)

            # Combine all metrics
            for node in G.nodes():
                metrics[node] = {
                    "pagerank": pagerank.get(node, 0.0),
                    "betweenness_centrality": betweenness.get(node, 0.0),
                    "in_degree": in_degree.get(node, 0),
                    "out_degree": out_degree.get(node, 0),
                    "closeness_centrality": closeness.get(node, 0.0),
                    "hub_score": hubs.get(node, 0.0),
                    "authority_score": authorities.get(node, 0.0),
                    "clustering_coefficient": clustering.get(node, 0.0),
                    "katz_centrality": katz_centrality.get(node, 0.0),
                    "total_degree": in_degree.get(node, 0) + out_degree.get(node, 0),
                }

            self.metrics = metrics
            logger.info("NetworkX metrics calculated successfully")

        except Exception as e:
            logger.error(f"NetworkX calculation error: {e}")
            # Return empty metrics to prevent failure
            for node in nodes:
                metrics[node["slug"]] = self._get_default_metrics()

        return metrics

    def _safe_pagerank(self, G: nx.DiGraph, **kwargs) -> Dict[str, float]:
        """Safely calculate PageRank with fallback."""
        try:
            return nx.pagerank(G, alpha=0.85, max_iter=100, **kwargs)
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return {node: 1.0 / len(G.nodes()) for node in G.nodes()}

    def _safe_betweenness_centrality(self, G: nx.DiGraph) -> Dict[str, float]:
        """Safely calculate betweenness centrality with fallback."""
        try:
            return nx.betweenness_centrality(G)
        except Exception as e:
            logger.warning(f"Betweenness centrality calculation failed: {e}")
            return {node: 0.0 for node in G.nodes()}

    def _safe_closeness_centrality(self, G: nx.DiGraph) -> Dict[str, float]:
        """Safely calculate closeness centrality with fallback."""
        try:
            return nx.closeness_centrality(G)
        except Exception as e:
            logger.warning(f"Closeness centrality calculation failed: {e}")
            return {node: 0.0 for node in G.nodes()}

    def _safe_hits(self, G: nx.DiGraph) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Safely calculate HITS algorithm with fallback."""
        try:
            return nx.hits(G, max_iter=100)
        except Exception as e:
            logger.warning(f"HITS calculation failed: {e}")
            default_dict = {node: 0.0 for node in G.nodes()}
            return default_dict, default_dict

    def _safe_clustering(self, G: nx.DiGraph) -> Dict[str, float]:
        """Safely calculate clustering coefficient with fallback."""
        try:
            return nx.clustering(G.to_undirected())
        except Exception as e:
            logger.warning(f"Clustering calculation failed: {e}")
            return {node: 0.0 for node in G.nodes()}

    def _safe_katz_centrality(self, G: nx.DiGraph) -> Dict[str, float]:
        """Safely calculate Katz centrality with fallback."""
        try:
            return nx.katz_centrality(G, alpha=0.1, max_iter=1000)
        except Exception as e:
            logger.warning(f"Katz centrality calculation failed: {e}")
            return {node: 0.0 for node in G.nodes()}

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics structure."""
        return {
            "pagerank": 0.0,
            "betweenness_centrality": 0.0,
            "in_degree": 0,
            "out_degree": 0,
            "closeness_centrality": 0.0,
            "hub_score": 0.0,
            "authority_score": 0.0,
            "clustering_coefficient": 0.0,
            "katz_centrality": 0.0,
            "total_degree": 0,
        }

    def generate_comprehensive_analysis(
        self, metrics: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive SEO analysis report.

        Args:
            metrics: Optional pre-calculated metrics, uses self.metrics if not provided

        Returns:
            Comprehensive analysis dictionary
        """
        if metrics:
            self.metrics = metrics

        logger.info("Generating comprehensive SEO analysis")

        analysis = {
            "summary": self._generate_summary_statistics(),
            "pillar_pages": self.identify_pillar_pages(),
            "underperforming_pages": self._identify_underperforming_pages(),
            "content_clusters": self._analyze_content_clusters(),  # Use internal dict version
            "link_opportunities": self._generate_link_opportunities(),
            "content_quality": self.analyze_content_quality(),
            "recommendations": self.generate_content_recommendations(),
        }

        self.analysis_cache = analysis
        logger.info("Comprehensive analysis completed")
        return analysis

    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the blog."""
        total_pages = len(self.posts_data)
        total_links = sum(
            len(post["internal_links"]) for post in self.posts_data.values()
        )

        # Calculate averages
        avg_links_per_page = total_links / total_pages if total_pages > 0 else 0

        # Word count statistics
        word_counts = [post["word_count"] for post in self.posts_data.values()]
        avg_word_count = statistics.mean(word_counts) if word_counts else 0
        median_word_count = statistics.median(word_counts) if word_counts else 0

        # Quality metrics
        readability_scores = [
            post["quality_metrics"]["readability_score"]
            for post in self.posts_data.values()
        ]
        avg_readability = (
            statistics.mean(readability_scores) if readability_scores else 0
        )

        return {
            "total_pages": total_pages,
            "total_internal_links": total_links,
            "avg_links_per_page": round(avg_links_per_page, 2),
            "link_density": round(
                total_links / (total_pages * (total_pages - 1))
                if total_pages > 1
                else 0,
                4,
            ),
            "avg_word_count": round(avg_word_count, 0),
            "median_word_count": round(median_word_count, 0),
            "avg_readability_score": round(avg_readability, 2),
            "content_coverage": self._calculate_content_coverage(),
        }

    def _calculate_content_coverage(self) -> Dict[str, Any]:
        """Calculate content coverage and diversity metrics."""
        # Keyword diversity
        all_keywords = []
        for post in self.posts_data.values():
            all_keywords.extend(post["keywords"])

        unique_keywords = len(set(all_keywords))
        total_keywords = len(all_keywords)
        keyword_diversity = unique_keywords / max(total_keywords, 1)

        # Author diversity
        authors = [post.get("author", "Unknown") for post in self.posts_data.values()]
        unique_authors = len(set(authors))

        return {
            "unique_keywords": unique_keywords,
            "total_keyword_mentions": total_keywords,
            "keyword_diversity": round(keyword_diversity, 3),
            "unique_authors": unique_authors,
            "avg_keywords_per_post": round(
                total_keywords / max(len(self.posts_data), 1), 2
            ),
        }

    def identify_pillar_pages(
        self, content_data: Optional[Dict[str, Dict]] = None, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify pillar pages based on multiple authority metrics."""
        # Use provided content_data or fallback to self.posts_data
        posts_data = content_data or self.posts_data
        pillar_candidates = []

        total_pages = len(posts_data)

        # If we have metrics, use them. Otherwise, use content data directly
        if self.metrics:
            for slug, metric in self.metrics.items():
                post = posts_data.get(slug, {})
                # Calculate composite pillar score
                pillar_score = self._calculate_pillar_score(metric, post, total_pages)

                pillar_candidates.append(
                    {
                        "slug": slug,
                        "title": post.get("title", slug),
                        "pillar_score": pillar_score,
                        "pagerank": metric.get("pagerank", 0),
                        "authority_score": metric.get("authority_score", 0),
                        "in_degree": metric.get("in_degree", 0),
                        "word_count": post.get("word_count", 0),
                        "readability_score": post.get("quality_metrics", {}).get(
                            "readability_score", 0
                        ),
                        "keywords_count": len(post.get("keywords", [])),
                        "quality_metrics": post.get("quality_metrics", {}),
                    }
                )
        else:
            # Generate basic pillar scores based on content alone
            for slug, post in posts_data.items():
                # Use simple heuristics when no graph metrics available
                basic_metric = {
                    "pagerank": 0.1,  # Default value
                    "authority_score": 0.1,  # Default value
                    "in_degree": 0,  # Default value
                }

                # Calculate composite pillar score
                pillar_score = self._calculate_pillar_score(
                    basic_metric, post, total_pages
                )

                pillar_candidates.append(
                    {
                        "slug": slug,
                        "title": post.get("title", slug),
                        "pillar_score": pillar_score,
                        "pagerank": basic_metric.get("pagerank", 0),
                        "authority_score": basic_metric.get("authority_score", 0),
                        "in_degree": basic_metric.get("in_degree", 0),
                        "word_count": post.get("word_count", 0),
                        "readability_score": post.get("quality_metrics", {}).get(
                            "readability_score", 0
                        ),
                        "keywords_count": len(post.get("keywords", [])),
                        "quality_metrics": post.get("quality_metrics", {}),
                    }
                )

        # Sort by pillar score and return top candidates
        pillar_candidates.sort(key=lambda x: x["pillar_score"], reverse=True)
        return pillar_candidates[:top_n]

    def _calculate_pillar_score(
        self, metric: Dict, post: Dict, total_pages: int
    ) -> float:
        """Calculate composite pillar score for a page."""
        # Normalize metrics
        pagerank_score = metric.get("pagerank", 0) * 0.3
        authority_score = metric.get("authority_score", 0) * 0.25
        link_score = metric.get("in_degree", 0) / max(total_pages, 1) * 0.2

        # Content quality factors
        word_count_score = min(post.get("word_count", 0) / 2000, 1.0) * 0.1
        readability_score = (
            post.get("quality_metrics", {}).get("readability_score", 0) / 100 * 0.05
        )
        keyword_score = min(len(post.get("keywords", [])) / 10, 1.0) * 0.1

        return (
            pagerank_score
            + authority_score
            + link_score
            + word_count_score
            + readability_score
            + keyword_score
        )

    def _identify_underperforming_pages(
        self, pagerank_threshold: float = 0.01, in_degree_threshold: int = 2
    ) -> List[Dict[str, Any]]:
        """Identify pages that need more internal links."""
        underperforming = []

        for slug, metric in self.metrics.items():
            post = self.posts_data.get(slug, {})

            pagerank = metric.get("pagerank", 0)
            in_degree = metric.get("in_degree", 0)

            if pagerank < pagerank_threshold and in_degree < in_degree_threshold:
                underperforming.append(
                    {
                        "slug": slug,
                        "title": post.get("title", slug),
                        "pagerank": pagerank,
                        "in_degree": in_degree,
                        "word_count": post.get("word_count", 0),
                        "keywords": post.get("keywords", []),
                        "readability_score": post.get("quality_metrics", {}).get(
                            "readability_score", 0
                        ),
                        "potential_score": self._calculate_potential_score(post),
                    }
                )

        # Sort by potential score (higher potential = higher priority)
        return sorted(underperforming, key=lambda x: x["potential_score"], reverse=True)

    def _calculate_potential_score(self, post: Dict) -> float:
        """Calculate potential score for underperforming content."""
        word_count = post.get("word_count", 0)
        keywords_count = len(post.get("keywords", []))
        readability = post.get("quality_metrics", {}).get("readability_score", 0)

        # Higher score = higher potential for improvement
        potential = (
            min(word_count / 1000, 1.0) * 0.4
            + min(keywords_count / 5, 1.0) * 0.3
            + readability / 100 * 0.3
        )

        return potential

    def detect_content_clusters(
        self, content_data: Optional[Dict[str, Dict]] = None, min_cluster_size: int = 2
    ) -> List[Dict]:
        """Analyze content clustering by keywords and topics."""
        # Use provided content_data or fallback to self.posts_data
        posts_data = content_data or self.posts_data
        keyword_clusters = defaultdict(list)

        # Group content by keywords
        for slug, post in posts_data.items():
            for keyword in post.get("keywords", []):
                if len(keyword) >= 3:  # Filter very short keywords
                    keyword_clusters[keyword.lower()].append(
                        {
                            "slug": slug,
                            "title": post.get("title", slug),
                            "pagerank": self.metrics.get(slug, {}).get("pagerank", 0),
                            "word_count": post.get("word_count", 0),
                            "authority_score": self.metrics.get(slug, {}).get(
                                "authority_score", 0
                            ),
                        }
                    )

        # Filter clusters and create the expected structure
        clusters = []
        for keyword, pages in keyword_clusters.items():
            if len(pages) >= min_cluster_size:
                # Sort pages by PageRank within cluster
                pages.sort(key=lambda x: x["pagerank"], reverse=True)

                # Calculate cluster strength
                cluster_strength = sum(page["pagerank"] for page in pages)

                clusters.append(
                    {
                        "topic": keyword,
                        "pages": pages,
                        "cluster_size": len(pages),
                        "cluster_strength": cluster_strength,
                        "pillar_page": pages[0]
                        if pages
                        else None,  # Highest PageRank page
                    }
                )

        # Sort by cluster strength
        clusters.sort(key=lambda x: x["cluster_strength"], reverse=True)
        return clusters

    def _analyze_content_clusters(self, min_cluster_size: int = 2) -> Dict[str, Dict]:
        """Internal method that returns clusters in dictionary format for compatibility."""
        # Use provided content_data or fallback to self.posts_data
        posts_data = self.posts_data
        keyword_clusters = defaultdict(list)

        # Group content by keywords
        for slug, post in posts_data.items():
            for keyword in post.get("keywords", []):
                if len(keyword) >= 3:  # Filter very short keywords
                    keyword_clusters[keyword.lower()].append(
                        {
                            "slug": slug,
                            "title": post.get("title", slug),
                            "pagerank": self.metrics.get(slug, {}).get("pagerank", 0),
                            "word_count": post.get("word_count", 0),
                            "authority_score": self.metrics.get(slug, {}).get(
                                "authority_score", 0
                            ),
                        }
                    )

        # Filter clusters and sort by authority
        filtered_clusters = {}
        for keyword, pages in keyword_clusters.items():
            if len(pages) >= min_cluster_size:
                # Sort pages by PageRank within cluster
                pages.sort(key=lambda x: x["pagerank"], reverse=True)

                # Calculate cluster strength
                cluster_strength = sum(page["pagerank"] for page in pages)

                filtered_clusters[keyword] = {
                    "pages": pages,
                    "cluster_size": len(pages),
                    "cluster_strength": cluster_strength,
                    "pillar_page": pages[0] if pages else None,  # Highest PageRank page
                }

        return filtered_clusters

    def analyze_content_quality(
        self, content_data: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """Analyze overall content quality metrics."""
        # Use provided content_data or fallback to self.posts_data
        posts_data = content_data or self.posts_data

        quality_scores = []
        quality_breakdown = {
            "excellent": [],  # >80
            "good": [],  # 60-80
            "fair": [],  # 40-60
            "poor": [],  # <40
        }

        for slug, post in posts_data.items():
            quality_metrics = post.get("quality_metrics", {})

            # Calculate composite quality score
            quality_score = self._calculate_content_quality_score(quality_metrics, post)
            quality_scores.append(quality_score)

            # Categorize quality
            if quality_score >= 80:
                category = "excellent"
            elif quality_score >= 60:
                category = "good"
            elif quality_score >= 40:
                category = "fair"
            else:
                category = "poor"

            quality_breakdown[category].append(
                {
                    "slug": slug,
                    "title": post.get("title", slug),
                    "quality_score": quality_score,
                    "word_count": post.get("word_count", 0),
                    "readability_score": quality_metrics.get("readability_score", 0),
                }
            )

        # Create quality_scores dictionary as expected by tests
        quality_scores_dict = {}
        for slug, post in posts_data.items():
            quality_metrics = post.get("quality_metrics", {})
            quality_score = self._calculate_content_quality_score(quality_metrics, post)
            quality_scores_dict[slug] = quality_score / 100  # Normalize to 0-1 range

        # Generate recommendations based on quality analysis
        recommendations = []
        poor_content = quality_breakdown["poor"]
        if poor_content:
            recommendations.append(
                {
                    "type": "content_quality",
                    "priority": "high",
                    "description": f"Improve quality for {len(poor_content)} low-scoring pages",
                    "action_items": [
                        f"Enhance content for {page['title']}"
                        for page in poor_content[:3]
                    ],
                }
            )

        fair_content = quality_breakdown["fair"]
        if fair_content:
            recommendations.append(
                {
                    "type": "content_enhancement",
                    "priority": "medium",
                    "description": f"Enhance {len(fair_content)} moderately-scoring pages",
                    "action_items": [
                        f"Add more depth to {page['title']}"
                        for page in fair_content[:3]
                    ],
                }
            )

        return {
            "quality_scores": quality_scores_dict,
            "recommendations": recommendations,
            "average_quality_score": statistics.mean(quality_scores)
            if quality_scores
            else 0,
            "quality_distribution": {k: len(v) for k, v in quality_breakdown.items()},
            "quality_breakdown": quality_breakdown,
            "improvement_candidates": quality_breakdown["poor"]
            + quality_breakdown["fair"],
        }

    def _calculate_content_quality_score(
        self, quality_metrics: Dict, post: Dict
    ) -> float:
        """Calculate composite content quality score."""
        word_count = post.get("word_count", 0)
        readability = quality_metrics.get("readability_score", 0)
        header_count = quality_metrics.get("header_count", 0)
        image_count = quality_metrics.get("image_count", 0)
        keyword_count = len(post.get("keywords", []))

        # Scoring components
        length_score = min(word_count / 1000, 1.0) * 25  # Up to 25 points for length
        readability_score = readability / 100 * 25  # Up to 25 points for readability
        structure_score = min(header_count * 3, 20)  # Up to 20 points for structure
        media_score = min(image_count * 5, 15)  # Up to 15 points for media
        keyword_score = min(keyword_count * 2, 15)  # Up to 15 points for keywords

        return (
            length_score
            + readability_score
            + structure_score
            + media_score
            + keyword_score
        )

    def _generate_link_opportunities(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Generate internal linking opportunities based on content relevance."""
        link_opportunities = []

        for source_slug, source_post in self.posts_data.items():
            source_keywords = set(kw.lower() for kw in source_post.get("keywords", []))
            current_targets = set(
                link["target_slug"] for link in source_post["internal_links"]
            )

            for target_slug, target_post in self.posts_data.items():
                if source_slug == target_slug or target_slug in current_targets:
                    continue

                target_keywords = set(
                    kw.lower() for kw in target_post.get("keywords", [])
                )

                # Calculate relevance and opportunity score
                opportunity = self._calculate_link_opportunity(
                    source_keywords, target_keywords, target_slug
                )

                if opportunity["score"] > 0:
                    opportunity.update(
                        {
                            "source_slug": source_slug,
                            "source_title": source_post.get("title", source_slug),
                            "target_slug": target_slug,
                            "target_title": target_post.get("title", target_slug),
                        }
                    )
                    link_opportunities.append(opportunity)

        # Sort by opportunity score and return top opportunities
        link_opportunities.sort(key=lambda x: x["score"], reverse=True)
        return link_opportunities[:top_n]

    def _calculate_link_opportunity(
        self, source_keywords: set, target_keywords: set, target_slug: str
    ) -> Dict[str, Any]:
        """Calculate link opportunity score between two pieces of content."""
        # Keyword overlap
        shared_keywords = source_keywords & target_keywords
        keyword_overlap = len(shared_keywords)

        if keyword_overlap == 0:
            return {"score": 0}

        # Target page authority
        target_metrics = self.metrics.get(target_slug, {})
        target_pagerank = target_metrics.get("pagerank", 0)
        target_authority = target_metrics.get("authority_score", 0)

        # Calculate opportunity score
        relevance_score = keyword_overlap * 2
        authority_bonus = (target_pagerank + target_authority) * 10
        opportunity_score = relevance_score + authority_bonus

        return {
            "score": opportunity_score,
            "keyword_overlap": keyword_overlap,
            "shared_keywords": list(shared_keywords),
            "target_pagerank": target_pagerank,
            "target_authority": target_authority,
            "relevance_score": relevance_score,
        }

    def generate_content_recommendations(
        self, content_data: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Generate actionable SEO recommendations with priority scoring."""
        recommendations = []

        # If content_data is provided, generate analysis on the fly
        if content_data:
            analysis = {
                "pillar_pages": self.identify_pillar_pages(content_data),
                "underperforming_pages": self._identify_underperforming_pages(),
                "content_clusters": self.detect_content_clusters(content_data),
                "content_quality": self.analyze_content_quality(content_data),
            }
        elif self.analysis_cache:
            analysis = self.analysis_cache
        else:
            logger.warning(
                "No cached analysis or content data provided for recommendations"
            )
            return recommendations

        # Recommendation 1: Strengthen pillar pages
        pillar_pages = analysis.get("pillar_pages", [])
        if pillar_pages:
            top_pillars = pillar_pages[:3]
            recommendations.append(
                {
                    "category": "Pillar Page Optimization",
                    "priority": "High",
                    "impact": "High",
                    "action": f"Focus on strengthening top pillar pages: {', '.join([p['title'] for p in top_pillars])}",
                    "details": "These pages have high authority and should be the primary hubs in your internal linking strategy.",
                    "metrics": {
                        "affected_pages": len(top_pillars),
                        "avg_pagerank": statistics.mean(
                            [p["pagerank"] for p in top_pillars]
                        ),
                    },
                }
            )

        # Recommendation 2: Improve underperforming pages
        underperforming = analysis.get("underperforming_pages", [])
        if underperforming:
            high_potential = [p for p in underperforming if p["potential_score"] > 0.5]
            recommendations.append(
                {
                    "category": "Link Building",
                    "priority": "High",
                    "impact": "Medium",
                    "action": f"Add internal links to {len(high_potential)} high-potential underperforming pages",
                    "details": "These pages have good content but low PageRank and need more internal links.",
                    "metrics": {
                        "affected_pages": len(high_potential),
                        "avg_potential": statistics.mean(
                            [p["potential_score"] for p in high_potential]
                        )
                        if high_potential
                        else 0,
                    },
                }
            )

        # Recommendation 3: Content clustering
        content_clusters = analysis.get("content_clusters", [])
        if content_clusters:
            strong_clusters = [
                c for c in content_clusters if c["cluster_strength"] > 0.1
            ]
            recommendations.append(
                {
                    "category": "Content Clustering",
                    "priority": "Medium",
                    "impact": "High",
                    "action": f"Create topic clusters around {len(strong_clusters)} strong keyword themes",
                    "details": "Link related pages together to create topical authority clusters.",
                    "metrics": {
                        "clusters_identified": len(strong_clusters),
                        "total_pages_in_clusters": sum(
                            c["cluster_size"] for c in strong_clusters
                        ),
                    },
                }
            )

        # Recommendation 4: Content quality improvement
        content_quality = analysis.get("content_quality", {})
        improvement_candidates = content_quality.get("improvement_candidates", [])
        if improvement_candidates:
            recommendations.append(
                {
                    "category": "Content Quality",
                    "priority": "Medium",
                    "impact": "Medium",
                    "action": f"Improve content quality for {len(improvement_candidates)} pages",
                    "details": "Focus on improving readability, structure, and keyword optimization.",
                    "metrics": {
                        "affected_pages": len(improvement_candidates),
                        "avg_quality_score": statistics.mean(
                            [p["quality_score"] for p in improvement_candidates]
                        )
                        if improvement_candidates
                        else 0,
                    },
                }
            )

        # Recommendation 5: Implement link opportunities
        link_opportunities = analysis.get("link_opportunities", [])
        if link_opportunities:
            high_value_opportunities = [
                opp for opp in link_opportunities if opp["score"] > 5
            ]
            recommendations.append(
                {
                    "category": "Internal Linking",
                    "priority": "Medium",
                    "impact": "Medium",
                    "action": f"Implement {len(high_value_opportunities)} high-value link opportunities",
                    "details": "Add contextually relevant internal links to improve link equity distribution.",
                    "metrics": {
                        "opportunities_identified": len(high_value_opportunities),
                        "avg_opportunity_score": statistics.mean(
                            [opp["score"] for opp in high_value_opportunities]
                        )
                        if high_value_opportunities
                        else 0,
                    },
                }
            )

        return recommendations

    def find_content_gaps(
        self, content_data: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Identify content gaps and opportunities.

        Args:
            content_data: Optional content data, uses self.posts_data if not provided

        Returns:
            Dictionary containing identified content gaps
        """
        # Use provided content_data or fallback to self.posts_data
        posts_data = content_data or self.posts_data

        gaps = {"missing_topics": [], "thin_content": [], "orphaned_content": []}

        # Identify thin content (low word count)
        for slug, post in posts_data.items():
            word_count = post.get("word_count", 0)
            if word_count > 0 and word_count < 500:
                gaps["thin_content"].append(
                    {
                        "slug": slug,
                        "title": post.get("title", slug),
                        "word_count": word_count,
                        "keywords": post.get("keywords", []),
                    }
                )

        # Identify orphaned content (no incoming links)
        # Count incoming links for each page
        incoming_link_counts = defaultdict(int)
        for slug, post in posts_data.items():
            for link in post.get("internal_links", []):
                target = link.get("target_slug", link.get("url", "").strip("/"))
                incoming_link_counts[target] += 1

        for slug, post in posts_data.items():
            if incoming_link_counts[slug] == 0 and slug != "index":
                gaps["orphaned_content"].append(
                    {
                        "slug": slug,
                        "title": post.get("title", slug),
                        "word_count": post.get("word_count", 0),
                        "keywords": post.get("keywords", []),
                    }
                )

        # Identify missing topics based on keyword analysis
        all_keywords = []
        for post in posts_data.values():
            all_keywords.extend(post.get("keywords", []))

        # Find underrepresented keyword combinations
        from collections import Counter

        keyword_counts = Counter(all_keywords)

        # Identify keywords that appear only once (potential gap topics)
        single_occurrence_keywords = [
            kw for kw, count in keyword_counts.items() if count == 1
        ]

        for keyword in single_occurrence_keywords[:10]:  # Limit to top 10
            gaps["missing_topics"].append(
                {
                    "keyword": keyword,
                    "current_coverage": 1,
                    "potential_opportunity": "high" if len(keyword) > 5 else "medium",
                }
            )

        return gaps

    def get_page_insights(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get detailed insights for a specific page."""
        if slug not in self.posts_data:
            return None

        post = self.posts_data[slug]
        metrics = self.metrics.get(slug, {})

        return {
            "basic_info": {
                "slug": slug,
                "title": post.get("title", slug),
                "word_count": post.get("word_count", 0),
                "author": post.get("author", ""),
                "keywords": post.get("keywords", []),
            },
            "graph_metrics": metrics,
            "quality_metrics": post.get("quality_metrics", {}),
            "linking": {
                "outbound_links": len(post.get("internal_links", [])),
                "inbound_links": metrics.get("in_degree", 0),
                "link_targets": [
                    link["target_slug"] for link in post.get("internal_links", [])
                ],
            },
            "seo_score": self._calculate_page_seo_score(post, metrics),
            "recommendations": self._get_page_recommendations(post, metrics),
        }

    def _calculate_page_seo_score(self, post: Dict, metrics: Dict) -> float:
        """Calculate comprehensive SEO score for a page."""
        content_score = (
            self._calculate_content_quality_score(post.get("quality_metrics", {}), post)
            / 100
        )
        authority_score = metrics.get("pagerank", 0) * 100
        linking_score = min(metrics.get("in_degree", 0) / 5, 1.0)

        return (content_score * 0.4 + authority_score * 0.4 + linking_score * 0.2) * 100

    def _get_page_recommendations(self, post: Dict, metrics: Dict) -> List[str]:
        """Get specific recommendations for a page."""
        recommendations = []

        # Content recommendations
        word_count = post.get("word_count", 0)
        if word_count < 500:
            recommendations.append("Expand content - aim for at least 500 words")

        # Link recommendations
        in_degree = metrics.get("in_degree", 0)
        if in_degree < 2:
            recommendations.append("Increase internal links pointing to this page")

        out_degree = metrics.get("out_degree", 0)
        if out_degree < 3:
            recommendations.append("Add more internal links to related content")

        # Quality recommendations
        quality_metrics = post.get("quality_metrics", {})
        if quality_metrics.get("header_count", 0) < 2:
            recommendations.append("Add more headers to improve content structure")

        if quality_metrics.get("readability_score", 0) < 60:
            recommendations.append(
                "Improve readability with shorter sentences and paragraphs"
            )

        return recommendations
