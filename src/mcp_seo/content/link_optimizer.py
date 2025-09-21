#!/usr/bin/env python3
"""
Link Optimization Engine for Blog Content
==========================================

Advanced internal link optimization engine that analyzes content relationships,
identifies linking opportunities, and provides strategic recommendations for
improving internal linking structure and PageRank distribution.

Features:
- Strategic link opportunity detection based on content relevance
- PageRank-aware link recommendations
- Content cluster optimization
- Authority distribution analysis
- Link equity flow optimization
- Contextual linking suggestions

Author: MCP SEO Team
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LinkOpportunity:
    """Represents a strategic internal linking opportunity."""
    source_slug: str
    source_title: str
    target_slug: str
    target_title: str
    opportunity_score: float
    relevance_score: float
    authority_benefit: float
    keyword_overlap: int
    shared_keywords: List[str]
    target_pagerank: float
    target_authority: float
    context_suggestions: List[str]
    priority: str = "Medium"


@dataclass
class ClusterOpportunity:
    """Represents a content cluster optimization opportunity."""
    cluster_keyword: str
    cluster_pages: List[Dict[str, Any]]
    pillar_page: Optional[Dict[str, Any]]
    cluster_strength: float
    missing_connections: List[Tuple[str, str]]
    optimization_potential: float


class LinkOptimizer:
    """Advanced internal link optimization engine."""

    def __init__(self,
                 posts_data: Dict[str, Dict],
                 metrics: Optional[Dict[str, Dict]] = None):
        """
        Initialize the link optimizer.

        Args:
            posts_data: Dictionary of post data keyed by slug
            metrics: Optional pre-calculated graph metrics
        """
        self.posts_data = posts_data
        self.metrics = metrics or {}
        self.link_graph = self._build_link_graph()
        self.keyword_index = self._build_keyword_index()

    def _build_link_graph(self) -> Dict[str, Set[str]]:
        """Build internal link graph for analysis."""
        graph = defaultdict(set)

        # Handle the case where posts_data might be a Mock object or None
        if not self.posts_data or not hasattr(self.posts_data, 'items'):
            logger.warning("posts_data is empty or not iterable, returning empty graph")
            return dict(graph)

        try:
            for source_slug, post in self.posts_data.items():
                # Handle case where post might be a Mock object
                if not isinstance(post, dict):
                    continue

                internal_links = post.get('internal_links', [])
                if not isinstance(internal_links, list):
                    continue

                for link in internal_links:
                    if not isinstance(link, dict):
                        continue

                    target_slug = link.get('target_slug')
                    if target_slug and target_slug in self.posts_data:
                        graph[source_slug].add(target_slug)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error building link graph: {e}")

        return dict(graph)

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """Build keyword to pages index."""
        keyword_index = defaultdict(list)

        # Handle the case where posts_data might be a Mock object or None
        if not self.posts_data or not hasattr(self.posts_data, 'items'):
            logger.warning("posts_data is empty or not iterable, returning empty keyword index")
            return dict(keyword_index)

        try:
            for slug, post in self.posts_data.items():
                # Handle case where post might be a Mock object
                if not isinstance(post, dict):
                    continue

                keywords = post.get('keywords', [])
                if not isinstance(keywords, list):
                    continue

                for keyword in keywords:
                    if isinstance(keyword, str) and keyword.strip():
                        keyword_index[keyword.lower()].append(slug)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error building keyword index: {e}")

        return dict(keyword_index)

    def identify_link_opportunities(self,
                                  max_opportunities: int = 50,
                                  min_relevance_score: float = 0.5) -> List[LinkOpportunity]:
        """
        Identify strategic internal linking opportunities.

        Args:
            max_opportunities: Maximum number of opportunities to return
            min_relevance_score: Minimum relevance score threshold

        Returns:
            List of link opportunities sorted by potential impact
        """
        logger.info("Identifying internal link opportunities")

        opportunities = []

        for source_slug, source_post in self.posts_data.items():
            source_keywords = set(kw.lower() for kw in source_post.get('keywords', []))
            current_targets = set(link['target_slug'] for link in source_post.get('internal_links', []))

            for target_slug, target_post in self.posts_data.items():
                if source_slug == target_slug or target_slug in current_targets:
                    continue

                opportunity = self._analyze_link_opportunity(
                    source_slug, source_post, target_slug, target_post, source_keywords
                )

                if opportunity and opportunity.relevance_score >= min_relevance_score:
                    opportunities.append(opportunity)

        # Sort by opportunity score and return top opportunities
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        return opportunities[:max_opportunities]

    def _analyze_link_opportunity(self,
                                source_slug: str,
                                source_post: Dict,
                                target_slug: str,
                                target_post: Dict,
                                source_keywords: Set[str]) -> Optional[LinkOpportunity]:
        """Analyze a potential link opportunity between two pages."""
        target_keywords = set(kw.lower() for kw in target_post.get('keywords', []))

        # Calculate keyword overlap
        shared_keywords = source_keywords & target_keywords
        keyword_overlap = len(shared_keywords)

        if keyword_overlap == 0:
            return None

        # Get target metrics
        target_metrics = self.metrics.get(target_slug, {})
        target_pagerank = target_metrics.get('pagerank', 0)
        target_authority = target_metrics.get('authority_score', 0)

        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            source_post, target_post, keyword_overlap, len(source_keywords), len(target_keywords)
        )

        # Calculate authority benefit
        authority_benefit = self._calculate_authority_benefit(target_pagerank, target_authority)

        # Calculate overall opportunity score
        opportunity_score = (
            relevance_score * 0.4 +
            authority_benefit * 0.3 +
            keyword_overlap * 0.2 +
            self._calculate_content_quality_factor(target_post) * 0.1
        )

        # Generate context suggestions
        context_suggestions = self._generate_context_suggestions(
            source_post, target_post, list(shared_keywords)
        )

        # Determine priority
        priority = self._determine_priority(opportunity_score, relevance_score, authority_benefit)

        return LinkOpportunity(
            source_slug=source_slug,
            source_title=source_post.get('title', source_slug),
            target_slug=target_slug,
            target_title=target_post.get('title', target_slug),
            opportunity_score=opportunity_score,
            relevance_score=relevance_score,
            authority_benefit=authority_benefit,
            keyword_overlap=keyword_overlap,
            shared_keywords=list(shared_keywords),
            target_pagerank=target_pagerank,
            target_authority=target_authority,
            context_suggestions=context_suggestions,
            priority=priority
        )

    def _calculate_relevance_score(self,
                                 source_post: Dict,
                                 target_post: Dict,
                                 keyword_overlap: int,
                                 source_keyword_count: int,
                                 target_keyword_count: int) -> float:
        """Calculate content relevance score between two pages."""
        # Keyword overlap ratio
        max_keywords = max(source_keyword_count, target_keyword_count, 1)
        keyword_ratio = keyword_overlap / max_keywords

        # Content similarity factors
        source_word_count = source_post.get('word_count', 0)
        target_word_count = target_post.get('word_count', 0)

        # Prefer linking to substantial content
        content_factor = min(target_word_count / 1000, 1.0)

        # Topic clustering factor (if pages share multiple keywords, they're more related)
        clustering_factor = min(keyword_overlap / 3, 1.0)

        return (keyword_ratio * 0.5 + content_factor * 0.3 + clustering_factor * 0.2)

    def _calculate_authority_benefit(self, pagerank: float, authority_score: float) -> float:
        """Calculate the authority benefit of linking to a target page."""
        # Normalize scores and combine
        pagerank_benefit = min(pagerank * 10, 1.0)  # Scale PageRank
        authority_benefit = min(authority_score * 2, 1.0)  # Scale authority

        return (pagerank_benefit * 0.6 + authority_benefit * 0.4)

    def _calculate_content_quality_factor(self, post: Dict) -> float:
        """Calculate content quality factor for link target."""
        quality_metrics = post.get('quality_metrics', {})

        readability = quality_metrics.get('readability_score', 0) / 100
        structure = min(quality_metrics.get('header_count', 0) / 5, 1.0)
        completeness = min(post.get('word_count', 0) / 1000, 1.0)

        return (readability * 0.4 + structure * 0.3 + completeness * 0.3)

    def _generate_context_suggestions(self,
                                    source_post: Dict,
                                    target_post: Dict,
                                    shared_keywords: List[str]) -> List[str]:
        """Generate contextual suggestions for where to place the link."""
        suggestions = []

        # Suggest linking near shared keywords
        for keyword in shared_keywords[:3]:  # Top 3 shared keywords
            suggestions.append(f"Consider linking near mentions of '{keyword}'")

        # Suggest based on content structure
        source_headers = self._extract_headers(source_post.get('content', ''))
        for header in source_headers:
            if any(keyword in header.lower() for keyword in shared_keywords):
                suggestions.append(f"Good placement opportunity in section: '{header}'")

        # Generic suggestions based on content type
        target_title = target_post.get('title', '')
        suggestions.append(f"Natural anchor text: 'learn more about {target_title.lower()}'")

        return suggestions[:5]  # Return top 5 suggestions

    def _extract_headers(self, content: str) -> List[str]:
        """Extract headers from markdown content."""
        import re
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        return headers

    def _determine_priority(self,
                          opportunity_score: float,
                          relevance_score: float,
                          authority_benefit: float) -> str:
        """Determine priority level for the opportunity."""
        if opportunity_score > 0.8 and relevance_score > 0.7:
            return "High"
        elif opportunity_score > 0.6 and (relevance_score > 0.6 or authority_benefit > 0.5):
            return "Medium"
        elif opportunity_score > 0.3:
            return "Low"
        else:
            return "Very Low"

    def identify_cluster_opportunities(self,
                                     min_cluster_size: int = 3,
                                     min_cluster_strength: float = 0.1) -> List[ClusterOpportunity]:
        """
        Identify content cluster optimization opportunities.

        Args:
            min_cluster_size: Minimum number of pages in a cluster
            min_cluster_strength: Minimum cluster strength threshold

        Returns:
            List of cluster optimization opportunities
        """
        logger.info("Identifying content cluster opportunities")

        cluster_opportunities = []

        # Analyze each keyword cluster
        for keyword, page_slugs in self.keyword_index.items():
            if len(page_slugs) < min_cluster_size:
                continue

            cluster_pages = []
            for slug in page_slugs:
                if slug in self.posts_data and slug in self.metrics:
                    page_info = {
                        'slug': slug,
                        'title': self.posts_data[slug].get('title', slug),
                        'pagerank': self.metrics[slug].get('pagerank', 0),
                        'authority': self.metrics[slug].get('authority_score', 0),
                        'word_count': self.posts_data[slug].get('word_count', 0)
                    }
                    cluster_pages.append(page_info)

            if len(cluster_pages) < min_cluster_size:
                continue

            # Calculate cluster strength
            cluster_strength = sum(page['pagerank'] for page in cluster_pages)

            if cluster_strength < min_cluster_strength:
                continue

            # Identify pillar page (highest PageRank)
            pillar_page = max(cluster_pages, key=lambda x: x['pagerank'])

            # Find missing connections
            missing_connections = self._find_missing_cluster_connections(
                cluster_pages, keyword
            )

            # Calculate optimization potential
            optimization_potential = self._calculate_cluster_optimization_potential(
                cluster_pages, missing_connections, cluster_strength
            )

            cluster_opportunity = ClusterOpportunity(
                cluster_keyword=keyword,
                cluster_pages=cluster_pages,
                pillar_page=pillar_page,
                cluster_strength=cluster_strength,
                missing_connections=missing_connections,
                optimization_potential=optimization_potential
            )

            cluster_opportunities.append(cluster_opportunity)

        # Sort by optimization potential
        cluster_opportunities.sort(key=lambda x: x.optimization_potential, reverse=True)
        return cluster_opportunities

    def _find_missing_cluster_connections(self,
                                        cluster_pages: List[Dict],
                                        keyword: str) -> List[Tuple[str, str]]:
        """Find missing internal links within a content cluster."""
        missing_connections = []
        cluster_slugs = {page['slug'] for page in cluster_pages}

        for source_page in cluster_pages:
            source_slug = source_page['slug']
            current_targets = set(
                link['target_slug']
                for link in self.posts_data[source_slug].get('internal_links', [])
            )

            for target_page in cluster_pages:
                target_slug = target_page['slug']

                if (source_slug != target_slug and
                    target_slug not in current_targets and
                    target_slug in cluster_slugs):
                    missing_connections.append((source_slug, target_slug))

        return missing_connections

    def _calculate_cluster_optimization_potential(self,
                                                cluster_pages: List[Dict],
                                                missing_connections: List[Tuple[str, str]],
                                                cluster_strength: float) -> float:
        """Calculate the optimization potential for a content cluster."""
        # Factor 1: Number of missing connections
        connection_factor = len(missing_connections) / max(len(cluster_pages) ** 2, 1)

        # Factor 2: Cluster authority distribution
        pageranks = [page['pagerank'] for page in cluster_pages]
        authority_variance = statistics.variance(pageranks) if len(pageranks) > 1 else 0
        distribution_factor = min(authority_variance * 10, 1.0)

        # Factor 3: Cluster size and strength
        size_factor = min(len(cluster_pages) / 10, 1.0)
        strength_factor = min(cluster_strength, 1.0)

        return (
            connection_factor * 0.4 +
            distribution_factor * 0.3 +
            size_factor * 0.2 +
            strength_factor * 0.1
        )

    def analyze_link_equity_flow(self) -> Dict[str, Any]:
        """Analyze how link equity flows through the content graph."""
        logger.info("Analyzing link equity flow")

        # Calculate flow metrics
        total_links = sum(len(targets) for targets in self.link_graph.values())
        total_pages = len(self.posts_data)

        # Page flow analysis
        page_flow = {}
        for slug in self.posts_data.keys():
            outbound_links = len(self.link_graph.get(slug, set()))
            inbound_links = sum(1 for targets in self.link_graph.values() if slug in targets)

            page_flow[slug] = {
                'outbound_links': outbound_links,
                'inbound_links': inbound_links,
                'flow_ratio': outbound_links / max(inbound_links, 1),
                'authority': self.metrics.get(slug, {}).get('pagerank', 0)
            }

        # Identify flow bottlenecks
        bottlenecks = self._identify_flow_bottlenecks(page_flow)

        # Identify authority sinks
        authority_sinks = self._identify_authority_sinks(page_flow)

        # Calculate overall flow health
        flow_health = self._calculate_flow_health(page_flow, total_links, total_pages)

        return {
            'total_links': total_links,
            'total_pages': total_pages,
            'average_links_per_page': total_links / total_pages,
            'page_flow': page_flow,
            'bottlenecks': bottlenecks,
            'authority_sinks': authority_sinks,
            'flow_health_score': flow_health,
            'recommendations': self._generate_flow_recommendations(
                bottlenecks, authority_sinks, flow_health
            )
        }

    def _identify_flow_bottlenecks(self, page_flow: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify pages that restrict link equity flow."""
        bottlenecks = []

        for slug, flow_data in page_flow.items():
            # High incoming links but low outgoing links
            if (flow_data['inbound_links'] > 3 and
                flow_data['outbound_links'] < 2 and
                flow_data['flow_ratio'] < 0.5):

                post = self.posts_data.get(slug, {})
                bottlenecks.append({
                    'slug': slug,
                    'title': post.get('title', slug),
                    'inbound_links': flow_data['inbound_links'],
                    'outbound_links': flow_data['outbound_links'],
                    'flow_ratio': flow_data['flow_ratio'],
                    'authority': flow_data['authority'],
                    'severity': 'High' if flow_data['inbound_links'] > 5 else 'Medium'
                })

        return sorted(bottlenecks, key=lambda x: x['inbound_links'], reverse=True)

    def _identify_authority_sinks(self, page_flow: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify pages that accumulate authority but don't distribute it."""
        authority_sinks = []

        # Sort pages by authority
        authority_pages = sorted(
            page_flow.items(),
            key=lambda x: x[1]['authority'],
            reverse=True
        )

        # Check top authority pages for poor outbound linking
        for slug, flow_data in authority_pages[:10]:  # Top 10 authority pages
            if (flow_data['authority'] > 0.02 and  # Has significant authority
                flow_data['outbound_links'] < 3):   # But few outbound links

                post = self.posts_data.get(slug, {})
                authority_sinks.append({
                    'slug': slug,
                    'title': post.get('title', slug),
                    'authority': flow_data['authority'],
                    'outbound_links': flow_data['outbound_links'],
                    'potential_benefit': flow_data['authority'] * 10,  # Estimated benefit
                    'recommendation': 'Add 3-5 internal links to distribute authority'
                })

        return authority_sinks

    def _calculate_flow_health(self,
                             page_flow: Dict[str, Dict],
                             total_links: int,
                             total_pages: int) -> float:
        """Calculate overall link equity flow health score."""
        if total_pages == 0:
            return 0.0

        # Factor 1: Link density
        max_possible_links = total_pages * (total_pages - 1)
        link_density = total_links / max_possible_links if max_possible_links > 0 else 0

        # Factor 2: Flow balance (pages with balanced in/out links)
        balanced_pages = sum(
            1 for flow in page_flow.values()
            if 0.3 <= flow['flow_ratio'] <= 3.0 and flow['outbound_links'] > 0
        )
        balance_ratio = balanced_pages / total_pages

        # Factor 3: Authority distribution
        authorities = [flow['authority'] for flow in page_flow.values()]
        if len(authorities) > 1:
            authority_gini = self._calculate_gini_coefficient(authorities)
            distribution_score = 1 - authority_gini  # Lower Gini = better distribution
        else:
            distribution_score = 1.0

        # Combine factors
        health_score = (
            link_density * 0.4 +
            balance_ratio * 0.4 +
            distribution_score * 0.2
        )

        return min(health_score, 1.0)

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        if not values or len(values) < 2:
            return 0.0

        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini coefficient
        cumsum = sum(sorted_values)
        if cumsum == 0:
            return 0.0

        gini_sum = sum((2 * i - n - 1) * val for i, val in enumerate(sorted_values, 1))
        return gini_sum / (n * cumsum)

    def _generate_flow_recommendations(self,
                                     bottlenecks: List[Dict],
                                     authority_sinks: List[Dict],
                                     flow_health: float) -> List[Dict[str, str]]:
        """Generate recommendations for improving link equity flow."""
        recommendations = []

        # Bottleneck recommendations
        if bottlenecks:
            high_severity_bottlenecks = [b for b in bottlenecks if b['severity'] == 'High']
            if high_severity_bottlenecks:
                recommendations.append({
                    'category': 'Flow Bottlenecks',
                    'priority': 'High',
                    'action': f"Add outbound links to {len(high_severity_bottlenecks)} bottleneck pages",
                    'details': f"Pages receiving many links but not distributing authority effectively"
                })

        # Authority sink recommendations
        if authority_sinks:
            recommendations.append({
                'category': 'Authority Distribution',
                'priority': 'Medium',
                'action': f"Improve outbound linking for {len(authority_sinks)} high-authority pages",
                'details': "These pages have authority but aren't distributing it effectively"
            })

        # Overall health recommendations
        if flow_health < 0.3:
            recommendations.append({
                'category': 'Overall Health',
                'priority': 'High',
                'action': "Systematic internal linking improvement needed",
                'details': f"Flow health score is {flow_health:.2f} - consider comprehensive linking audit"
            })
        elif flow_health < 0.6:
            recommendations.append({
                'category': 'Overall Health',
                'priority': 'Medium',
                'action': "Moderate internal linking improvements needed",
                'details': f"Flow health score is {flow_health:.2f} - focus on key optimization areas"
            })

        return recommendations

    def generate_implementation_plan(self,
                                   link_opportunities: List[LinkOpportunity],
                                   cluster_opportunities: List[ClusterOpportunity]) -> Dict[str, Any]:
        """Generate a structured implementation plan for link optimizations."""
        logger.info("Generating link optimization implementation plan")

        # Prioritize opportunities
        high_priority_links = [opp for opp in link_opportunities if opp.priority == "High"]
        medium_priority_links = [opp for opp in link_opportunities if opp.priority == "Medium"]

        # Organize by phases
        implementation_plan = {
            'phase_1_immediate': {
                'description': 'High-impact, easy-to-implement opportunities',
                'link_opportunities': high_priority_links[:10],
                'cluster_work': cluster_opportunities[:3],
                'estimated_effort': 'Low',
                'expected_impact': 'High'
            },
            'phase_2_strategic': {
                'description': 'Medium-priority systematic improvements',
                'link_opportunities': medium_priority_links[:15],
                'cluster_work': cluster_opportunities[3:8],
                'estimated_effort': 'Medium',
                'expected_impact': 'Medium'
            },
            'phase_3_comprehensive': {
                'description': 'Comprehensive linking strategy implementation',
                'link_opportunities': link_opportunities[25:],
                'cluster_work': cluster_opportunities[8:],
                'estimated_effort': 'High',
                'expected_impact': 'High'
            }
        }

        # Generate action items
        action_items = self._generate_action_items(implementation_plan)

        # Calculate expected ROI
        expected_roi = self._calculate_expected_roi(link_opportunities, cluster_opportunities)

        return {
            'implementation_phases': implementation_plan,
            'action_items': action_items,
            'expected_roi': expected_roi,
            'tracking_metrics': self._define_tracking_metrics(),
            'success_criteria': self._define_success_criteria()
        }

    def _generate_action_items(self, implementation_plan: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific action items for implementation."""
        action_items = []

        for phase_name, phase_data in implementation_plan.items():
            phase_num = phase_name.split('_')[1]

            # Link opportunity actions
            for opp in phase_data['link_opportunities'][:5]:  # Top 5 per phase
                action_items.append({
                    'phase': phase_num,
                    'type': 'Link Addition',
                    'action': f"Add link from '{opp.source_title}' to '{opp.target_title}'",
                    'priority': opp.priority,
                    'context': opp.context_suggestions[0] if opp.context_suggestions else "Review content for best placement"
                })

            # Cluster opportunity actions
            for cluster in phase_data['cluster_work'][:2]:  # Top 2 per phase
                action_items.append({
                    'phase': phase_num,
                    'type': 'Cluster Optimization',
                    'action': f"Optimize '{cluster.cluster_keyword}' content cluster",
                    'priority': 'Medium',
                    'context': f"Connect {len(cluster.missing_connections)} missing page relationships"
                })

        return action_items

    def _calculate_expected_roi(self,
                              link_opportunities: List[LinkOpportunity],
                              cluster_opportunities: List[ClusterOpportunity]) -> Dict[str, float]:
        """Calculate expected ROI for optimization efforts."""
        # Simplified ROI calculation based on opportunity scores
        total_link_value = sum(opp.opportunity_score for opp in link_opportunities)
        total_cluster_value = sum(cluster.optimization_potential for cluster in cluster_opportunities)

        # Effort estimation (simplified)
        link_effort = len(link_opportunities) * 0.5  # 0.5 hours per link
        cluster_effort = len(cluster_opportunities) * 2  # 2 hours per cluster

        total_effort = link_effort + cluster_effort
        total_value = total_link_value + total_cluster_value

        return {
            'total_expected_value': total_value,
            'total_estimated_effort_hours': total_effort,
            'value_per_hour': total_value / max(total_effort, 1),
            'link_opportunities_value': total_link_value,
            'cluster_opportunities_value': total_cluster_value
        }

    def _define_tracking_metrics(self) -> List[str]:
        """Define metrics to track optimization success."""
        return [
            'Average PageRank scores',
            'Internal link density',
            'Pages with 0 inbound links',
            'Authority distribution (Gini coefficient)',
            'Cluster connectivity scores',
            'Organic search traffic growth',
            'Page-to-page navigation patterns',
            'Content discovery rates'
        ]

    def _define_success_criteria(self) -> Dict[str, str]:
        """Define success criteria for optimization efforts."""
        return {
            'link_density_improvement': 'Increase internal link density by 20%',
            'orphaned_pages_reduction': 'Reduce pages with 0 inbound links by 50%',
            'authority_distribution': 'Improve authority distribution (lower Gini coefficient)',
            'cluster_connectivity': 'Achieve 80% connectivity within content clusters',
            'pagerank_improvement': 'Increase average PageRank scores by 15%',
            'traffic_impact': 'Measure organic traffic improvement over 3 months'
        }