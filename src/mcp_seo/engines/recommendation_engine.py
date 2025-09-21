"""
SEO Recommendation Engine for comprehensive SEO analysis and optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity levels for SEO issues and recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RecommendationType(Enum):
    """Types of SEO recommendations."""
    TECHNICAL = "technical"
    CONTENT = "content"
    KEYWORDS = "keywords"
    LINKS = "links"
    PERFORMANCE = "performance"
    UX = "user_experience"


@dataclass
class SEORecommendation:
    """Individual SEO recommendation."""
    title: str
    description: str
    priority: SeverityLevel
    category: RecommendationType
    impact: str
    effort: str
    affected_pages: Optional[int] = None
    specific_issues: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    resources: Optional[List[str]] = None


@dataclass
class SEOScore:
    """SEO scoring breakdown."""
    overall_score: int  # 0-100
    technical_score: int
    content_score: int
    keywords_score: int
    links_score: int
    performance_score: int
    breakdown: Dict[str, Any]


class SEORecommendationEngine:
    """Advanced SEO recommendation engine with comprehensive analysis."""

    def __init__(self):
        self.recommendations = []
        self.score_weights = {
            'technical': 0.25,
            'content': 0.20,
            'keywords': 0.20,
            'links': 0.20,
            'performance': 0.15
        }

    def analyze_keyword_performance(self, keyword_data: Dict[str, Any]) -> List[SEORecommendation]:
        """Analyze keyword performance and generate recommendations."""
        recommendations = []

        # Check for missing keywords in top positions
        missing_rankings = []
        low_rankings = []

        for keyword, data in keyword_data.items():
            position = data.get('gitalchemy_position')
            search_volume = data.get('search_volume', {}).get('search_volume', 0)
            difficulty = data.get('difficulty', {}).get('difficulty', 0)

            if not position:
                missing_rankings.append({
                    'keyword': keyword,
                    'search_volume': search_volume,
                    'difficulty': difficulty
                })
            elif position > 20:
                low_rankings.append({
                    'keyword': keyword,
                    'position': position,
                    'search_volume': search_volume
                })

        # Generate recommendations for missing rankings
        if missing_rankings:
            high_volume_missing = [k for k in missing_rankings if k['search_volume'] > 1000]
            if high_volume_missing:
                recommendations.append(SEORecommendation(
                    title="Target High-Volume Missing Keywords",
                    description=f"Create dedicated content for {len(high_volume_missing)} high-volume keywords where you're not ranking",
                    priority=SeverityLevel.HIGH,
                    category=RecommendationType.CONTENT,
                    impact="High - Could drive significant new organic traffic",
                    effort="Medium - Requires content creation and optimization",
                    affected_pages=len(high_volume_missing),
                    specific_issues=[k['keyword'] for k in high_volume_missing[:5]],
                    action_items=[
                        "Create comprehensive content for each target keyword",
                        "Optimize page titles and meta descriptions",
                        "Build internal links to new content",
                        "Monitor ranking progress weekly"
                    ]
                ))

        # Generate recommendations for low rankings
        if low_rankings:
            recommendations.append(SEORecommendation(
                title="Improve Low-Ranking Keyword Positions",
                description=f"Optimize content for {len(low_rankings)} keywords ranking below position 20",
                priority=SeverityLevel.MEDIUM,
                category=RecommendationType.CONTENT,
                impact="Medium - Could improve existing traffic",
                effort="Low - Requires content optimization",
                affected_pages=len(low_rankings),
                specific_issues=[f"{k['keyword']} (position {k['position']})" for k in low_rankings[:5]],
                action_items=[
                    "Enhance content depth and quality",
                    "Improve keyword optimization",
                    "Add relevant internal links",
                    "Optimize user engagement signals"
                ]
            ))

        return recommendations

    def analyze_technical_issues(self, onpage_data: Dict[str, Any]) -> List[SEORecommendation]:
        """Analyze technical SEO issues and generate recommendations."""
        recommendations = []

        if not onpage_data or 'summary' not in onpage_data:
            return recommendations

        summary = onpage_data['summary']

        # Critical technical issues
        critical_issues = summary.get('critical_issues', 0)
        if critical_issues > 0:
            recommendations.append(SEORecommendation(
                title="Fix Critical Technical Issues",
                description=f"Address {critical_issues} critical technical SEO issues immediately",
                priority=SeverityLevel.CRITICAL,
                category=RecommendationType.TECHNICAL,
                impact="Critical - These issues prevent proper indexing",
                effort="High - Requires technical implementation",
                affected_pages=critical_issues,
                action_items=[
                    "Review and fix broken pages (404 errors)",
                    "Implement proper redirects for moved content",
                    "Fix crawl errors and blocked resources",
                    "Ensure all pages are accessible to search engines"
                ]
            ))

        # Duplicate content issues
        duplicate_titles = summary.get('duplicate_title_tags', 0)
        duplicate_descriptions = summary.get('duplicate_meta_descriptions', 0)

        if duplicate_titles > 5 or duplicate_descriptions > 5:
            recommendations.append(SEORecommendation(
                title="Eliminate Duplicate Content Issues",
                description=f"Fix {duplicate_titles} duplicate titles and {duplicate_descriptions} duplicate meta descriptions",
                priority=SeverityLevel.HIGH,
                category=RecommendationType.TECHNICAL,
                impact="High - Duplicate content confuses search engines",
                effort="Medium - Requires content review and updates",
                affected_pages=max(duplicate_titles, duplicate_descriptions),
                action_items=[
                    "Create unique title tags for each page",
                    "Write unique meta descriptions",
                    "Implement canonical tags where appropriate",
                    "Review content for uniqueness"
                ]
            ))

        return recommendations

    def analyze_content_opportunities(self, content_data: Dict[str, Any]) -> List[SEORecommendation]:
        """Analyze content opportunities and generate recommendations."""
        recommendations = []

        # Check for thin content
        if 'pages' in content_data:
            thin_content_pages = [
                page for page in content_data['pages']
                if page.get('word_count', 0) < 300
            ]

            if len(thin_content_pages) > 0:
                recommendations.append(SEORecommendation(
                    title="Expand Thin Content Pages",
                    description=f"Enhance {len(thin_content_pages)} pages with insufficient content",
                    priority=SeverityLevel.MEDIUM,
                    category=RecommendationType.CONTENT,
                    impact="Medium - Better content improves rankings",
                    effort="Medium - Requires content expansion",
                    affected_pages=len(thin_content_pages),
                    action_items=[
                        "Add comprehensive information to each page",
                        "Include relevant images and media",
                        "Add FAQ sections or detailed explanations",
                        "Ensure content serves user intent"
                    ]
                ))

        return recommendations

    def analyze_link_opportunities(self, pagerank_data: Dict[str, Any]) -> List[SEORecommendation]:
        """Analyze internal linking opportunities."""
        recommendations = []

        # Check for orphaned pages
        orphaned_pages = pagerank_data.get('orphaned_pages', [])
        if len(orphaned_pages) > 0:
            recommendations.append(SEORecommendation(
                title="Fix Orphaned Pages",
                description=f"Add internal links to {len(orphaned_pages)} orphaned pages",
                priority=SeverityLevel.HIGH,
                category=RecommendationType.LINKS,
                impact="High - Orphaned pages can't be discovered or indexed properly",
                effort="Low - Add links from relevant content",
                affected_pages=len(orphaned_pages),
                action_items=[
                    "Identify relevant pages to link from",
                    "Add contextual internal links",
                    "Include pages in navigation if appropriate",
                    "Add to XML sitemap"
                ]
            ))

        # Check for link equity distribution
        if 'link_opportunities' in pagerank_data:
            high_authority_pages = pagerank_data['link_opportunities'].get('high_authority_pages', [])
            if len(high_authority_pages) > 0:
                recommendations.append(SEORecommendation(
                    title="Leverage High-Authority Pages",
                    description=f"Use {len(high_authority_pages)} high-authority pages to boost other content",
                    priority=SeverityLevel.MEDIUM,
                    category=RecommendationType.LINKS,
                    impact="Medium - Strategic link placement can boost rankings",
                    effort="Low - Add strategic internal links",
                    affected_pages=len(high_authority_pages),
                    action_items=[
                        "Link from high-authority pages to target content",
                        "Create topic clusters around authority pages",
                        "Add relevant contextual links",
                        "Monitor link equity flow"
                    ]
                ))

        return recommendations

    def generate_comprehensive_recommendations(
        self,
        keyword_data: Optional[Dict[str, Any]] = None,
        onpage_data: Optional[Dict[str, Any]] = None,
        content_data: Optional[Dict[str, Any]] = None,
        pagerank_data: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive SEO recommendations from all available data."""

        all_recommendations = []

        # Analyze each data source
        if keyword_data:
            all_recommendations.extend(self.analyze_keyword_performance(keyword_data))

        if onpage_data:
            all_recommendations.extend(self.analyze_technical_issues(onpage_data))

        if content_data:
            all_recommendations.extend(self.analyze_content_opportunities(content_data))

        if pagerank_data:
            all_recommendations.extend(self.analyze_link_opportunities(pagerank_data))

        # Calculate SEO score
        seo_score = self._calculate_seo_score(
            keyword_data, onpage_data, content_data, pagerank_data, performance_data
        )

        # Prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(all_recommendations)

        # Generate action plan
        action_plan = self._generate_action_plan(prioritized_recommendations)

        return {
            'seo_score': seo_score,
            'recommendations': [rec.__dict__ for rec in prioritized_recommendations],
            'action_plan': action_plan,
            'summary': {
                'total_recommendations': len(prioritized_recommendations),
                'critical_issues': len([r for r in prioritized_recommendations if r.priority == SeverityLevel.CRITICAL]),
                'high_priority': len([r for r in prioritized_recommendations if r.priority == SeverityLevel.HIGH]),
                'medium_priority': len([r for r in prioritized_recommendations if r.priority == SeverityLevel.MEDIUM]),
                'low_priority': len([r for r in prioritized_recommendations if r.priority == SeverityLevel.LOW]),
                'categories': {
                    'technical': len([r for r in prioritized_recommendations if r.category == RecommendationType.TECHNICAL]),
                    'content': len([r for r in prioritized_recommendations if r.category == RecommendationType.CONTENT]),
                    'keywords': len([r for r in prioritized_recommendations if r.category == RecommendationType.KEYWORDS]),
                    'links': len([r for r in prioritized_recommendations if r.category == RecommendationType.LINKS]),
                    'performance': len([r for r in prioritized_recommendations if r.category == RecommendationType.PERFORMANCE])
                }
            }
        }

    def _calculate_seo_score(
        self,
        keyword_data: Optional[Dict[str, Any]] = None,
        onpage_data: Optional[Dict[str, Any]] = None,
        content_data: Optional[Dict[str, Any]] = None,
        pagerank_data: Optional[Dict[str, Any]] = None,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> SEOScore:
        """Calculate overall SEO score based on available data."""

        # Technical score
        technical_score = 100
        if onpage_data and 'summary' in onpage_data:
            summary = onpage_data['summary']
            critical_issues = summary.get('critical_issues', 0)
            high_issues = summary.get('high_priority_issues', 0)

            # Deduct points for issues
            technical_score -= min(critical_issues * 20, 60)  # Max 60 points deduction
            technical_score -= min(high_issues * 10, 30)      # Max 30 points deduction

        # Content score
        content_score = 80  # Default assuming decent content
        if content_data and 'pages' in content_data:
            thin_pages = len([p for p in content_data['pages'] if p.get('word_count', 0) < 300])
            total_pages = len(content_data['pages'])
            if total_pages > 0:
                thin_ratio = thin_pages / total_pages
                content_score = max(40, 100 - int(thin_ratio * 100))

        # Keywords score
        keywords_score = 70  # Default
        if keyword_data:
            total_keywords = len(keyword_data)
            ranking_keywords = len([k for k, v in keyword_data.items() if v.get('gitalchemy_position')])
            if total_keywords > 0:
                ranking_ratio = ranking_keywords / total_keywords
                keywords_score = int(ranking_ratio * 100)

        # Links score
        links_score = 80  # Default
        if pagerank_data:
            orphaned_count = len(pagerank_data.get('orphaned_pages', []))
            if 'basic_metrics' in pagerank_data:
                total_pages = pagerank_data['basic_metrics'].get('total_pages', 1)
                orphaned_ratio = orphaned_count / total_pages
                links_score = max(40, 100 - int(orphaned_ratio * 100))

        # Performance score
        performance_score = 75  # Default
        if performance_data and 'lighthouse_analysis' in performance_data:
            lighthouse = performance_data['lighthouse_analysis']
            performance_score = lighthouse.get('performance_score', 75)

        # Calculate weighted overall score
        overall_score = int(
            technical_score * self.score_weights['technical'] +
            content_score * self.score_weights['content'] +
            keywords_score * self.score_weights['keywords'] +
            links_score * self.score_weights['links'] +
            performance_score * self.score_weights['performance']
        )

        return SEOScore(
            overall_score=overall_score,
            technical_score=technical_score,
            content_score=content_score,
            keywords_score=keywords_score,
            links_score=links_score,
            performance_score=performance_score,
            breakdown={
                'technical': {'score': technical_score, 'weight': self.score_weights['technical']},
                'content': {'score': content_score, 'weight': self.score_weights['content']},
                'keywords': {'score': keywords_score, 'weight': self.score_weights['keywords']},
                'links': {'score': links_score, 'weight': self.score_weights['links']},
                'performance': {'score': performance_score, 'weight': self.score_weights['performance']}
            }
        )

    def _prioritize_recommendations(self, recommendations: List[SEORecommendation]) -> List[SEORecommendation]:
        """Sort recommendations by priority and impact."""
        priority_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3,
            SeverityLevel.INFO: 4
        }

        return sorted(recommendations, key=lambda r: (priority_order[r.priority], r.affected_pages or 0), reverse=True)

    def _generate_action_plan(self, recommendations: List[SEORecommendation]) -> Dict[str, Any]:
        """Generate a structured action plan based on recommendations."""

        # Group by timeframe
        immediate_actions = [r for r in recommendations if r.priority in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        short_term_actions = [r for r in recommendations if r.priority == SeverityLevel.MEDIUM]
        long_term_actions = [r for r in recommendations if r.priority == SeverityLevel.LOW]

        return {
            'immediate_actions': {
                'timeframe': '0-2 weeks',
                'count': len(immediate_actions),
                'actions': [{'title': r.title, 'category': r.category.value} for r in immediate_actions[:5]]
            },
            'short_term_actions': {
                'timeframe': '2-8 weeks',
                'count': len(short_term_actions),
                'actions': [{'title': r.title, 'category': r.category.value} for r in short_term_actions[:5]]
            },
            'long_term_actions': {
                'timeframe': '2-6 months',
                'count': len(long_term_actions),
                'actions': [{'title': r.title, 'category': r.category.value} for r in long_term_actions[:5]]
            },
            'estimated_impact': {
                'traffic_increase': '15-40%',
                'ranking_improvement': '10-25 positions',
                'technical_health': 'Significant improvement'
            }
        }