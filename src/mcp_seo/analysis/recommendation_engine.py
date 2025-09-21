"""
SEO Recommendation Engine Module

This module provides intelligent SEO recommendation generation based on comprehensive
analysis results. Extracted and enhanced from legacy SEO analyzer scripts, it offers
customizable recommendation strategies for different domains and industries.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class RecommendationPriority(Enum):
    """Priority levels for SEO recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(Enum):
    """Types of SEO recommendations."""

    KEYWORD_TARGETING = "keyword_targeting"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    TECHNICAL_SEO = "technical_seo"
    CONTENT_STRATEGY = "content_strategy"
    LINK_BUILDING = "link_building"
    PERFORMANCE = "performance"
    LOCAL_SEO = "local_seo"
    MOBILE_SEO = "mobile_seo"


@dataclass
class SEORecommendation:
    """Individual SEO recommendation with metadata."""

    title: str
    description: str
    priority: RecommendationPriority
    recommendation_type: RecommendationType
    effort_level: str  # "low", "medium", "high"
    estimated_impact: str  # "low", "medium", "high"
    keywords_affected: List[str]
    implementation_steps: List[str]
    success_metrics: List[str]
    timeframe: str  # e.g., "1-2 weeks", "1-3 months"
    confidence_score: float  # 0-100


@dataclass
class RecommendationConfig:
    """Configuration for recommendation generation."""

    industry_vertical: Optional[str] = None
    business_type: str = "generic"  # "ecommerce", "saas", "blog", "local", "generic"
    target_audience: Optional[str] = None
    available_resources: str = "medium"  # "low", "medium", "high"
    technical_capability: str = "medium"  # "low", "medium", "high"
    content_strategy: str = "balanced"  # "aggressive", "balanced", "conservative"
    exclude_types: List[RecommendationType] = None


class SEORecommendationEngine:
    """
    Intelligent SEO recommendation engine with customizable strategies.

    Enhanced version of the legacy _generate_recommendations method with
    comprehensive analysis capabilities and industry-specific recommendations.
    """

    def __init__(self, config: Optional[RecommendationConfig] = None):
        """
        Initialize recommendation engine.

        Args:
            config: Configuration for recommendation generation
        """
        self.config = config or RecommendationConfig()
        self.custom_templates = {}
        self.keyword_analyzers = []

    def generate_comprehensive_recommendations(
        self, analysis_results: Dict[str, Any], target_domain: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive SEO recommendations based on analysis results.

        Enhanced version of the legacy recommendation generation with structured
        output and configurable recommendation strategies.

        Args:
            analysis_results: Results from keyword ranking analysis
            target_domain: Primary domain being analyzed

        Returns:
            Comprehensive recommendations with priorities and implementation details
        """
        recommendations = []

        # Analyze current state
        analysis_summary = self._analyze_current_state(analysis_results, target_domain)

        # Generate keyword-focused recommendations
        keyword_recs = self._generate_keyword_recommendations(
            analysis_results, analysis_summary
        )
        recommendations.extend(keyword_recs)

        # Generate competitor-focused recommendations
        competitor_recs = self._generate_competitor_recommendations(
            analysis_results, analysis_summary
        )
        recommendations.extend(competitor_recs)

        # Generate technical SEO recommendations
        technical_recs = self._generate_technical_recommendations(
            analysis_results, analysis_summary
        )
        recommendations.extend(technical_recs)

        # Generate content strategy recommendations
        content_recs = self._generate_content_recommendations(
            analysis_results, analysis_summary
        )
        recommendations.extend(content_recs)

        # Generate performance recommendations
        performance_recs = self._generate_performance_recommendations(
            analysis_results, analysis_summary
        )
        recommendations.extend(performance_recs)

        # Add industry-specific recommendations
        if self.config.industry_vertical:
            industry_recs = self._generate_industry_specific_recommendations(
                analysis_results, analysis_summary, self.config.industry_vertical
            )
            recommendations.extend(industry_recs)

        # Filter and prioritize recommendations
        filtered_recs = self._filter_recommendations(recommendations)
        prioritized_recs = self._prioritize_recommendations(
            filtered_recs, analysis_summary
        )

        # Generate implementation roadmap
        roadmap = self._generate_implementation_roadmap(prioritized_recs)

        return {
            "target_domain": target_domain,
            "analysis_summary": analysis_summary,
            "recommendations": [
                self._recommendation_to_dict(rec) for rec in prioritized_recs
            ],
            "implementation_roadmap": roadmap,
            "quick_wins": [
                rec
                for rec in prioritized_recs
                if rec.effort_level == "low"
                and rec.estimated_impact in ["medium", "high"]
            ],
            "high_impact_initiatives": [
                rec for rec in prioritized_recs if rec.estimated_impact == "high"
            ],
            "total_recommendations": len(prioritized_recs),
            "generated_at": datetime.now().isoformat(),
            "config_used": self._config_to_dict(),
        }

    def generate_quick_recommendations(
        self, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate quick, actionable recommendations similar to legacy format.

        Maintains compatibility with legacy recommendation format while providing
        enhanced analysis capabilities.

        Args:
            analysis_results: Results from keyword ranking analysis

        Returns:
            List of formatted recommendation strings
        """
        recommendations = []

        # Extract keyword results
        keyword_results = {
            k: v for k, v in analysis_results.items() if not k.startswith("_")
        }

        # Check for missing rankings (enhanced logic)
        missing_rankings = []
        poor_rankings = []

        for keyword, data in keyword_results.items():
            position = data.get("target_position")
            if not position:
                missing_rankings.append(keyword)
            elif position > 20:
                poor_rankings.append((keyword, position))

        if missing_rankings:
            recommendations.append(
                f"🔍 Target missing keywords: {', '.join(missing_rankings[:3])}"
                + (
                    f" (+{len(missing_rankings) - 3} more)"
                    if len(missing_rankings) > 3
                    else ""
                )
            )

        if poor_rankings:
            worst_keywords = sorted(poor_rankings, key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(
                f"📈 Improve rankings for: {', '.join([f'{kw} (#{pos})' for kw, pos in worst_keywords])}"
            )

        # Check for competitor opportunities (enhanced)
        total_competitors = sum(
            len(data.get("competitor_analysis", []))
            for data in keyword_results.values()
        )
        high_competitor_keywords = sum(
            1
            for data in keyword_results.values()
            if len(data.get("competitor_analysis", [])) > 3
        )

        if total_competitors > 0:
            recommendations.append(
                f"🏆 {total_competitors} competitor opportunities identified across {high_competitor_keywords} high-competition keywords"
            )

        # Opportunity-based recommendations
        high_opportunity_keywords = [
            kw
            for kw, data in keyword_results.items()
            if data.get("opportunity_score", 0) > 70
        ]

        if high_opportunity_keywords:
            recommendations.append(
                f"💡 High-opportunity keywords: {', '.join(high_opportunity_keywords[:3])}"
                + (
                    f" (+{len(high_opportunity_keywords) - 3} more)"
                    if len(high_opportunity_keywords) > 3
                    else ""
                )
            )

        # SERP features opportunities
        featured_snippet_opportunities = [
            kw
            for kw, data in keyword_results.values()
            if data.get("serp_features", {}).get("featured_snippet")
            and data.get("target_position", 999) <= 10
        ]

        if featured_snippet_opportunities:
            recommendations.append(
                f"⭐ Featured snippet opportunities: {', '.join(featured_snippet_opportunities[:2])}"
            )

        # Generic recommendations based on business type
        recommendations.extend(self._get_generic_recommendations())

        return recommendations

    def add_custom_template(
        self, name: str, template_func: Callable[[Dict, Dict], List[SEORecommendation]]
    ):
        """Add custom recommendation template."""
        self.custom_templates[name] = template_func

    def add_keyword_analyzer(self, analyzer_func: Callable[[str, Dict], Dict]):
        """Add custom keyword analysis function."""
        self.keyword_analyzers.append(analyzer_func)

    def _analyze_current_state(
        self, analysis_results: Dict[str, Any], target_domain: str
    ) -> Dict[str, Any]:
        """Analyze current SEO state from results."""
        keyword_results = {
            k: v for k, v in analysis_results.items() if not k.startswith("_")
        }
        summary = analysis_results.get("_summary", {})

        if not keyword_results:
            return {"status": "insufficient_data"}

        # Ranking analysis
        total_keywords = len(keyword_results)
        ranked_keywords = sum(
            1 for data in keyword_results.values() if data.get("target_position")
        )
        top_10_rankings = sum(
            1
            for data in keyword_results.values()
            if data.get("target_position", 999) <= 10
        )
        top_3_rankings = sum(
            1
            for data in keyword_results.values()
            if data.get("target_position", 999) <= 3
        )

        # Opportunity analysis
        avg_opportunity = (
            sum(data.get("opportunity_score", 0) for data in keyword_results.values())
            / total_keywords
        )
        high_opportunity_count = sum(
            1
            for data in keyword_results.values()
            if data.get("opportunity_score", 0) > 70
        )

        # Competitive analysis
        total_competitors = sum(
            len(data.get("competitor_analysis", []))
            for data in keyword_results.values()
        )
        avg_competitors = (
            total_competitors / total_keywords if total_keywords > 0 else 0
        )

        # Difficulty analysis
        difficulty_scores = [
            data.get("difficulty_score")
            for data in keyword_results.values()
            if data.get("difficulty_score")
        ]
        avg_difficulty = (
            sum(difficulty_scores) / len(difficulty_scores)
            if difficulty_scores
            else None
        )

        return {
            "target_domain": target_domain,
            "total_keywords": total_keywords,
            "ranking_performance": {
                "ranked_keywords": ranked_keywords,
                "ranking_percentage": (ranked_keywords / total_keywords) * 100,
                "top_10_percentage": (top_10_rankings / total_keywords) * 100,
                "top_3_percentage": (top_3_rankings / total_keywords) * 100,
            },
            "opportunity_metrics": {
                "average_opportunity_score": avg_opportunity,
                "high_opportunity_keywords": high_opportunity_count,
                "opportunity_percentage": (high_opportunity_count / total_keywords)
                * 100,
            },
            "competitive_landscape": {
                "average_competitors_per_keyword": avg_competitors,
                "total_competitor_instances": total_competitors,
                "high_competition_keywords": sum(
                    1
                    for data in keyword_results.values()
                    if len(data.get("competitor_analysis", [])) > 3
                ),
            },
            "difficulty_assessment": {
                "average_difficulty": avg_difficulty,
                "difficulty_distribution": self._analyze_difficulty_distribution(
                    difficulty_scores
                ),
            },
            "serp_features": self._analyze_serp_features_summary(keyword_results),
            "status": "analyzed",
        }

    def _generate_keyword_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Generate keyword-focused recommendations."""
        recommendations = []
        keyword_results = {
            k: v for k, v in analysis_results.items() if not k.startswith("_")
        }

        if analysis_summary.get("status") != "analyzed":
            return recommendations

        ranking_perf = analysis_summary["ranking_performance"]
        opportunity_metrics = analysis_summary["opportunity_metrics"]

        # Missing keywords recommendation
        missing_keywords = [
            kw
            for kw, data in keyword_results.items()
            if not data.get("target_position")
        ]
        if missing_keywords:
            recommendations.append(
                SEORecommendation(
                    title="Target Missing Keywords",
                    description=f"Create content targeting {len(missing_keywords)} keywords where you currently have no ranking.",
                    priority=RecommendationPriority.HIGH,
                    recommendation_type=RecommendationType.KEYWORD_TARGETING,
                    effort_level="medium",
                    estimated_impact="high",
                    keywords_affected=missing_keywords[:10],  # Limit for display
                    implementation_steps=[
                        "Conduct keyword research for content gaps",
                        "Create comprehensive content for high-volume missing keywords",
                        "Optimize existing pages for related missing keywords",
                        "Build internal linking structure to new content",
                    ],
                    success_metrics=[
                        "Rankings for target keywords",
                        "Organic traffic increase",
                        "SERP visibility improvement",
                    ],
                    timeframe="2-4 months",
                    confidence_score=85.0,
                )
            )

        # Low-hanging fruit recommendations
        improvement_opportunities = [
            (kw, data)
            for kw, data in keyword_results.items()
            if data.get("target_position", 999) in range(11, 21)
            and data.get("opportunity_score", 0) > 60
        ]

        if improvement_opportunities:
            recommendations.append(
                SEORecommendation(
                    title="Optimize Page 2 Rankings",
                    description=f"Improve {len(improvement_opportunities)} keywords ranking on page 2 to reach page 1.",
                    priority=RecommendationPriority.HIGH,
                    recommendation_type=RecommendationType.KEYWORD_TARGETING,
                    effort_level="low",
                    estimated_impact="medium",
                    keywords_affected=[kw for kw, _ in improvement_opportunities[:5]],
                    implementation_steps=[
                        "Analyze top-ranking competitor content",
                        "Enhance content quality and depth",
                        "Improve on-page SEO optimization",
                        "Build relevant internal links",
                    ],
                    success_metrics=[
                        "Keyword ranking improvements",
                        "Page 1 rankings increase",
                        "Click-through rate improvement",
                    ],
                    timeframe="1-2 months",
                    confidence_score=75.0,
                )
            )

        # High opportunity keywords
        if opportunity_metrics["high_opportunity_keywords"] > 0:
            high_opp_keywords = [
                kw
                for kw, data in keyword_results.items()
                if data.get("opportunity_score", 0) > 70
            ]

            recommendations.append(
                SEORecommendation(
                    title="Focus on High-Opportunity Keywords",
                    description=f"Prioritize {len(high_opp_keywords)} keywords with high opportunity scores for maximum impact.",
                    priority=RecommendationPriority.MEDIUM,
                    recommendation_type=RecommendationType.KEYWORD_TARGETING,
                    effort_level="medium",
                    estimated_impact="high",
                    keywords_affected=high_opp_keywords[:8],
                    implementation_steps=[
                        "Analyze search intent for high-opportunity keywords",
                        "Create targeted landing pages",
                        "Optimize content for user experience",
                        "Monitor and iterate based on performance",
                    ],
                    success_metrics=[
                        "Organic traffic growth",
                        "Conversion rate improvement",
                        "Keyword ranking progress",
                    ],
                    timeframe="2-3 months",
                    confidence_score=80.0,
                )
            )

        return recommendations

    def _generate_competitor_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Generate competitor-focused recommendations."""
        recommendations = []
        keyword_results = {
            k: v for k, v in analysis_results.items() if not k.startswith("_")
        }

        competitive_landscape = analysis_summary.get("competitive_landscape", {})
        avg_competitors = competitive_landscape.get(
            "average_competitors_per_keyword", 0
        )

        if avg_competitors > 2:
            # High competition analysis
            high_comp_keywords = [
                kw
                for kw, data in keyword_results.items()
                if len(data.get("competitor_analysis", [])) > 3
            ]

            if high_comp_keywords:
                recommendations.append(
                    SEORecommendation(
                        title="Competitive Intelligence Strategy",
                        description=f"Analyze {len(high_comp_keywords)} highly competitive keywords to identify content gaps and opportunities.",
                        priority=RecommendationPriority.MEDIUM,
                        recommendation_type=RecommendationType.COMPETITOR_ANALYSIS,
                        effort_level="medium",
                        estimated_impact="medium",
                        keywords_affected=high_comp_keywords[:5],
                        implementation_steps=[
                            "Conduct detailed competitor content analysis",
                            "Identify unique value propositions",
                            "Create differentiated content strategies",
                            "Monitor competitor ranking changes",
                        ],
                        success_metrics=[
                            "Competitive keyword rankings",
                            "Content engagement metrics",
                            "Market share indicators",
                        ],
                        timeframe="3-6 months",
                        confidence_score=70.0,
                    )
                )

        # Featured snippet opportunities from competitors
        snippet_opportunities = []
        for kw, data in keyword_results.items():
            competitors = data.get("competitor_analysis", [])
            if (
                any(comp.get("featured_snippet") for comp in competitors)
                and data.get("target_position", 999) <= 10
            ):
                snippet_opportunities.append(kw)

        if snippet_opportunities:
            recommendations.append(
                SEORecommendation(
                    title="Featured Snippet Optimization",
                    description=f"Target {len(snippet_opportunities)} featured snippet opportunities where competitors currently rank.",
                    priority=RecommendationPriority.HIGH,
                    recommendation_type=RecommendationType.COMPETITOR_ANALYSIS,
                    effort_level="low",
                    estimated_impact="high",
                    keywords_affected=snippet_opportunities[:3],
                    implementation_steps=[
                        "Analyze competitor featured snippet content",
                        "Structure content with clear answers",
                        "Optimize for question-based queries",
                        "Use proper schema markup",
                    ],
                    success_metrics=[
                        "Featured snippet captures",
                        "Zero-click search visibility",
                        "CTR improvement",
                    ],
                    timeframe="1-2 months",
                    confidence_score=85.0,
                )
            )

        return recommendations

    def _generate_technical_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Generate technical SEO recommendations."""
        recommendations = []

        # Always include core technical recommendations
        recommendations.append(
            SEORecommendation(
                title="Core Web Vitals Optimization",
                description="Optimize page loading speed, interactivity, and visual stability for better user experience and rankings.",
                priority=RecommendationPriority.HIGH,
                recommendation_type=RecommendationType.TECHNICAL_SEO,
                effort_level="medium",
                estimated_impact="high",
                keywords_affected=[],
                implementation_steps=[
                    "Audit current Core Web Vitals scores",
                    "Optimize images and compress assets",
                    "Implement lazy loading",
                    "Minimize JavaScript and CSS",
                    "Optimize server response times",
                ],
                success_metrics=[
                    "Core Web Vitals scores",
                    "Page speed improvements",
                    "User experience metrics",
                ],
                timeframe="2-4 weeks",
                confidence_score=90.0,
            )
        )

        recommendations.append(
            SEORecommendation(
                title="Mobile-First Optimization",
                description="Ensure website is fully optimized for mobile devices and mobile-first indexing.",
                priority=RecommendationPriority.HIGH,
                recommendation_type=RecommendationType.MOBILE_SEO,
                effort_level="medium",
                estimated_impact="high",
                keywords_affected=[],
                implementation_steps=[
                    "Audit mobile usability",
                    "Optimize touch targets and navigation",
                    "Ensure responsive design consistency",
                    "Test mobile page speed",
                    "Optimize mobile user experience",
                ],
                success_metrics=[
                    "Mobile usability scores",
                    "Mobile traffic growth",
                    "Mobile ranking improvements",
                ],
                timeframe="3-6 weeks",
                confidence_score=85.0,
            )
        )

        return recommendations

    def _generate_content_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Generate content strategy recommendations."""
        recommendations = []

        ranking_perf = analysis_summary.get("ranking_performance", {})

        if ranking_perf.get("top_10_percentage", 0) < 30:
            recommendations.append(
                SEORecommendation(
                    title="Content Quality Enhancement",
                    description="Improve content quality and depth to increase top 10 rankings from current low percentage.",
                    priority=RecommendationPriority.HIGH,
                    recommendation_type=RecommendationType.CONTENT_STRATEGY,
                    effort_level="high",
                    estimated_impact="high",
                    keywords_affected=[],
                    implementation_steps=[
                        "Conduct content audit for top-performing pages",
                        "Identify content gaps and improvement opportunities",
                        "Create comprehensive, in-depth content",
                        "Implement content optimization best practices",
                        "Regular content updates and maintenance",
                    ],
                    success_metrics=[
                        "Top 10 ranking percentage",
                        "Content engagement metrics",
                        "Time on page improvements",
                    ],
                    timeframe="3-6 months",
                    confidence_score=80.0,
                )
            )

        # Industry-specific content recommendations
        if self.config.business_type == "saas":
            recommendations.append(
                SEORecommendation(
                    title="SaaS-Specific Content Strategy",
                    description="Create content that addresses software solution searches and comparison queries.",
                    priority=RecommendationPriority.MEDIUM,
                    recommendation_type=RecommendationType.CONTENT_STRATEGY,
                    effort_level="medium",
                    estimated_impact="medium",
                    keywords_affected=[],
                    implementation_steps=[
                        "Create comparison and alternative pages",
                        "Develop use case and implementation guides",
                        "Build integration and API documentation",
                        "Create customer success stories",
                    ],
                    success_metrics=[
                        "Lead generation increase",
                        "Demo request improvements",
                        "Feature-specific traffic",
                    ],
                    timeframe="2-4 months",
                    confidence_score=75.0,
                )
            )

        return recommendations

    def _generate_performance_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Generate performance-focused recommendations."""
        recommendations = []

        # Always include performance recommendations
        recommendations.append(
            SEORecommendation(
                title="SEO Performance Monitoring",
                description="Implement comprehensive tracking and monitoring for SEO performance metrics.",
                priority=RecommendationPriority.MEDIUM,
                recommendation_type=RecommendationType.PERFORMANCE,
                effort_level="low",
                estimated_impact="medium",
                keywords_affected=[],
                implementation_steps=[
                    "Set up Google Search Console monitoring",
                    "Implement rank tracking for target keywords",
                    "Configure traffic and conversion tracking",
                    "Create SEO performance dashboards",
                    "Establish regular reporting schedules",
                ],
                success_metrics=[
                    "Tracking accuracy",
                    "Reporting efficiency",
                    "Data-driven decision making",
                ],
                timeframe="1-2 weeks",
                confidence_score=95.0,
            )
        )

        return recommendations

    def _generate_industry_specific_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict, industry: str
    ) -> List[SEORecommendation]:
        """Generate industry-specific recommendations."""
        recommendations = []

        industry_templates = {
            "developer_tools": self._developer_tools_recommendations,
            "mobile_apps": self._mobile_apps_recommendations,
            "ecommerce": self._ecommerce_recommendations,
            "local_business": self._local_business_recommendations,
        }

        if industry in industry_templates:
            recommendations.extend(
                industry_templates[industry](analysis_results, analysis_summary)
            )

        return recommendations

    def _developer_tools_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Recommendations specific to developer tools and software."""
        return [
            SEORecommendation(
                title="Developer Community Engagement",
                description="Build presence in developer communities and platforms for increased visibility.",
                priority=RecommendationPriority.MEDIUM,
                recommendation_type=RecommendationType.LINK_BUILDING,
                effort_level="medium",
                estimated_impact="medium",
                keywords_affected=[],
                implementation_steps=[
                    "Participate in Stack Overflow and GitHub discussions",
                    "Create open source contributions",
                    "Engage with developer communities on Reddit and Discord",
                    "Write technical blog posts and tutorials",
                ],
                success_metrics=[
                    "Community engagement metrics",
                    "Referral traffic from dev platforms",
                    "Brand mention increase",
                ],
                timeframe="3-6 months",
                confidence_score=70.0,
            )
        ]

    def _mobile_apps_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Recommendations specific to mobile apps."""
        return [
            SEORecommendation(
                title="App Store Optimization (ASO)",
                description="Optimize app store presence to complement web SEO efforts.",
                priority=RecommendationPriority.HIGH,
                recommendation_type=RecommendationType.CONTENT_STRATEGY,
                effort_level="medium",
                estimated_impact="high",
                keywords_affected=[],
                implementation_steps=[
                    "Research app store keyword opportunities",
                    "Optimize app titles and descriptions",
                    "Improve app screenshots and videos",
                    "Encourage user reviews and ratings",
                ],
                success_metrics=[
                    "App store rankings",
                    "App downloads",
                    "App store conversion rates",
                ],
                timeframe="1-2 months",
                confidence_score=80.0,
            )
        ]

    def _ecommerce_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Recommendations specific to ecommerce sites."""
        return [
            SEORecommendation(
                title="Product Page SEO Optimization",
                description="Optimize product pages for transactional keywords and user experience.",
                priority=RecommendationPriority.HIGH,
                recommendation_type=RecommendationType.KEYWORD_TARGETING,
                effort_level="medium",
                estimated_impact="high",
                keywords_affected=[],
                implementation_steps=[
                    "Optimize product titles and descriptions",
                    "Implement schema markup for products",
                    "Create detailed product specifications",
                    "Optimize product images with alt text",
                ],
                success_metrics=[
                    "Product page rankings",
                    "E-commerce conversion rates",
                    "Product visibility",
                ],
                timeframe="2-3 months",
                confidence_score=85.0,
            )
        ]

    def _local_business_recommendations(
        self, analysis_results: Dict, analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Recommendations specific to local businesses."""
        return [
            SEORecommendation(
                title="Local SEO Optimization",
                description="Optimize for local search visibility and Google My Business presence.",
                priority=RecommendationPriority.HIGH,
                recommendation_type=RecommendationType.LOCAL_SEO,
                effort_level="medium",
                estimated_impact="high",
                keywords_affected=[],
                implementation_steps=[
                    "Optimize Google My Business profile",
                    "Build local citations and directories",
                    "Create location-specific landing pages",
                    "Encourage customer reviews",
                ],
                success_metrics=[
                    "Local search rankings",
                    "GMB insights metrics",
                    "Local traffic increase",
                ],
                timeframe="1-3 months",
                confidence_score=90.0,
            )
        ]

    def _get_generic_recommendations(self) -> List[str]:
        """Get generic recommendations based on business type."""
        base_recommendations = [
            "🔗 Build quality backlinks from relevant industry websites",
            "📱 Optimize for mobile-first indexing and user experience",
            "⚡ Improve Core Web Vitals performance metrics",
            "🎯 Create comprehensive content for long-tail keywords",
        ]

        business_specific = {
            "saas": [
                "💼 Create comparison pages for competitor alternatives",
                "📚 Develop comprehensive documentation and guides",
                "🎓 Build educational content for your target market",
            ],
            "ecommerce": [
                "🛍️ Optimize product pages for commercial keywords",
                "📦 Create detailed product category descriptions",
                "⭐ Implement structured data for products and reviews",
            ],
            "blog": [
                "📝 Focus on informational and how-to content",
                "🔍 Target question-based keywords and featured snippets",
                "📅 Maintain consistent content publishing schedule",
            ],
            "local": [
                "📍 Optimize for local search and Google My Business",
                "🏪 Create location-specific landing pages",
                "📞 Ensure consistent NAP (Name, Address, Phone) information",
            ],
        }

        business_recs = business_specific.get(self.config.business_type, [])
        return base_recommendations + business_recs

    def _filter_recommendations(
        self, recommendations: List[SEORecommendation]
    ) -> List[SEORecommendation]:
        """Filter recommendations based on configuration."""
        if not self.config.exclude_types:
            return recommendations

        return [
            rec
            for rec in recommendations
            if rec.recommendation_type not in self.config.exclude_types
        ]

    def _prioritize_recommendations(
        self, recommendations: List[SEORecommendation], analysis_summary: Dict
    ) -> List[SEORecommendation]:
        """Prioritize recommendations based on impact and effort."""

        def priority_score(rec):
            # Priority level scoring
            priority_scores = {
                RecommendationPriority.CRITICAL: 100,
                RecommendationPriority.HIGH: 80,
                RecommendationPriority.MEDIUM: 60,
                RecommendationPriority.LOW: 40,
            }

            # Impact scoring
            impact_scores = {"high": 30, "medium": 20, "low": 10}

            # Effort scoring (inverse - lower effort gets higher score)
            effort_scores = {"low": 20, "medium": 15, "high": 10}

            return (
                priority_scores.get(rec.priority, 40)
                + impact_scores.get(rec.estimated_impact, 10)
                + effort_scores.get(rec.effort_level, 10)
                + rec.confidence_score * 0.1
            )

        return sorted(recommendations, key=priority_score, reverse=True)

    def _generate_implementation_roadmap(
        self, recommendations: List[SEORecommendation]
    ) -> Dict[str, Any]:
        """Generate implementation roadmap for recommendations."""
        roadmap = {
            "immediate_actions": [],  # 0-2 weeks
            "short_term": [],  # 2-8 weeks
            "medium_term": [],  # 2-6 months
            "long_term": [],  # 6+ months
        }

        timeframe_mapping = {
            "immediate_actions": ["1-2 weeks", "immediate", "1 week"],
            "short_term": ["2-4 weeks", "3-6 weeks", "1-2 months"],
            "medium_term": ["2-3 months", "3-6 months", "2-4 months"],
            "long_term": ["6+ months", "6-12 months", "long-term"],
        }

        for rec in recommendations:
            categorized = False
            for category, timeframes in timeframe_mapping.items():
                if any(tf in rec.timeframe.lower() for tf in timeframes):
                    roadmap[category].append(
                        {
                            "title": rec.title,
                            "priority": rec.priority.value,
                            "effort": rec.effort_level,
                            "impact": rec.estimated_impact,
                        }
                    )
                    categorized = True
                    break

            if not categorized:
                roadmap["medium_term"].append(
                    {
                        "title": rec.title,
                        "priority": rec.priority.value,
                        "effort": rec.effort_level,
                        "impact": rec.estimated_impact,
                    }
                )

        return roadmap

    def _analyze_difficulty_distribution(
        self, difficulty_scores: List[float]
    ) -> Dict[str, int]:
        """Analyze distribution of keyword difficulty scores."""
        if not difficulty_scores:
            return {}

        distribution = {"easy": 0, "medium": 0, "hard": 0}

        for score in difficulty_scores:
            if score < 30:
                distribution["easy"] += 1
            elif score < 70:
                distribution["medium"] += 1
            else:
                distribution["hard"] += 1

        return distribution

    def _analyze_serp_features_summary(self, keyword_results: Dict) -> Dict[str, Any]:
        """Analyze SERP features across all keywords."""
        features_summary = {
            "featured_snippet_opportunities": 0,
            "knowledge_panel_presence": 0,
            "people_also_ask_presence": 0,
            "total_serp_features": 0,
        }

        for data in keyword_results.values():
            serp_features = data.get("serp_features", {})
            if serp_features.get("featured_snippet"):
                features_summary["featured_snippet_opportunities"] += 1
            if serp_features.get("knowledge_panel"):
                features_summary["knowledge_panel_presence"] += 1
            if serp_features.get("people_also_ask"):
                features_summary["people_also_ask_presence"] += 1

            features_summary["total_serp_features"] += serp_features.get(
                "total_features", 0
            )

        return features_summary

    def _recommendation_to_dict(self, rec: SEORecommendation) -> Dict[str, Any]:
        """Convert recommendation object to dictionary."""
        return {
            "title": rec.title,
            "description": rec.description,
            "priority": rec.priority.value,
            "type": rec.recommendation_type.value,
            "effort_level": rec.effort_level,
            "estimated_impact": rec.estimated_impact,
            "keywords_affected": rec.keywords_affected,
            "implementation_steps": rec.implementation_steps,
            "success_metrics": rec.success_metrics,
            "timeframe": rec.timeframe,
            "confidence_score": rec.confidence_score,
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "industry_vertical": self.config.industry_vertical,
            "business_type": self.config.business_type,
            "target_audience": self.config.target_audience,
            "available_resources": self.config.available_resources,
            "technical_capability": self.config.technical_capability,
            "content_strategy": self.config.content_strategy,
            "exclude_types": [t.value for t in (self.config.exclude_types or [])],
        }
