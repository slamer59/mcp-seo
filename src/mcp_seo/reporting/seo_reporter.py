"""
Rich reporting utilities for SEO analysis with professional output formatting.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SEOReporter:
    """Professional SEO reporting with rich formatting."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()
        else:
            self.console = None

    def generate_keyword_analysis_report(self, keyword_data: Dict[str, Any]) -> str:
        """Generate a formatted keyword analysis report."""
        if not keyword_data:
            return "No keyword data available for analysis."

        if self.use_rich:
            return self._generate_rich_keyword_report(keyword_data)
        else:
            return self._generate_plain_keyword_report(keyword_data)

    def generate_onpage_analysis_report(self, onpage_data: Dict[str, Any]) -> str:
        """Generate a formatted on-page analysis report."""
        if not onpage_data:
            return "No on-page data available for analysis."

        if self.use_rich:
            return self._generate_rich_onpage_report(onpage_data)
        else:
            return self._generate_plain_onpage_report(onpage_data)

    def generate_pagerank_analysis_report(self, pagerank_data: Dict[str, Any]) -> str:
        """Generate a formatted PageRank analysis report."""
        if not pagerank_data:
            return "No PageRank data available for analysis."

        if self.use_rich:
            return self._generate_rich_pagerank_report(pagerank_data)
        else:
            return self._generate_plain_pagerank_report(pagerank_data)

    def generate_comprehensive_seo_report(self, seo_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive SEO analysis report."""
        if not seo_analysis:
            return "No SEO analysis data available."

        if self.use_rich:
            return self._generate_rich_comprehensive_report(seo_analysis)
        else:
            return self._generate_plain_comprehensive_report(seo_analysis)

    def _generate_rich_keyword_report(self, keyword_data: Dict[str, Any]) -> str:
        """Generate rich-formatted keyword analysis report."""
        output = []

        # Header
        header_panel = Panel.fit(
            "[bold blue]🔍 Keyword Analysis Report[/bold blue]\n"
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
        )
        output.append(str(header_panel))

        # Summary statistics
        total_keywords = len(keyword_data.get("keywords_data", []))
        avg_volume = keyword_data.get("analysis_summary", {}).get(
            "avg_search_volume", 0
        )
        high_volume_count = keyword_data.get("analysis_summary", {}).get(
            "high_volume_keywords", 0
        )

        summary_table = Table(title="📊 Keyword Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Keywords", str(total_keywords))
        summary_table.add_row("Average Search Volume", f"{avg_volume:,.0f}")
        summary_table.add_row("High Volume Keywords (>10k)", str(high_volume_count))

        output.append(str(summary_table))

        # Top performing keywords
        if keyword_data.get("keywords_data"):
            keywords_table = Table(title="🎯 Keyword Performance")
            keywords_table.add_column("Keyword", style="cyan")
            keywords_table.add_column("Search Volume", style="green")
            keywords_table.add_column("CPC", style="yellow")
            keywords_table.add_column("Competition", style="red")

            for keyword in keyword_data["keywords_data"][:10]:  # Top 10
                volume = keyword.get("search_volume", 0)
                cpc = keyword.get("cpc", 0)
                competition = keyword.get("competition_level", "Unknown")

                keywords_table.add_row(
                    keyword.get("keyword", "N/A"),
                    f"{volume:,}" if volume else "N/A",
                    f"${cpc:.2f}" if cpc else "N/A",
                    str(competition),
                )

            output.append(str(keywords_table))

        # Suggestions if available
        if keyword_data.get("suggestions"):
            suggestions_panel = Panel(
                "\n".join(
                    [
                        f"• {suggestion.get('keyword', 'N/A')}"
                        for suggestion in keyword_data["suggestions"][:5]
                    ]
                ),
                title="💡 Keyword Suggestions",
                border_style="green",
            )
            output.append(str(suggestions_panel))

        return "\n\n".join(output)

    def _generate_rich_onpage_report(self, onpage_data: Dict[str, Any]) -> str:
        """Generate rich-formatted on-page analysis report."""
        output = []

        # Header
        header_panel = Panel.fit(
            "[bold blue]📋 On-Page SEO Analysis Report[/bold blue]\n"
            f"Target: {onpage_data.get('target', 'N/A')}\n"
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
        )
        output.append(str(header_panel))

        summary = onpage_data.get("summary", {})

        # Issues summary
        issues_table = Table(title="🚨 Issues Summary")
        issues_table.add_column("Severity", style="red")
        issues_table.add_column("Count", style="yellow")
        issues_table.add_column("Description", style="white")

        issues_data = [
            (
                "Critical",
                summary.get("critical_issues", 0),
                "Issues that prevent proper indexing",
            ),
            (
                "High",
                summary.get("high_priority_issues", 0),
                "Issues that significantly impact SEO",
            ),
            (
                "Medium",
                summary.get("medium_priority_issues", 0),
                "Issues that moderately impact SEO",
            ),
            (
                "Low",
                summary.get("low_priority_issues", 0),
                "Minor optimization opportunities",
            ),
        ]

        for severity, count, description in issues_data:
            style = (
                "red"
                if severity == "Critical"
                else "yellow"
                if severity == "High"
                else "green"
            )
            issues_table.add_row(
                f"[{style}]{severity}[/{style}]", str(count), description
            )

        output.append(str(issues_table))

        # Page statistics
        page_stats_table = Table(title="📊 Page Statistics")
        page_stats_table.add_column("Metric", style="cyan")
        page_stats_table.add_column("Value", style="green")

        page_stats_table.add_row(
            "Total Pages Crawled", str(summary.get("crawled_pages", 0))
        )
        page_stats_table.add_row("Broken Pages", str(summary.get("broken_pages", 0)))
        page_stats_table.add_row(
            "Duplicate Titles", str(summary.get("duplicate_title_tags", 0))
        )
        page_stats_table.add_row(
            "Duplicate Meta Descriptions",
            str(summary.get("duplicate_meta_descriptions", 0)),
        )

        output.append(str(page_stats_table))

        # Individual issues
        if summary.get("issues"):
            issues_panel = Panel(
                "\n".join(
                    [
                        f"• [{issue.get('severity', 'low')}]{issue.get('description', 'N/A')}[/{issue.get('severity', 'low')}]"
                        for issue in summary["issues"][:5]
                    ]
                ),
                title="🔍 Top Issues",
                border_style="red",
            )
            output.append(str(issues_panel))

        return "\n\n".join(output)

    def _generate_rich_pagerank_report(self, pagerank_data: Dict[str, Any]) -> str:
        """Generate rich-formatted PageRank analysis report."""
        output = []

        # Header
        header_panel = Panel.fit(
            "[bold blue]🔗 PageRank Analysis Report[/bold blue]\n"
            f"Domain: {pagerank_data.get('domain', 'N/A')}\n"
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
        )
        output.append(str(header_panel))

        # Basic metrics
        if "basic_metrics" in pagerank_data:
            metrics = pagerank_data["basic_metrics"]
            metrics_table = Table(title="📊 Link Graph Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            metrics_table.add_row("Total Pages", str(metrics.get("total_pages", 0)))
            metrics_table.add_row("Total Links", str(metrics.get("total_links", 0)))
            metrics_table.add_row(
                "Average In-Degree", f"{metrics.get('average_in_degree', 0):.2f}"
            )
            metrics_table.add_row(
                "Average Out-Degree", f"{metrics.get('average_out_degree', 0):.2f}"
            )
            metrics_table.add_row(
                "Orphaned Pages", str(metrics.get("orphaned_pages_count", 0))
            )
            metrics_table.add_row(
                "Hub Pages (>10 links)", str(metrics.get("hub_pages_count", 0))
            )

            output.append(str(metrics_table))

        # Top pages by PageRank
        if "top_pages_by_links" in pagerank_data:
            top_pages_table = Table(title="🏆 Top Pages by Link Authority")
            top_pages_table.add_column("URL", style="cyan")
            top_pages_table.add_column("Incoming Links", style="green")
            top_pages_table.add_column("Outgoing Links", style="yellow")

            for page in pagerank_data["top_pages_by_links"][:10]:
                url = page.get("url", "N/A")
                # Truncate long URLs
                if len(url) > 50:
                    url = url[:47] + "..."

                top_pages_table.add_row(
                    url, str(page.get("in_degree", 0)), str(page.get("out_degree", 0))
                )

            output.append(str(top_pages_table))

        # Orphaned pages warning
        orphaned_count = len(pagerank_data.get("orphaned_pages", []))
        if orphaned_count > 0:
            orphaned_panel = Panel(
                f"Found {orphaned_count} orphaned pages with no incoming links.\n"
                "These pages are difficult to discover and may not be indexed properly.\n"
                "Consider adding internal links to improve page discovery.",
                title="⚠️ Orphaned Pages Warning",
                border_style="yellow",
            )
            output.append(str(orphaned_panel))

        return "\n\n".join(output)

    def _generate_rich_comprehensive_report(self, seo_analysis: Dict[str, Any]) -> str:
        """Generate rich-formatted comprehensive SEO report."""
        output = []

        # Header
        header_panel = Panel.fit(
            "[bold blue]📈 Comprehensive SEO Analysis Report[/bold blue]\n"
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
        )
        output.append(str(header_panel))

        # SEO Score
        if "seo_score" in seo_analysis:
            score = seo_analysis["seo_score"]
            overall_score = score.get("overall_score", 0)

            # Color code based on score
            if overall_score >= 80:
                score_color = "green"
                grade = "A"
            elif overall_score >= 70:
                score_color = "yellow"
                grade = "B"
            elif overall_score >= 60:
                score_color = "orange"
                grade = "C"
            else:
                score_color = "red"
                grade = "D"

            score_panel = Panel(
                f"[bold {score_color}]{overall_score}/100 (Grade: {grade})[/bold {score_color}]",
                title="🎯 Overall SEO Score",
                border_style=score_color,
            )
            output.append(str(score_panel))

            # Score breakdown
            score_table = Table(title="📊 Score Breakdown")
            score_table.add_column("Category", style="cyan")
            score_table.add_column("Score", style="green")
            score_table.add_column("Weight", style="yellow")

            breakdown = score.get("breakdown", {})
            for category, data in breakdown.items():
                score_value = data.get("score", 0)
                weight = data.get("weight", 0)
                score_table.add_row(
                    category.title(), f"{score_value}/100", f"{weight:.0%}"
                )

            output.append(str(score_table))

        # Recommendations summary
        if "summary" in seo_analysis:
            summary = seo_analysis["summary"]
            recommendations_table = Table(title="📋 Recommendations Summary")
            recommendations_table.add_column("Priority", style="red")
            recommendations_table.add_column("Count", style="yellow")

            recommendations_table.add_row(
                "Critical Issues", str(summary.get("critical_issues", 0))
            )
            recommendations_table.add_row(
                "High Priority", str(summary.get("high_priority", 0))
            )
            recommendations_table.add_row(
                "Medium Priority", str(summary.get("medium_priority", 0))
            )
            recommendations_table.add_row(
                "Low Priority", str(summary.get("low_priority", 0))
            )

            output.append(str(recommendations_table))

        # Top recommendations
        if "recommendations" in seo_analysis:
            recommendations = seo_analysis["recommendations"][:5]  # Top 5
            rec_panel = Panel(
                "\n".join(
                    [
                        f"[{rec.get('priority', 'medium')}]• {rec.get('title', 'N/A')}[/{rec.get('priority', 'medium')}]"
                        for rec in recommendations
                    ]
                ),
                title="🎯 Top Recommendations",
                border_style="green",
            )
            output.append(str(rec_panel))

        # Action plan
        if "action_plan" in seo_analysis:
            action_plan = seo_analysis["action_plan"]
            action_tree = Tree("📅 Action Plan")

            if "immediate_actions" in action_plan:
                immediate = action_tree.add("🚨 Immediate Actions (0-2 weeks)")
                for action in action_plan["immediate_actions"].get("actions", [])[:3]:
                    immediate.add(action.get("title", "N/A"))

            if "short_term_actions" in action_plan:
                short_term = action_tree.add("⏰ Short-term Actions (2-8 weeks)")
                for action in action_plan["short_term_actions"].get("actions", [])[:3]:
                    short_term.add(action.get("title", "N/A"))

            if "long_term_actions" in action_plan:
                long_term = action_tree.add("📅 Long-term Actions (2-6 months)")
                for action in action_plan["long_term_actions"].get("actions", [])[:3]:
                    long_term.add(action.get("title", "N/A"))

            output.append(str(action_tree))

        return "\n\n".join(output)

    def _generate_plain_keyword_report(self, keyword_data: Dict[str, Any]) -> str:
        """Generate plain text keyword analysis report."""
        lines = []
        lines.append("KEYWORD ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        total_keywords = len(keyword_data.get("keywords_data", []))
        avg_volume = keyword_data.get("analysis_summary", {}).get(
            "avg_search_volume", 0
        )
        lines.append("SUMMARY:")
        lines.append(f"- Total Keywords: {total_keywords}")
        lines.append(f"- Average Search Volume: {avg_volume:,.0f}")
        lines.append("")

        # Keywords
        if keyword_data.get("keywords_data"):
            lines.append("TOP KEYWORDS:")
            for i, keyword in enumerate(keyword_data["keywords_data"][:10], 1):
                volume = keyword.get("search_volume", 0)
                lines.append(
                    f"{i}. {keyword.get('keyword', 'N/A')} - Volume: {volume:,}"
                )

        return "\n".join(lines)

    def _generate_plain_onpage_report(self, onpage_data: Dict[str, Any]) -> str:
        """Generate plain text on-page analysis report."""
        lines = []
        lines.append("ON-PAGE SEO ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Target: {onpage_data.get('target', 'N/A')}")
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        summary = onpage_data.get("summary", {})
        lines.append("ISSUES SUMMARY:")
        lines.append(f"- Critical Issues: {summary.get('critical_issues', 0)}")
        lines.append(
            f"- High Priority Issues: {summary.get('high_priority_issues', 0)}"
        )
        lines.append(
            f"- Medium Priority Issues: {summary.get('medium_priority_issues', 0)}"
        )
        lines.append(f"- Low Priority Issues: {summary.get('low_priority_issues', 0)}")
        lines.append("")

        lines.append("PAGE STATISTICS:")
        lines.append(f"- Total Pages Crawled: {summary.get('crawled_pages', 0)}")
        lines.append(f"- Broken Pages: {summary.get('broken_pages', 0)}")
        lines.append(f"- Duplicate Titles: {summary.get('duplicate_title_tags', 0)}")

        return "\n".join(lines)

    def _generate_plain_pagerank_report(self, pagerank_data: Dict[str, Any]) -> str:
        """Generate plain text PageRank analysis report."""
        lines = []
        lines.append("PAGERANK ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Domain: {pagerank_data.get('domain', 'N/A')}")
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if "basic_metrics" in pagerank_data:
            metrics = pagerank_data["basic_metrics"]
            lines.append("LINK GRAPH METRICS:")
            lines.append(f"- Total Pages: {metrics.get('total_pages', 0)}")
            lines.append(f"- Total Links: {metrics.get('total_links', 0)}")
            lines.append(f"- Orphaned Pages: {metrics.get('orphaned_pages_count', 0)}")

        return "\n".join(lines)

    def _generate_plain_comprehensive_report(self, seo_analysis: Dict[str, Any]) -> str:
        """Generate plain text comprehensive SEO report."""
        lines = []
        lines.append("COMPREHENSIVE SEO ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if "seo_score" in seo_analysis:
            score = seo_analysis["seo_score"]
            lines.append(f"OVERALL SEO SCORE: {score.get('overall_score', 0)}/100")
            lines.append("")

        if "summary" in seo_analysis:
            summary = seo_analysis["summary"]
            lines.append("RECOMMENDATIONS SUMMARY:")
            lines.append(f"- Critical Issues: {summary.get('critical_issues', 0)}")
            lines.append(f"- High Priority: {summary.get('high_priority', 0)}")
            lines.append(f"- Medium Priority: {summary.get('medium_priority', 0)}")
            lines.append(f"- Low Priority: {summary.get('low_priority', 0)}")

        return "\n".join(lines)

    def save_report_to_file(
        self, report_content: str, filename: str, format_type: str = "txt"
    ) -> bool:
        """Save report content to file."""
        try:
            file_path = Path(filename)
            if format_type.lower() == "json" and isinstance(report_content, dict):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(report_content, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
            return True
        except Exception as e:
            logger.error(f"Failed to save report to {filename}: {e}")
            return False

    def print_report(self, report_content: str) -> None:
        """Print report to console with appropriate formatting."""
        if self.use_rich and self.console:
            # For rich content, we need to render it properly
            if report_content.startswith("["):  # Rich markup detected
                self.console.print(report_content)
            else:
                print(report_content)
        else:
            print(report_content)
