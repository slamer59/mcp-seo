"""
Rich console reporting utilities for MCP SEO.

Provides professional styled console output, progress tracking, and
comprehensive report generation capabilities for enhanced SEO analysis.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


class MockableMethod:
    """A wrapper for methods that allows setting side_effect for test compatibility."""

    def __init__(self, method):
        self._method = method
        self.side_effect = None

    def __call__(self, *args, **kwargs):
        if self.side_effect:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            elif callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
        return self._method(*args, **kwargs)


class SEOReporter:
    """Professional SEO analysis reporter with rich console output."""

    def __init__(self):
        self.console = Console()

        # Create mock-compatible method wrappers
        self._setup_mock_support()

    def display_header(self, title: str, subtitle: str = None) -> None:
        """Display professional styled header."""
        header_text = f"[bold blue]{title}[/bold blue]"
        if subtitle:
            header_text += f"\n[cyan]{subtitle}[/cyan]"

        header_panel = Panel.fit(header_text, border_style="blue")
        self.console.print(header_panel)

    def display_analysis_progress(
        self, items: List[str], analysis_func, desc: str = "Analyzing"
    ) -> Dict:
        """Display progress while analyzing items with rate limiting."""
        results = {}

        self.console.print(f"[blue]🔍 {desc} {len(items)} items...[/blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            for item in items:
                task = progress.add_task(f"{desc}: {item}", total=1)

                # Execute analysis function
                result = analysis_func(item)
                results[item] = result

                progress.update(task, completed=1)
                time.sleep(1)  # Rate limiting

        return results

    def display_keyword_analysis_table(
        self, analysis_results: Dict, title: str = "SEO Performance Analysis"
    ) -> None:
        """Display keyword analysis results in a professional table."""
        summary_table = Table(title=title)
        summary_table.add_column("Keyword", style="cyan", no_wrap=True)
        summary_table.add_column("Position", style="yellow")
        summary_table.add_column("Search Volume", style="green")
        summary_table.add_column("Difficulty", style="red")
        summary_table.add_column("Top Competitor", style="magenta")

        for keyword, data in analysis_results.items():
            # Extract position
            position = data.get("position", "Not Found")
            position_str = (
                str(position)
                if position and position != "Not Found"
                else "Not in Top 100"
            )

            # Extract metrics (with fallbacks)
            search_volume = self._extract_search_volume(data)
            difficulty = self._extract_difficulty(data)
            top_competitor = self._extract_top_competitor(data)

            summary_table.add_row(
                keyword, position_str, search_volume, difficulty, top_competitor
            )

        self.console.print(summary_table)

    def display_pagerank_analysis_table(
        self, pagerank_data: Dict, title: str = "PageRank Authority Analysis"
    ) -> None:
        """Display PageRank analysis results."""
        pagerank_table = Table(title=title)
        pagerank_table.add_column("Page", style="cyan")
        pagerank_table.add_column("PageRank Score", style="green")
        pagerank_table.add_column("Internal Links", style="yellow")
        pagerank_table.add_column("Category", style="magenta")

        # Sort by PageRank score descending
        sorted_pages = sorted(
            pagerank_data.items(), key=lambda x: x[1].get("pagerank", 0), reverse=True
        )

        for page, data in sorted_pages[:20]:  # Show top 20
            pagerank_score = f"{data.get('pagerank', 0):.4f}"
            internal_links = str(data.get("internal_links_count", 0))
            category = data.get("category", "Unknown")

            pagerank_table.add_row(page, pagerank_score, internal_links, category)

        self.console.print(pagerank_table)

    def display_recommendations_panel(
        self, recommendations: List[str], title: str = "🎯 SEO Recommendations"
    ) -> None:
        """Display SEO recommendations in a styled panel."""
        rec_text = "\n".join(f"• {rec}" for rec in recommendations)

        rec_panel = Panel(rec_text, title=title, border_style="green")
        self.console.print(rec_panel)

    def display_link_opportunities_table(self, opportunities: List[Dict]) -> None:
        """Display internal link opportunities."""
        if not opportunities:
            self.console.print("[yellow]No link opportunities found.[/yellow]")
            return

        link_table = Table(title="🔗 Internal Link Opportunities")
        link_table.add_column("Source Page", style="cyan")
        link_table.add_column("Target Page", style="green")
        link_table.add_column("Opportunity Score", style="yellow")
        link_table.add_column("Keyword Overlap", style="magenta")

        # Sort by opportunity score
        sorted_opps = sorted(
            opportunities, key=lambda x: x.get("opportunity_score", 0), reverse=True
        )

        for opp in sorted_opps[:15]:  # Show top 15
            source = opp.get("source_title", opp.get("source_slug", "Unknown"))[:40]
            target = opp.get("target_title", opp.get("target_slug", "Unknown"))[:40]
            score = f"{opp.get('opportunity_score', 0):.2f}"
            overlap = str(opp.get("keyword_overlap", 0))

            link_table.add_row(source, target, score, overlap)

        self.console.print(link_table)

    def display_summary_stats(
        self, stats: Dict, title: str = "📊 Analysis Summary"
    ) -> None:
        """Display summary statistics."""
        stats_table = Table(title=title)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        for metric, value in stats.items():
            # Format different value types
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, (list, dict)):
                formatted_value = str(len(value))
            else:
                formatted_value = str(value)

            stats_table.add_row(metric.replace("_", " ").title(), formatted_value)

        self.console.print(stats_table)

    def display_error(self, message: str, details: str = None) -> None:
        """Display error message."""
        error_text = f"[red]❌ Error: {message}[/red]"
        if details:
            error_text += f"\n[dim]{details}[/dim]"
        self.console.print(error_text)

    def display_success(self, message: str) -> None:
        """Display success message."""
        self.console.print(f"[green]✅ {message}[/green]")

    def display_warning(self, message: str) -> None:
        """Display warning message."""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")

    def display_info(self, message: str) -> None:
        """Display info message."""
        self.console.print(f"[cyan]ℹ️  {message}[/cyan]")

    def save_analysis_report(
        self, data: Dict, filename: str, format: str = "json"
    ) -> None:
        """Save analysis results to file."""
        output_path = Path(filename)

        try:
            if format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                self.display_success(f"Report saved to {output_path}")
            else:
                self.display_error(f"Unsupported format: {format}")

        except Exception as e:
            self.display_error(f"Failed to save report: {e}")

    def generate_comprehensive_report(self, analysis_data: Dict) -> Dict:
        """Generate comprehensive analysis report with timestamp."""
        return {
            "analysis_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "comprehensive_seo_analysis",
                "total_items_analyzed": len(analysis_data),
            },
            "analysis_results": analysis_data,
            "summary_statistics": self._calculate_summary_stats(analysis_data),
        }

    def _extract_search_volume(self, data: Dict) -> str:
        """Extract search volume from analysis data."""
        volume_data = data.get("search_volume", {})
        if isinstance(volume_data, dict) and "results" in volume_data:
            try:
                results = volume_data["results"]
                if results and len(results) > 0:
                    tasks = results[0].get("result", [])
                    if tasks and len(tasks) > 0:
                        volume = tasks[0].get("search_volume")
                        return str(volume) if volume else "N/A"
            except (IndexError, KeyError, TypeError):
                pass
        return "N/A"

    def _extract_difficulty(self, data: Dict) -> str:
        """Extract keyword difficulty from analysis data."""
        difficulty_data = data.get("difficulty", {})
        if isinstance(difficulty_data, dict) and "results" in difficulty_data:
            try:
                results = difficulty_data["results"]
                if results and len(results) > 0:
                    tasks = results[0].get("result", [])
                    if tasks and len(tasks) > 0:
                        difficulty = tasks[0].get("keyword_difficulty")
                        if difficulty is not None:
                            if difficulty < 30:
                                return "Low"
                            elif difficulty < 60:
                                return "Medium"
                            else:
                                return "High"
            except (IndexError, KeyError, TypeError):
                pass
        return "N/A"

    def _extract_top_competitor(self, data: Dict) -> str:
        """Extract top competitor from analysis data."""
        competitors = data.get("competitor_analysis", [])
        if competitors and len(competitors) > 0:
            return competitors[0].get("type", "Unknown")
        return "None"

    def _calculate_summary_stats(self, data: Dict) -> Dict:
        """Calculate summary statistics from analysis data."""
        stats = {
            "total_items": len(data),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Count different metrics if available
        if data:
            sample_item = next(iter(data.values()))
            if "pagerank" in sample_item:
                pagerank_values = [item.get("pagerank", 0) for item in data.values()]
                stats.update(
                    {
                        "avg_pagerank": sum(pagerank_values) / len(pagerank_values),
                        "max_pagerank": max(pagerank_values),
                        "min_pagerank": min(pagerank_values),
                    }
                )

        return stats

    def _setup_mock_support(self):
        """Setup mock support for testing - allows setting side_effect on methods."""
        # Wrap key methods with mockable wrappers for test compatibility
        self.generate_keyword_analysis_report = MockableMethod(self.generate_keyword_analysis_report)
        self.generate_pagerank_analysis_report = MockableMethod(self.generate_pagerank_analysis_report)
        self.generate_onpage_analysis_report = MockableMethod(self.generate_onpage_analysis_report)
        self.generate_comprehensive_seo_report = MockableMethod(self.generate_comprehensive_seo_report)
        self.create_progress_tracker = MockableMethod(self.create_progress_tracker)

    def generate_keyword_analysis_report(self, keyword_data: Dict[str, Any]) -> str:
        """Generate keyword analysis report with Rich formatting."""
        # Capture console output to string
        import io

        from rich.console import Console

        output = io.StringIO()
        temp_console = Console(file=output, width=120)

        # Create table
        table = self._create_keyword_table(keyword_data.get("keywords_data", []))
        temp_console.print(table)

        # Create summary panel
        if "analysis_summary" in keyword_data:
            panel = self._create_summary_panel(keyword_data["analysis_summary"])
            temp_console.print(panel)

        return output.getvalue()

    def generate_pagerank_analysis_report(self, pagerank_data: Dict[str, Any]) -> str:
        """Generate PageRank analysis report with Rich formatting."""
        import io

        from rich.console import Console

        output = io.StringIO()
        temp_console = Console(file=output, width=120)

        # Basic metrics
        if "basic_metrics" in pagerank_data:
            metrics_table = Table(title="PageRank Analysis")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            metrics = pagerank_data["basic_metrics"]
            metrics_table.add_row("Total Pages", str(metrics.get("total_pages", 0)))
            metrics_table.add_row("Total Links", str(metrics.get("total_links", 0)))
            metrics_table.add_row(
                "Avg PageRank", f"{metrics.get('avg_pagerank', 0):.4f}"
            )

            temp_console.print(metrics_table)

        # Top pages
        if "top_pages" in pagerank_data:
            top_table = Table(title="Top Pages")
            top_table.add_column("URL", style="cyan")
            top_table.add_column("PageRank", style="green")
            top_table.add_column("Inbound Links", style="yellow")

            for page in pagerank_data["top_pages"][:10]:
                top_table.add_row(
                    page.get("url", "N/A"),
                    f"{page.get('pagerank', 0):.4f}",
                    str(page.get("inbound_links", 0)),
                )

            temp_console.print(top_table)

        return output.getvalue()

    def generate_onpage_analysis_report(self, onpage_data: Dict[str, Any]) -> str:
        """Generate on-page analysis report with Rich formatting."""
        import io

        from rich.console import Console

        output = io.StringIO()
        temp_console = Console(file=output, width=120)

        # Summary
        if "summary" in onpage_data:
            summary_table = Table(title="On-Page Analysis Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Count", style="green")

            summary = onpage_data["summary"]
            summary_table.add_row(
                "Total Pages", str(summary.get("total_pages_analyzed", 0))
            )
            summary_table.add_row(
                "Critical Issues", str(summary.get("critical_issues", 0))
            )
            summary_table.add_row(
                "High Priority", str(summary.get("high_priority_issues", 0))
            )
            summary_table.add_row(
                "Medium Priority", str(summary.get("medium_priority_issues", 0))
            )
            summary_table.add_row(
                "Low Priority", str(summary.get("low_priority_issues", 0))
            )

            temp_console.print(summary_table)

        return output.getvalue()

    def generate_comprehensive_seo_report(
        self,
        keyword_data: Dict[str, Any] = None,
        pagerank_data: Dict[str, Any] = None,
        onpage_data: Dict[str, Any] = None,
    ) -> str:
        """Generate comprehensive SEO report with Rich formatting."""
        import io

        from rich.console import Console

        output = io.StringIO()
        temp_console = Console(file=output, width=120)

        # Header
        header = Panel.fit(
            "[bold blue]Comprehensive SEO Analysis Report[/bold blue]",
            border_style="blue",
        )
        temp_console.print(header)

        # Individual sections
        if keyword_data:
            temp_console.print("\n[bold cyan]Keyword Analysis[/bold cyan]")
            keyword_report = self.generate_keyword_analysis_report(keyword_data)
            temp_console.print(keyword_report)

        if pagerank_data:
            temp_console.print("\n[bold cyan]PageRank Analysis[/bold cyan]")
            pagerank_report = self.generate_pagerank_analysis_report(pagerank_data)
            temp_console.print(pagerank_report)

        if onpage_data:
            temp_console.print("\n[bold cyan]On-Page Analysis[/bold cyan]")
            onpage_report = self.generate_onpage_analysis_report(onpage_data)
            temp_console.print(onpage_report)

        return output.getvalue()

    def _create_keyword_table(self, keywords_data: List[Dict]) -> Table:
        """Create keyword data table."""
        table = Table(title="Keyword Analysis")
        table.add_column("Keyword", style="cyan")
        table.add_column("Search Volume", style="green")
        table.add_column("Competition", style="yellow")
        table.add_column("CPC", style="magenta")

        for keyword in keywords_data[:10]:  # Top 10
            table.add_row(
                keyword.get("keyword", "N/A"),
                str(keyword.get("search_volume", 0)),
                str(keyword.get("competition", "N/A")),
                f"${keyword.get('cpc', 0):.2f}",
            )

        return table

    def _create_summary_panel(self, summary_data: Dict) -> Panel:
        """Create summary panel."""
        content = "\n".join(
            [
                f"Total Keywords: {summary_data.get('total_keywords', 0)}",
                f"Avg Search Volume: {summary_data.get('avg_search_volume', 0):.0f}",
                f"Avg Competition: {summary_data.get('avg_competition', 0):.2f}",
            ]
        )

        return Panel(content, title="Summary", border_style="green")

    def create_progress_tracker(self, task_name: str, total_steps: int):
        """Create progress tracker."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        )
        return progress

    def _create_recommendations_table(self, recommendations: List[Dict]) -> Table:
        """Create recommendations table."""
        table = Table(title="SEO Recommendations")
        table.add_column("Title", style="cyan")
        table.add_column("Priority", style="red")
        table.add_column("Category", style="green")
        table.add_column("Impact", style="yellow")
        table.add_column("Effort", style="magenta")

        for rec in recommendations:
            priority_color = self._get_priority_color(rec.get("priority", "medium"))
            table.add_row(
                rec.get("title", "N/A"),
                f"[{priority_color}]{rec.get('priority', 'medium')}[/{priority_color}]",
                rec.get("category", "N/A"),
                rec.get("impact", "N/A"),
                rec.get("effort", "N/A"),
            )

        return table

    def _create_metrics_panel(self, metrics: Dict) -> Panel:
        """Create metrics panel."""
        content = "\n".join(
            [
                f"Overall Score: {metrics.get('overall_score', 0)}/100",
                f"Technical Score: {metrics.get('technical_score', 0)}/100",
                f"Content Score: {metrics.get('content_score', 0)}/100",
                f"Keywords Score: {metrics.get('keywords_score', 0)}/100",
                f"Links Score: {metrics.get('links_score', 0)}/100",
            ]
        )

        return Panel(content, title="SEO Metrics", border_style="blue")

    def _get_priority_color(self, priority: str) -> str:
        """Get color for priority level."""
        colors = {
            "critical": "red",
            "high": "orange",
            "medium": "yellow",
            "low": "green",
        }
        return colors.get(priority.lower(), "white")

    def export_report_to_file(self, data: Dict, filename: str, format: str = "txt"):
        """Export report to file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                if format.lower() == "html":
                    # Simple HTML export
                    f.write("<html><body><pre>")
                    f.write(str(data))
                    f.write("</pre></body></html>")
                else:
                    f.write(str(data))
            return True
        except Exception as e:
            self.display_error(f"Failed to export report: {e}")
            return False


class ProgressTracker:
    """Enhanced progress tracking for long-running operations."""

    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.current_operation = None

    def start_operation(self, operation_name: str, total_steps: int = None):
        """Start tracking a new operation."""
        self.current_operation = operation_name
        if total_steps:
            self.console.print(
                f"[blue]🚀 Starting {operation_name} ({total_steps} steps)[/blue]"
            )
        else:
            self.console.print(f"[blue]🚀 Starting {operation_name}[/blue]")

    def update_step(self, step_name: str, step_number: int = None):
        """Update current step."""
        if step_number:
            self.console.print(f"[cyan]📊 Step {step_number}: {step_name}[/cyan]")
        else:
            self.console.print(f"[cyan]📊 {step_name}[/cyan]")

    def complete_operation(self, summary: str = None):
        """Complete the current operation."""
        message = f"🎉 {self.current_operation} completed!"
        if summary:
            message += f" {summary}"
        self.console.print(f"[green]{message}[/green]")
        self.current_operation = None


# Convenience function for quick reporting
def create_seo_reporter() -> SEOReporter:
    """Create a new SEO reporter instance."""
    return SEOReporter()
