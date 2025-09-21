#!/usr/bin/env python3
"""
GitAlchemy PageRank Analyzer using Kùzu Graph Database

This script analyzes the internal link structure of GitAlchemy website using PageRank
to identify pillar pages and optimize link equity distribution.

Features:
- Sitemap parsing from https://www.gitalchemy.app/sitemap.xml
- Web crawling for internal links discovery
- Kùzu graph database for efficient graph operations
- PageRank calculation for authority analysis
- Rich console output with progress tracking
- Export capabilities (JSON, CSV)
- Link structure optimization recommendations
"""

import asyncio
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from xml.etree import ElementTree as ET

import aiohttp
import kuzu
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree


class PageRankAnalyzer:
    """Main class for analyzing website PageRank using Kùzu graph database."""
    
    def __init__(self, base_url: str = "https://www.gitalchemy.app", max_pages: int = 100):
        self.base_url = base_url
        self.max_pages = max_pages
        self.console = Console()
        self.pages: Set[str] = set()
        self.links: List[Tuple[str, str]] = []
        self.pagerank_scores: Dict[str, float] = {}
        self.db_path = "pagerank_analysis.db"
        self.db_connection: Optional[kuzu.Connection] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pagerank_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters for consistency."""
        parsed = urlparse(url)
        # Remove fragment and normalize
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            parsed.params,
            '',  # Remove query parameters for consistency
            ''   # Remove fragments
        ))
        return normalized
        
    def is_internal_url(self, url: str) -> bool:
        """Check if URL is internal to the base domain."""
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        
        # Extract base domain (remove www. if present)
        base_domain = parsed_base.netloc.replace('www.', '') if parsed_base.netloc.startswith('www.') else parsed_base.netloc
        url_domain = parsed_url.netloc.replace('www.', '') if parsed_url.netloc.startswith('www.') else parsed_url.netloc
        
        return (
            url_domain == base_domain or 
            parsed_url.netloc == '' or
            url_domain.endswith(base_domain)
        )
        
    async def fetch_sitemap(self, progress: Progress, task_id: TaskID) -> List[str]:
        """Fetch and parse sitemap.xml to get initial page list."""
        sitemap_url = f"{self.base_url}/sitemap.xml"
        pages = []
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'GitAlchemy-PageRank-Analyzer/1.0'}
            ) as session:
                progress.update(task_id, description="Fetching sitemap.xml...")
                async with session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)
                        
                        # Handle sitemap namespace
                        namespace = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                        urls = root.findall('.//sm:url/sm:loc', namespace)
                        
                        if not urls:  # Try without namespace
                            urls = root.findall('.//url/loc')
                        
                        for url_elem in urls:
                            if url_elem.text:
                                normalized_url = self.normalize_url(url_elem.text)
                                if self.is_internal_url(normalized_url):
                                    pages.append(normalized_url)
                                    
                        progress.update(task_id, description=f"Found {len(pages)} pages in sitemap")
                        self.logger.info(f"Successfully parsed sitemap: {len(pages)} pages found")
                    else:
                        self.logger.error(f"Failed to fetch sitemap: HTTP {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error fetching sitemap: {e}")
            
        return pages[:self.max_pages]  # Limit to max_pages
        
    async def crawl_page_links(self, session: aiohttp.ClientSession, url: str, 
                              progress: Progress, task_id: TaskID) -> List[str]:
        """Crawl a single page to extract internal links."""
        internal_links = []
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract all links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, href)
                        normalized_url = self.normalize_url(absolute_url)
                        
                        if self.is_internal_url(normalized_url) and normalized_url != url:
                            internal_links.append(normalized_url)
                            
                    progress.update(task_id, advance=1, 
                                  description=f"Crawled {url} → {len(internal_links)} links")
                else:
                    self.logger.warning(f"Failed to crawl {url}: HTTP {response.status}")
                    progress.update(task_id, advance=1)
                    
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            progress.update(task_id, advance=1)
            
        return internal_links
        
    async def build_link_graph(self, pages: List[str], progress: Progress) -> None:
        """Build the link graph by crawling all pages."""
        crawl_task = progress.add_task("Crawling pages for links...", total=len(pages))
        
        # Add all pages to our set
        self.pages.update(pages)
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'GitAlchemy-PageRank-Analyzer/1.0'},
            connector=aiohttp.TCPConnector(limit=10)  # Limit concurrent connections
        ) as session:
            
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            
            async def crawl_with_semaphore(url: str) -> List[str]:
                async with semaphore:
                    return await self.crawl_page_links(session, url, progress, crawl_task)
                    
            # Crawl all pages concurrently
            tasks = [crawl_with_semaphore(url) for url in pages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and build link graph
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing {pages[i]}: {result}")
                    continue
                    
                source_url = pages[i]
                target_urls = result
                
                for target_url in target_urls:
                    # Add target URL to pages if not already present
                    self.pages.add(target_url)
                    # Add link to graph
                    self.links.append((source_url, target_url))
                    
        progress.update(crawl_task, description=f"Completed crawling. Found {len(self.links)} links between {len(self.pages)} pages")
        
    def setup_kuzu_database(self, progress: Progress) -> None:
        """Set up Kùzu graph database with schema."""
        db_task = progress.add_task("Setting up Kùzu database...", total=3)
        
        try:
            # Create database
            database = kuzu.Database(self.db_path)
            self.db_connection = kuzu.Connection(database)
            progress.update(db_task, advance=1, description="Database created")
            
            # Create node table for pages
            self.db_connection.execute("""
                CREATE NODE TABLE Page(
                    url STRING, 
                    title STRING, 
                    domain STRING,
                    path STRING,
                    pagerank DOUBLE,
                    in_degree INT64,
                    out_degree INT64,
                    PRIMARY KEY (url)
                )
            """)
            progress.update(db_task, advance=1, description="Page node table created")
            
            # Create relationship table for links
            self.db_connection.execute("""
                CREATE REL TABLE Links(
                    FROM Page TO Page,
                    anchor_text STRING,
                    link_type STRING
                )
            """)
            progress.update(db_task, advance=1, description="Links relationship table created")
            
            self.logger.info("Kùzu database setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up Kùzu database: {e}")
            raise
            
    def populate_database(self, progress: Progress) -> None:
        """Populate database with pages and links."""
        populate_task = progress.add_task("Populating database...", total=len(self.pages) + len(self.links))
        
        try:
            # Insert pages
            for url in self.pages:
                parsed = urlparse(url)
                title = parsed.path.replace('/', ' ').strip() or 'Home'
                
                self.db_connection.execute("""
                    CREATE (p:Page {
                        url: $url, 
                        title: $title, 
                        domain: $domain,
                        path: $path,
                        pagerank: 0.0,
                        in_degree: 0,
                        out_degree: 0
                    })
                """, parameters={"url": url, "title": title, "domain": parsed.netloc, "path": parsed.path})
                
                progress.update(populate_task, advance=1)
                
            # Insert links
            for source, target in self.links:
                try:
                    self.db_connection.execute("""
                        MATCH (s:Page {url: $source}), (t:Page {url: $target})
                        CREATE (s)-[:Links {anchor_text: "", link_type: "internal"}]->(t)
                    """, parameters={"source": source, "target": target})
                    progress.update(populate_task, advance=1)
                except Exception as e:
                    # Skip if nodes don't exist
                    progress.update(populate_task, advance=1)
                    continue
                    
            self.logger.info("Database population completed")
            
        except Exception as e:
            self.logger.error(f"Error populating database: {e}")
            raise
            
    def calculate_degree_centrality(self, progress: Progress) -> None:
        """Calculate and update in-degree and out-degree for all pages."""
        degree_task = progress.add_task("Calculating degree centrality...", total=2)
        
        try:
            # Calculate and update in-degree
            self.db_connection.execute("""
                MATCH (p:Page)<-[:Links]-(other:Page)
                WITH p, COUNT(other) as in_deg
                SET p.in_degree = in_deg
            """)
            progress.update(degree_task, advance=1, description="In-degree calculated")
            
            # Calculate and update out-degree
            self.db_connection.execute("""
                MATCH (p:Page)-[:Links]->(other:Page)
                WITH p, COUNT(other) as out_deg
                SET p.out_degree = out_deg
            """)
            progress.update(degree_task, advance=1, description="Out-degree calculated")
            
            self.logger.info("Degree centrality calculation completed")
            
        except Exception as e:
            self.logger.error(f"Error calculating degree centrality: {e}")
            
    def calculate_pagerank(self, progress: Progress, damping_factor: float = 0.85, 
                          max_iterations: int = 100, tolerance: float = 1e-6) -> None:
        """Calculate PageRank using power iteration method."""
        pagerank_task = progress.add_task("Calculating PageRank...", total=max_iterations)
        
        try:
            # Get all pages
            result = self.db_connection.execute("MATCH (p:Page) RETURN p.url as url")
            pages = [row[0] for row in result]
            n_pages = len(pages)
            
            if n_pages == 0:
                self.logger.error("No pages found in database")
                return
                
            # Initialize PageRank scores
            initial_score = 1.0 / n_pages
            for page in pages:
                self.pagerank_scores[page] = initial_score
                
            # Power iteration
            for iteration in range(max_iterations):
                new_scores = {}
                total_change = 0.0
                
                for page in pages:
                    # Get incoming links
                    result = self.db_connection.execute("""
                        MATCH (source:Page)-[:Links]->(target:Page {url: $url})
                        RETURN source.url as source_url, source.out_degree as out_degree
                    """, parameters={"url": page})
                    
                    incoming_links = list(result)
                    
                    # Calculate new PageRank score
                    rank_sum = 0.0
                    for source_url, out_degree in incoming_links:
                        if out_degree > 0:
                            rank_sum += self.pagerank_scores[source_url] / out_degree
                            
                    new_score = (1 - damping_factor) / n_pages + damping_factor * rank_sum
                    new_scores[page] = new_score
                    total_change += abs(new_score - self.pagerank_scores[page])
                    
                # Update scores
                self.pagerank_scores.update(new_scores)
                
                progress.update(pagerank_task, advance=1, 
                              description=f"PageRank iteration {iteration + 1}, change: {total_change:.6f}")
                
                # Check convergence
                if total_change < tolerance:
                    progress.update(pagerank_task, completed=max_iterations,
                                  description=f"PageRank converged after {iteration + 1} iterations")
                    break
                    
            # Update database with PageRank scores
            for url, score in self.pagerank_scores.items():
                self.db_connection.execute("""
                    MATCH (p:Page {url: $url})
                    SET p.pagerank = $score
                """, parameters={"url": url, "score": score})
                
            self.logger.info(f"PageRank calculation completed. Converged in {iteration + 1} iterations")
            
        except Exception as e:
            self.logger.error(f"Error calculating PageRank: {e}")
            
    def generate_insights(self) -> Dict:
        """Generate insights and recommendations from PageRank analysis."""
        try:
            # Get page data
            result = self.db_connection.execute("""
                MATCH (p:Page) 
                RETURN p.url as url, p.pagerank as pagerank, p.in_degree as in_degree, 
                       p.out_degree as out_degree, p.title as title, p.path as path
                ORDER BY p.pagerank DESC
            """)
            
            pages_data = []
            for row in result:
                pages_data.append({
                    'url': row[0],
                    'pagerank': row[1],
                    'in_degree': row[2],
                    'out_degree': row[3],
                    'title': row[4],
                    'path': row[5]
                })
                
            if not pages_data:
                return {"error": "No page data found"}
                
            # Calculate statistics
            pagerank_scores = [p['pagerank'] for p in pages_data]
            in_degrees = [p['in_degree'] for p in pages_data]
            out_degrees = [p['out_degree'] for p in pages_data]
            
            # Identify pillar pages (top 10% by PageRank)
            pillar_threshold = np.percentile(pagerank_scores, 90)
            pillar_pages = [p for p in pages_data if p['pagerank'] >= pillar_threshold]
            
            # Identify orphaned pages (no incoming links)
            orphaned_pages = [p for p in pages_data if p['in_degree'] == 0]
            
            # Identify pages with low internal links (bottom 25% by out_degree)
            low_outlink_threshold = np.percentile(out_degrees, 25)
            low_outlink_pages = [p for p in pages_data if p['out_degree'] <= low_outlink_threshold]
            
            insights = {
                'summary': {
                    'total_pages': len(pages_data),
                    'total_links': len(self.links),
                    'avg_pagerank': np.mean(pagerank_scores),
                    'avg_in_degree': np.mean(in_degrees),
                    'avg_out_degree': np.mean(out_degrees),
                    'max_pagerank': max(pagerank_scores),
                    'min_pagerank': min(pagerank_scores)
                },
                'pillar_pages': sorted(pillar_pages, key=lambda x: x['pagerank'], reverse=True)[:10],
                'orphaned_pages': orphaned_pages,
                'low_outlink_pages': low_outlink_pages[:10],
                'all_pages': pages_data,
                'recommendations': self.generate_recommendations(pages_data, pillar_pages, orphaned_pages, low_outlink_pages)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
            
    def generate_recommendations(self, all_pages: List[Dict], pillar_pages: List[Dict], 
                               orphaned_pages: List[Dict], low_outlink_pages: List[Dict]) -> List[str]:
        """Generate specific recommendations for link structure optimization."""
        recommendations = []
        
        # Pillar page recommendations
        if pillar_pages:
            recommendations.append(
                f"✅ Identified {len(pillar_pages)} pillar pages with high authority. "
                f"Ensure these pages are easily accessible from your main navigation."
            )
            
            top_pillar = pillar_pages[0]
            recommendations.append(
                f"🎯 Top authority page: {top_pillar['title']} (PageRank: {top_pillar['pagerank']:.4f}). "
                f"Consider featuring this prominently in your site structure."
            )
        
        # Orphaned pages recommendations
        if orphaned_pages:
            recommendations.append(
                f"⚠️  Found {len(orphaned_pages)} orphaned pages with no incoming links. "
                f"These pages are missing valuable link equity and may be hard to discover."
            )
            
            for orphan in orphaned_pages[:3]:
                recommendations.append(
                    f"   → Add internal links to: {orphan['title']} ({orphan['path']})"
                )
        
        # Low outlink recommendations
        if low_outlink_pages:
            recommendations.append(
                f"🔗 {len(low_outlink_pages)} pages have very few outgoing links. "
                f"Consider adding relevant internal links to distribute link equity better."
            )
        
        # Hub page opportunities
        high_inlink_pages = sorted(all_pages, key=lambda x: x['in_degree'], reverse=True)[:5]
        recommendations.append(
            f"📊 Create content hubs around high-authority pages: "
            f"{', '.join([p['title'] for p in high_inlink_pages])}"
        )
        
        return recommendations
        
    def create_visualization(self, insights: Dict) -> Table:
        """Create rich table visualization of top pages."""
        table = Table(title="🏆 Top Pages by PageRank Authority", show_header=True, header_style="bold magenta")
        
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Page Title", style="bold blue", min_width=30)
        table.add_column("PageRank", justify="right", style="green")
        table.add_column("In Links", justify="center", style="yellow")
        table.add_column("Out Links", justify="center", style="cyan")
        table.add_column("URL", style="dim", max_width=40)
        
        for i, page in enumerate(insights['pillar_pages'], 1):
            table.add_row(
                str(i),
                page['title'][:30] + "..." if len(page['title']) > 30 else page['title'],
                f"{page['pagerank']:.4f}",
                str(page['in_degree']),
                str(page['out_degree']),
                page['path']
            )
            
        return table
        
    def export_results(self, insights: Dict, output_dir: str = "pagerank_results") -> None:
        """Export results to JSON and CSV files."""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export full insights to JSON
            json_path = Path(output_dir) / f"pagerank_insights_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
                
            # Export page data to CSV
            csv_path = Path(output_dir) / f"pagerank_pages_{timestamp}.csv"
            df = pd.DataFrame(insights['all_pages'])
            df.to_csv(csv_path, index=False)
            
            # Export recommendations to text file
            rec_path = Path(output_dir) / f"recommendations_{timestamp}.txt"
            with open(rec_path, 'w', encoding='utf-8') as f:
                f.write("GitAlchemy PageRank Analysis Recommendations\n")
                f.write("=" * 50 + "\n\n")
                for rec in insights['recommendations']:
                    f.write(f"• {rec}\n\n")
                    
            self.console.print(f"✅ Results exported to {output_dir}/")
            self.logger.info(f"Results exported to {output_dir}/")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            
    async def run_analysis(self) -> Dict:
        """Run the complete PageRank analysis."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            
            try:
                # Step 1: Fetch sitemap
                sitemap_task = progress.add_task("Fetching sitemap...", total=1)
                pages = await self.fetch_sitemap(progress, sitemap_task)
                
                if not pages:
                    self.console.print("❌ No pages found in sitemap")
                    return {"error": "No pages found in sitemap"}
                
                # Step 2: Build link graph
                await self.build_link_graph(pages, progress)
                
                # Step 3: Setup database
                self.setup_kuzu_database(progress)
                
                # Step 4: Populate database
                self.populate_database(progress)
                
                # Step 5: Calculate metrics
                self.calculate_degree_centrality(progress)
                self.calculate_pagerank(progress)
                
                # Step 6: Generate insights
                analysis_task = progress.add_task("Generating insights...", total=1)
                insights = self.generate_insights()
                progress.update(analysis_task, advance=1, description="Analysis completed")
                
                return insights
                
            except Exception as e:
                self.logger.error(f"Analysis failed: {e}")
                return {"error": str(e)}
                
    def display_results(self, insights: Dict) -> None:
        """Display analysis results in rich format."""
        if "error" in insights:
            self.console.print(f"❌ Analysis failed: {insights['error']}")
            return
            
        # Summary panel
        summary = insights['summary']
        summary_text = f"""
📊 [bold]Analysis Summary[/bold]
   • Total Pages: {summary['total_pages']}
   • Total Links: {summary['total_links']}
   • Average PageRank: {summary['avg_pagerank']:.4f}
   • Average Incoming Links: {summary['avg_in_degree']:.1f}
   • Average Outgoing Links: {summary['avg_out_degree']:.1f}
        """
        
        self.console.print(Panel(summary_text, title="GitAlchemy PageRank Analysis", border_style="blue"))
        self.console.print()
        
        # Top pages table
        self.console.print(self.create_visualization(insights))
        self.console.print()
        
        # Issues panel
        orphaned = len(insights['orphaned_pages'])
        low_outlinks = len(insights['low_outlink_pages'])
        
        issues_text = f"""
⚠️  [bold red]Issues Found[/bold red]
   • {orphaned} orphaned pages (no incoming links)
   • {low_outlinks} pages with few outgoing links
        """
        
        self.console.print(Panel(issues_text, title="Link Structure Issues", border_style="red"))
        self.console.print()
        
        # Recommendations
        self.console.print("[bold green]🎯 Recommendations:[/bold green]")
        for rec in insights['recommendations']:
            self.console.print(f"  {rec}")
        
    def cleanup(self) -> None:
        """Cleanup database connection."""
        if self.db_connection:
            self.db_connection.close()
            
        # Remove database file if it exists
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()


async def main():
    """Main entry point."""
    console = Console()
    console.print("🚀 GitAlchemy PageRank Analyzer", style="bold blue")
    console.print("Using Kùzu Graph Database for efficient analysis\n")
    
    analyzer = PageRankAnalyzer()
    
    try:
        # Run analysis
        insights = await analyzer.run_analysis()
        
        # Display results
        analyzer.display_results(insights)
        
        # Export results
        if "error" not in insights:
            analyzer.export_results(insights)
            
    except KeyboardInterrupt:
        console.print("\n❌ Analysis interrupted by user")
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())