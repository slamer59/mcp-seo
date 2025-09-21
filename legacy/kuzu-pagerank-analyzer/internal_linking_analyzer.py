#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
#   "beautifulsoup4>=4.12.0",
#   "python-dotenv>=1.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
#   "click>=8.0.0",
#   "networkx>=3.0.0",
#   "matplotlib>=3.7.0"
# ]
# ///

"""
GitAlchemy Internal Linking Analyzer
Analyzes and optimizes internal link structure for better SEO
"""

import os
import requests
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import click
import pandas as pd
import networkx as nx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

load_dotenv('.env.local')

console = Console()

class InternalLinkingAnalyzer:
    """Analyze and optimize internal linking structure for GitAlchemy"""
    
    def __init__(self, base_url: str = "https://www.gitalchemy.app"):
        self.base_url = base_url.rstrip('/')
        self.pages = {}
        self.link_graph = nx.DiGraph()
        
    def fetch_sitemap(self) -> List[str]:
        """Fetch all URLs from sitemap"""
        sitemap_url = f"{self.base_url}/sitemap.xml"
        
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            urls = []
            
            # Handle namespace
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            for url in root.findall('ns:url', namespace):
                loc = url.find('ns:loc', namespace)
                if loc is not None:
                    urls.append(loc.text)
                    
            console.print(f"[green]✅ Found {len(urls)} URLs in sitemap[/green]")
            return urls
            
        except Exception as e:
            console.print(f"[red]❌ Error fetching sitemap: {e}[/red]")
            return []
    
    def analyze_page(self, url: str) -> Dict:
        """Analyze a single page for internal links and content"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract page data
            page_data = {
                'url': url,
                'title': soup.find('title').get_text().strip() if soup.find('title') else '',
                'h1': soup.find('h1').get_text().strip() if soup.find('h1') else '',
                'h2_tags': [h2.get_text().strip() for h2 in soup.find_all('h2')],
                'meta_description': '',
                'word_count': len(soup.get_text().split()),
                'internal_links': [],
                'external_links': [],
                'content_keywords': []
            }
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                page_data['meta_description'] = meta_desc.get('content', '')
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                
                if parsed.netloc == urlparse(self.base_url).netloc:
                    # Internal link
                    page_data['internal_links'].append({
                        'url': full_url,
                        'anchor_text': link.get_text().strip(),
                        'title': link.get('title', '')
                    })
                else:
                    # External link
                    page_data['external_links'].append({
                        'url': full_url,
                        'anchor_text': link.get_text().strip()
                    })
            
            # Extract content keywords (simplified)
            content_text = soup.get_text().lower()
            gitlab_keywords = [
                'gitlab', 'mobile', 'android', 'client', 'devops', 'git',
                'merge request', 'repository', 'project management', 'ci/cd'
            ]
            
            for keyword in gitlab_keywords:
                if keyword in content_text:
                    count = content_text.count(keyword)
                    page_data['content_keywords'].append({
                        'keyword': keyword,
                        'count': count
                    })
            
            return page_data
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Error analyzing {url}: {e}[/yellow]")
            return None
    
    def build_link_graph(self, pages_data: List[Dict]) -> None:
        """Build network graph of internal links"""
        # Add all pages as nodes
        for page in pages_data:
            if page:
                self.link_graph.add_node(page['url'], **{
                    'title': page['title'],
                    'h1': page['h1'],
                    'word_count': page['word_count'],
                    'internal_links_count': len(page['internal_links'])
                })
        
        # Add edges for internal links
        for page in pages_data:
            if page:
                for link in page['internal_links']:
                    target_url = link['url']
                    if self.link_graph.has_node(target_url):
                        self.link_graph.add_edge(
                            page['url'], 
                            target_url,
                            anchor_text=link['anchor_text']
                        )
    
    def analyze_link_structure(self) -> Dict:
        """Analyze the internal link structure"""
        if not self.link_graph.nodes():
            return {}
        
        analysis = {
            'total_pages': len(self.link_graph.nodes()),
            'total_internal_links': len(self.link_graph.edges()),
            'average_internal_links': len(self.link_graph.edges()) / len(self.link_graph.nodes()),
            'orphaned_pages': [],
            'highly_linked_pages': [],
            'poorly_linked_pages': [],
            'hub_pages': [],
            'authority_pages': []
        }
        
        # Find orphaned pages (no incoming links)
        for node in self.link_graph.nodes():
            if self.link_graph.in_degree(node) == 0:
                analysis['orphaned_pages'].append({
                    'url': node,
                    'title': self.link_graph.nodes[node].get('title', '')
                })
        
        # Find highly linked pages (top 10 by incoming links)
        in_degree = dict(self.link_graph.in_degree())
        sorted_pages = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        
        analysis['highly_linked_pages'] = [
            {
                'url': url,
                'title': self.link_graph.nodes[url].get('title', ''),
                'incoming_links': count
            }
            for url, count in sorted_pages[:10]
        ]
        
        # Find poorly linked pages (bottom 10 by incoming links)
        analysis['poorly_linked_pages'] = [
            {
                'url': url,
                'title': self.link_graph.nodes[url].get('title', ''),
                'incoming_links': count
            }
            for url, count in sorted_pages[-10:] if count > 0
        ]
        
        # Find hub pages (high outgoing links)
        out_degree = dict(self.link_graph.out_degree())
        hub_pages = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis['hub_pages'] = [
            {
                'url': url,
                'title': self.link_graph.nodes[url].get('title', ''),
                'outgoing_links': count
            }
            for url, count in hub_pages
        ]
        
        return analysis
    
    def suggest_internal_links(self) -> List[Dict]:
        """Suggest internal linking opportunities based on content similarity"""
        suggestions = []
        
        # Simple keyword-based suggestions
        pages = list(self.pages.values())
        
        for page in pages:
            if not page:
                continue
                
            page_keywords = {kw['keyword'] for kw in page.get('content_keywords', [])}
            related_pages = []
            
            # Find pages with similar keywords
            for other_page in pages:
                if not other_page or other_page['url'] == page['url']:
                    continue
                
                other_keywords = {kw['keyword'] for kw in other_page.get('content_keywords', [])}
                common_keywords = page_keywords.intersection(other_keywords)
                
                if len(common_keywords) >= 2:  # At least 2 common keywords
                    # Check if link doesn't already exist
                    existing_links = {link['url'] for link in page.get('internal_links', [])}
                    if other_page['url'] not in existing_links:
                        related_pages.append({
                            'url': other_page['url'],
                            'title': other_page['title'],
                            'common_keywords': list(common_keywords),
                            'similarity_score': len(common_keywords)
                        })
            
            # Sort by similarity and take top 5
            related_pages.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            if related_pages:
                suggestions.append({
                    'source_page': {
                        'url': page['url'],
                        'title': page['title']
                    },
                    'suggested_links': related_pages[:5]
                })
        
        return suggestions
    
    def generate_report(self, output_file: str = None) -> None:
        """Generate comprehensive internal linking report"""
        console.print("\n[green]📊 Generating Internal Linking Analysis Report[/green]")
        
        analysis = self.analyze_link_structure()
        
        # Summary statistics
        stats_table = Table(title="Internal Linking Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        
        stats_table.add_row("Total Pages", str(analysis.get('total_pages', 0)))
        stats_table.add_row("Total Internal Links", str(analysis.get('total_internal_links', 0)))
        stats_table.add_row("Average Links per Page", f"{analysis.get('average_internal_links', 0):.1f}")
        stats_table.add_row("Orphaned Pages", str(len(analysis.get('orphaned_pages', []))))
        
        console.print(stats_table)
        
        # Top linked pages
        if analysis.get('highly_linked_pages'):
            top_pages_table = Table(title="Most Linked Pages")
            top_pages_table.add_column("Page Title", style="green")
            top_pages_table.add_column("Incoming Links", style="yellow")
            
            for page in analysis['highly_linked_pages'][:5]:
                title = page['title'][:60] + "..." if len(page['title']) > 60 else page['title']
                top_pages_table.add_row(title, str(page['incoming_links']))
            
            console.print(top_pages_table)
        
        # Orphaned pages
        if analysis.get('orphaned_pages'):
            orphaned_table = Table(title="Orphaned Pages (No Incoming Links)")
            orphaned_table.add_column("Page Title", style="red")
            orphaned_table.add_column("URL", style="dim")
            
            for page in analysis['orphaned_pages'][:10]:
                title = page['title'][:50] + "..." if len(page['title']) > 50 else page['title']
                url = page['url'].replace(self.base_url, "")
                orphaned_table.add_row(title, url)
            
            console.print(orphaned_table)
        
        # Link suggestions
        suggestions = self.suggest_internal_links()
        if suggestions:
            console.print(f"\n[blue]🔗 Found {len(suggestions)} pages with internal linking opportunities[/blue]")
            
            # Show top 5 suggestions
            for i, suggestion in enumerate(suggestions[:5]):
                source_title = suggestion['source_page']['title'][:50]
                console.print(f"\n[cyan]Page: {source_title}[/cyan]")
                
                for link in suggestion['suggested_links'][:3]:
                    target_title = link['title'][:40]
                    keywords = ", ".join(link['common_keywords'])
                    console.print(f"  → Link to: {target_title} (Keywords: {keywords})")
        
        # Save detailed report
        if output_file:
            report_data = {
                'analysis_summary': analysis,
                'link_suggestions': suggestions,
                'analyzed_at': pd.Timestamp.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            console.print(f"[green]✅ Detailed report saved to {output_file}[/green]")
    
    def run_analysis(self, limit_pages: int = None) -> None:
        """Run complete internal linking analysis"""
        console.print(Panel.fit(
            "[bold blue]GitAlchemy Internal Linking Analyzer[/bold blue]\n"
            "Analyzing internal link structure and opportunities",
            border_style="blue"
        ))
        
        # Fetch sitemap
        urls = self.fetch_sitemap()
        
        if limit_pages:
            urls = urls[:limit_pages]
            console.print(f"[yellow]📝 Limiting analysis to {limit_pages} pages[/yellow]")
        
        # Analyze each page
        console.print(f"[blue]🔍 Analyzing {len(urls)} pages...[/blue]")
        
        pages_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for url in urls:
                task = progress.add_task(f"Analyzing: {url.split('/')[-1]}", total=1)
                
                page_data = self.analyze_page(url)
                pages_data.append(page_data)
                
                if page_data:
                    self.pages[url] = page_data
                
                progress.update(task, completed=1)
        
        # Build link graph
        console.print("[blue]🕸️  Building link graph...[/blue]")
        self.build_link_graph(pages_data)
        
        # Generate report
        self.generate_report()

@click.command()
@click.option('--limit', '-l', type=int, help='Limit number of pages to analyze')
@click.option('--output', '-o', help='Output file for detailed report')
@click.option('--base-url', '-u', default='https://www.gitalchemy.app', help='Base URL to analyze')
def main(limit: int, output: str, base_url: str):
    """
    GitAlchemy Internal Linking Analyzer
    
    Analyze internal link structure and suggest optimization opportunities
    for better SEO and user navigation.
    """
    
    analyzer = InternalLinkingAnalyzer(base_url)
    analyzer.run_analysis(limit_pages=limit)
    
    if output:
        analyzer.generate_report(output)

if __name__ == "__main__":
    main()