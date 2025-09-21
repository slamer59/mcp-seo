#!/usr/bin/env python3
"""
Kùzu PageRank SEO Analyzer for GitAlchemy Blog Content
=====================================================

A comprehensive SEO analysis tool that uses Kùzu graph database to analyze
internal linking structure and calculate PageRank scores for blog posts.

Features:
- Parse local markdown files from src/content/blog/
- Extract internal links and frontmatter metadata
- Build Kùzu graph database with BlogPage nodes and LINKS_TO relationships
- Calculate PageRank using Kùzu native algo extension
- Export comprehensive SEO analysis with actionable recommendations

Author: GitAlchemy Team
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

import click
import kuzu
import networkx as nx
import numpy as np
import polars as pl
import frontmatter
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()


class MarkdownParser:
    """Parse markdown files and extract metadata and internal links."""
    
    def __init__(self, blog_dir: Path):
        self.blog_dir = blog_dir
        self.posts = {}
        self.link_pattern = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
        
    def parse_all_posts(self) -> Dict[str, Dict]:
        """Parse all markdown files in the blog directory."""
        markdown_files = list(self.blog_dir.glob("*.md"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Parsing markdown files...", total=len(markdown_files))
            
            for md_file in markdown_files:
                try:
                    post_data = self._parse_single_post(md_file)
                    if post_data:
                        self.posts[post_data['slug']] = post_data
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Error parsing {md_file}: {e}")
                    progress.advance(task)
        
        console.print(f"[green]✅ Successfully parsed {len(self.posts)} blog posts[/green]")
        return self.posts
    
    def _parse_single_post(self, file_path: Path) -> Optional[Dict]:
        """Parse a single markdown file and extract metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            # Extract frontmatter
            metadata = post.metadata
            content = post.content
            
            # Skip unpublished posts
            if not metadata.get('published', True):
                return None
            
            # Extract basic metadata
            slug = metadata.get('slug', file_path.stem)
            title = metadata.get('title', slug.replace('-', ' ').title())
            
            # Calculate word count
            word_count = len(content.split())
            
            # Extract keywords
            keywords = self._extract_keywords(metadata, content)
            
            # Extract internal links
            internal_links = self._extract_internal_links(content)
            
            # Calculate content quality metrics
            quality_metrics = self._calculate_quality_metrics(content)
            
            return {
                'slug': slug,
                'title': title,
                'file_path': str(file_path),
                'date': str(metadata.get('date', '')),
                'author': metadata.get('author', ''),
                'description': metadata.get('description', ''),
                'keywords': keywords,
                'word_count': word_count,
                'internal_links': internal_links,
                'frontmatter': metadata,
                'content': content,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_keywords(self, metadata: Dict, content: str) -> List[str]:
        """Extract keywords from metadata and content."""
        keywords = []
        
        # Keywords from frontmatter
        if 'keywords' in metadata:
            if isinstance(metadata['keywords'], str):
                keywords.extend([k.strip() for k in metadata['keywords'].split(',')])
            elif isinstance(metadata['keywords'], list):
                keywords.extend(metadata['keywords'])
        
        # Extract from title and headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        for header in headers:
            # Simple keyword extraction from headers
            words = re.findall(r'\b[a-zA-Z]{3,}\b', header.lower())
            keywords.extend(words[:3])  # Take first 3 words from each header
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_internal_links(self, content: str) -> List[Dict[str, str]]:
        """Extract internal links using [[filename.md|anchor_text]] pattern."""
        links = []
        matches = self.link_pattern.findall(content)
        
        for match in matches:
            target_file, anchor_text = match if len(match) == 2 else (match[0], match[0])
            
            # Clean up the target file (remove .md extension if present)
            target_slug = target_file.replace('.md', '')
            if not anchor_text:
                anchor_text = target_slug.replace('-', ' ').title()
                
            links.append({
                'target_slug': target_slug,
                'anchor_text': anchor_text.strip()
            })
        
        return links
    
    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Calculate content quality metrics."""
        # Remove HTML and get clean text
        soup = BeautifulSoup(content, 'html.parser')
        clean_text = soup.get_text()
        
        word_count = len(clean_text.split())
        char_count = len(clean_text)
        
        # Count headers
        header_count = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        
        # Count images
        image_count = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        
        # Calculate readability score (simple)
        sentences = len(re.findall(r'[.!?]+', clean_text))
        avg_sentence_length = word_count / max(sentences, 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'header_count': header_count,
            'image_count': image_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': max(0, 100 - (avg_sentence_length * 2))  # Simple readability
        }


class KuzuGraphBuilder:
    """Build and manage Kùzu graph database for SEO analysis."""
    
    def __init__(self, db_path: str = "blog_seo_analysis"):
        self.db_path = db_path
        self.db = None
        self.conn = None
        
    def __enter__(self):
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def create_schema(self):
        """Create the graph schema for blog posts and links."""
        console.print("[cyan]📊 Creating Kùzu graph schema...[/cyan]")
        
        # Drop existing tables if they exist
        try:
            self.conn.execute("DROP TABLE LINKS_TO")
            self.conn.execute("DROP TABLE BlogPage")
        except:
            pass  # Tables might not exist
        
        # Create BlogPage node table
        self.conn.execute("""
            CREATE NODE TABLE BlogPage(
                slug STRING,
                title STRING,
                word_count INT64,
                date STRING,
                author STRING,
                description STRING,
                keywords STRING[],
                file_path STRING,
                readability_score DOUBLE,
                header_count INT64,
                image_count INT64,
                PRIMARY KEY(slug)
            )
        """)
        
        # Create LINKS_TO relationship table
        self.conn.execute("""
            CREATE REL TABLE LINKS_TO(
                FROM BlogPage TO BlogPage,
                anchor_text STRING,
                link_type STRING DEFAULT 'internal'
            )
        """)
        
        console.print("[green]✅ Schema created successfully[/green]")
    
    def populate_graph(self, posts_data: Dict[str, Dict]):
        """Populate the graph with blog post data and relationships."""
        console.print("[cyan]📝 Populating graph with blog post data...[/cyan]")
        
        # Insert blog posts as nodes
        with Progress(console=console) as progress:
            task = progress.add_task("Inserting blog posts...", total=len(posts_data))
            
            for slug, post in posts_data.items():
                try:
                    self.conn.execute("""
                        CREATE (p:BlogPage {
                            slug: $slug,
                            title: $title,
                            word_count: $word_count,
                            date: $date,
                            author: $author,
                            description: $description,
                            keywords: $keywords,
                            file_path: $file_path,
                            readability_score: $readability_score,
                            header_count: $header_count,
                            image_count: $image_count
                        })
                    """, {
                        'slug': slug,
                        'title': post['title'],
                        'word_count': post['word_count'],
                        'date': post['date'],
                        'author': post['author'],
                        'description': post['description'],
                        'keywords': post['keywords'],
                        'file_path': post['file_path'],
                        'readability_score': post['quality_metrics']['readability_score'],
                        'header_count': post['quality_metrics']['header_count'],
                        'image_count': post['quality_metrics']['image_count']
                    })
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Error inserting post {slug}: {e}")
                    progress.advance(task)
        
        # Insert relationships
        console.print("[cyan]🔗 Creating internal link relationships...[/cyan]")
        
        total_links = sum(len(post['internal_links']) for post in posts_data.values())
        
        with Progress(console=console) as progress:
            task = progress.add_task("Creating relationships...", total=total_links)
            
            for source_slug, post in posts_data.items():
                for link in post['internal_links']:
                    target_slug = link['target_slug']
                    
                    # Check if target exists in our posts
                    if target_slug in posts_data:
                        try:
                            self.conn.execute("""
                                MATCH (source:BlogPage {slug: $source_slug}),
                                      (target:BlogPage {slug: $target_slug})
                                CREATE (source)-[l:LINKS_TO {
                                    anchor_text: $anchor_text,
                                    link_type: 'internal'
                                }]->(target)
                            """, {
                                'source_slug': source_slug,
                                'target_slug': target_slug,
                                'anchor_text': link['anchor_text']
                            })
                        except Exception as e:
                            logger.error(f"Error creating link {source_slug} -> {target_slug}: {e}")
                    
                    progress.advance(task)
        
        console.print("[green]✅ Graph populated successfully[/green]")
    
    def install_algo_extension(self):
        """Install and load the Kùzu algo extension."""
        try:
            console.print("[cyan]🔧 Installing Kùzu algo extension...[/cyan]")
            self.conn.execute("INSTALL algo")
            self.conn.execute("LOAD EXTENSION algo")
            console.print("[green]✅ Algo extension loaded successfully[/green]")
        except Exception as e:
            logger.warning(f"Could not install algo extension: {e}")
            console.print("[yellow]⚠️  Algo extension not available, will use NetworkX as fallback[/yellow]")
            return False
        return True
    
    def calculate_pagerank_kuzu(self) -> Optional[pl.DataFrame]:
        """Calculate PageRank using Kùzu's native algorithm."""
        try:
            console.print("[cyan]🧮 Calculating PageRank using Kùzu algo extension...[/cyan]")
            
            # Project the graph for PageRank calculation
            self.conn.execute("CALL project_graph('BlogGraph', ['BlogPage'], ['LINKS_TO'])")
            
            # Run PageRank algorithm - try different syntax
            try:
                result = self.conn.execute("CALL page_rank('BlogGraph') RETURN *")
            except:
                # Try alternative syntax
                result = self.conn.execute("SELECT * FROM page_rank('BlogGraph')")
            
            # Convert to Polars DataFrame
            df_data = []
            while result.has_next():
                row = result.get_next()
                df_data.append({
                    'node_id': row[0],
                    'pagerank_score': row[1]
                })
            
            if df_data:
                pagerank_df = pl.DataFrame(df_data)
                console.print("[green]✅ PageRank calculated successfully with Kùzu[/green]")
                return pagerank_df
            else:
                console.print("[yellow]⚠️  No PageRank results from Kùzu[/yellow]")
                return None
                
        except Exception as e:
            logger.error(f"Kùzu PageRank calculation failed: {e}")
            console.print(f"[red]❌ Kùzu PageRank failed: {e}[/red]")
            return None
    
    def get_all_nodes_and_edges(self) -> Tuple[List[Dict], List[Dict]]:
        """Get all nodes and edges for NetworkX fallback."""
        # Get all nodes
        nodes_result = self.conn.execute("""
            MATCH (p:BlogPage)
            RETURN p.slug, p.title, p.word_count, p.readability_score
        """)
        
        nodes = []
        while nodes_result.has_next():
            row = nodes_result.get_next()
            nodes.append({
                'slug': row[0],
                'title': row[1],
                'word_count': row[2],
                'readability_score': row[3]
            })
        
        # Get all edges
        edges_result = self.conn.execute("""
            MATCH (source:BlogPage)-[l:LINKS_TO]->(target:BlogPage)
            RETURN source.slug, target.slug, l.anchor_text
        """)
        
        edges = []
        while edges_result.has_next():
            row = edges_result.get_next()
            edges.append({
                'source': row[0],
                'target': row[1],
                'anchor_text': row[2]
            })
        
        return nodes, edges


class SEOAnalyzer:
    """Comprehensive SEO analysis using graph metrics."""
    
    def __init__(self, posts_data: Dict[str, Dict]):
        self.posts_data = posts_data
        self.pagerank_scores = {}
        self.centrality_scores = {}
        
    def calculate_networkx_metrics(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict]:
        """Calculate graph metrics using NetworkX as fallback."""
        console.print("[cyan]🔄 Calculating metrics using NetworkX...[/cyan]")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['slug'], **node)
        
        # Add edges
        for edge in edges:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'], anchor_text=edge['anchor_text'])
        
        # Calculate metrics
        metrics = {}
        
        try:
            # PageRank
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(G)
            
            # In-degree centrality
            in_degree = dict(G.in_degree())
            
            # Out-degree centrality  
            out_degree = dict(G.out_degree())
            
            # Closeness centrality
            try:
                closeness = nx.closeness_centrality(G)
            except:
                closeness = {node: 0.0 for node in G.nodes()}
            
            # Authority and hub scores
            try:
                hubs, authorities = nx.hits(G, max_iter=100)
            except:
                hubs = {node: 0.0 for node in G.nodes()}
                authorities = {node: 0.0 for node in G.nodes()}
            
            # Combine all metrics
            for node in G.nodes():
                metrics[node] = {
                    'pagerank': pagerank.get(node, 0.0),
                    'betweenness_centrality': betweenness.get(node, 0.0),
                    'in_degree': in_degree.get(node, 0),
                    'out_degree': out_degree.get(node, 0),
                    'closeness_centrality': closeness.get(node, 0.0),
                    'hub_score': hubs.get(node, 0.0),
                    'authority_score': authorities.get(node, 0.0)
                }
            
            console.print("[green]✅ NetworkX metrics calculated successfully[/green]")
            
        except Exception as e:
            logger.error(f"NetworkX calculation error: {e}")
            console.print(f"[red]❌ NetworkX calculation failed: {e}[/red]")
            
        return metrics
    
    def generate_comprehensive_analysis(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive SEO analysis report."""
        console.print("[cyan]📊 Generating comprehensive SEO analysis...[/cyan]")
        
        analysis = {
            'summary': {},
            'pillar_pages': [],
            'underperforming_pages': [],
            'content_clusters': {},
            'link_opportunities': [],
            'recommendations': []
        }
        
        # Summary statistics
        total_pages = len(self.posts_data)
        total_links = sum(len(post['internal_links']) for post in self.posts_data.values())
        avg_links_per_page = total_links / total_pages if total_pages > 0 else 0
        
        analysis['summary'] = {
            'total_pages': total_pages,
            'total_internal_links': total_links,
            'avg_links_per_page': round(avg_links_per_page, 2),
            'link_density': round(total_links / (total_pages * (total_pages - 1)) if total_pages > 1 else 0, 4)
        }
        
        # Identify pillar pages (high PageRank + authority)
        pillar_candidates = []
        for slug, metric in metrics.items():
            post = self.posts_data.get(slug, {})
            pillar_score = (
                metric.get('pagerank', 0) * 0.4 +
                metric.get('authority_score', 0) * 0.3 +
                metric.get('in_degree', 0) / max(total_pages, 1) * 0.2 +
                min(post.get('word_count', 0) / 2000, 1.0) * 0.1
            )
            
            pillar_candidates.append({
                'slug': slug,
                'title': post.get('title', slug),
                'pillar_score': pillar_score,
                'pagerank': metric.get('pagerank', 0),
                'authority_score': metric.get('authority_score', 0),
                'in_degree': metric.get('in_degree', 0),
                'word_count': post.get('word_count', 0)
            })
        
        # Sort and get top pillar pages
        pillar_candidates.sort(key=lambda x: x['pillar_score'], reverse=True)
        analysis['pillar_pages'] = pillar_candidates[:10]
        
        # Identify underperforming pages (low PageRank + low in-degree)
        underperforming = []
        for slug, metric in metrics.items():
            post = self.posts_data.get(slug, {})
            if metric.get('pagerank', 0) < 0.01 and metric.get('in_degree', 0) < 2:
                underperforming.append({
                    'slug': slug,
                    'title': post.get('title', slug),
                    'pagerank': metric.get('pagerank', 0),
                    'in_degree': metric.get('in_degree', 0),
                    'word_count': post.get('word_count', 0),
                    'keywords': post.get('keywords', [])
                })
        
        analysis['underperforming_pages'] = sorted(underperforming, key=lambda x: x['pagerank'])
        
        # Content clustering by keywords
        keyword_clusters = defaultdict(list)
        for slug, post in self.posts_data.items():
            for keyword in post.get('keywords', []):
                if len(keyword) > 3:  # Filter short keywords
                    keyword_clusters[keyword.lower()].append({
                        'slug': slug,
                        'title': post.get('title', slug),
                        'pagerank': metrics.get(slug, {}).get('pagerank', 0)
                    })
        
        # Keep clusters with at least 2 pages
        analysis['content_clusters'] = {
            keyword: pages for keyword, pages in keyword_clusters.items()
            if len(pages) >= 2
        }
        
        # Generate link opportunities
        link_opportunities = []
        for source_slug, source_post in self.posts_data.items():
            source_keywords = set(kw.lower() for kw in source_post.get('keywords', []))
            current_targets = set(link['target_slug'] for link in source_post['internal_links'])
            
            for target_slug, target_post in self.posts_data.items():
                if source_slug != target_slug and target_slug not in current_targets:
                    target_keywords = set(kw.lower() for kw in target_post.get('keywords', []))
                    
                    # Calculate relevance based on keyword overlap
                    keyword_overlap = len(source_keywords & target_keywords)
                    if keyword_overlap > 0:
                        target_pagerank = metrics.get(target_slug, {}).get('pagerank', 0)
                        opportunity_score = keyword_overlap * (1 + target_pagerank * 10)
                        
                        link_opportunities.append({
                            'source_slug': source_slug,
                            'source_title': source_post.get('title', source_slug),
                            'target_slug': target_slug,
                            'target_title': target_post.get('title', target_slug),
                            'keyword_overlap': keyword_overlap,
                            'shared_keywords': list(source_keywords & target_keywords),
                            'opportunity_score': opportunity_score,
                            'target_pagerank': target_pagerank
                        })
        
        # Sort opportunities by score
        link_opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        analysis['link_opportunities'] = link_opportunities[:20]  # Top 20 opportunities
        
        # Generate actionable recommendations
        recommendations = []
        
        # Recommendation 1: Strengthen pillar pages
        top_pillars = analysis['pillar_pages'][:3]
        if top_pillars:
            recommendations.append({
                'category': 'Pillar Page Optimization',
                'priority': 'High',
                'action': f"Focus on strengthening top pillar pages: {', '.join([p['title'] for p in top_pillars])}",
                'details': 'These pages have high authority and should be the primary hubs in your internal linking strategy.'
            })
        
        # Recommendation 2: Improve underperforming pages
        if analysis['underperforming_pages']:
            recommendations.append({
                'category': 'Link Building',
                'priority': 'High',
                'action': f"Add internal links to {len(analysis['underperforming_pages'])} underperforming pages",
                'details': 'These pages have low PageRank and need more internal links to improve their authority.'
            })
        
        # Recommendation 3: Content clustering
        if analysis['content_clusters']:
            top_clusters = sorted(analysis['content_clusters'].items(), 
                                key=lambda x: len(x[1]), reverse=True)[:3]
            recommendations.append({
                'category': 'Content Clustering',
                'priority': 'Medium',
                'action': f"Create topic clusters around: {', '.join([cluster[0] for cluster in top_clusters])}",
                'details': 'Link related pages together to create topical authority clusters.'
            })
        
        # Recommendation 4: Link opportunities
        if analysis['link_opportunities']:
            recommendations.append({
                'category': 'Internal Linking',
                'priority': 'Medium',
                'action': f"Implement {len(analysis['link_opportunities'])} identified link opportunities",
                'details': 'Add contextually relevant internal links to improve link equity distribution.'
            })
        
        analysis['recommendations'] = recommendations
        
        console.print("[green]✅ Comprehensive analysis completed[/green]")
        return analysis


def display_results(analysis: Dict[str, Any], metrics: Dict[str, Dict]):
    """Display analysis results in rich format."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🚀 GitAlchemy Blog SEO Analysis Results[/bold cyan]",
        style="cyan"
    ))
    
    # Summary Table
    summary_table = Table(title="📊 Summary Statistics", style="cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="green")
    
    for key, value in analysis['summary'].items():
        summary_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(summary_table)
    console.print()
    
    # Top Pillar Pages
    if analysis['pillar_pages']:
        pillar_table = Table(title="🏛️ Top Pillar Pages (Content Hubs)", style="green")
        pillar_table.add_column("Title", style="bold", min_width=30)
        pillar_table.add_column("PageRank", style="cyan")
        pillar_table.add_column("Authority", style="blue")
        pillar_table.add_column("Incoming Links", style="magenta")
        pillar_table.add_column("Word Count", style="yellow")
        
        for page in analysis['pillar_pages'][:5]:
            pillar_table.add_row(
                page['title'][:50] + "..." if len(page['title']) > 50 else page['title'],
                f"{page['pagerank']:.4f}",
                f"{page['authority_score']:.4f}",
                str(page['in_degree']),
                str(page['word_count'])
            )
        
        console.print(pillar_table)
        console.print()
    
    # Underperforming Pages
    if analysis['underperforming_pages']:
        under_table = Table(title="⚠️  Pages Needing More Internal Links", style="yellow")
        under_table.add_column("Title", style="bold", min_width=30)
        under_table.add_column("PageRank", style="red")
        under_table.add_column("Incoming Links", style="red")
        under_table.add_column("Keywords", style="cyan")
        
        for page in analysis['underperforming_pages'][:5]:
            keywords_str = ", ".join(page['keywords'][:3])
            under_table.add_row(
                page['title'][:50] + "..." if len(page['title']) > 50 else page['title'],
                f"{page['pagerank']:.4f}",
                str(page['in_degree']),
                keywords_str
            )
        
        console.print(under_table)
        console.print()
    
    # Link Opportunities
    if analysis['link_opportunities']:
        opp_table = Table(title="🔗 Top Internal Linking Opportunities", style="blue")
        opp_table.add_column("From Page", style="bold", min_width=20)
        opp_table.add_column("To Page", style="bold", min_width=20)
        opp_table.add_column("Shared Keywords", style="green")
        opp_table.add_column("Opportunity Score", style="cyan")
        
        for opp in analysis['link_opportunities'][:5]:
            shared = ", ".join(opp['shared_keywords'][:2])
            opp_table.add_row(
                opp['source_title'][:25] + "..." if len(opp['source_title']) > 25 else opp['source_title'],
                opp['target_title'][:25] + "..." if len(opp['target_title']) > 25 else opp['target_title'],
                shared,
                f"{opp['opportunity_score']:.2f}"
            )
        
        console.print(opp_table)
        console.print()
    
    # Recommendations
    if analysis['recommendations']:
        console.print(Panel(
            Text("🎯 SEO Recommendations", style="bold yellow"),
            style="yellow"
        ))
        
        for i, rec in enumerate(analysis['recommendations'], 1):
            console.print(f"[bold cyan]{i}. {rec['category']} ({rec['priority']} Priority)[/bold cyan]")
            console.print(f"   Action: {rec['action']}")
            console.print(f"   Details: {rec['details']}")
            console.print()


@click.command()
@click.option(
    '--blog-dir', 
    type=click.Path(exists=True, path_type=Path),
    default='../../src/content/blog',
    help='Directory containing markdown blog posts'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default='output',
    help='Directory to save analysis results'
)
@click.option(
    '--export-format',
    type=click.Choice(['json', 'csv', 'both']),
    default='both',
    help='Export format for results'
)
@click.option(
    '--db-path',
    default='blog_seo_analysis',
    help='Kùzu database path'
)
def main(blog_dir: Path, output_dir: Path, export_format: str, db_path: str):
    """
    Comprehensive SEO PageRank Analysis for GitAlchemy Blog Content.
    
    This tool analyzes your blog's internal linking structure using Kùzu graph database
    and provides actionable SEO recommendations to improve search rankings.
    """
    console.print(Panel.fit(
        "[bold green]🚀 GitAlchemy SEO PageRank Analyzer[/bold green]\n"
        "[cyan]Powered by Kùzu Graph Database[/cyan]",
        style="green"
    ))
    
    # Validate blog directory
    if not blog_dir.exists():
        console.print(f"[red]❌ Blog directory not found: {blog_dir}[/red]")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Parse all markdown files
        parser = MarkdownParser(blog_dir)
        posts_data = parser.parse_all_posts()
        
        if not posts_data:
            console.print("[red]❌ No blog posts found to analyze[/red]")
            sys.exit(1)
        
        # Step 2: Build Kùzu graph and calculate metrics
        with KuzuGraphBuilder(db_path) as graph_builder:
            graph_builder.create_schema()
            graph_builder.populate_graph(posts_data)
            
            # Try to use Kùzu native PageRank
            algo_available = graph_builder.install_algo_extension()
            pagerank_df = None
            
            if algo_available:
                pagerank_df = graph_builder.calculate_pagerank_kuzu()
            
            # Get graph data for NetworkX fallback
            nodes, edges = graph_builder.get_all_nodes_and_edges()
        
        # Step 3: Calculate comprehensive metrics
        analyzer = SEOAnalyzer(posts_data)
        
        if pagerank_df is not None:
            # Use Kùzu PageRank results + NetworkX for other metrics
            console.print("[cyan]🔄 Combining Kùzu PageRank with NetworkX metrics...[/cyan]")
            metrics = analyzer.calculate_networkx_metrics(nodes, edges)
            
            # Update with Kùzu PageRank scores
            pagerank_dict = {row[0]: row[1] for row in pagerank_df.to_dicts()}
            for node_slug in metrics:
                if node_slug in pagerank_dict:
                    metrics[node_slug]['pagerank'] = pagerank_dict[node_slug]
        else:
            # Use NetworkX for all metrics
            metrics = analyzer.calculate_networkx_metrics(nodes, edges)
        
        # Step 4: Generate comprehensive analysis
        analysis = analyzer.generate_comprehensive_analysis(metrics)
        
        # Step 5: Display results
        display_results(analysis, metrics)
        
        # Step 6: Export results
        if export_format in ['json', 'both']:
            json_file = output_dir / 'seo_analysis.json'
            export_data = {
                'analysis': analysis,
                'metrics': metrics,
                'posts_summary': {
                    slug: {
                        'title': post['title'],
                        'word_count': post['word_count'],
                        'keywords': post['keywords'],
                        'internal_links_count': len(post['internal_links'])
                    }
                    for slug, post in posts_data.items()
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]✅ JSON results exported to {json_file}[/green]")
        
        if export_format in ['csv', 'both']:
            # Export pillar pages CSV
            if analysis['pillar_pages']:
                pillar_df = pl.DataFrame(analysis['pillar_pages'])
                csv_file = output_dir / 'pillar_pages.csv'
                pillar_df.write_csv(csv_file)
                console.print(f"[green]✅ Pillar pages CSV exported to {csv_file}[/green]")
            
            # Export link opportunities CSV
            if analysis['link_opportunities']:
                # Flatten the data for CSV export
                flattened_opportunities = []
                for opp in analysis['link_opportunities']:
                    flat_opp = {
                        'source_slug': opp['source_slug'],
                        'source_title': opp['source_title'],
                        'target_slug': opp['target_slug'],
                        'target_title': opp['target_title'],
                        'keyword_overlap': opp['keyword_overlap'],
                        'shared_keywords': ', '.join(opp['shared_keywords']),  # Join list to string
                        'opportunity_score': opp['opportunity_score'],
                        'target_pagerank': opp['target_pagerank']
                    }
                    flattened_opportunities.append(flat_opp)
                
                opp_df = pl.DataFrame(flattened_opportunities)
                csv_file = output_dir / 'link_opportunities.csv'
                opp_df.write_csv(csv_file)
                console.print(f"[green]✅ Link opportunities CSV exported to {csv_file}[/green]")
        
        console.print("\n[bold green]🎉 Analysis completed successfully![/bold green]")
        console.print(f"[cyan]Total pages analyzed: {len(posts_data)}[/cyan]")
        console.print(f"[cyan]Database saved to: {db_path}[/cyan]")
        console.print(f"[cyan]Results exported to: {output_dir}[/cyan]")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()