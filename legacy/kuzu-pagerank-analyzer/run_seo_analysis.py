#!/usr/bin/env python3
"""
GitAlchemy Blog SEO PageRank Analyzer
====================================

A comprehensive SEO analysis tool that uses Kùzu graph database to analyze
internal linking structure and calculate PageRank scores for blog posts.

Usage:
    python run_seo_analysis.py [--output-dir DIR] [--export-format FORMAT]

This script provides a working wrapper around the Kùzu PageRank analyzer.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src'))

from kuzu_pagerank_analyzer.main import MarkdownParser, KuzuGraphBuilder, SEOAnalyzer, display_results
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="GitAlchemy Blog SEO PageRank Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_seo_analysis.py
  python run_seo_analysis.py --output-dir ./seo-results
  python run_seo_analysis.py --export-format json
        """
    )
    
    parser.add_argument(
        '--blog-dir',
        type=Path,
        default=Path('../../src/content/blog'),
        help='Directory containing markdown blog posts (default: ../../src/content/blog)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./analysis-results'),
        help='Directory to save analysis results (default: ./analysis-results)'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['json', 'csv', 'both'],
        default='both',
        help='Export format for results (default: both)'
    )
    
    parser.add_argument(
        '--db-path',
        default='blog_seo_analysis',
        help='Kùzu database path (default: blog_seo_analysis)'
    )
    
    args = parser.parse_args()
    
    # Display header
    console.print(Panel.fit(
        "[bold green]🚀 GitAlchemy SEO PageRank Analyzer[/bold green]\n"
        "[cyan]Powered by Kùzu Graph Database[/cyan]",
        style="green"
    ))
    
    # Validate blog directory
    if not args.blog_dir.exists():
        console.print(f"[red]❌ Blog directory not found: {args.blog_dir}[/red]")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Parse all markdown files
        console.print("[cyan]📚 Parsing markdown files...[/cyan]")
        parser = MarkdownParser(args.blog_dir)
        posts_data = parser.parse_all_posts()
        
        if not posts_data:
            console.print("[red]❌ No blog posts found to analyze[/red]")
            return 1
        
        # Step 2: Build Kùzu graph and calculate metrics
        console.print("[cyan]🔧 Building graph database...[/cyan]")
        with KuzuGraphBuilder(args.db_path) as graph_builder:
            graph_builder.create_schema()
            graph_builder.populate_graph(posts_data)
            
            # Try to use Kùzu native PageRank
            console.print("[cyan]⚡ Attempting Kùzu native PageRank...[/cyan]")
            algo_available = graph_builder.install_algo_extension()
            pagerank_df = None
            
            if algo_available:
                pagerank_df = graph_builder.calculate_pagerank_kuzu()
            
            # Get graph data for NetworkX
            console.print("[cyan]📊 Extracting graph data...[/cyan]")
            nodes, edges = graph_builder.get_all_nodes_and_edges()
        
        # Step 3: Calculate comprehensive metrics
        console.print("[cyan]🧮 Calculating SEO metrics...[/cyan]")
        analyzer = SEOAnalyzer(posts_data)
        
        if pagerank_df is not None:
            # Use Kùzu PageRank results + NetworkX for other metrics
            console.print("[cyan]🔄 Combining Kùzu PageRank with NetworkX metrics...[/cyan]")
            metrics = analyzer.calculate_networkx_metrics(nodes, edges)
            
            # Update with Kùzu PageRank scores
            try:
                pagerank_dict = {str(row['node_id']): row['pagerank_score'] for row in pagerank_df.to_dicts()}
                for node_slug in metrics:
                    if node_slug in pagerank_dict:
                        metrics[node_slug]['pagerank'] = pagerank_dict[node_slug]
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not merge Kùzu PageRank: {e}[/yellow]")
        else:
            # Use NetworkX for all metrics
            metrics = analyzer.calculate_networkx_metrics(nodes, edges)
        
        # Step 4: Generate comprehensive analysis
        console.print("[cyan]📈 Generating SEO analysis...[/cyan]")
        analysis = analyzer.generate_comprehensive_analysis(metrics)
        
        # Step 5: Display results
        display_results(analysis, metrics)
        
        # Step 6: Export results
        console.print("[cyan]💾 Exporting results...[/cyan]")
        
        if args.export_format in ['json', 'both']:
            json_file = args.output_dir / 'seo_analysis.json'
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
        
        if args.export_format in ['csv', 'both']:
            try:
                import polars as pl
                
                # Export pillar pages CSV
                if analysis['pillar_pages']:
                    pillar_df = pl.DataFrame(analysis['pillar_pages'])
                    csv_file = args.output_dir / 'pillar_pages.csv'
                    pillar_df.write_csv(csv_file)
                    console.print(f"[green]✅ Pillar pages CSV exported to {csv_file}[/green]")
                
                # Export link opportunities CSV (flattened)
                if analysis['link_opportunities']:
                    flattened_opportunities = []
                    for opp in analysis['link_opportunities']:
                        flat_opp = {
                            'source_slug': opp['source_slug'],
                            'source_title': opp['source_title'],
                            'target_slug': opp['target_slug'],
                            'target_title': opp['target_title'],
                            'keyword_overlap': opp['keyword_overlap'],
                            'shared_keywords': ', '.join(opp.get('shared_keywords', [])),
                            'opportunity_score': opp['opportunity_score'],
                            'target_pagerank': opp['target_pagerank']
                        }
                        flattened_opportunities.append(flat_opp)
                    
                    opp_df = pl.DataFrame(flattened_opportunities)
                    csv_file = args.output_dir / 'link_opportunities.csv'
                    opp_df.write_csv(csv_file)
                    console.print(f"[green]✅ Link opportunities CSV exported to {csv_file}[/green]")
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  CSV export failed: {e}[/yellow]")
        
        # Summary
        console.print("\n[bold green]🎉 Analysis completed successfully![/bold green]")
        console.print(f"[cyan]📊 Total pages analyzed: {len(posts_data)}[/cyan]")
        console.print(f"[cyan]💾 Database saved to: {args.db_path}[/cyan]")
        console.print(f"[cyan]📁 Results exported to: {args.output_dir}[/cyan]")
        
        # Key insights
        console.print(f"\n[bold yellow]🔍 Key Insights:[/bold yellow]")
        console.print(f"• Internal linking density: {analysis['summary']['link_density']:.4f}")
        console.print(f"• Average links per page: {analysis['summary']['avg_links_per_page']}")
        console.print(f"• Pillar pages identified: {len(analysis['pillar_pages'])}")
        console.print(f"• Link opportunities found: {len(analysis['link_opportunities'])}")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())