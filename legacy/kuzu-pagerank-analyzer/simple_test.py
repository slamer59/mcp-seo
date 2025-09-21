#!/usr/bin/env python3
"""
Simple test version to debug issues
"""
import sys
import os
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, 'src')

from kuzu_pagerank_analyzer.main import MarkdownParser, KuzuGraphBuilder, SEOAnalyzer

def main():
    blog_dir = Path("../../src/content/blog")
    output_dir = Path("./test-simple")
    output_dir.mkdir(exist_ok=True)
    
    try:
        print("1. Parsing markdown files...")
        parser = MarkdownParser(blog_dir)
        posts_data = parser.parse_all_posts()
        print(f"   Parsed {len(posts_data)} posts")
        
        print("2. Building graph...")
        with KuzuGraphBuilder("simple_test_db") as graph_builder:
            graph_builder.create_schema()
            graph_builder.populate_graph(posts_data)
            
            print("3. Getting data for NetworkX...")
            nodes, edges = graph_builder.get_all_nodes_and_edges()
            print(f"   Got {len(nodes)} nodes and {len(edges)} edges")
        
        print("4. Running analysis...")
        analyzer = SEOAnalyzer(posts_data)
        metrics = analyzer.calculate_networkx_metrics(nodes, edges)
        print(f"   Calculated metrics for {len(metrics)} pages")
        
        print("5. Generating comprehensive analysis...")
        analysis = analyzer.generate_comprehensive_analysis(metrics)
        
        print("6. Exporting results...")
        result = {
            'analysis': analysis,
            'metrics': metrics
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("✅ Success!")
        print(f"Analysis saved to {output_dir / 'results.json'}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total pages: {len(posts_data)}")
        print(f"- Total internal links: {sum(len(post['internal_links']) for post in posts_data.values())}")
        print(f"- Top pillar pages: {len(analysis.get('pillar_pages', []))}")
        print(f"- Link opportunities: {len(analysis.get('link_opportunities', []))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())