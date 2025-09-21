#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31.0",
#   "python-dotenv>=1.0.0",
#   "rich>=13.0.0",
#   "pandas>=2.0.0",
#   "click>=8.0.0"
# ]
# ///

"""
GitAlchemy SEO Analyzer using DataForSEO API
Comprehensive SEO analysis tool for GitAlchemy mobile GitLab client
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

import click
import requests
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

console = Console()

class DataForSEOClient:
    """DataForSEO API Client for comprehensive SEO analysis"""
    
    def __init__(self):
        self.base_url = "https://api.dataforseo.com/v3"
        self.login = os.getenv('DATAFORSEO_LOGIN')
        self.password = os.getenv('DATAFORSEO_PASSWORD')
        
        if not self.login or not self.password:
            console.print("[red]Error: DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD must be set in .env.local[/red]")
            raise ValueError("Missing DataForSEO credentials")
    
    def _make_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to DataForSEO API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=data,
                auth=(self.login, self.password),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]API Request failed: {e}[/red]")
            return {}

class GitAlchemySEOAnalyzer:
    """Main SEO analyzer for GitAlchemy"""
    
    def __init__(self):
        self.client = DataForSEOClient()
        self.target_keywords = [
            "gitlab mobile client",
            "gitlab android app", 
            "gitlab ios app",
            "mobile gitlab management",
            "gitlab client app",
            "gitalchemy",
            "gitlab merge request mobile",
            "gitlab ci cd mobile",
            "gitlab project management android",
            "mobile devops gitlab"
        ]
        self.competitors = [
            "labcoat gitlab",
            "git+ gitlab", 
            "mgvora gitplus",
            "gitlab mobile devops"
        ]
        self.target_domain = "gitalchemy.app"
    
    def analyze_keyword_rankings(self, keywords: List[str], location: str = "United States") -> Dict:
        """Analyze keyword rankings for GitAlchemy and competitors"""
        
        console.print(f"[blue]🔍 Analyzing rankings for {len(keywords)} keywords...[/blue]")
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for keyword in keywords:
                task = progress.add_task(f"Analyzing: {keyword}", total=1)
                
                # SERP Analysis
                serp_data = self._get_serp_data(keyword, location)
                
                # Keyword Difficulty
                difficulty_data = self._get_keyword_difficulty(keyword, location)
                
                # Search Volume
                volume_data = self._get_search_volume(keyword, location)
                
                results[keyword] = {
                    'serp_results': serp_data,
                    'difficulty': difficulty_data,
                    'search_volume': volume_data,
                    'gitalchemy_position': self._find_domain_position(serp_data, self.target_domain),
                    'competitor_analysis': self._analyze_competitors(serp_data),
                    'analyzed_at': datetime.now(timezone.utc).isoformat()
                }
                
                progress.update(task, completed=1)
                time.sleep(1)  # Rate limiting
        
        return results
    
    def _get_serp_data(self, keyword: str, location: str) -> Dict:
        """Get SERP data for keyword"""
        data = [{
            "keyword": keyword,
            "location_name": location,
            "language_name": "English",
            "device": "desktop",
            "os": "windows"
        }]
        
        return self.client._make_request("serp/google/organic/task_post", data)
    
    def _get_keyword_difficulty(self, keyword: str, location: str) -> Dict:
        """Get keyword difficulty metrics"""
        data = [{
            "keyword": keyword,
            "location_name": location,
            "language_name": "English"
        }]
        
        return self.client._make_request("dataforseo_labs/google/keyword_difficulty/task_post", data)
    
    def _get_search_volume(self, keyword: str, location: str) -> Dict:
        """Get search volume data"""
        data = [{
            "keywords": [keyword],
            "location_name": location,
            "language_name": "English"
        }]
        
        return self.client._make_request("keywords_data/google_ads/search_volume/task_post", data)
    
    def _find_domain_position(self, serp_data: Dict, domain: str) -> Optional[int]:
        """Find GitAlchemy's position in SERP results"""
        if not serp_data.get('results'):
            return None
            
        for i, result in enumerate(serp_data.get('results', []), 1):
            if domain in result.get('url', ''):
                return i
        return None
    
    def _analyze_competitors(self, serp_data: Dict) -> List[Dict]:
        """Analyze competitor positions in SERP"""
        competitors = []
        
        if not serp_data.get('results'):
            return competitors
            
        for i, result in enumerate(serp_data.get('results', [])[:10], 1):
            url = result.get('url', '')
            title = result.get('title', '')
            
            # Identify known competitors
            competitor_type = None
            if 'github.com/mgvora/gitplus' in url:
                competitor_type = 'Git+ Mobile App'
            elif 'labcoat' in title.lower():
                competitor_type = 'LabCoat GitLab'
            elif 'docs.gitlab.com' in url:
                competitor_type = 'Official GitLab Docs'
            elif 'reddit.com' in url and 'gitlab' in title.lower():
                competitor_type = 'Community Discussion'
            
            if competitor_type:
                competitors.append({
                    'position': i,
                    'url': url,
                    'title': title,
                    'type': competitor_type
                })
        
        return competitors
    
    def generate_seo_report(self, analysis_results: Dict, output_file: str = None) -> None:
        """Generate comprehensive SEO report"""
        
        console.print("\n[green]📊 Generating SEO Analysis Report[/green]")
        
        # Summary Table
        summary_table = Table(title="GitAlchemy SEO Performance Summary")
        summary_table.add_column("Keyword", style="cyan")
        summary_table.add_column("Position", style="yellow") 
        summary_table.add_column("Search Volume", style="green")
        summary_table.add_column("Difficulty", style="red")
        summary_table.add_column("Top Competitor", style="magenta")
        
        for keyword, data in analysis_results.items():
            position = data.get('gitalchemy_position', 'Not Found')
            position_str = str(position) if position else "Not in Top 100"
            
            # Extract metrics (simplified for demo)
            search_volume = "N/A"  # Would extract from volume_data
            difficulty = "N/A"     # Would extract from difficulty_data
            
            top_competitor = "None"
            competitors = data.get('competitor_analysis', [])
            if competitors:
                top_competitor = competitors[0]['type']
            
            summary_table.add_row(
                keyword,
                position_str,
                search_volume,
                difficulty,
                top_competitor
            )
        
        console.print(summary_table)
        
        # Recommendations Panel
        recommendations = self._generate_recommendations(analysis_results)
        rec_panel = Panel(
            recommendations,
            title="🎯 SEO Recommendations",
            border_style="green"
        )
        console.print(rec_panel)
        
        # Save to file if requested
        if output_file:
            self._save_report(analysis_results, output_file)
    
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate SEO recommendations based on analysis"""
        recommendations = []
        
        # Check for missing rankings
        missing_rankings = [k for k, v in results.items() if not v.get('gitalchemy_position')]
        if missing_rankings:
            recommendations.append(
                f"🔍 Target missing keywords: {', '.join(missing_rankings[:3])}"
            )
        
        # Check for competitor opportunities
        competitor_count = sum(len(v.get('competitor_analysis', [])) for v in results.values())
        if competitor_count > 0:
            recommendations.append(
                f"🏆 {competitor_count} competitor opportunities identified"
            )
        
        # Generic recommendations
        recommendations.extend([
            "📱 Optimize for mobile-first indexing",
            "🔗 Build quality backlinks from developer communities", 
            "📄 Create comprehensive GitLab tutorial content",
            "⚡ Improve Core Web Vitals performance",
            "🎯 Target long-tail keywords with lower competition"
        ])
        
        return "\n".join(f"• {rec}" for rec in recommendations)
    
    def _save_report(self, results: Dict, filename: str) -> None:
        """Save analysis results to JSON file"""
        output_path = Path(filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✅ Report saved to {output_path}[/green]")

@click.command()
@click.option('--keywords', '-k', help='Comma-separated keywords to analyze')
@click.option('--location', '-l', default='United States', help='Target location for analysis')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--competitors', '-c', is_flag=True, help='Include competitor analysis')
@click.option('--full-report', '-f', is_flag=True, help='Generate full comprehensive report')
def main(keywords: str, location: str, output: str, competitors: bool, full_report: bool):
    """
    GitAlchemy SEO Analyzer using DataForSEO API
    
    Analyze keyword rankings, competitor positions, and SEO opportunities
    for GitAlchemy mobile GitLab client.
    """
    
    console.print(Panel.fit(
        "[bold blue]GitAlchemy SEO Analyzer[/bold blue]\n"
        "Powered by DataForSEO API",
        border_style="blue"
    ))
    
    try:
        analyzer = GitAlchemySEOAnalyzer()
        
        # Determine keywords to analyze
        target_keywords = analyzer.target_keywords
        if keywords:
            target_keywords = [k.strip() for k in keywords.split(',')]
        
        if competitors:
            target_keywords.extend(analyzer.competitors)
        
        # Perform analysis
        console.print(f"[yellow]🚀 Starting SEO analysis for {len(target_keywords)} keywords[/yellow]")
        
        results = analyzer.analyze_keyword_rankings(target_keywords, location)
        
        # Generate report
        output_file = output or f"gitalchemy_seo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.generate_seo_report(results, output_file if full_report else None)
        
        console.print(f"\n[green]✅ Analysis complete! Analyzed {len(results)} keywords[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise click.Abort()

if __name__ == "__main__":
    main()