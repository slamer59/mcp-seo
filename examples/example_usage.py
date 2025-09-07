#!/usr/bin/env python3
"""
Example usage of FastMCP SEO Analysis Server.
Demonstrates various SEO analysis capabilities.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mcp_seo.server import get_clients
from mcp_seo.models.seo_models import (
    OnPageAnalysisRequest, 
    KeywordAnalysisRequest, 
    SERPAnalysisRequest,
    DomainAnalysisRequest,
    DeviceType
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def example_account_info():
    """Example: Get DataForSEO account information."""
    print_section("Account Information")
    
    try:
        client, _, _, _ = get_clients()
        result = client.get_account_info()
        
        if "tasks" in result and result["tasks"]:
            account_data = result["tasks"][0]["result"][0]
            print(f"✅ Account connected successfully!")
            print(f"   Login: {account_data.get('login', 'N/A')}")
            print(f"   Money: ${account_data.get('money', 0):.2f}")
            print(f"   Rates: {len(account_data.get('rates', []))} rate plans available")
        else:
            print("❌ Failed to get account information")
            print(f"   Response: {json.dumps(result, indent=2)}")
    
    except Exception as e:
        print(f"❌ Error getting account info: {e}")


def example_keyword_analysis():
    """Example: Analyze keywords for search volume and competition."""
    print_section("Keyword Analysis")
    
    try:
        _, _, keyword_analyzer, _ = get_clients()
        
        # Sample keywords to analyze
        keywords = ["seo tools", "keyword research", "competitor analysis", "technical seo"]
        
        request = KeywordAnalysisRequest(
            keywords=keywords,
            location="usa",
            language="english",
            device=DeviceType.DESKTOP,
            include_suggestions=True,
            suggestion_limit=10
        )
        
        print(f"🔍 Analyzing {len(keywords)} keywords...")
        result = keyword_analyzer.analyze_keywords(request)
        
        if "keywords_data" in result:
            print("✅ Keyword analysis completed!")
            
            for kw_data in result["keywords_data"]:
                keyword = kw_data["keyword"]
                volume = kw_data.get("search_volume", "N/A")
                cpc = kw_data.get("cpc", 0)
                competition = kw_data.get("competition", 0)
                
                print(f"   📊 {keyword}:")
                print(f"      Search Volume: {volume}")
                print(f"      CPC: ${cpc:.2f}" if cpc else "      CPC: N/A")
                print(f"      Competition: {competition:.2f}" if competition else "      Competition: N/A")
        
            # Show suggestions if available
            if "suggestions" in result and result["suggestions"]:
                print(f"\n   💡 Related keyword suggestions:")
                for suggestion in result["suggestions"][:5]:  # Show top 5
                    print(f"      • {suggestion['keyword']} (Volume: {suggestion.get('search_volume', 'N/A')})")
        
        else:
            print("❌ Keyword analysis failed")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error in keyword analysis: {e}")


def main():
    """Run all examples."""
    print("🚀 FastMCP SEO Analysis Server - Example Usage")
    print("=" * 60)
    
    # Check if credentials are set up
    if not os.getenv("DATAFORSEO_LOGIN") or not os.getenv("DATAFORSEO_PASSWORD"):
        print("❌ DataForSEO credentials not found!")
        print("   Please set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment variables")
        print("   or create a .env file with your credentials.")
        return
    
    # Run examples
    try:
        example_account_info()
        time.sleep(2)  # Small delay between examples
        
        example_keyword_analysis()
        
        print_section("Examples Completed")
        print("✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   - Modify the examples with your own domains and keywords")
        print("   - Explore additional MCP tools in the server")
        print("   - Check out the README.md for detailed tool documentation")
    
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()