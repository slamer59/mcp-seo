#!/usr/bin/env python3
"""
Simple test script for PageRank Analyzer to validate basic functionality.
Tests with a small subset of pages to ensure all components work correctly.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pagerank_analyzer import PageRankAnalyzer


async def test_basic_functionality():
    """Test basic functionality with limited pages."""
    print("🧪 Testing PageRank Analyzer basic functionality...")
    
    # Create analyzer with limited pages for testing
    analyzer = PageRankAnalyzer(
        base_url="https://www.gitalchemy.app",
        max_pages=10  # Limit to 10 pages for quick testing
    )
    
    try:
        # Run basic analysis
        print("📊 Running analysis...")
        insights = await analyzer.run_analysis()
        
        # Check results
        if "error" in insights:
            print(f"❌ Test failed: {insights['error']}")
            return False
            
        # Validate structure
        expected_keys = ['summary', 'pillar_pages', 'orphaned_pages', 'low_outlink_pages', 'all_pages', 'recommendations']
        for key in expected_keys:
            if key not in insights:
                print(f"❌ Missing key in results: {key}")
                return False
                
        # Print results
        summary = insights['summary']
        print(f"✅ Analysis completed successfully!")
        print(f"   • Total pages analyzed: {summary['total_pages']}")
        print(f"   • Total links found: {summary['total_links']}")
        print(f"   • Average PageRank: {summary['avg_pagerank']:.4f}")
        print(f"   • Pillar pages identified: {len(insights['pillar_pages'])}")
        print(f"   • Orphaned pages found: {len(insights['orphaned_pages'])}")
        
        if insights['pillar_pages']:
            top_page = insights['pillar_pages'][0]
            print(f"   • Top authority page: {top_page['title']} (PageRank: {top_page['pagerank']:.4f})")
            
        print(f"   • Generated {len(insights['recommendations'])} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        # Cleanup
        analyzer.cleanup()


def test_url_normalization():
    """Test URL normalization functionality."""
    print("\n🔧 Testing URL normalization...")
    
    analyzer = PageRankAnalyzer()
    
    test_cases = [
        ("https://example.com/page/", "https://example.com/page"),
        ("https://example.com/page?param=1", "https://example.com/page"),
        ("https://example.com/page#section", "https://example.com/page"),
        ("https://example.com/", "https://example.com/"),
        ("https://example.com/page/?param=1#section", "https://example.com/page"),
    ]
    
    for input_url, expected in test_cases:
        result = analyzer.normalize_url(input_url)
        if result == expected:
            print(f"✅ {input_url} → {result}")
        else:
            print(f"❌ {input_url} → {result} (expected: {expected})")
            return False
            
    return True


def test_internal_url_detection():
    """Test internal URL detection."""
    print("\n🔍 Testing internal URL detection...")
    
    analyzer = PageRankAnalyzer(base_url="https://www.gitalchemy.app")
    
    test_cases = [
        ("https://www.gitalchemy.app/features", True),
        ("https://gitalchemy.app/about", True),
        ("/pricing", True),
        ("https://external.com/page", False),
        ("mailto:test@example.com", False),
        ("tel:+1234567890", False),
    ]
    
    for url, expected in test_cases:
        result = analyzer.is_internal_url(url)
        status = "✅" if result == expected else "❌"
        print(f"{status} {url} → {'internal' if result else 'external'}")
        if result != expected:
            return False
            
    return True


async def main():
    """Run all tests."""
    print("🚀 GitAlchemy PageRank Analyzer - Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("URL Normalization", test_url_normalization),
        ("Internal URL Detection", test_internal_url_detection),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The analyzer is ready for production use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)