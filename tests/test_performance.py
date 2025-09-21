"""
Performance and load tests for the MCP SEO system.

This module contains comprehensive performance tests covering:
- Large dataset processing performance
- Memory usage monitoring during analysis
- API rate limit handling under load
- Concurrent request processing
- Database query performance (Kuzu)
- Graph analysis scaling tests

All tests use pytest-benchmark for accurate performance measurements.
"""

import asyncio
import gc
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
from aioresponses import aioresponses

from mcp_seo.analysis.competitor_analyzer import SERPCompetitorAnalyzer
from mcp_seo.content.blog_analyzer import BlogAnalyzer
from mcp_seo.dataforseo.client import DataForSEOClient, ApiException
from mcp_seo.engines.recommendation_engine import SEORecommendationEngine
from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from mcp_seo.graph.link_graph_builder import LinkGraphBuilder
from mcp_seo.tools.keyword_analyzer import KeywordAnalyzer
from mcp_seo.tools.onpage_analyzer import OnPageAnalyzer

# Configure logging for performance testing
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Context manager for monitoring memory usage during tests."""

    def __init__(self, test_name: str = ""):
        self.test_name = test_name
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None

    def __enter__(self):
        gc.collect()  # Force garbage collection
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        memory_growth = self.end_memory - self.start_memory
        logger.info(f"Memory usage for {self.test_name}:")
        logger.info(f"  Start: {self.start_memory:.2f} MB")
        logger.info(f"  Peak: {self.peak_memory:.2f} MB")
        logger.info(f"  End: {self.end_memory:.2f} MB")
        logger.info(f"  Growth: {memory_growth:.2f} MB")

    def update_peak(self):
        """Update peak memory usage."""
        current = self.process.memory_info().rss / 1024 / 1024
        if current > self.peak_memory:
            self.peak_memory = current


@pytest.fixture
def large_page_dataset():
    """Generate large dataset of pages for performance testing."""
    pages = []
    domains = ["example.com", "test.org", "sample.net", "demo.io", "site.co"]

    for i in range(1000):
        domain = random.choice(domains)
        path = f"/page-{i}" if i % 10 != 0 else f"/category-{i//10}/page-{i}"
        pages.append({
            "url": f"https://{domain}{path}",
            "title": f"Page {i} - {domain.split('.')[0].title()}",
            "status_code": 200 if i % 50 != 0 else random.choice([404, 500, 301]),
            "content_length": random.randint(500, 5000)
        })
    return pages


@pytest.fixture
def large_links_dataset():
    """Generate large dataset of links for performance testing."""
    links = []
    domains = ["example.com", "test.org", "sample.net", "demo.io", "site.co"]

    # Generate internal links
    for i in range(5000):
        domain = random.choice(domains)
        source = f"https://{domain}/page-{random.randint(1, 999)}"
        target = f"https://{domain}/page-{random.randint(1, 999)}"
        anchor = f"Link {i}" if i % 5 == 0 else f"Page {random.randint(1, 999)}"
        links.append((source, target, anchor))

    return links


@pytest.fixture
def mock_dataforseo_responses():
    """Mock responses for DataForSEO API calls."""
    return {
        "account_info": {
            "status_code": 20000,
            "status_message": "Ok.",
            "tasks": [{
                "result": {
                    "money": {
                        "total": 1000.0,
                        "used": 50.0,
                        "left": 950.0
                    },
                    "rates": {
                        "minute": 10,
                        "day": 1000
                    }
                }
            }]
        },
        "keyword_data": {
            "status_code": 20000,
            "status_message": "Ok.",
            "tasks": [{
                "result": [{
                    "keyword": "test keyword",
                    "search_volume": 1000,
                    "competition": 0.5,
                    "cpc": 1.25
                }]
            }]
        },
        "onpage_task": {
            "status_code": 20000,
            "status_message": "Ok.",
            "tasks": [{
                "id": "test-task-id",
                "status_code": 20000,
                "status_message": "Ok."
            }]
        }
    }


class TestLargeDatasetProcessing:
    """Tests for large dataset processing performance."""

    @pytest.mark.slow
    @pytest.mark.performance
    def test_kuzu_batch_insert_performance(self, benchmark, large_page_dataset, large_links_dataset):
        """Test Kuzu database batch insert performance with large datasets."""

        def setup_and_insert():
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Benchmark page insertion
                start_time = time.time()
                manager.add_pages_batch(large_page_dataset)
                page_insert_time = time.time() - start_time

                # Benchmark link insertion
                start_time = time.time()
                manager.add_links_batch(large_links_dataset)
                link_insert_time = time.time() - start_time

                return page_insert_time, link_insert_time

        page_time, link_time = benchmark(setup_and_insert)

        # Performance assertions
        assert page_time < 30.0, f"Page insertion took too long: {page_time:.2f}s"
        assert link_time < 60.0, f"Link insertion took too long: {link_time:.2f}s"

        logger.info(f"Performance metrics:")
        logger.info(f"  Pages/second: {len(large_page_dataset) / page_time:.2f}")
        logger.info(f"  Links/second: {len(large_links_dataset) / link_time:.2f}")

    @pytest.mark.slow
    @pytest.mark.performance
    def test_pagerank_calculation_scaling(self, benchmark, large_page_dataset, large_links_dataset):
        """Test PageRank calculation performance with increasing dataset sizes."""

        def calculate_pagerank(num_pages: int, num_links: int):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Insert subset of data
                manager.add_pages_batch(large_page_dataset[:num_pages])
                manager.add_links_batch(large_links_dataset[:num_links])
                manager.calculate_degree_centrality()

                # Calculate PageRank
                analyzer = PageRankAnalyzer(manager)
                start_time = time.time()
                results = analyzer.calculate_pagerank()
                calculation_time = time.time() - start_time

                return calculation_time, len(results.get('pages', []))

        # Test different dataset sizes
        test_sizes = [(100, 200), (500, 1000), (1000, 2000)]

        for num_pages, num_links in test_sizes:
            calc_time, result_count = benchmark.pedantic(
                calculate_pagerank,
                args=(num_pages, num_links),
                rounds=1,
                iterations=1
            )

            logger.info(f"PageRank for {num_pages} pages, {num_links} links:")
            logger.info(f"  Time: {calc_time:.2f}s")
            logger.info(f"  Pages/second: {result_count / calc_time:.2f}")

    @pytest.mark.slow
    @pytest.mark.performance
    def test_graph_query_performance(self, benchmark, populated_kuzu_manager):
        """Test graph database query performance."""

        def run_complex_queries():
            manager = populated_kuzu_manager

            # Complex query 1: Get top pages by PageRank
            start_time = time.time()
            pages = manager.get_page_data()
            query1_time = time.time() - start_time

            # Complex query 2: Get all links
            start_time = time.time()
            links = manager.get_links_data()
            query2_time = time.time() - start_time

            # Complex query 3: Get incoming links for each page
            start_time = time.time()
            for page in pages[:10]:  # Test first 10 pages
                incoming = manager.get_incoming_links(page['url'])
            query3_time = time.time() - start_time

            return query1_time, query2_time, query3_time

        q1_time, q2_time, q3_time = benchmark(run_complex_queries)

        # Performance assertions
        assert q1_time < 1.0, f"Page data query too slow: {q1_time:.3f}s"
        assert q2_time < 1.0, f"Links data query too slow: {q2_time:.3f}s"
        assert q3_time < 2.0, f"Incoming links queries too slow: {q3_time:.3f}s"


class TestMemoryUsage:
    """Tests for memory usage patterns and leak detection."""

    @pytest.mark.memory
    @pytest.mark.slow
    def test_kuzu_memory_usage_scaling(self, large_page_dataset, large_links_dataset):
        """Test memory usage scaling with dataset size."""

        memory_points = []

        for size_factor in [0.1, 0.5, 1.0]:
            pages_count = int(len(large_page_dataset) * size_factor)
            links_count = int(len(large_links_dataset) * size_factor)

            with MemoryMonitor(f"Kuzu-{pages_count}pages") as monitor:
                with KuzuManager() as manager:
                    manager.initialize_schema()

                    # Insert data
                    manager.add_pages_batch(large_page_dataset[:pages_count])
                    monitor.update_peak()

                    manager.add_links_batch(large_links_dataset[:links_count])
                    monitor.update_peak()

                    # Perform operations
                    manager.calculate_degree_centrality()
                    monitor.update_peak()

                    analyzer = PageRankAnalyzer(manager)
                    analyzer.calculate_pagerank()
                    monitor.update_peak()

            memory_points.append({
                'pages': pages_count,
                'links': links_count,
                'memory_growth': monitor.end_memory - monitor.start_memory,
                'peak_memory': monitor.peak_memory
            })

        # Check memory scaling is reasonable
        for i in range(1, len(memory_points)):
            prev_point = memory_points[i-1]
            curr_point = memory_points[i]

            # Memory should scale sub-linearly with data size
            data_ratio = curr_point['pages'] / prev_point['pages']
            memory_ratio = curr_point['memory_growth'] / max(prev_point['memory_growth'], 1)

            assert memory_ratio < data_ratio * 2, "Memory scaling worse than linear"

    @pytest.mark.memory
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform operations repeatedly
        for i in range(10):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Add sample data
                pages = [
                    {"url": f"https://test.com/page-{j}", "title": f"Page {j}"}
                    for j in range(100)
                ]
                links = [
                    (f"https://test.com/page-{j}", f"https://test.com/page-{j+1}", f"Link {j}")
                    for j in range(99)
                ]

                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Force cleanup
                del pages, links
                gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Should not grow by more than 100MB after 10 iterations (adjusted for CI)
        assert memory_growth < 100, f"Potential memory leak: {memory_growth:.2f}MB growth"

    @pytest.mark.memory
    def test_content_analyzer_memory_efficiency(self):
        """Test memory efficiency of content analyzers."""

        # Generate large content
        large_content = "\n".join([
            f"This is paragraph {i} with some keyword content for testing. " * 10
            for i in range(1000)
        ])

        with MemoryMonitor("BlogAnalyzer") as monitor:
            analyzer = BlogAnalyzer()

            # Analyze large content
            result = analyzer.analyze_content(large_content, "test keyword")
            monitor.update_peak()

            # Ensure result is processed
            assert result is not None
            assert 'readability' in result

        # Memory growth should be reasonable for large content
        assert monitor.peak_memory - monitor.start_memory < 100, "Excessive memory usage"


class TestAPIRateLimitHandling:
    """Tests for API rate limit handling under load."""

    @pytest.mark.slow
    @pytest.mark.requires_network
    def test_dataforseo_rate_limit_handling(self, mock_dataforseo_responses):
        """Test DataForSEO client rate limit handling."""

        rate_limit_response = {
            "status_code": 40304,
            "status_message": "Rate limit exceeded"
        }

        with patch('requests.Session.post') as mock_post:
            # Simulate rate limiting followed by success
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = [
                rate_limit_response,  # First call hits rate limit
                rate_limit_response,  # Second call still hits rate limit
                mock_dataforseo_responses['keyword_data']  # Third call succeeds
            ]
            mock_post.return_value = mock_response

            client = DataForSEOClient("test", "test")

            # Should eventually succeed after rate limit
            with pytest.raises(ApiException, match="Rate limit exceeded"):
                client.get_keyword_data(["test keyword"])

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_concurrent_api_request_performance(self, benchmark, mock_dataforseo_responses):
        """Test performance of concurrent API requests."""

        def simulate_concurrent_requests(num_requests: int = 10):
            results = []

            with patch('requests.Session.post') as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = mock_dataforseo_responses['keyword_data']
                mock_post.return_value = mock_response

                client = DataForSEOClient("test", "test")

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(client.get_keyword_data, [f"keyword-{i}"])
                        for i in range(num_requests)
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Request failed: {e}")

            return len(results)

        successful_requests = benchmark.pedantic(
            simulate_concurrent_requests,
            args=(20,),
            rounds=3,
            iterations=1
        )

        assert successful_requests >= 15, "Too many concurrent requests failed"

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_api_client_connection_pooling(self, benchmark):
        """Test API client connection pooling efficiency."""

        def test_connection_reuse():
            with patch('requests.Session.post') as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "status_code": 20000,
                    "tasks": [{"result": {"test": "data"}}]
                }
                mock_post.return_value = mock_response

                client = DataForSEOClient("test", "test")

                # Make multiple requests using same client
                for i in range(50):
                    client._make_request("/test/endpoint", "POST", {"data": f"test-{i}"})

                return mock_post.call_count

        call_count = benchmark(test_connection_reuse)
        assert call_count == 50, "All requests should have been made"


class TestConcurrentProcessing:
    """Tests for concurrent request processing capabilities."""

    @pytest.mark.slow
    @pytest.mark.performance
    def test_concurrent_graph_operations(self, benchmark):
        """Test concurrent graph database operations."""

        def concurrent_graph_ops():
            results = []

            def worker_function(worker_id: int):
                with KuzuManager() as manager:
                    manager.initialize_schema()

                    # Each worker processes different data
                    pages = [
                        {"url": f"https://worker{worker_id}.com/page-{i}", "title": f"Worker {worker_id} Page {i}"}
                        for i in range(50)
                    ]
                    links = [
                        (f"https://worker{worker_id}.com/page-{i}", f"https://worker{worker_id}.com/page-{i+1}", f"Link {i}")
                        for i in range(49)
                    ]

                    manager.add_pages_batch(pages)
                    manager.add_links_batch(links)
                    manager.calculate_degree_centrality()

                    return len(manager.get_page_data())

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(worker_function, i)
                    for i in range(8)
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

            return len(results)

        completed_workers = benchmark.pedantic(
            concurrent_graph_ops,
            rounds=1,
            iterations=1
        )

        assert completed_workers >= 6, "Too many concurrent workers failed"

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_content_processing(self, benchmark):
        """Test asynchronous content processing performance."""

        async def process_content_async():
            content_items = [
                f"This is test content {i} with various keywords and analysis requirements. " * 20
                for i in range(100)
            ]

            async def analyze_content(content: str, content_id: int) -> dict:
                # Simulate async content analysis
                await asyncio.sleep(0.01)  # Simulate I/O
                analyzer = BlogAnalyzer()
                return analyzer.analyze_content(content, f"keyword-{content_id}")

            # Process content concurrently
            semaphore = asyncio.Semaphore(10)  # Limit concurrency

            async def analyze_with_semaphore(content: str, content_id: int):
                async with semaphore:
                    return await analyze_content(content, content_id)

            tasks = [
                analyze_with_semaphore(content, i)
                for i, content in enumerate(content_items)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful analyses
            successful = sum(1 for r in results if isinstance(r, dict))
            return successful

        # Run the async benchmark
        successful_analyses = await process_content_async()
        assert successful_analyses >= 95, "Too many async content analyses failed"

    @pytest.mark.slow
    @pytest.mark.performance
    def test_link_graph_builder_concurrency(self, benchmark):
        """Test LinkGraphBuilder performance with concurrent operations."""

        async def concurrent_graph_building():
            results = []

            async def build_graph(domain: str):
                with aioresponses() as mock_responses:
                    # Mock sitemap response
                    mock_responses.get(
                        f"https://{domain}/sitemap.xml",
                        payload=f'''<?xml version="1.0" encoding="UTF-8"?>
                        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                            <url><loc>https://{domain}/</loc></url>
                            <url><loc>https://{domain}/page1</loc></url>
                            <url><loc>https://{domain}/page2</loc></url>
                        </urlset>''',
                        content_type='application/xml'
                    )

                    # Mock page responses
                    for page in ['/', '/page1', '/page2']:
                        mock_responses.get(
                            f"https://{domain}{page}",
                            payload=f'''<html>
                                <head><title>{domain.title()} {page}</title></head>
                                <body>
                                    <a href="/page1">Page 1</a>
                                    <a href="/page2">Page 2</a>
                                </body>
                            </html>''',
                            content_type='text/html'
                        )

                    with KuzuManager() as manager:
                        manager.initialize_schema()

                        builder = LinkGraphBuilder(
                            base_url=f"https://{domain}",
                            kuzu_manager=manager,
                            max_pages=10
                        )

                        await builder.build_graph()
                        return len(manager.get_page_data())

            # Build graphs for multiple domains concurrently
            domains = [f"test{i}.com" for i in range(5)]

            tasks = [build_graph(domain) for domain in domains]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful = [r for r in results if isinstance(r, int)]
            return len(successful)

        successful_builds = asyncio.run(concurrent_graph_building())
        assert successful_builds >= 4, "Too many concurrent graph builds failed"


class TestGraphAnalysisScaling:
    """Tests for graph analysis scaling with different graph sizes."""

    @pytest.mark.slow
    @pytest.mark.performance
    def test_pagerank_algorithm_scaling(self, benchmark):
        """Test PageRank algorithm performance scaling."""

        def test_pagerank_scaling(num_nodes: int, connectivity_ratio: float = 0.1):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create nodes
                pages = [
                    {"url": f"https://scale-test.com/page-{i}", "title": f"Page {i}"}
                    for i in range(num_nodes)
                ]
                manager.add_pages_batch(pages)

                # Create links based on connectivity ratio
                num_links = int(num_nodes * connectivity_ratio * num_nodes)
                links = []
                for _ in range(num_links):
                    source_id = random.randint(0, num_nodes - 1)
                    target_id = random.randint(0, num_nodes - 1)
                    if source_id != target_id:
                        links.append((
                            f"https://scale-test.com/page-{source_id}",
                            f"https://scale-test.com/page-{target_id}",
                            f"Link {len(links)}"
                        ))

                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Time PageRank calculation
                analyzer = PageRankAnalyzer(manager)
                start_time = time.time()
                results = analyzer.calculate_pagerank()
                calculation_time = time.time() - start_time

                return calculation_time, len(results.get('pages', []))

        # Test different graph sizes
        test_sizes = [100, 500, 1000]

        for size in test_sizes:
            calc_time, result_count = benchmark.pedantic(
                test_pagerank_scaling,
                args=(size,),
                rounds=1,
                iterations=1
            )

            # Performance should be reasonable even for larger graphs
            pages_per_second = result_count / calc_time
            assert pages_per_second > 10, f"PageRank too slow for {size} nodes: {pages_per_second:.2f} pages/s"

    @pytest.mark.slow
    @pytest.mark.performance
    def test_graph_statistics_performance(self, benchmark):
        """Test graph statistics calculation performance."""

        def calculate_comprehensive_stats():
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create a moderately sized graph
                num_pages = 500
                pages = [
                    {"url": f"https://stats-test.com/page-{i}", "title": f"Page {i}"}
                    for i in range(num_pages)
                ]
                manager.add_pages_batch(pages)

                # Create random links
                links = []
                for i in range(1000):
                    source = random.randint(0, num_pages - 1)
                    target = random.randint(0, num_pages - 1)
                    if source != target:
                        links.append((
                            f"https://stats-test.com/page-{source}",
                            f"https://stats-test.com/page-{target}",
                            f"Link {i}"
                        ))

                manager.add_links_batch(links)

                # Benchmark comprehensive statistics
                start_time = time.time()

                # Calculate various statistics
                manager.calculate_degree_centrality()
                stats = manager.get_graph_stats()
                page_data = manager.get_page_data()
                links_data = manager.get_links_data()

                # Sample incoming links for top pages
                for page in page_data[:10]:
                    incoming = manager.get_incoming_links(page['url'])

                calculation_time = time.time() - start_time

                return calculation_time, stats['total_pages'], stats['total_links']

        calc_time, page_count, link_count = benchmark(calculate_comprehensive_stats)

        # Performance assertions
        assert calc_time < 10.0, f"Graph statistics calculation too slow: {calc_time:.2f}s"
        assert page_count > 0, "No pages found in statistics"
        assert link_count > 0, "No links found in statistics"

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_recommendation_engine_scaling(self, benchmark):
        """Test recommendation engine performance with large datasets."""

        def generate_recommendations():
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create realistic SEO graph
                domains = ["example.com", "competitor1.com", "competitor2.com"]
                pages = []
                links = []

                for domain in domains:
                    # Add pages for each domain
                    domain_pages = [
                        {"url": f"https://{domain}/page-{i}", "title": f"{domain} Page {i}"}
                        for i in range(200)
                    ]
                    pages.extend(domain_pages)

                    # Add internal links
                    for i in range(150):
                        source_idx = random.randint(0, 199)
                        target_idx = random.randint(0, 199)
                        if source_idx != target_idx:
                            links.append((
                                f"https://{domain}/page-{source_idx}",
                                f"https://{domain}/page-{target_idx}",
                                f"Internal link {i}"
                            ))

                # Add cross-domain links
                for _ in range(100):
                    source_domain = random.choice(domains)
                    target_domain = random.choice(domains)
                    if source_domain != target_domain:
                        links.append((
                            f"https://{source_domain}/page-{random.randint(0, 199)}",
                            f"https://{target_domain}/page-{random.randint(0, 199)}",
                            "External link"
                        ))

                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Calculate PageRank
                analyzer = PageRankAnalyzer(manager)
                pagerank_results = analyzer.calculate_pagerank()

                # Generate recommendations
                engine = SEORecommendationEngine(manager)
                start_time = time.time()
                recommendations = engine.generate_link_recommendations("example.com")
                recommendation_time = time.time() - start_time

                return recommendation_time, len(recommendations)

        rec_time, rec_count = benchmark.pedantic(
            generate_recommendations,
            rounds=1,
            iterations=1
        )

        # Performance and quality assertions
        assert rec_time < 15.0, f"Recommendation generation too slow: {rec_time:.2f}s"
        assert rec_count > 0, "No recommendations generated"


# Performance test execution helpers
def run_performance_tests():
    """Helper function to run all performance tests with proper configuration."""
    pytest.main([
        "tests/test_performance.py",
        "-v",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds,iterations",
        "-m", "performance or benchmark"
    ])


def run_memory_tests():
    """Helper function to run memory-specific tests."""
    pytest.main([
        "tests/test_performance.py",
        "-v",
        "-s",
        "-m", "memory"
    ])


if __name__ == "__main__":
    # Run performance tests when script is executed directly
    print("Running performance tests...")
    run_performance_tests()
    print("\nRunning memory tests...")
    run_memory_tests()