"""
Database-specific performance tests for Kuzu graph database operations.

This module focuses specifically on testing database performance characteristics:
- Query execution times
- Index performance
- Batch operation efficiency
- Connection pooling and management
- Memory usage patterns in database operations
"""

import pytest
import time
import gc
import logging
from typing import List, Dict
from pathlib import Path

from mcp_seo.graph.kuzu_manager import KuzuManager
from mcp_seo.graph.pagerank_analyzer import PageRankAnalyzer
from .performance_config import perf_config, PerformanceDataGenerator, PerformanceMetrics

logger = logging.getLogger(__name__)


class TestKuzuQueryPerformance:
    """Test Kuzu database query performance characteristics."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_batch_insert_scaling(self, benchmark):
        """Test how batch insert performance scales with data size."""

        def insert_batch(batch_size: int):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Generate test data
                pages = PerformanceDataGenerator.generate_pages(batch_size, domain_count=5)
                links = PerformanceDataGenerator.generate_links(pages, density=2.0)

                # Time page insertion
                start_time = time.time()
                manager.add_pages_batch(pages)
                page_time = time.time() - start_time

                # Time link insertion
                start_time = time.time()
                manager.add_links_batch(links)
                link_time = time.time() - start_time

                return {
                    'pages_per_second': len(pages) / page_time if page_time > 0 else 0,
                    'links_per_second': len(links) / link_time if link_time > 0 else 0,
                    'total_time': page_time + link_time
                }

        # Test different batch sizes
        for batch_size in [100, 500, 1000]:
            result = benchmark.pedantic(
                insert_batch,
                args=(batch_size,),
                rounds=3,
                iterations=1
            )

            # Performance expectations
            assert result['pages_per_second'] > 50, f"Page insertion too slow: {result['pages_per_second']:.1f} pages/s"
            assert result['links_per_second'] > 100, f"Link insertion too slow: {result['links_per_second']:.1f} links/s"

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_query_complexity_scaling(self, benchmark):
        """Test how query performance scales with graph complexity."""

        def setup_and_query(num_pages: int, link_density: float):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create test graph
                pages = PerformanceDataGenerator.generate_pages(num_pages, domain_count=3)
                links = PerformanceDataGenerator.generate_links(pages, density=link_density)

                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Test various query types
                metrics = PerformanceMetrics()

                # Simple query: get all pages
                metrics.start_timing('get_pages')
                all_pages = manager.get_page_data()
                metrics.end_timing('get_pages')

                # Complex query: get all links
                metrics.start_timing('get_links')
                all_links = manager.get_links_data()
                metrics.end_timing('get_links')

                # Expensive query: get incoming links for top pages
                top_pages = all_pages[:min(10, len(all_pages))]
                metrics.start_timing('incoming_links')
                for page in top_pages:
                    incoming = manager.get_incoming_links(page['url'])
                metrics.end_timing('incoming_links')

                # Graph statistics
                metrics.start_timing('graph_stats')
                stats = manager.get_graph_stats()
                metrics.end_timing('graph_stats')

                return {
                    'page_query_time': metrics.get_metric('get_pages_duration'),
                    'link_query_time': metrics.get_metric('get_links_duration'),
                    'incoming_query_time': metrics.get_metric('incoming_links_duration'),
                    'stats_query_time': metrics.get_metric('graph_stats_duration'),
                    'total_pages': len(all_pages),
                    'total_links': len(all_links)
                }

        # Test different graph complexities
        test_configs = [
            (200, 1.0),   # Small, sparse graph
            (500, 2.0),   # Medium graph
            (1000, 3.0)   # Large, dense graph
        ]

        for num_pages, density in test_configs:
            result = benchmark.pedantic(
                setup_and_query,
                args=(num_pages, density),
                rounds=2,
                iterations=1
            )

            # Query time should be reasonable even for complex graphs
            assert result['page_query_time'] < 2.0, f"Page query too slow: {result['page_query_time']:.3f}s"
            assert result['link_query_time'] < 3.0, f"Link query too slow: {result['link_query_time']:.3f}s"
            assert result['incoming_query_time'] < 5.0, f"Incoming links query too slow: {result['incoming_query_time']:.3f}s"

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_concurrent_database_access(self, benchmark):
        """Test database performance under concurrent access."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def concurrent_operations():
            # Create shared database
            db_path = "/tmp/test_concurrent_kuzu.db"
            results = []
            errors = []

            def worker_operation(worker_id: int):
                try:
                    with KuzuManager(db_path) as manager:
                        if worker_id == 0:  # First worker initializes schema
                            manager.initialize_schema()

                        # Each worker operates on different data
                        pages = PerformanceDataGenerator.generate_pages(
                            50, domain_count=1
                        )
                        # Unique URLs for each worker
                        for page in pages:
                            page['url'] = page['url'].replace('test-domain-0', f'worker-{worker_id}')

                        links = PerformanceDataGenerator.generate_links(pages, density=1.5)

                        start_time = time.time()
                        manager.add_pages_batch(pages)
                        manager.add_links_batch(links)
                        manager.calculate_degree_centrality()

                        # Query operations
                        page_data = manager.get_page_data()
                        operation_time = time.time() - start_time

                        return {
                            'worker_id': worker_id,
                            'operation_time': operation_time,
                            'pages_processed': len(pages),
                            'final_page_count': len(page_data)
                        }

                except Exception as e:
                    errors.append(f"Worker {worker_id}: {str(e)}")
                    return None

            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker_operation, i) for i in range(6)]

                for future in as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        errors.append(str(e))

            # Cleanup
            try:
                Path(db_path).unlink(missing_ok=True)
            except Exception:
                pass

            return len(results), len(errors), results

        successful_workers, error_count, worker_results = benchmark.pedantic(
            concurrent_operations,
            rounds=1,
            iterations=1
        )

        # Should handle concurrent access gracefully
        assert error_count <= 1, f"Too many concurrent access errors: {error_count}"
        assert successful_workers >= 4, f"Too few workers completed: {successful_workers}"

        # Performance should be reasonable under concurrent load
        if worker_results:
            avg_time = sum(r['operation_time'] for r in worker_results) / len(worker_results)
            assert avg_time < 10.0, f"Concurrent operations too slow: {avg_time:.2f}s average"

    @pytest.mark.memory
    @pytest.mark.performance
    def test_database_memory_efficiency(self):
        """Test database memory usage patterns."""
        import psutil

        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_points = []

        # Test memory usage with increasing data sizes
        for size_multiplier in [1, 2, 4, 8]:
            gc.collect()
            before_memory = process.memory_info().rss / 1024 / 1024

            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create progressively larger datasets
                pages = PerformanceDataGenerator.generate_pages(
                    100 * size_multiplier, domain_count=3
                )
                links = PerformanceDataGenerator.generate_links(pages, density=2.0)

                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Perform some queries to load data
                page_data = manager.get_page_data()
                link_data = manager.get_links_data()

                during_memory = process.memory_info().rss / 1024 / 1024

            gc.collect()
            after_memory = process.memory_info().rss / 1024 / 1024

            memory_points.append({
                'size_multiplier': size_multiplier,
                'data_size': len(pages),
                'before_memory': before_memory,
                'during_memory': during_memory,
                'after_memory': after_memory,
                'memory_growth': during_memory - before_memory,
                'memory_released': during_memory - after_memory
            })

        # Analyze memory patterns
        for i, point in enumerate(memory_points):
            logger.info(f"Size {point['size_multiplier']}x: "
                       f"Data={point['data_size']}, "
                       f"Growth={point['memory_growth']:.1f}MB, "
                       f"Released={point['memory_released']:.1f}MB")

            # Memory growth should be reasonable
            memory_per_page = point['memory_growth'] / point['data_size'] * 1024 * 1024  # bytes per page
            assert memory_per_page < 50000, f"Excessive memory per page: {memory_per_page:.0f} bytes"

            # Should release most memory after closing
            if point['memory_growth'] > 10:  # Only check if significant growth
                release_ratio = point['memory_released'] / point['memory_growth']
                assert release_ratio > 0.7, f"Poor memory cleanup: {release_ratio:.2f} release ratio"

        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        # Overall memory growth should be minimal
        assert total_growth < 100, f"Excessive total memory growth: {total_growth:.1f}MB"


class TestPageRankPerformance:
    """Test PageRank algorithm performance characteristics."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_pagerank_algorithm_scaling(self, benchmark):
        """Test PageRank calculation performance scaling."""

        def calculate_pagerank_with_timing(num_pages: int, link_density: float):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create test graph
                pages = PerformanceDataGenerator.generate_pages(num_pages, domain_count=5)
                links = PerformanceDataGenerator.generate_links(pages, density=link_density)

                # Setup timing
                setup_start = time.time()
                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()
                setup_time = time.time() - setup_start

                # PageRank calculation timing
                analyzer = PageRankAnalyzer(manager)
                calc_start = time.time()
                results = analyzer.calculate_pagerank()
                calc_time = time.time() - calc_start

                return {
                    'setup_time': setup_time,
                    'calculation_time': calc_time,
                    'total_time': setup_time + calc_time,
                    'pages_processed': len(results.get('pages', [])),
                    'pages_per_second': len(results.get('pages', [])) / calc_time if calc_time > 0 else 0
                }

        # Test different graph sizes
        test_sizes = [
            (200, 1.5),   # Small graph
            (500, 2.0),   # Medium graph
            (1000, 2.5)   # Large graph
        ]

        scaling_results = []

        for num_pages, density in test_sizes:
            result = benchmark.pedantic(
                calculate_pagerank_with_timing,
                args=(num_pages, density),
                rounds=2,
                iterations=1
            )

            scaling_results.append({
                'size': num_pages,
                'density': density,
                **result
            })

            # Performance thresholds
            assert result['pages_per_second'] > 20, f"PageRank too slow: {result['pages_per_second']:.1f} pages/s"
            assert result['calculation_time'] < 15.0, f"PageRank calculation too slow: {result['calculation_time']:.2f}s"

        # Check scaling characteristics
        for i in range(1, len(scaling_results)):
            prev_result = scaling_results[i-1]
            curr_result = scaling_results[i]

            size_ratio = curr_result['size'] / prev_result['size']
            time_ratio = curr_result['calculation_time'] / prev_result['calculation_time']

            # Time complexity should be reasonable (not exponential)
            assert time_ratio < size_ratio ** 1.5, f"Poor time complexity scaling: {time_ratio:.2f} vs {size_ratio:.2f}"

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_pagerank_convergence_performance(self, benchmark):
        """Test PageRank convergence performance with different parameters."""

        def test_convergence_params(damping_factor: float, tolerance: float, max_iterations: int):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create fixed test graph
                pages = PerformanceDataGenerator.generate_pages(300, domain_count=3)
                links = PerformanceDataGenerator.generate_links(pages, density=2.0)

                manager.add_pages_batch(pages)
                manager.add_links_batch(links)
                manager.calculate_degree_centrality()

                # Test PageRank with different parameters
                analyzer = PageRankAnalyzer(manager)

                start_time = time.time()
                results = analyzer.calculate_pagerank(
                    damping_factor=damping_factor,
                    tolerance=tolerance,
                    max_iterations=max_iterations
                )
                calc_time = time.time() - start_time

                return {
                    'calculation_time': calc_time,
                    'converged': results.get('converged', False),
                    'iterations': results.get('iterations', 0),
                    'final_difference': results.get('max_difference', 1.0)
                }

        # Test different convergence parameters
        param_sets = [
            (0.85, 1e-6, 100),   # Standard parameters
            (0.85, 1e-4, 50),    # Looser tolerance, fewer iterations
            (0.90, 1e-6, 100),   # Higher damping factor
        ]

        for damping, tolerance, max_iter in param_sets:
            result = benchmark.pedantic(
                test_convergence_params,
                args=(damping, tolerance, max_iter),
                rounds=3,
                iterations=1
            )

            # Should converge in reasonable time
            assert result['calculation_time'] < 10.0, f"Convergence too slow: {result['calculation_time']:.2f}s"

            # Should actually converge
            if max_iter >= 50:  # Only check convergence for reasonable iteration limits
                assert result['converged'], f"Failed to converge with tolerance {tolerance}"


class TestDatabaseIndexing:
    """Test database indexing and optimization performance."""

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_url_lookup_performance(self, benchmark):
        """Test URL-based lookup performance (primary key access)."""

        def setup_and_test_lookups(num_pages: int):
            with KuzuManager() as manager:
                manager.initialize_schema()

                # Create test data
                pages = PerformanceDataGenerator.generate_pages(num_pages, domain_count=5)
                manager.add_pages_batch(pages)

                # Test individual URL lookups
                test_urls = [page['url'] for page in pages[:100]]  # Test first 100

                lookup_times = []
                for url in test_urls:
                    start_time = time.time()
                    incoming_links = manager.get_incoming_links(url)
                    lookup_time = time.time() - start_time
                    lookup_times.append(lookup_time)

                avg_lookup_time = sum(lookup_times) / len(lookup_times) if lookup_times else 0
                max_lookup_time = max(lookup_times) if lookup_times else 0

                return {
                    'avg_lookup_time': avg_lookup_time,
                    'max_lookup_time': max_lookup_time,
                    'lookups_per_second': 1.0 / avg_lookup_time if avg_lookup_time > 0 else 0
                }

        # Test with different dataset sizes
        for size in [500, 1000, 2000]:
            result = benchmark.pedantic(
                setup_and_test_lookups,
                args=(size,),
                rounds=2,
                iterations=1
            )

            # Lookups should be fast regardless of dataset size
            assert result['avg_lookup_time'] < 0.1, f"URL lookup too slow: {result['avg_lookup_time']:.4f}s average"
            assert result['max_lookup_time'] < 0.5, f"Worst URL lookup too slow: {result['max_lookup_time']:.4f}s"
            assert result['lookups_per_second'] > 20, f"Lookup throughput too low: {result['lookups_per_second']:.1f}/s"