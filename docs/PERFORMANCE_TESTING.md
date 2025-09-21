# Performance Testing Guide for MCP SEO

This guide covers the comprehensive performance testing suite for the MCP SEO system, including setup, execution, and interpretation of results.

## Overview

The performance testing suite covers:

- **Large Dataset Processing**: How the system handles thousands of pages and links
- **Memory Usage Monitoring**: Memory consumption patterns and leak detection
- **API Rate Limit Handling**: Behavior under API constraints and concurrent requests
- **Database Performance**: Kuzu graph database query and operation performance
- **Graph Analysis Scaling**: PageRank and other algorithms with increasing data sizes
- **Concurrent Processing**: System behavior under concurrent load

## Quick Start

### Prerequisites

Install the performance testing dependencies:

```bash
# Install test dependencies with performance tools
pip install -e ".[test]"

# Or install specific performance packages
pip install pytest-benchmark psutil aioresponses
```

### Running Tests

Use the performance test runner for different scenarios:

```bash
# Quick performance tests (for development)
python scripts/run_performance_tests.py --quick

# Full performance benchmarks (for release testing)
python scripts/run_performance_tests.py --full

# Memory usage tests
python scripts/run_performance_tests.py --memory

# Load and stress tests
python scripts/run_performance_tests.py --load

# All performance tests
python scripts/run_performance_tests.py --all
```

### Manual Test Execution

Run specific test categories manually:

```bash
# Run all performance tests
pytest tests/test_performance.py -v -m "performance"

# Run only benchmark tests
pytest tests/test_performance.py -v -m "benchmark" --benchmark-only

# Run memory tests with detailed output
pytest tests/test_performance.py -v -m "memory" -s

# Run database-specific performance tests
pytest tests/test_database_performance.py -v

# Run slow/load tests
pytest tests/test_performance.py -v -m "slow"
```

## Test Categories

### 1. Large Dataset Processing Tests

**Location**: `tests/test_performance.py::TestLargeDatasetProcessing`

Tests system performance with large amounts of data:

- **Kuzu Batch Insert Performance**: Measures database insertion speed
- **PageRank Calculation Scaling**: Tests algorithm performance with increasing data
- **Graph Query Performance**: Complex database query execution times

**Key Metrics**:
- Pages/second insertion rate
- Links/second insertion rate
- PageRank calculation time
- Query response times

### 2. Memory Usage Tests

**Location**: `tests/test_performance.py::TestMemoryUsage`

Monitors memory consumption patterns:

- **Memory Scaling**: How memory usage grows with data size
- **Memory Leak Detection**: Ensures proper cleanup after operations
- **Content Analyzer Efficiency**: Memory usage in content processing

**Key Metrics**:
- Memory growth per operation
- Memory release ratio
- Peak memory usage
- Memory per page/link

### 3. API Rate Limit Tests

**Location**: `tests/test_performance.py::TestAPIRateLimitHandling`

Tests API client behavior under constraints:

- **Rate Limit Handling**: Response to API rate limiting
- **Concurrent Request Performance**: Multiple simultaneous API calls
- **Connection Pooling**: Efficiency of connection reuse

**Key Metrics**:
- Request success rate
- Response time under load
- Connection reuse efficiency

### 4. Database Performance Tests

**Location**: `tests/test_database_performance.py`

Comprehensive database performance analysis:

- **Query Complexity Scaling**: Performance with complex graph queries
- **Concurrent Database Access**: Multiple simultaneous database operations
- **Index Performance**: URL lookup and search performance
- **Memory Efficiency**: Database memory usage patterns

**Key Metrics**:
- Query execution time
- Concurrent operation success rate
- Index lookup speed
- Database memory footprint

### 5. Concurrent Processing Tests

**Location**: `tests/test_performance.py::TestConcurrentProcessing`

Tests system behavior under concurrent load:

- **Concurrent Graph Operations**: Multiple graph builders running simultaneously
- **Async Content Processing**: Asynchronous content analysis performance
- **Thread Safety**: System stability under concurrent access

**Key Metrics**:
- Concurrent operation success rate
- Thread safety violations
- Performance degradation under load

### 6. Graph Analysis Scaling Tests

**Location**: `tests/test_performance.py::TestGraphAnalysisScaling`

Tests algorithm performance with different graph sizes:

- **PageRank Algorithm Scaling**: Performance vs. graph size
- **Graph Statistics Performance**: Metrics calculation speed
- **Recommendation Engine Scaling**: Recommendation generation performance

**Key Metrics**:
- Algorithm time complexity
- Memory usage scaling
- Recommendation quality vs. speed

## Configuration

### Environment Variables

Configure performance tests using environment variables:

```bash
# Dataset sizes
export PERF_LARGE_DATASET_SIZE=1000
export PERF_XLARGE_DATASET_SIZE=5000

# Concurrency levels
export PERF_HIGH_CONCURRENCY=10

# API rate limits
export PERF_API_REQUESTS_PER_MINUTE=100

# Benchmark settings
export PERF_BENCHMARK_ROUNDS=3
export PERF_LONG_TEST_TIMEOUT=300
```

### Performance Configuration

Edit `tests/performance_config.py` to adjust:

- Dataset sizes for different test categories
- Performance thresholds and expectations
- Concurrency levels for load testing
- Memory usage limits
- Benchmark parameters

### CI/CD Configuration

The system automatically adjusts for CI environments:

- Reduced dataset sizes
- Lower concurrency limits
- Shorter timeouts
- Fewer benchmark rounds

## Interpreting Results

### Benchmark Output

pytest-benchmark provides detailed performance metrics:

```
Name (time in ms)                     Min      Max     Mean   StdDev  Median     IQR  Outliers  Ops
test_kuzu_batch_insert               45.2     52.3     48.1     2.1    47.8     1.9      2;0   20.8
test_pagerank_calculation           125.4    145.2    135.2     7.8   134.1     9.2      1;1    7.4
test_graph_query_performance         12.3     18.7     14.2     2.3    13.8     2.1      3;0   70.4
```

**Key Columns**:
- **Mean**: Average execution time (most important for typical performance)
- **Median**: Middle value (good for understanding typical performance)
- **StdDev**: Standard deviation (consistency indicator)
- **Ops**: Operations per second (throughput)
- **Outliers**: Number of outlier measurements

### Memory Test Output

Memory tests show detailed usage patterns:

```
Memory usage for Kuzu-1000pages:
  Start: 45.23 MB
  Peak: 78.45 MB
  End: 47.12 MB
  Growth: 1.89 MB
```

**What to Look For**:
- **Growth**: Should be reasonable for data size
- **Peak**: Maximum memory usage during operation
- **Cleanup**: End memory should be close to start memory

### Performance Thresholds

The tests include automatic performance assertions:

```python
# Example thresholds
assert page_time < 30.0, f"Page insertion took too long: {page_time:.2f}s"
assert memory_growth < 50, f"Potential memory leak: {memory_growth:.2f}MB growth"
assert concurrent_success_rate > 0.95, f"Too many concurrent failures: {concurrent_success_rate:.2f}"
```

## Performance Optimization Tips

### Database Performance

1. **Batch Operations**: Use batch inserts instead of individual operations
2. **Connection Reuse**: Use context managers for proper connection handling
3. **Query Optimization**: Use specific queries instead of broad scans
4. **Index Usage**: Ensure URL-based lookups use primary key indexes

### Memory Optimization

1. **Cleanup**: Explicitly close database connections and clear large objects
2. **Garbage Collection**: Force GC after large operations
3. **Streaming**: Process large datasets in chunks
4. **Connection Pooling**: Reuse database connections

### Concurrent Processing

1. **Semaphores**: Limit concurrent operations to prevent resource exhaustion
2. **Thread Safety**: Ensure thread-safe database access
3. **Error Handling**: Implement proper error handling for concurrent failures
4. **Resource Limits**: Set appropriate limits for concurrent operations

## Troubleshooting

### Common Issues

1. **Tests Timing Out**:
   - Reduce dataset sizes in configuration
   - Increase timeout values
   - Check system resources

2. **Memory Test Failures**:
   - Ensure proper cleanup in test fixtures
   - Check for memory leaks in application code
   - Verify garbage collection is working

3. **Benchmark Inconsistency**:
   - Run on dedicated test machine
   - Ensure consistent system load
   - Increase benchmark rounds for stability

4. **Database Connection Errors**:
   - Check file permissions for temporary databases
   - Verify proper connection cleanup
   - Ensure no concurrent access conflicts

### Performance Regression

If performance regresses:

1. **Compare Benchmarks**: Use `pytest --benchmark-compare` to compare results
2. **Profile Code**: Use Python profilers to identify bottlenecks
3. **Check Dependencies**: Ensure no dependency version conflicts
4. **Review Changes**: Examine recent code changes for performance impacts

## Continuous Integration

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Run quick performance tests
      run: |
        python scripts/run_performance_tests.py --quick

    - name: Run memory tests
      run: |
        python scripts/run_performance_tests.py --memory

    - name: Archive benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: .benchmarks/
```

### Performance Monitoring

Set up regular performance monitoring:

1. **Nightly Builds**: Run full performance tests on nightly builds
2. **Benchmark History**: Track performance trends over time
3. **Alert Thresholds**: Set up alerts for significant performance regressions
4. **Resource Monitoring**: Monitor CI system resources during tests

## Advanced Usage

### Custom Performance Tests

Create custom performance tests for specific scenarios:

```python
@pytest.mark.benchmark
def test_custom_scenario(benchmark):
    def my_operation():
        # Your custom performance test
        return result

    result = benchmark(my_operation)
    assert result meets_expectations()
```

### Profiling Integration

Integrate with Python profilers:

```python
import cProfile
import pstats

def test_with_profiling():
    pr = cProfile.Profile()
    pr.enable()

    # Your operation here

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

### Load Testing

For more comprehensive load testing, consider:

1. **Locust**: For web application load testing
2. **Artillery**: For API load testing
3. **Custom Scripts**: For specific MCP protocol testing

This comprehensive performance testing suite ensures the MCP SEO system maintains optimal performance characteristics across different usage scenarios and data sizes.