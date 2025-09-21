# Performance Considerations and Optimization Strategy

## Performance Requirements

### **Target Performance Metrics**

| Operation | Target Time | Memory Limit | Concurrent Users |
|-----------|-------------|--------------|------------------|
| OnPage Analysis (100 pages) | <60 seconds | <500MB | 5 concurrent |
| Keyword Analysis (50 keywords) | <30 seconds | <200MB | 10 concurrent |
| PageRank Calculation (1000 pages) | <45 seconds | <1GB | 2 concurrent |
| Content Analysis (500 posts) | <90 seconds | <750MB | 3 concurrent |
| Comprehensive Audit | <180 seconds | <1.5GB | 1 concurrent |

### **Scalability Targets**

- **Site Size**: Support analysis of up to 10,000 pages
- **Keyword Volume**: Handle up to 1,000 keywords per analysis
- **Concurrent Analysis**: Support 5+ simultaneous analyses
- **Data Retention**: 30-day analysis history with <5s retrieval

## Bottleneck Analysis and Mitigation

### **1. External API Rate Limiting**

#### **Problem**
DataForSEO API has rate limits that can cause delays:
- 2000 API calls per minute
- Task completion can take 30-60 seconds
- Bulk operations may hit rate limits

#### **Solution: Request Batching and Caching**

```python
# infrastructure/adapters/dataforseo_adapter.py

class OptimizedDataForSEOAdapter:
    """Optimized DataForSEO adapter with caching and batching."""

    def __init__(self, client: DataForSEOClient, cache_manager: CacheManager):
        self._client = client
        self._cache = cache_manager
        self._request_queue = asyncio.Queue()
        self._rate_limiter = RateLimiter(requests_per_minute=1800)  # Safety margin

    async def batch_keyword_analysis(self, keywords: List[str]) -> Dict[str, KeywordData]:
        """Batch keyword requests for efficiency."""
        # Check cache first
        cached_results = {}
        uncached_keywords = []

        for keyword in keywords:
            cache_key = f"keyword:{keyword}"
            cached = await self._cache.get(cache_key)
            if cached:
                cached_results[keyword] = cached
            else:
                uncached_keywords.append(keyword)

        if not uncached_keywords:
            return cached_results

        # Batch uncached requests
        batch_size = 100  # DataForSEO batch limit
        batched_results = {}

        for i in range(0, len(uncached_keywords), batch_size):
            batch = uncached_keywords[i:i + batch_size]

            async with self._rate_limiter:
                result = await self._client.get_keyword_data_batch(batch)

                # Cache results
                for keyword, data in result.items():
                    cache_key = f"keyword:{keyword}"
                    await self._cache.set(cache_key, data, ttl=3600)  # 1 hour TTL
                    batched_results[keyword] = data

        return {**cached_results, **batched_results}
```

### **2. Graph Database Performance**

#### **Problem**
Kuzu graph operations can be slow for large sites:
- PageRank iteration complexity: O(n²)
- Graph construction time increases with link density
- Memory usage grows with graph size

#### **Solution: Incremental Updates and Indexing**

```python
# graph/optimized_pagerank_analyzer.py

class OptimizedPageRankAnalyzer:
    """High-performance PageRank analyzer with incremental updates."""

    def __init__(self, kuzu_manager: KuzuManager):
        self._kuzu = kuzu_manager
        self._pagerank_cache = {}
        self._graph_hash = None

    async def calculate_pagerank_incremental(
        self,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """Calculate PageRank with incremental updates."""

        # Check if graph has changed
        current_hash = await self._calculate_graph_hash()
        if current_hash == self._graph_hash and self._pagerank_cache:
            logger.info("Using cached PageRank results")
            return self._pagerank_cache

        # Use sparse matrix representation for large graphs
        if await self._get_graph_size() > 5000:
            return await self._calculate_pagerank_sparse(
                damping_factor, max_iterations, tolerance
            )
        else:
            return await self._calculate_pagerank_dense(
                damping_factor, max_iterations, tolerance
            )

    async def _calculate_pagerank_sparse(self, damping_factor, max_iterations, tolerance):
        """Sparse matrix PageRank for large graphs."""
        import scipy.sparse as sp
        from scipy.sparse.linalg import norm

        # Build sparse adjacency matrix
        pages = await self._kuzu.get_page_list()
        n = len(pages)
        page_to_idx = {page: i for i, page in enumerate(pages)}

        # Use coordinate format for efficient construction
        row_indices = []
        col_indices = []
        data = []

        async for edge in self._kuzu.get_edges_iterator():
            if edge.source in page_to_idx and edge.target in page_to_idx:
                row_indices.append(page_to_idx[edge.target])
                col_indices.append(page_to_idx[edge.source])
                data.append(1.0)

        # Normalize by out-degree
        out_degrees = await self._kuzu.get_out_degrees()
        for i, (row, col, _) in enumerate(zip(row_indices, col_indices, data)):
            source_page = pages[col]
            out_degree = out_degrees.get(source_page, 1)
            data[i] = 1.0 / out_degree

        # Create sparse transition matrix
        transition_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)), shape=(n, n)
        )

        # Power iteration with sparse operations
        pagerank = np.ones(n) / n
        dangling_weights = np.ones(n) / n

        for iteration in range(max_iterations):
            prev_pagerank = pagerank.copy()

            # Sparse matrix-vector multiplication
            pagerank = damping_factor * transition_matrix.dot(pagerank)
            pagerank += (1 - damping_factor) * dangling_weights / n

            # Check convergence
            if norm(pagerank - prev_pagerank, 1) < tolerance:
                logger.info(f"PageRank converged after {iteration + 1} iterations")
                break

        # Convert back to dictionary
        result = {pages[i]: score for i, score in enumerate(pagerank)}

        # Cache results
        self._pagerank_cache = result
        self._graph_hash = current_hash

        return result
```

### **3. Memory Management**

#### **Problem**
Large site analysis can consume excessive memory:
- Full site data loaded into memory
- Multiple analyzer instances
- Graph structures for PageRank

#### **Solution: Streaming and Memory Pools**

```python
# application/workflows/memory_optimized_workflow.py

class MemoryOptimizedComprehensiveAudit:
    """Memory-efficient comprehensive audit workflow."""

    def __init__(self, factory: AnalyzerFactoryProtocol, memory_limit_mb: int = 1000):
        self._factory = factory
        self._memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self._memory_monitor = MemoryMonitor()

    async def execute_stream(self, target: str) -> AsyncIterator[AuditProgressUpdate]:
        """Execute audit with memory management and streaming results."""

        with self._memory_monitor.track_memory():
            # Stage 1: OnPage Analysis (streaming)
            yield AuditProgressUpdate("onpage", "Starting OnPage analysis", 10)

            async for page_result in self._stream_onpage_analysis(target):
                # Process page-by-page to control memory
                yield AuditProgressUpdate("onpage", f"Analyzed {page_result.url}", 30)

                # Check memory usage
                if self._memory_monitor.current_usage() > self._memory_limit:
                    await self._clear_intermediate_cache()

            # Stage 2: Keyword Analysis (batched)
            yield AuditProgressUpdate("keywords", "Starting keyword analysis", 50)

            keyword_results = await self._batch_keyword_analysis(target)
            yield AuditProgressUpdate("keywords", "Keyword analysis complete", 70)

            # Stage 3: Graph Analysis (chunked)
            yield AuditProgressUpdate("graph", "Building link graph", 80)

            graph_results = await self._chunked_graph_analysis(target)
            yield AuditProgressUpdate("graph", "Graph analysis complete", 90)

            # Final consolidation
            yield AuditProgressUpdate("consolidation", "Generating recommendations", 95)

            final_result = await self._consolidate_results(
                onpage_result, keyword_results, graph_results
            )

            yield AuditProgressUpdate("complete", "Audit complete", 100, final_result)

    async def _stream_onpage_analysis(self, target: str) -> AsyncIterator[PageAnalysisResult]:
        """Stream OnPage analysis results page-by-page."""
        analyzer = self._factory.create_onpage_analyzer()

        async for page_data in analyzer.analyze_pages_stream(target):
            # Yield individual page results
            yield page_data

            # Clear page data from memory after processing
            del page_data

    async def _clear_intermediate_cache(self):
        """Clear intermediate cache to free memory."""
        import gc
        gc.collect()
        logger.info("Cleared intermediate cache to free memory")
```

### **4. Concurrent Processing**

#### **Problem**
Sequential processing is slow for large analyses:
- Single-threaded API calls
- Sequential page processing
- No parallelization of independent operations

#### **Solution: Async Concurrency with Semaphores**

```python
# application/services/concurrent_analyzer.py

class ConcurrentAnalysisService:
    """High-performance concurrent analysis service."""

    def __init__(self, max_concurrent_requests: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._session_pool = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=50,
                limit_per_host=10,
                keepalive_timeout=30
            )
        )

    async def analyze_pages_concurrent(
        self,
        urls: List[str],
        analyzer: OnPageAnalyzer
    ) -> List[PageAnalysisResult]:
        """Analyze multiple pages concurrently."""

        async def analyze_single_page(url: str) -> PageAnalysisResult:
            async with self._semaphore:  # Limit concurrency
                try:
                    return await analyzer.analyze_page_async(url)
                except Exception as e:
                    logger.error(f"Failed to analyze {url}: {e}")
                    return PageAnalysisResult.error(url, str(e))

        # Process pages in batches to control memory
        batch_size = 50
        all_results = []

        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]

            # Process batch concurrently
            tasks = [analyze_single_page(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis error: {result}")
                else:
                    all_results.append(result)

            # Progress reporting
            progress = min(100, (i + batch_size) * 100 // len(urls))
            logger.info(f"Processed {i + batch_size}/{len(urls)} pages ({progress}%)")

        return all_results

    async def analyze_keywords_concurrent(
        self,
        keywords: List[str],
        analyzer: KeywordAnalyzer
    ) -> List[KeywordAnalysisResult]:
        """Analyze keywords with optimal batching."""

        # Determine optimal batch size based on API limits
        optimal_batch_size = min(100, len(keywords) // 4 + 1)

        async def analyze_keyword_batch(keyword_batch: List[str]) -> List[KeywordAnalysisResult]:
            async with self._semaphore:
                return await analyzer.analyze_keywords_batch_async(keyword_batch)

        # Create batches
        batches = [
            keywords[i:i + optimal_batch_size]
            for i in range(0, len(keywords), optimal_batch_size)
        ]

        # Process batches concurrently
        batch_tasks = [analyze_keyword_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)

        return all_results
```

## Caching Strategy

### **Multi-Level Caching Architecture**

```python
# infrastructure/caching/cache_manager.py

class MultiLevelCacheManager:
    """Multi-level caching for optimal performance."""

    def __init__(self):
        self._l1_cache = {}  # In-memory cache (LRU)
        self._l2_cache = SQLiteCache("cache.db")  # Persistent cache
        self._l3_cache = FileCache("cache_dir")  # Large object cache

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2 -> L3)."""
        # Try L1 cache first (fastest)
        if key in self._l1_cache:
            return self._l1_cache[key]

        # Try L2 cache (database)
        value = await self._l2_cache.get(key)
        if value is not None:
            # Promote to L1 cache
            self._l1_cache[key] = value
            return value

        # Try L3 cache (file system)
        value = await self._l3_cache.get(key)
        if value is not None:
            # Promote to L1 and L2 caches
            self._l1_cache[key] = value
            await self._l2_cache.set(key, value)
            return value

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in appropriate cache level based on size."""
        value_size = self._estimate_size(value)

        if value_size < 1024:  # Small objects (< 1KB) -> L1
            self._l1_cache[key] = value
            await self._l2_cache.set(key, value, ttl)
        elif value_size < 1024 * 1024:  # Medium objects (< 1MB) -> L2
            await self._l2_cache.set(key, value, ttl)
        else:  # Large objects (> 1MB) -> L3
            await self._l3_cache.set(key, value, ttl)

### **Cache Key Strategy**

```python
# infrastructure/caching/cache_keys.py

class CacheKeyGenerator:
    """Generate consistent cache keys for different data types."""

    @staticmethod
    def onpage_analysis_key(target: str, config: OnPageAnalysisRequest) -> str:
        """Generate cache key for OnPage analysis."""
        config_hash = hashlib.md5(
            json.dumps(config.dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"onpage:{target}:{config_hash}"

    @staticmethod
    def keyword_analysis_key(keywords: List[str], location: str, language: str) -> str:
        """Generate cache key for keyword analysis."""
        keywords_hash = hashlib.md5(
            "|".join(sorted(keywords)).encode()
        ).hexdigest()[:8]
        return f"keywords:{keywords_hash}:{location}:{language}"

    @staticmethod
    def pagerank_key(domain: str, damping_factor: float, max_iterations: int) -> str:
        """Generate cache key for PageRank analysis."""
        return f"pagerank:{domain}:{damping_factor}:{max_iterations}"
```

## Database Optimization

### **Kuzu Graph Database Tuning**

```python
# graph/optimized_kuzu_manager.py

class OptimizedKuzuManager:
    """Optimized Kuzu manager with performance tuning."""

    def __init__(self, db_path: str):
        # Kuzu performance settings
        self._db = kuzu.Database(
            db_path,
            buffer_pool_size=1024 * 1024 * 1024,  # 1GB buffer pool
            enable_semi_mask=True,
            enable_z_order=True
        )
        self._connection = kuzu.Connection(self._db)

        # Create optimized schema with indexes
        self._setup_optimized_schema()

    def _setup_optimized_schema(self):
        """Create schema with performance optimizations."""
        # Create node table with indexes
        self._connection.execute("""
            CREATE NODE TABLE IF NOT EXISTS Page(
                url STRING PRIMARY KEY,
                title STRING,
                content_length INT64,
                pagerank_score DOUBLE DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create edge table with composite index
        self._connection.execute("""
            CREATE REL TABLE IF NOT EXISTS Links(
                FROM Page TO Page,
                anchor_text STRING,
                link_type STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for performance
        self._connection.execute("""
            CREATE INDEX ON Page(pagerank_score)
        """)

        self._connection.execute("""
            CREATE INDEX ON Links(link_type)
        """)

    async def bulk_insert_pages(self, pages: List[Dict]) -> None:
        """Optimized bulk page insertion."""
        if not pages:
            return

        # Use batch insertion for better performance
        batch_size = 1000
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]

            # Prepare batch insert statement
            values = []
            for page in batch:
                values.append(f"('{page['url']}', '{page['title']}', {page['content_length']})")

            query = f"""
                INSERT INTO Page (url, title, content_length) VALUES
                {', '.join(values)}
            """

            self._connection.execute(query)

    async def get_pagerank_scores_optimized(self) -> Dict[str, float]:
        """Optimized PageRank score retrieval."""
        result = self._connection.execute("""
            MATCH (p:Page)
            WHERE p.pagerank_score > 0
            RETURN p.url, p.pagerank_score
            ORDER BY p.pagerank_score DESC
        """)

        return {row[0]: row[1] for row in result}
```

## Monitoring and Metrics

### **Performance Monitoring**

```python
# shared/metrics.py

class PerformanceMonitor:
    """Real-time performance monitoring for SEO analysis."""

    def __init__(self):
        self._metrics = {}
        self._start_times = {}

    def start_operation(self, operation: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self._start_times[operation_id] = time.time()
        return operation_id

    def end_operation(self, operation_id: str, metadata: Dict = None):
        """End timing an operation and record metrics."""
        if operation_id not in self._start_times:
            return

        duration = time.time() - self._start_times[operation_id]
        del self._start_times[operation_id]

        operation = operation_id.split('_')[0]
        if operation not in self._metrics:
            self._metrics[operation] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }

        metrics = self._metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)

        if metadata:
            metrics.setdefault('metadata', []).append(metadata)

        logger.info(f"Operation {operation} completed in {duration:.2f}s")

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        return {
            'operations': self._metrics,
            'timestamp': time.time(),
            'memory_usage': self._get_memory_usage(),
            'active_operations': len(self._start_times)
        }

    def _get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        return {
            'rss': process.memory_info().rss,
            'vms': process.memory_info().vms,
            'percent': process.memory_percent()
        }

# Usage in analyzers
class InstrumentedOnPageAnalyzer:
    def __init__(self, client: DataForSEOClient):
        self._client = client
        self._monitor = PerformanceMonitor()

    async def analyze(self, request: OnPageAnalysisRequest) -> OnPageAnalysisResult:
        op_id = self._monitor.start_operation("onpage_analysis")

        try:
            result = await self._perform_analysis(request)
            self._monitor.end_operation(op_id, {
                'target': request.target,
                'pages_analyzed': len(result.pages),
                'issues_found': len(result.issues)
            })
            return result
        except Exception as e:
            self._monitor.end_operation(op_id, {'error': str(e)})
            raise
```

This performance strategy ensures:
- **Optimal resource utilization** with controlled memory usage
- **Fast response times** through caching and concurrency
- **Scalable architecture** that can handle large sites
- **Real-time monitoring** for performance optimization
- **Graceful degradation** under high load conditions