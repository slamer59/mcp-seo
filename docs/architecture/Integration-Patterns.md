# Integration Patterns for MCP SEO System

## Overview

The MCP SEO system supports three primary integration patterns:
1. **MCP Server Integration** - For Claude and other MCP clients
2. **CLI Integration** - For command-line usage and automation
3. **Direct API Integration** - For programmatic access in Python applications

## MCP Server Integration Pattern

### **FastMCP Server Architecture**

```python
# interfaces/mcp/server.py
from fastmcp import FastMCP
from mcp_seo.infrastructure.config import ProductionAnalyzerFactory

mcp = FastMCP("MCP SEO Analysis Server")

# Global factory for dependency injection
_factory: Optional[AnalyzerFactoryProtocol] = None

def get_factory() -> AnalyzerFactoryProtocol:
    global _factory
    if _factory is None:
        config = get_settings()
        _factory = ProductionAnalyzerFactory(config)
    return _factory

@mcp.tool()
async def onpage_analysis_start(params: OnPageAnalysisParams) -> Dict[str, Any]:
    """Start comprehensive OnPage SEO analysis for a website."""
    factory = get_factory()
    analyzer = factory.create_onpage_analyzer()

    request = OnPageAnalysisRequest(
        target=params.target,
        max_crawl_pages=params.max_crawl_pages,
        respect_sitemap=params.respect_sitemap
    )

    return await analyzer.analyze_async(request)
```

### **MCP Tool Categories**

#### **Analysis Tools**
```python
# interfaces/mcp/tools/analysis_tools.py

@mcp.tool()
async def onpage_analysis_start(params: OnPageAnalysisParams) -> Dict[str, Any]:
    """Start comprehensive OnPage SEO analysis."""

@mcp.tool()
async def keyword_analysis(params: KeywordAnalysisParams) -> Dict[str, Any]:
    """Analyze keyword search volume and competition."""

@mcp.tool()
async def content_gap_analysis(params: ContentGapParams) -> Dict[str, Any]:
    """Identify content gaps vs competitors."""

@mcp.tool()
async def comprehensive_seo_audit(params: SEOAuditParams) -> Dict[str, Any]:
    """Run complete SEO audit combining multiple analyses."""
```

#### **Graph Analysis Tools**
```python
# interfaces/mcp/tools/graph_tools.py

@mcp.tool()
async def analyze_pagerank(domain: str, max_pages: int = 100) -> Dict[str, Any]:
    """Analyze PageRank for internal link structure."""

@mcp.tool()
async def find_pillar_pages(domain: str, limit: int = 10) -> Dict[str, Any]:
    """Identify pillar pages with high authority."""

@mcp.tool()
async def optimize_internal_links(domain: str) -> Dict[str, Any]:
    """Generate internal link optimization recommendations."""

@mcp.tool()
async def analyze_centrality(request: CentralityAnalysisRequest) -> Dict[str, Any]:
    """Comprehensive centrality analysis for authority identification."""
```

#### **Content Tools**
```python
# interfaces/mcp/tools/content_tools.py

@mcp.tool()
async def analyze_blog_content(site_path: str) -> Dict[str, Any]:
    """Analyze blog content for SEO optimization opportunities."""

@mcp.tool()
async def optimize_content_quality(content: str) -> Dict[str, Any]:
    """Analyze and optimize content quality metrics."""

@mcp.tool()
async def generate_content_strategy(competitor_analysis: Dict) -> Dict[str, Any]:
    """Generate content strategy based on competitive analysis."""
```

### **Error Handling and Resilience**

```python
# interfaces/mcp/error_handling.py

class MCPErrorHandler:
    """Centralized error handling for MCP tools."""

    @staticmethod
    def handle_api_exception(func):
        """Decorator for handling API exceptions in MCP tools."""
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ApiException as e:
                return {
                    "status": "error",
                    "error_type": "api_error",
                    "message": str(e),
                    "details": getattr(e, 'details', {})
                }
            except ValidationError as e:
                return {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": "Invalid request parameters",
                    "details": e.errors()
                }
            except Exception as e:
                logger.exception("Unexpected error in MCP tool")
                return {
                    "status": "error",
                    "error_type": "internal_error",
                    "message": "An unexpected error occurred"
                }
        return wrapper

@MCPErrorHandler.handle_api_exception
@mcp.tool()
async def resilient_analysis_tool(params: AnalysisParams) -> Dict[str, Any]:
    """Example of resilient MCP tool with error handling."""
    # Tool implementation
```

## CLI Integration Pattern

### **Command Structure**

```python
# interfaces/cli/main.py
import click
from mcp_seo.infrastructure.config import ProductionAnalyzerFactory

@click.group()
@click.pass_context
def cli(ctx):
    """MCP SEO Analysis CLI."""
    ctx.ensure_object(dict)
    config = get_settings()
    ctx.obj['factory'] = ProductionAnalyzerFactory(config)

@cli.group()
def analyze():
    """SEO analysis commands."""
    pass

@analyze.command()
@click.option('--target', required=True, help='Target website URL')
@click.option('--max-pages', default=100, help='Maximum pages to crawl')
@click.option('--output', type=click.Choice(['json', 'table', 'html']), default='table')
@click.pass_context
def onpage(ctx, target: str, max_pages: int, output: str):
    """Run OnPage SEO analysis."""
    factory = ctx.obj['factory']
    analyzer = factory.create_onpage_analyzer()

    request = OnPageAnalysisRequest(target=target, max_crawl_pages=max_pages)
    result = analyzer.analyze(request)

    formatter = factory.create_output_formatter(output)
    formatter.display(result)
```

### **Progress Tracking and Streaming**

```python
# interfaces/cli/progress.py
import click
from rich.progress import Progress, TaskID

class CLIProgressReporter:
    """Rich progress reporting for CLI operations."""

    def __init__(self):
        self.progress = Progress()
        self.tasks: Dict[str, TaskID] = {}

    def start_task(self, name: str, total: Optional[int] = None) -> str:
        """Start a new progress task."""
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        return name

    def update_task(self, name: str, advance: int = 1, description: str = None):
        """Update task progress."""
        if name in self.tasks:
            self.progress.update(
                self.tasks[name],
                advance=advance,
                description=description
            )

    def complete_task(self, name: str):
        """Mark task as completed."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], completed=True)

# Usage in CLI commands
@analyze.command()
@click.option('--target', required=True)
@click.pass_context
def comprehensive_audit(ctx, target: str):
    """Run comprehensive SEO audit with progress tracking."""
    factory = ctx.obj['factory']
    progress_reporter = CLIProgressReporter()

    with progress_reporter.progress:
        # Create workflow with progress reporter
        workflow = factory.create_comprehensive_audit_workflow(
            progress_reporter=progress_reporter
        )

        result = workflow.execute(target)

        # Display results
        formatter = factory.create_output_formatter('table')
        formatter.display(result)
```

## Direct API Integration Pattern

### **Python Client Library**

```python
# interfaces/api/client.py

class SEOAnalysisClient:
    """Python client for direct SEO analysis integration."""

    def __init__(self, config: Optional[SEOConfig] = None):
        """Initialize client with optional configuration."""
        self._config = config or get_settings()
        self._factory = ProductionAnalyzerFactory(self._config)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        # Close database connections, etc.
        pass

    # High-level analysis methods
    def analyze_onpage(self, target: str, **kwargs) -> OnPageAnalysisResult:
        """Run OnPage SEO analysis."""
        analyzer = self._factory.create_onpage_analyzer()
        request = OnPageAnalysisRequest(target=target, **kwargs)
        return analyzer.analyze(request)

    def analyze_keywords(self, keywords: List[str], **kwargs) -> KeywordAnalysisResult:
        """Analyze keyword metrics."""
        analyzer = self._factory.create_keyword_analyzer()
        request = KeywordAnalysisRequest(keywords=keywords, **kwargs)
        return analyzer.analyze(request)

    def analyze_content(self, content_path: str, **kwargs) -> ContentAnalysisResult:
        """Analyze blog/content directory."""
        analyzer = self._factory.create_content_analyzer()
        return analyzer.analyze_directory(content_path, **kwargs)

    # Workflow methods
    def comprehensive_audit(self, target: str, **kwargs) -> ComprehensiveAuditResult:
        """Run complete SEO audit."""
        workflow = self._factory.create_comprehensive_audit_workflow()
        return workflow.execute(target, **kwargs)

    # Graph analysis methods
    def analyze_pagerank(self, domain: str, **kwargs) -> PageRankResult:
        """Analyze PageRank for internal links."""
        analyzer = self._factory.create_pagerank_analyzer()
        return analyzer.analyze(domain, **kwargs)
```

### **Async Client Support**

```python
# interfaces/api/async_client.py

class AsyncSEOAnalysisClient:
    """Async Python client for concurrent SEO analysis."""

    def __init__(self, config: Optional[SEOConfig] = None):
        self._config = config or get_settings()
        self._factory = ProductionAnalyzerFactory(self._config)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Clean up async resources."""
        pass

    async def analyze_onpage(self, target: str, **kwargs) -> OnPageAnalysisResult:
        """Async OnPage analysis."""
        analyzer = self._factory.create_onpage_analyzer()
        request = OnPageAnalysisRequest(target=target, **kwargs)
        return await analyzer.analyze_async(request)

    async def analyze_multiple_sites(self, targets: List[str], **kwargs) -> List[OnPageAnalysisResult]:
        """Analyze multiple sites concurrently."""
        tasks = [
            self.analyze_onpage(target, **kwargs)
            for target in targets
        ]
        return await asyncio.gather(*tasks)

    async def comprehensive_audit_stream(self, target: str, **kwargs) -> AsyncIterator[AuditProgressUpdate]:
        """Stream comprehensive audit progress."""
        workflow = self._factory.create_comprehensive_audit_workflow()
        async for update in workflow.execute_stream(target, **kwargs):
            yield update
```

### **Integration Examples**

#### **Basic Usage**
```python
# Basic synchronous usage
from mcp_seo import SEOAnalysisClient

with SEOAnalysisClient() as client:
    # OnPage analysis
    onpage_result = client.analyze_onpage("https://example.com")
    print(f"SEO Score: {onpage_result.seo_score}")

    # Keyword analysis
    keyword_result = client.analyze_keywords(["seo tools", "keyword research"])
    for keyword in keyword_result.keywords:
        print(f"{keyword.term}: {keyword.search_volume}")

    # Comprehensive audit
    audit_result = client.comprehensive_audit("https://example.com")
    for recommendation in audit_result.recommendations:
        print(f"[{recommendation.priority}] {recommendation.title}")
```

#### **Async Usage**
```python
# Async usage for better performance
import asyncio
from mcp_seo import AsyncSEOAnalysisClient

async def analyze_competitor_sites():
    async with AsyncSEOAnalysisClient() as client:
        competitors = [
            "https://competitor1.com",
            "https://competitor2.com",
            "https://competitor3.com"
        ]

        # Analyze all competitors concurrently
        results = await client.analyze_multiple_sites(competitors)

        # Compare results
        for site, result in zip(competitors, results):
            print(f"{site}: Score {result.seo_score}")

asyncio.run(analyze_competitor_sites())
```

#### **Streaming Progress**
```python
# Stream comprehensive audit progress
async def audit_with_progress():
    async with AsyncSEOAnalysisClient() as client:
        async for update in client.comprehensive_audit_stream("https://example.com"):
            print(f"[{update.stage}] {update.message} ({update.progress}%)")

asyncio.run(audit_with_progress())
```

## Configuration and Environment Management

### **Environment-Specific Configurations**

```python
# infrastructure/config/environments.py

class DevelopmentConfig(SEOConfig):
    """Development environment configuration."""
    debug = True
    log_level = "DEBUG"
    use_cache = True
    max_crawl_pages = 10  # Limit for development
    mock_external_apis = True

class ProductionConfig(SEOConfig):
    """Production environment configuration."""
    debug = False
    log_level = "INFO"
    use_cache = True
    max_crawl_pages = 1000
    mock_external_apis = False

    # Production-specific settings
    rate_limiting = True
    connection_pooling = True
    metrics_collection = True

class TestConfig(SEOConfig):
    """Test environment configuration."""
    debug = True
    log_level = "DEBUG"
    use_cache = False
    max_crawl_pages = 5
    mock_external_apis = True

    # Test-specific settings
    use_memory_storage = True
    reset_state_between_tests = True

def get_config_for_environment(env: str) -> SEOConfig:
    """Get configuration for specific environment."""
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig
    }

    config_class = configs.get(env, DevelopmentConfig)
    return config_class()
```

### **Factory Selection Pattern**

```python
# infrastructure/config/factory_selector.py

def create_factory_for_environment(env: str) -> AnalyzerFactoryProtocol:
    """Create appropriate factory for environment."""
    config = get_config_for_environment(env)

    if env == "test":
        return TestAnalyzerFactory(config)
    elif env == "development":
        return DevelopmentAnalyzerFactory(config)
    else:
        return ProductionAnalyzerFactory(config)

# Usage in different contexts
def setup_mcp_server():
    """Setup MCP server with environment-specific factory."""
    env = os.getenv("SEO_ENV", "production")
    factory = create_factory_for_environment(env)
    return create_mcp_server(factory)

def setup_cli():
    """Setup CLI with environment-specific factory."""
    env = os.getenv("SEO_ENV", "development")
    factory = create_factory_for_environment(env)
    return create_cli_app(factory)
```

This integration pattern ensures:
- **Consistent behavior** across all integration methods
- **Flexible configuration** for different environments
- **Proper resource management** with cleanup
- **Error handling** appropriate for each integration type
- **Performance optimization** with async support where beneficial