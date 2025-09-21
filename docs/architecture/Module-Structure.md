# MCP SEO Production Module Structure

## Directory Structure

```
src/mcp_seo/
├── domain/                     # Core business logic (no external dependencies)
│   ├── entities/              # Core SEO domain objects
│   │   ├── __init__.py
│   │   ├── page.py           # Page entity with SEO properties
│   │   ├── keyword.py        # Keyword entity with metrics
│   │   ├── link.py           # Link entity for graph analysis
│   │   └── site.py           # Site aggregate root
│   ├── services/             # Domain services for complex operations
│   │   ├── __init__.py
│   │   ├── seo_scoring.py    # SEO health scoring algorithms
│   │   ├── link_analysis.py  # PageRank and graph algorithms
│   │   └── content_quality.py # Content quality assessment
│   ├── repositories/         # Abstract data access interfaces
│   │   ├── __init__.py
│   │   ├── page_repository.py
│   │   ├── keyword_repository.py
│   │   └── graph_repository.py
│   └── value_objects/        # Immutable value objects
│       ├── __init__.py
│       ├── seo_metrics.py    # SEO score and metric values
│       ├── analysis_config.py # Configuration value objects
│       └── url_info.py       # URL parsing and validation
│
├── application/              # Use cases and workflows (orchestration)
│   ├── analyzers/           # High-level analysis orchestrators
│   │   ├── __init__.py
│   │   ├── onpage_analyzer.py # OnPage analysis use case
│   │   ├── keyword_analyzer.py # Keyword analysis use case
│   │   ├── content_analyzer.py # Content analysis use case
│   │   └── competitive_analyzer.py # Competitive analysis use case
│   ├── workflows/           # Multi-step analysis processes
│   │   ├── __init__.py
│   │   ├── comprehensive_audit.py # Full SEO audit workflow
│   │   ├── content_optimization.py # Content optimization workflow
│   │   └── link_building.py # Link building workflow
│   ├── handlers/            # Command/query handlers
│   │   ├── __init__.py
│   │   ├── analysis_commands.py # Analysis command handlers
│   │   └── reporting_queries.py # Reporting query handlers
│   └── services/            # Application services
│       ├── __init__.py
│       ├── recommendation_engine.py # SEO recommendation orchestration
│       └── report_generator.py # Report generation coordination
│
├── infrastructure/          # External integrations and persistence
│   ├── adapters/           # External service adapters
│   │   ├── __init__.py
│   │   ├── dataforseo_adapter.py # DataForSEO API integration
│   │   ├── web_crawler_adapter.py # Web crawling functionality
│   │   └── nlp_adapter.py  # NLP service integration
│   ├── persistence/        # Data persistence implementations
│   │   ├── __init__.py
│   │   ├── kuzu_graph_store.py # Kuzu graph database
│   │   ├── sqlite_store.py # SQLite for caching
│   │   └── memory_store.py # In-memory for testing
│   ├── reporting/          # Output formatters and visualizers
│   │   ├── __init__.py
│   │   ├── console_reporter.py # Rich console output
│   │   ├── json_reporter.py # JSON format output
│   │   └── html_reporter.py # HTML report generation
│   └── config/             # Configuration management
│       ├── __init__.py
│       ├── settings.py     # Application settings
│       └── factories.py    # Dependency injection factories
│
├── interfaces/             # Entry points and external APIs
│   ├── mcp/               # MCP server implementation
│   │   ├── __init__.py
│   │   ├── server.py      # FastMCP server setup
│   │   ├── tools/         # MCP tool implementations
│   │   │   ├── __init__.py
│   │   │   ├── onpage_tools.py
│   │   │   ├── keyword_tools.py
│   │   │   ├── content_tools.py
│   │   │   └── graph_tools.py
│   │   └── schemas/       # MCP request/response schemas
│   │       ├── __init__.py
│   │       ├── analysis_schemas.py
│   │       └── reporting_schemas.py
│   ├── cli/               # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py        # CLI entry point
│   │   ├── commands/      # CLI command implementations
│   │   │   ├── __init__.py
│   │   │   ├── analyze.py
│   │   │   └── report.py
│   │   └── formatters/    # CLI output formatters
│   │       ├── __init__.py
│   │       └── table_formatter.py
│   └── api/               # Direct API for programmatic usage
│       ├── __init__.py
│       ├── client.py      # Python client library
│       └── exceptions.py  # API-specific exceptions
│
├── shared/                # Shared utilities and common code
│   ├── __init__.py
│   ├── logging.py         # Logging configuration
│   ├── exceptions.py      # Common exception types
│   ├── metrics.py         # Performance metrics collection
│   └── utils.py           # General utility functions
│
└── __init__.py            # Package entry point
```

## Module Responsibilities

### **Domain Layer**
- **Pure business logic** with no external dependencies
- **Entities:** Core SEO objects with behavior and invariants
- **Services:** Complex domain operations that don't belong to entities
- **Repositories:** Abstract interfaces for data access
- **Value Objects:** Immutable data structures

### **Application Layer**
- **Use case orchestration** without business logic
- **Analyzers:** High-level analysis coordination
- **Workflows:** Multi-step processes across multiple analyzers
- **Handlers:** Command/query processing
- **Services:** Application-level coordination

### **Infrastructure Layer**
- **External service integration** (DataForSEO, web crawling)
- **Data persistence** implementations
- **Reporting and visualization**
- **Configuration management**

### **Interface Layer**
- **MCP server** with protocol-compliant tools
- **CLI** for command-line usage
- **API client** for programmatic access

## Dependency Rules

1. **Domain** depends on nothing external
2. **Application** depends only on Domain
3. **Infrastructure** can depend on Domain and Application
4. **Interfaces** can depend on all layers
5. **No circular dependencies** between modules
6. **Shared utilities** can be used by any layer

## Key Design Patterns

### **Repository Pattern**
```python
# Abstract repository in domain
class PageRepositoryProtocol(Protocol):
    def save_page(self, page: Page) -> None: ...
    def find_by_url(self, url: str) -> Optional[Page]: ...

# Concrete implementation in infrastructure
class KuzuPageRepository:
    def __init__(self, kuzu_manager: KuzuManager):
        self._kuzu = kuzu_manager

    def save_page(self, page: Page) -> None:
        # Kuzu-specific implementation
```

### **Factory Pattern**
```python
# Abstract factory in domain/application
class AnalyzerFactoryProtocol(Protocol):
    def create_onpage_analyzer(self) -> OnPageAnalyzer: ...

# Concrete factory in infrastructure
class ProductionAnalyzerFactory:
    def create_onpage_analyzer(self) -> OnPageAnalyzer:
        # Wire up dependencies for production
```

### **Adapter Pattern**
```python
# External service adapter in infrastructure
class DataForSEOAdapter:
    def __init__(self, client: DataForSEOClient):
        self._client = client

    def analyze_onpage(self, request: OnPageRequest) -> OnPageResult:
        # Adapt external API to domain objects
```

## Import Guidelines

### **Allowed Imports by Layer**

#### Domain Layer
```python
# ✅ Allowed
from typing import Protocol, Dict, List
from dataclasses import dataclass
from enum import Enum

# ❌ Not allowed
from mcp_seo.infrastructure import *  # No infrastructure imports
from requests import *  # No external libraries
```

#### Application Layer
```python
# ✅ Allowed
from mcp_seo.domain.entities import Page, Keyword
from mcp_seo.domain.services import SEOScoringService
from typing import Protocol

# ❌ Not allowed
from mcp_seo.infrastructure.adapters import DataForSEOAdapter  # Direct infrastructure usage
```

#### Infrastructure Layer
```python
# ✅ Allowed
from mcp_seo.domain.repositories import PageRepositoryProtocol
from mcp_seo.application.analyzers import OnPageAnalyzer
import requests  # External libraries OK

# ❌ Not allowed
from mcp_seo.interfaces.mcp import MCPServer  # No interface layer imports
```

#### Interface Layer
```python
# ✅ Allowed
from mcp_seo.application.analyzers import OnPageAnalyzer
from mcp_seo.infrastructure.config import ProductionAnalyzerFactory
from fastmcp import FastMCP

# ❌ Not allowed - but this is the top layer, so generally more permissive
```

This module structure ensures:
- **Clean separation of concerns**
- **Easy testing** with clear boundaries
- **Flexible deployment** options
- **Maintainable codebase** with single responsibilities