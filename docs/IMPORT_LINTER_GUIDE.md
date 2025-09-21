# Import Linter Architecture Enforcement Guide

## Overview

This project uses `import-linter` to enforce clean architecture boundaries and prevent architectural drift. The configuration ensures that our Enhanced MCP SEO system maintains proper separation of concerns and dependency flow.

## Architecture Layers

Our import-linter configuration enforces a **4-layer clean architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                          │
│  mcp_seo.server - MCP server endpoints & FastMCP integration│
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  mcp_seo.tools    - Enhanced analyzers (OnPage, Keyword)    │
│  mcp_seo.engines  - Recommendation & processing engines     │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                            │
│  mcp_seo.content  - Content analysis business logic         │
│  mcp_seo.models   - Data models and contracts               │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                        │
│  mcp_seo.reporting   - Output formatting (Rich console)     │
│  mcp_seo.dataforseo  - External API clients                 │
│  mcp_seo.config      - Configuration management             │
└─────────────────────────────────────────────────────────────┘
```

## Contracts Enforced

### 1. **Layered Architecture**
- **Rule**: Higher layers can import from lower layers, but not vice versa
- **Purpose**: Prevents circular dependencies and maintains clean dependency flow
- **Example**: `mcp_seo.tools` can import from `mcp_seo.content`, but `mcp_seo.content` cannot import from `mcp_seo.tools`

### 2. **Domain Layer Independence**
- **Rule**: Domain layer (content, models) cannot import infrastructure layer
- **Purpose**: Keeps business logic free from external dependencies
- **Forbidden**: `mcp_seo.content` → `mcp_seo.dataforseo`, `mcp_seo.reporting`, `mcp_seo.config`

### 3. **Infrastructure Isolation**
- **Rule**: Infrastructure cannot import from application layer
- **Purpose**: Prevents infrastructure from depending on business logic
- **Forbidden**: `mcp_seo.dataforseo` → `mcp_seo.tools`, `mcp_seo.engines`

### 4. **Application Services Independence**
- **Rule**: Application services should not depend on each other
- **Purpose**: Enables independent evolution of services
- **Independent**: `mcp_seo.engines` ↔ `mcp_seo.content`

### 5. **Interface Layer Dependencies**
- **Rule**: Server should only depend on application layer, not infrastructure directly
- **Purpose**: Proper dependency injection and testability
- **Forbidden**: `mcp_seo.server` → `mcp_seo.dataforseo.client`, `mcp_seo.reporting.seo_reporter`

## Running Import Linter

### Install Dependencies
```bash
pip install -e ".[dev]"  # Installs import-linter with dev dependencies
```

### Run All Architecture Checks
```bash
lint-imports
```

### Run Specific Contract
```bash
lint-imports --contract "Layered Architecture"
lint-imports --contract "Domain Layer Independence"
```

### Verbose Output
```bash
lint-imports --verbose
```

### Show Timing Information
```bash
lint-imports --show_timings
```

## Common Violations and Fixes

### ❌ **Domain Layer Importing Infrastructure**
```python
# WRONG: Content analysis importing reporting
from mcp_seo.reporting import SEOReporter

class BlogAnalyzer:
    def __init__(self):
        self.reporter = SEOReporter()  # ❌ Violates Domain Independence
```

```python
# CORRECT: Use dependency injection
class BlogAnalyzer:
    def __init__(self, reporter=None):
        self.reporter = reporter  # ✅ Dependency injected from application layer
```

### ❌ **Infrastructure Importing Application Logic**
```python
# WRONG: DataForSEO client importing analyzer
from mcp_seo.tools import OnPageAnalyzer

class DataForSEOClient:
    def process_data(self, data):
        analyzer = OnPageAnalyzer()  # ❌ Violates Infrastructure Isolation
        return analyzer.analyze(data)
```

```python
# CORRECT: Keep infrastructure pure
class DataForSEOClient:
    def process_data(self, data):
        return self.clean_api_response(data)  # ✅ Pure infrastructure logic
```

### ❌ **Server Directly Importing Infrastructure**
```python
# WRONG: Server importing client directly
from mcp_seo.dataforseo.client import DataForSEOClient

class ServerEndpoint:
    def __init__(self):
        self.client = DataForSEOClient()  # ❌ Violates Interface Dependencies
```

```python
# CORRECT: Server imports application layer
from mcp_seo.tools import OnPageAnalyzer

class ServerEndpoint:
    def __init__(self):
        self.analyzer = OnPageAnalyzer()  # ✅ Application layer dependency
```

## Integration with CI/CD

### GitHub Actions
Add to `.github/workflows/ci.yml`:
```yaml
- name: Check Architecture Contracts
  run: lint-imports
```

### Pre-commit Hook
Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: import-linter
      name: import-linter
      entry: lint-imports
      language: system
      pass_filenames: false
```

### Make Target
Add to `Makefile`:
```make
lint-architecture:
	lint-imports

test: lint-architecture
	pytest tests/
```

## Benefits

### 🏗️ **Architectural Integrity**
- Prevents architectural drift over time
- Enforces separation of concerns automatically
- Catches violations before they reach production

### 🧪 **Better Testability**
- Forces dependency injection patterns
- Enables easier mocking and testing
- Reduces coupling between components

### 🔄 **Maintainability**
- Clear module responsibilities
- Predictable dependency flow
- Easier refactoring and evolution

### 👥 **Team Alignment**
- Enforces architectural decisions automatically
- Provides immediate feedback on violations
- Documents intended architecture through contracts

## Troubleshooting

### False Positives
If import-linter flags legitimate imports, add them to `ignore_imports`:
```toml
ignore_imports = [
    "mcp_seo.server -> mcp_seo.config.settings",
    "mcp_seo.tools.* -> mcp_seo.models.*"
]
```

### Performance Issues
For large codebases, use caching:
```bash
lint-imports --cache-dir .import_linter_cache
```

### Debugging Contract Failures
Use verbose mode to understand violations:
```bash
lint-imports --verbose --contract "Domain Layer Independence"
```

## Next Steps

1. **Run Initial Check**: `lint-imports` to see current violations
2. **Fix Violations**: Refactor code to match architecture contracts
3. **Add to CI**: Integrate into continuous integration pipeline
4. **Monitor**: Regular checks to prevent architectural drift

The import-linter ensures our Enhanced MCP SEO system maintains clean architecture principles and prevents technical debt accumulation.