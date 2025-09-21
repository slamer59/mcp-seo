# ADR-002: Dependency Injection Strategy

**Status:** Proposed
**Date:** 2025-09-21
**Context:** Phase 2 - Clean Architecture Design

## Context

Current enhanced components use direct instantiation and have hard dependencies:
- Difficult to test in isolation
- Cannot swap implementations
- Tight coupling between layers

## Problem

Need a dependency injection strategy that:
- Enables easy testing with mocks
- Supports configuration-based component selection
- Maintains performance for production usage
- Remains simple for developers to understand

## Considered Options

### **Option 1: Constructor Injection with Manual Wiring**
```python
class OnPageAnalyzer:
    def __init__(self, client: DataForSEOClient, reporter: SEOReporter):
        self.client = client
        self.reporter = reporter
```

**Pros:** Simple, explicit, no framework dependency
**Cons:** Manual wiring complexity, difficult to manage large dependency graphs

### **Option 2: Service Container with Registry**
```python
container = ServiceContainer()
container.register(DataForSEOClient, singleton=True)
container.register(OnPageAnalyzer, dependencies=[DataForSEOClient, SEOReporter])
```

**Pros:** Centralized configuration, automatic dependency resolution
**Cons:** Framework dependency, magic behavior, harder to debug

### **Option 3: Factory Pattern with Protocols**
```python
@protocol
class AnalyzerFactory:
    def create_onpage_analyzer(self) -> OnPageAnalyzer: ...
    def create_keyword_analyzer(self) -> KeywordAnalyzer: ...
```

**Pros:** Explicit interfaces, easy testing, no framework dependency
**Cons:** More boilerplate, manual factory implementation

## Decision

**Choose Option 3: Factory Pattern with Protocols**

Implement a hybrid approach:
1. **Protocol-based interfaces** for clean abstractions
2. **Factory classes** for component creation
3. **Configuration-driven** factory selection
4. **Context managers** for resource lifecycle

### **Implementation Pattern:**

```python
# Domain interfaces
class SEOAnalyzerProtocol(Protocol):
    def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...

# Factory interface
class AnalyzerFactoryProtocol(Protocol):
    def create_onpage_analyzer(self) -> SEOAnalyzerProtocol: ...

# Production factory
class ProductionAnalyzerFactory:
    def __init__(self, config: SEOConfig):
        self._config = config
        self._client_cache = {}

    def create_onpage_analyzer(self) -> OnPageAnalyzer:
        client = self._get_dataforseo_client()
        reporter = self._create_reporter()
        return OnPageAnalyzer(client, reporter)

# Test factory
class TestAnalyzerFactory:
    def create_onpage_analyzer(self) -> Mock:
        return Mock(spec=SEOAnalyzerProtocol)
```

## Consequences

### **Positive:**
- Easy to test with factory swapping
- Clear interfaces and contracts
- No external framework dependencies
- Explicit component creation
- Easy to understand and debug

### **Negative:**
- More boilerplate code
- Manual factory implementation
- Need to maintain protocol compliance

## Implementation Guidelines

1. Define protocols for all major interfaces
2. Create production and test factory implementations
3. Use context managers for resource cleanup
4. Implement configuration-driven factory selection
5. Provide factory injection for MCP server and CLI