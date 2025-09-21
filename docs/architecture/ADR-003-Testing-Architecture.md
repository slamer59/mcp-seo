# ADR-003: Testing Architecture and Strategy

**Status:** Proposed
**Date:** 2025-09-21
**Context:** Phase 2 - Clean Architecture Design

## Context

Current testing has 65% average coverage with mixed testing approaches:
- Unit tests mixed with integration tests
- Inconsistent mocking strategies
- Difficult to isolate components
- Slow test execution due to external dependencies

## Problem

Need a comprehensive testing strategy that:
- Achieves >90% code coverage
- Provides fast feedback during development
- Enables confidence in production deployments
- Supports both unit and integration testing

## Decision

Implement a **Layered Testing Strategy** with clear boundaries:

### **1. Unit Tests (Fast, Isolated)**
- **Scope:** Single component behavior
- **Dependencies:** All external dependencies mocked
- **Location:** `tests/unit/`
- **Execution Time:** <1 second per test
- **Coverage Target:** 95%

```python
# Example: Unit test with mocked dependencies
def test_onpage_analyzer_scoring():
    mock_client = Mock(spec=DataForSEOClient)
    mock_reporter = Mock(spec=SEOReporter)

    analyzer = OnPageAnalyzer(mock_client, mock_reporter)

    # Test business logic only
    result = analyzer.calculate_health_score(mock_data)

    assert result.score == expected_score
    mock_client.assert_not_called()  # Verify isolation
```

### **2. Integration Tests (Medium Speed)**
- **Scope:** Component interaction within bounded contexts
- **Dependencies:** Real implementations within domain boundaries
- **Location:** `tests/integration/`
- **Execution Time:** <5 seconds per test
- **Coverage Target:** 80%

```python
# Example: Integration test with real domain components
def test_content_analysis_workflow():
    factory = TestAnalyzerFactory()  # Uses real domain objects
    workflow = ContentAnalysisWorkflow(factory)

    result = workflow.analyze_blog_content(test_posts)

    assert len(result.pillar_pages) > 0
    assert result.content_clusters is not None
```

### **3. Contract Tests (API Compatibility)**
- **Scope:** External API contracts and MCP protocol compliance
- **Dependencies:** Mock external services with realistic responses
- **Location:** `tests/contracts/`
- **Execution Time:** <10 seconds per test
- **Coverage Target:** 100% of external interfaces

```python
# Example: Contract test for DataForSEO API
def test_dataforseo_onpage_contract():
    mock_responses.add(
        responses.POST,
        "https://api.dataforseo.com/v3/on_page/task_post",
        json=EXPECTED_RESPONSE_SCHEMA
    )

    client = DataForSEOClient()
    result = client.create_onpage_task("example.com")

    validate_response_schema(result, OnPageTaskSchema)
```

### **4. End-to-End Tests (Slow, Comprehensive)**
- **Scope:** Complete user workflows
- **Dependencies:** Real external services (limited subset)
- **Location:** `tests/e2e/`
- **Execution Time:** <30 seconds per test
- **Coverage Target:** Critical user paths only

## Testing Patterns

### **Test Data Management**
```python
# Fixtures for consistent test data
@pytest.fixture
def sample_onpage_data():
    return OnPageData.from_file("tests/fixtures/onpage_sample.json")

@pytest.fixture
def mock_dataforseo_responses():
    return MockResponseBuilder().with_onpage_data().build()
```

### **Mock Strategy**
```python
# Protocol-based mocking for clean interfaces
class MockDataForSEOClient:
    def __init__(self, responses: Dict[str, Any]):
        self._responses = responses

    def create_onpage_task(self, target: str) -> Dict[str, Any]:
        return self._responses.get("onpage_task", {})
```

### **Test Categories by Layer**

#### **Domain Layer Tests**
- Entity behavior validation
- Business rule enforcement
- Value object immutability
- Domain service logic

#### **Application Layer Tests**
- Use case orchestration
- Command/query handling
- Workflow coordination
- Error handling paths

#### **Infrastructure Layer Tests**
- Adapter functionality
- Data persistence
- External API communication
- Configuration management

#### **Interface Layer Tests**
- MCP protocol compliance
- Request/response validation
- Error handling
- Authentication/authorization

## Performance Testing Strategy

### **Load Testing**
- Large site analysis (1000+ pages)
- Concurrent analysis requests
- Memory usage patterns
- Database performance

### **Stress Testing**
- Resource exhaustion scenarios
- Network failure recovery
- Database connection limits
- API rate limiting

## Test Execution Strategy

### **Local Development**
```bash
# Fast feedback loop
pytest tests/unit/ -x --cov=src/

# Full test suite
pytest tests/ --cov=src/ --cov-report=html
```

### **CI/CD Pipeline**
1. **Unit Tests** - Every commit (< 30 seconds)
2. **Integration Tests** - Every PR (< 2 minutes)
3. **Contract Tests** - Every PR (< 1 minute)
4. **E2E Tests** - Nightly builds (< 5 minutes)

## Quality Gates

### **Coverage Requirements**
- Unit Tests: 95% line coverage
- Integration Tests: 80% of critical paths
- Contract Tests: 100% of external interfaces

### **Performance Requirements**
- Unit test suite: < 30 seconds
- Integration test suite: < 2 minutes
- Memory usage: < 500MB for large site analysis

## Consequences

### **Positive:**
- Fast feedback during development
- High confidence in deployments
- Clear testing boundaries
- Easy to identify test failures
- Scalable testing strategy

### **Negative:**
- Initial setup complexity
- More test code to maintain
- Need for comprehensive test data management
- Training required for team adoption