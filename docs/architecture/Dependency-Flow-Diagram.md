# Dependency Flow and Component Relationships

## System Context Diagram (C4 Level 1)

```mermaid
graph TB
    User[SEO Analyst]
    MCP[MCP Client]
    CLI[CLI User]
    Dev[Developer]

    SEOSystem[MCP SEO System]

    DataForSEO[DataForSEO API]
    KuzuDB[(Kuzu Graph DB)]
    WebSites[Target Websites]

    User -->|analyze website| MCP
    CLI -->|command line| SEOSystem
    Dev -->|programmatic| SEOSystem
    MCP -->|MCP protocol| SEOSystem

    SEOSystem -->|API calls| DataForSEO
    SEOSystem -->|graph queries| KuzuDB
    SEOSystem -->|crawl/fetch| WebSites

    DataForSEO -->|SEO data| SEOSystem
    KuzuDB -->|link analysis| SEOSystem
    WebSites -->|content| SEOSystem
```

## Container Diagram (C4 Level 2)

```mermaid
graph TB
    subgraph "MCP SEO System"
        MCPServer[MCP Server<br/>FastMCP]
        CLIApp[CLI Application<br/>Click/Typer]
        APIClient[Python API Client<br/>Direct Usage]

        AppLayer[Application Layer<br/>Use Cases & Workflows]
        DomainLayer[Domain Layer<br/>Business Logic]
        InfraLayer[Infrastructure Layer<br/>External Integrations]
    end

    subgraph "External Systems"
        DataForSEO[DataForSEO API<br/>RESTful Service]
        KuzuDB[(Kuzu Database<br/>Graph Storage)]
        WebCrawler[Web Crawler<br/>Site Content)]
    end

    MCPServer -->|orchestrates| AppLayer
    CLIApp -->|orchestrates| AppLayer
    APIClient -->|orchestates| AppLayer

    AppLayer -->|uses| DomainLayer
    AppLayer -->|uses| InfraLayer

    InfraLayer -->|HTTP calls| DataForSEO
    InfraLayer -->|graph operations| KuzuDB
    InfraLayer -->|content fetching| WebCrawler
```

## Component Diagram (C4 Level 3) - Application Layer

```mermaid
graph TB
    subgraph "Application Layer"
        subgraph "Analyzers"
            OnPageAnalyzer[OnPage Analyzer<br/>SEO Health Analysis]
            KeywordAnalyzer[Keyword Analyzer<br/>Search Volume & Competition]
            ContentAnalyzer[Content Analyzer<br/>Blog & Content Quality]
            CompetitorAnalyzer[Competitor Analyzer<br/>Competitive Intelligence]
        end

        subgraph "Workflows"
            ComprehensiveAudit[Comprehensive Audit<br/>Full SEO Analysis]
            ContentOptimization[Content Optimization<br/>Content Strategy]
            LinkBuilding[Link Building<br/>Link Opportunity Analysis]
        end

        subgraph "Services"
            RecommendationEngine[Recommendation Engine<br/>Priority-based Suggestions]
            ReportGenerator[Report Generator<br/>Multi-format Output]
        end
    end

    subgraph "Domain Layer"
        DomainServices[Domain Services<br/>SEO Algorithms]
        Entities[Entities<br/>Page, Keyword, Link]
        Repositories[Repository Interfaces<br/>Abstract Data Access]
    end

    subgraph "Infrastructure Layer"
        DataForSEOAdapter[DataForSEO Adapter<br/>External API Integration]
        KuzuGraphStore[Kuzu Graph Store<br/>Link Graph Persistence]
        ConsoleReporter[Console Reporter<br/>Rich CLI Output]
    end

    %% Analyzer dependencies
    OnPageAnalyzer -->|uses| DomainServices
    OnPageAnalyzer -->|uses| Repositories
    KeywordAnalyzer -->|uses| DomainServices
    ContentAnalyzer -->|uses| DomainServices
    CompetitorAnalyzer -->|uses| DomainServices

    %% Workflow orchestration
    ComprehensiveAudit -->|orchestrates| OnPageAnalyzer
    ComprehensiveAudit -->|orchestrates| KeywordAnalyzer
    ComprehensiveAudit -->|orchestrates| ContentAnalyzer
    ContentOptimization -->|orchestrates| ContentAnalyzer
    LinkBuilding -->|orchestrates| ContentAnalyzer

    %% Service dependencies
    RecommendationEngine -->|consolidates| OnPageAnalyzer
    RecommendationEngine -->|consolidates| KeywordAnalyzer
    ReportGenerator -->|formats| RecommendationEngine

    %% Infrastructure implementations
    Repositories -.->|implemented by| DataForSEOAdapter
    Repositories -.->|implemented by| KuzuGraphStore
    ReportGenerator -->|outputs via| ConsoleReporter
```

## Dependency Flow by Feature

### OnPage Analysis Flow

```mermaid
sequenceDiagram
    participant MCP as MCP Server
    participant APP as OnPage Analyzer
    participant DOM as SEO Scoring Service
    participant INF as DataForSEO Adapter
    participant API as DataForSEO API

    MCP->>APP: analyze_onpage(request)
    APP->>INF: create_analysis_task(target)
    INF->>API: POST /v3/on_page/task_post
    API-->>INF: task_id
    INF-->>APP: task_id

    APP->>INF: wait_for_completion(task_id)
    INF->>API: GET /v3/on_page/summary
    API-->>INF: analysis_results
    INF-->>APP: parsed_results

    APP->>DOM: calculate_seo_score(results)
    DOM-->>APP: seo_score

    APP->>APP: generate_recommendations()
    APP-->>MCP: analysis_report
```

### Content Analysis Flow

```mermaid
sequenceDiagram
    participant CLI as CLI Command
    participant WF as Content Workflow
    participant CA as Content Analyzer
    participant BA as Blog Analyzer
    participant LA as Link Analyzer
    participant KG as Kuzu Graph Store

    CLI->>WF: optimize_content(site_path)
    WF->>CA: analyze_content_quality(posts)
    CA->>BA: calculate_networkx_metrics(posts)
    BA-->>CA: content_clusters
    CA-->>WF: quality_analysis

    WF->>LA: analyze_internal_links(posts)
    LA->>KG: build_link_graph(links)
    KG-->>LA: graph_data
    LA->>LA: calculate_pagerank()
    LA-->>WF: link_optimization

    WF->>WF: consolidate_recommendations()
    WF-->>CLI: optimization_report
```

### Keyword Research Flow

```mermaid
sequenceDiagram
    participant API as Python API Client
    participant KA as Keyword Analyzer
    participant RE as Recommendation Engine
    participant DA as DataForSEO Adapter
    participant EXT as DataForSEO API

    API->>KA: analyze_keywords(keyword_list)
    KA->>DA: get_keyword_data(keywords)
    DA->>EXT: POST /v3/keywords_data/google/search_volume/task_post
    EXT-->>DA: search_volume_data
    DA-->>KA: processed_metrics

    KA->>DA: get_serp_data(keywords)
    DA->>EXT: POST /v3/serp/google/organic/task_post
    EXT-->>DA: serp_results
    DA-->>KA: competition_analysis

    KA->>RE: generate_keyword_recommendations(data)
    RE-->>KA: prioritized_suggestions

    KA-->>API: keyword_analysis_report
```

## Factory and Dependency Injection Pattern

```mermaid
graph TB
    subgraph "Configuration Layer"
        Config[SEO Config<br/>Environment Settings]
        ProductionFactory[Production Factory<br/>Real Dependencies]
        TestFactory[Test Factory<br/>Mock Dependencies]
    end

    subgraph "Application Context"
        AnalyzerContext[Analyzer Context<br/>Factory + Config]
    end

    subgraph "Created Components"
        OnPageAnalyzer[OnPage Analyzer]
        KeywordAnalyzer[Keyword Analyzer]
        ContentAnalyzer[Content Analyzer]
        ReportGenerator[Report Generator]
    end

    subgraph "Infrastructure Dependencies"
        DataForSEOClient[DataForSEO Client]
        KuzuManager[Kuzu Manager]
        ConsoleReporter[Console Reporter]
    end

    Config -->|configures| ProductionFactory
    Config -->|configures| TestFactory

    ProductionFactory -->|creates| AnalyzerContext
    TestFactory -->|creates| AnalyzerContext

    AnalyzerContext -->|instantiates| OnPageAnalyzer
    AnalyzerContext -->|instantiates| KeywordAnalyzer
    AnalyzerContext -->|instantiates| ContentAnalyzer
    AnalyzerContext -->|instantiates| ReportGenerator

    ProductionFactory -->|wires| DataForSEOClient
    ProductionFactory -->|wires| KuzuManager
    ProductionFactory -->|wires| ConsoleReporter

    OnPageAnalyzer -.->|injected| DataForSEOClient
    ContentAnalyzer -.->|injected| KuzuManager
    ReportGenerator -.->|injected| ConsoleReporter
```

## Data Flow Architecture

### Analysis Request Processing

```mermaid
graph LR
    subgraph "Input Layer"
        MCPRequest[MCP Request]
        CLICommand[CLI Command]
        APICall[API Call]
    end

    subgraph "Processing Pipeline"
        Validation[Request Validation]
        Orchestration[Analysis Orchestration]
        DataCollection[Data Collection]
        Processing[Data Processing]
        Recommendation[Recommendation Generation]
        Formatting[Report Formatting]
    end

    subgraph "Output Layer"
        MCPResponse[MCP Response]
        ConsoleOutput[Console Output]
        APIResponse[API Response]
    end

    MCPRequest --> Validation
    CLICommand --> Validation
    APICall --> Validation

    Validation --> Orchestration
    Orchestration --> DataCollection
    DataCollection --> Processing
    Processing --> Recommendation
    Recommendation --> Formatting

    Formatting --> MCPResponse
    Formatting --> ConsoleOutput
    Formatting --> APIResponse
```

### Graph Analysis Data Flow

```mermaid
graph TB
    subgraph "Content Sources"
        WebCrawl[Web Crawling]
        LocalFiles[Local Markdown Files]
        Sitemap[XML Sitemap]
    end

    subgraph "Data Processing"
        ContentParsing[Content Parsing]
        LinkExtraction[Link Extraction]
        GraphBuilding[Graph Building]
    end

    subgraph "Graph Storage"
        KuzuDB[(Kuzu Database)]
        MemoryGraph[NetworkX Graph]
    end

    subgraph "Analysis Algorithms"
        PageRank[PageRank Calculation]
        Centrality[Centrality Analysis]
        Community[Community Detection]
        PathAnalysis[Path Analysis]
    end

    subgraph "Results"
        PillarPages[Pillar Page Identification]
        ContentClusters[Content Clusters]
        LinkOpportunities[Link Opportunities]
        NavigationOptimization[Navigation Optimization]
    end

    WebCrawl --> ContentParsing
    LocalFiles --> ContentParsing
    Sitemap --> ContentParsing

    ContentParsing --> LinkExtraction
    LinkExtraction --> GraphBuilding

    GraphBuilding --> KuzuDB
    GraphBuilding --> MemoryGraph

    KuzuDB --> PageRank
    MemoryGraph --> Centrality
    MemoryGraph --> Community
    MemoryGraph --> PathAnalysis

    PageRank --> PillarPages
    Centrality --> PillarPages
    Community --> ContentClusters
    PathAnalysis --> NavigationOptimization

    PillarPages --> LinkOpportunities
    ContentClusters --> LinkOpportunities
```

This dependency flow ensures:
- **Clear data flow** from input to output
- **Proper separation** of concerns across layers
- **Flexible component substitution** via dependency injection
- **Testable architecture** with well-defined boundaries
- **Scalable processing** with async/parallel capabilities