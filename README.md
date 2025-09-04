# FastMCP SEO Analysis Server

A comprehensive SEO analysis server built on FastMCP that integrates with DataForSEO's API to provide professional-grade SEO analysis capabilities through the Model Context Protocol.

## Features

- **OnPage SEO Analysis**: Technical SEO audit, content optimization, performance analysis
- **Keyword Research**: Search volume data, competition analysis, SERP analysis
- **Competitor Intelligence**: Domain analysis, content gap analysis, competitive positioning  
- **Comprehensive Auditing**: Multi-dimensional analysis with priority scoring

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up DataForSEO credentials:
   ```bash
   export DATAFORSEO_USERNAME="your_username"
   export DATAFORSEO_PASSWORD="your_password"
   ```

3. Run the server:
   ```bash
   uv run python -m fastmcp_seo.server
   ```

## MCP Tools

The server provides comprehensive SEO analysis tools:

- `onpage_analysis_start` - Start OnPage SEO analysis
- `keyword_analysis` - Keyword research and analysis  
- `serp_analysis` - SERP results analysis
- `domain_analysis` - Domain performance analysis
- `competitor_comparison` - Compare domains
- `content_gap_analysis` - Find content opportunities
- `comprehensive_seo_audit` - Full SEO audit

## Configuration

Set environment variables or create a `.env` file:

```
DATAFORSEO_USERNAME=your_username
DATAFORSEO_PASSWORD=your_password
DEFAULT_LOCATION_CODE=2840
DEFAULT_LANGUAGE_CODE=en
```

## License

MIT License