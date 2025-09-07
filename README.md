# MCP SEO - Comprehensive SEO Analysis Server

A powerful MCP (Model Context Protocol) server that provides **comprehensive SEO analysis** using DataForSEO API with advanced **PageRank and internal link structure analysis** powered by Kuzu graph database.

## 🚀 Features

### **Core SEO Analysis**
- **OnPage SEO Analysis** - Technical SEO audit, content optimization, performance analysis
- **Keyword Research** - Search volume data, competition analysis, SERP analysis  
- **Competitor Intelligence** - Domain analysis, content gap analysis, competitive positioning
- **Comprehensive Auditing** - Multi-dimensional analysis with priority scoring

### **🔥 Advanced PageRank Analysis** 
- **PageRank Calculation** - Authority analysis using Kuzu graph database
- **Internal Link Optimization** - Link equity distribution and structure analysis
- **Pillar Page Identification** - Find high-authority pages for content strategy
- **Orphaned Page Detection** - Discover pages missing from internal linking
- **Link Graph Visualization** - Complete internal link structure mapping

## 📦 Installation

### **For Claude Code, OpenCoder & AI Development**

#### **Option 1: Direct from GitHub (Recommended)**
```bash
# Install directly from GitHub repository
uvx --from git+https://github.com/slamer59/mcp-seo mcp-seo

# Or install in development mode
uvx pip install git+https://github.com/slamer59/mcp-seo
```

#### **Option 2: Local Development Install**
```bash
# Clone the repository
git clone https://github.com/slamer59/mcp-seo
cd mcp-seo

# Install with uvx
uvx pip install -e .

# Or with uv (if you have uv installed)
uv pip install -e .
```

#### **Option 3: Python Package Install**
```bash
# Install from PyPI (when published)
uvx install mcp-seo

# Or with pip
pip install mcp-seo
```

### **MCP Server Configuration**

Add to your Claude Code/OpenCoder MCP configuration:

```json
{
  "mcpServers": {
    "mcp-seo": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/slamer59/mcp-seo", "mcp-seo"],
      "env": {
        "DATAFORSEO_LOGIN": "your_login",
        "DATAFORSEO_PASSWORD": "your_password"
      }
    }
  }
}
```

## 🔧 Configuration

### **DataForSEO API Setup**
1. Sign up at [DataForSEO](https://dataforseo.com/)
2. Get your API credentials
3. Set environment variables:

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"
```

### **Optional Configuration**
```bash
export DEFAULT_LOCATION_CODE=2840  # United States
export DEFAULT_LANGUAGE_CODE="en"  # English
```

## 🛠️ MCP Tools

### **Core SEO Tools**
- `onpage_analysis_start` - Start comprehensive OnPage SEO analysis
- `keyword_analysis` - Keyword research and search volume analysis
- `serp_analysis` - SERP results and ranking analysis
- `domain_analysis` - Domain performance and metrics analysis
- `competitor_comparison` - Compare multiple domains
- `content_gap_analysis` - Find content opportunities vs competitors
- `comprehensive_seo_audit` - Full multi-dimensional SEO audit

### **🔥 New PageRank & Link Analysis Tools**

#### **`analyze_pagerank`** - Complete PageRank Analysis
Calculate PageRank scores and analyze internal link authority distribution.

```json
{
  "domain": "https://example.com",
  "max_pages": 100,
  "damping_factor": 0.85,
  "use_sitemap": true
}
```

**Returns**: Complete analysis with pillar pages, orphaned pages, and optimization recommendations.

#### **`build_link_graph`** - Internal Link Graph Construction
Build and analyze internal link structure from sitemap or custom URLs.

```json
{
  "domain": "https://example.com",
  "max_pages": 50,
  "use_sitemap": true
}
```

**Returns**: Link graph statistics and structural metrics.

#### **`find_pillar_pages`** - High Authority Page Identification
Identify pages with highest PageRank scores for content strategy.

```json
{
  "domain": "https://example.com", 
  "percentile": 90.0,
  "limit": 10
}
```

**Returns**: Top authority pages with strategic recommendations.

#### **`find_orphaned_pages`** - Missing Link Detection
Find pages with no incoming internal links that need optimization.

```json
{
  "domain": "https://example.com"
}
```

**Returns**: Orphaned pages categorized by URL structure with fix recommendations.

#### **`optimize_internal_links`** - Link Optimization Strategy
Generate comprehensive internal linking optimization plan.

```json
{
  "domain": "https://example.com",
  "max_pages": 100
}
```

**Returns**: Priority-based optimization plan with specific actionable recommendations.

## 🎯 Usage Examples

### **Basic SEO Analysis**
```python
# Through MCP in Claude Code/OpenCoder
"Run a comprehensive SEO audit for https://example.com focusing on technical SEO and content optimization"
```

### **Advanced PageRank Analysis**
```python
# Analyze internal link structure
"Analyze the PageRank and internal linking structure for https://mysite.com, identify pillar pages and orphaned content"

# Find optimization opportunities  
"Find internal linking opportunities for https://mysite.com and create an optimization plan"
```

### **Content Strategy**
```python
# Identify high-authority pages
"Find the top 10 pillar pages on https://mysite.com and suggest how to leverage them for content strategy"

# Fix structural issues
"Identify orphaned pages on https://mysite.com and recommend how to integrate them into the site structure"
```

## 🧪 Testing & Development

### **Install Development Dependencies**
```bash
# Install with test dependencies
uv pip install -e ".[test]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mcp_seo --cov-report=html
```

### **Test Suite**
- **52+ Tests** covering all functionality
- **Unit Tests** for core components (KuzuManager, PageRankAnalyzer, LinkGraphBuilder)
- **Integration Tests** for end-to-end workflows
- **Async Testing** with proper mocking

## 📊 Technical Architecture

### **Core Components**
- **FastMCP Server** - MCP protocol implementation
- **DataForSEO Client** - Professional SEO data integration
- **Kuzu Graph Database** - High-performance graph analysis
- **PageRank Engine** - Mathematical authority calculation
- **Link Analysis** - Internal structure optimization

### **Dependencies**
- `fastmcp>=2.12.2` - MCP server framework
- `dataforseo-client>=2.0.0` - SEO data API
- `kuzu>=0.5.0` - Graph database for PageRank
- `aiohttp>=3.9.0` - Async web crawling
- `beautifulsoup4>=4.12.0` - HTML parsing
- `numpy>=1.24.0` - Mathematical operations
- `pydantic>=2.11.7` - Data validation

## 🚀 Why Choose MCP SEO?

### **✅ Professional Grade**
- **DataForSEO Integration** - Enterprise SEO data provider
- **Advanced Mathematics** - Proper PageRank algorithm implementation
- **Graph Database** - Kuzu for high-performance link analysis
- **Comprehensive Testing** - 52+ tests with proper async handling

### **✅ AI-Optimized** 
- **MCP Protocol** - Native integration with Claude Code, OpenCoder
- **Structured Data** - Clean JSON responses perfect for AI analysis
- **Actionable Insights** - AI can directly implement recommendations
- **Batch Operations** - Efficient multi-domain analysis

### **✅ Next.js & Modern Web Ready**
- **Sitemap.xml Processing** - Works with all modern frameworks
- **Beautiful Soup Parsing** - Handles server-side rendered content
- **Async Architecture** - Non-blocking operations for large sites
- **Extensible** - Ready for Playwright integration for SPA analysis

## 📈 Use Cases

### **Content Strategy**
- Identify high-authority pillar pages for content hubs
- Find orphaned content that needs internal linking
- Optimize link equity distribution across site sections

### **Technical SEO**
- Complete technical audit with DataForSEO professional data
- Internal link structure analysis and optimization
- Page authority analysis for navigation planning

### **Competitive Analysis**
- Multi-domain PageRank comparison
- Competitor link structure analysis  
- Content gap identification with authority metrics

### **Agency & Enterprise**
- Scalable multi-client SEO analysis
- Automated reporting with actionable insights
- Integration with AI workflows for optimization

## 🔗 Links

- **Repository**: https://github.com/slamer59/mcp-seo
- **Documentation**: [Full API Documentation](./docs/)
- **DataForSEO**: https://dataforseo.com/
- **MCP Protocol**: https://modelcontextprotocol.io/

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for AI-powered SEO analysis** | **Next.js Ready** | **Enterprise Grade** | **52+ Tests**
