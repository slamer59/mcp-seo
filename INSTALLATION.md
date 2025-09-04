# MCP SEO Installation Guide

Quick installation guide for Claude Code, OpenCoder, and other MCP-compatible environments.

## 🚀 Quick Install

### **Option 1: Direct GitHub Install (Recommended)**
```bash
uvx --from git+https://github.com/thomaspedot/mcp-seo mcp-seo
```

### **Option 2: Local Development**
```bash
git clone https://github.com/thomaspedot/mcp-seo
cd mcp-seo
uvx pip install -e .
```

## 🔧 MCP Server Configuration

### **Claude Code Configuration**
Add to your `.claude_code_config.json`:

```json
{
  "mcpServers": {
    "mcp-seo": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/thomaspedot/mcp-seo", "mcp-seo"],
      "env": {
        "DATAFORSEO_LOGIN": "your_dataforseo_login",
        "DATAFORSEO_PASSWORD": "your_dataforseo_password"
      }
    }
  }
}
```

### **OpenCoder Configuration**
Add to your MCP servers configuration:

```json
{
  "servers": {
    "mcp-seo": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/thomaspedot/mcp-seo", "mcp-seo"],
      "environment": {
        "DATAFORSEO_LOGIN": "your_dataforseo_login", 
        "DATAFORSEO_PASSWORD": "your_dataforseo_password"
      }
    }
  }
}
```

### **Generic MCP Client Configuration**
```json
{
  "name": "mcp-seo",
  "command": ["uvx", "--from", "git+https://github.com/thomaspedot/mcp-seo", "mcp-seo"],
  "env": {
    "DATAFORSEO_LOGIN": "your_login",
    "DATAFORSEO_PASSWORD": "your_password",
    "DEFAULT_LOCATION_CODE": "2840",
    "DEFAULT_LANGUAGE_CODE": "en"
  }
}
```

## 🔑 DataForSEO API Setup

1. **Sign Up**: Register at [DataForSEO](https://dataforseo.com/)
2. **Get Credentials**: Find your login/password in the dashboard
3. **Set Environment Variables**:
   ```bash
   export DATAFORSEO_LOGIN="your_login"
   export DATAFORSEO_PASSWORD="your_password"
   ```

## ✅ Verify Installation

Test that the server starts correctly:

```bash
# Test the server
uvx --from git+https://github.com/thomaspedot/mcp-seo mcp-seo --help

# Or if installed locally
mcp-seo --help
```

## 🎯 Usage Examples

Once installed, you can use these tools in Claude Code/OpenCoder:

### **PageRank Analysis**
```
"Analyze the PageRank and internal linking structure for https://mysite.com"
```

### **SEO Audit**
```  
"Run a comprehensive SEO audit for https://example.com focusing on technical SEO"
```

### **Link Optimization**
```
"Find internal linking opportunities for https://mysite.com and create an optimization plan"
```

## 🛠️ Development Setup

For development and testing:

```bash
# Clone and install with test dependencies
git clone https://github.com/thomaspedot/mcp-seo
cd mcp-seo
uv pip install -e ".[test]"

# Run tests
pytest tests/ -v
```

## 🔍 Troubleshooting

### **Common Issues**

**Server Not Starting**:
- Check DataForSEO credentials are set
- Verify uvx is installed: `uvx --version`
- Check network connectivity

**Import Errors**:
- Ensure all dependencies are installed: `uv pip install -e .`
- Check Python version: requires Python 3.10+

**API Errors**:
- Verify DataForSEO credentials in dashboard
- Check API quota and billing status
- Ensure correct location/language codes

### **Environment Variables**
```bash
# Required
DATAFORSEO_LOGIN=your_login
DATAFORSEO_PASSWORD=your_password

# Optional
DEFAULT_LOCATION_CODE=2840  # United States
DEFAULT_LANGUAGE_CODE=en    # English
```

### **Supported Environments**
- ✅ Claude Code
- ✅ OpenCoder  
- ✅ Any MCP-compatible client
- ✅ Python 3.10+
- ✅ uvx/uv package manager
- ✅ Linux, macOS, Windows

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/thomaspedot/mcp-seo/issues)
- **Documentation**: [Full README](./README.md)
- **DataForSEO Docs**: [API Documentation](https://docs.dataforseo.com/)

---

**Ready to analyze! 🚀** Your MCP SEO server is now configured and ready for AI-powered SEO analysis.