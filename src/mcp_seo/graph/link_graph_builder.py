"""
Link Graph Builder for SEO Analysis

Builds internal link graphs from sitemaps and web crawling for PageRank analysis.
Integrates with sitemap-markitdown package and async web crawling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from xml.etree import ElementTree as ET

import aiohttp
from bs4 import BeautifulSoup

from .kuzu_manager import KuzuManager

logger = logging.getLogger(__name__)


class LinkGraphBuilder:
    """Builds link graphs for SEO analysis using sitemap and web crawling."""

    def __init__(self, base_url: str, kuzu_manager: KuzuManager, max_pages: int = 100):
        """
        Initialize link graph builder.
        
        Args:
            base_url: Base URL of the website
            kuzu_manager: Initialized KuzuManager instance
            max_pages: Maximum number of pages to process
        """
        self.base_url = base_url.rstrip('/')
        self.kuzu_manager = kuzu_manager
        self.max_pages = max_pages
        self.processed_urls: Set[str] = set()

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and query parameters.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/') if parsed.path != '/' else '/',
            parsed.params,
            '',  # Remove query parameters
            ''   # Remove fragments
        ))
        return normalized

    def is_internal_url(self, url: str) -> bool:
        """
        Check if URL is internal to the base domain.

        Args:
            url: URL to check

        Returns:
            True if URL is internal
        """
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)

        # Special schemes are considered external
        special_schemes = {'mailto', 'tel', 'ftp', 'file', 'javascript', 'data'}
        if parsed_url.scheme in special_schemes:
            return False

        # Extract base domain (remove www. if present)
        base_domain = parsed_base.netloc.replace('www.', '') if parsed_base.netloc.startswith('www.') else parsed_base.netloc
        url_domain = parsed_url.netloc.replace('www.', '') if parsed_url.netloc.startswith('www.') else parsed_url.netloc

        return (
            url_domain == base_domain or
            parsed_url.netloc == ''
        )

    async def fetch_sitemap(self) -> List[str]:
        """
        Fetch and parse sitemap.xml to get initial page list.
        
        Returns:
            List of URLs from sitemap
        """
        sitemap_url = f"{self.base_url}/sitemap.xml"
        pages = []

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'MCP-SEO-PageRank-Analyzer/1.0'}
            ) as session:
                logger.info(f"Fetching sitemap from {sitemap_url}")
                async with session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)

                        # Handle sitemap namespace
                        namespace = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                        urls = root.findall('.//sm:url/sm:loc', namespace)

                        if not urls:  # Try without namespace
                            urls = root.findall('.//url/loc')

                        for url_elem in urls:
                            if url_elem.text:
                                normalized_url = self.normalize_url(url_elem.text)
                                if self.is_internal_url(normalized_url):
                                    pages.append(normalized_url)

                        logger.info(f"Found {len(pages)} pages in sitemap")
                    else:
                        logger.error(f"Failed to fetch sitemap: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error fetching sitemap: {e}")

        return pages[:self.max_pages]

    async def crawl_page_links(self, session: aiohttp.ClientSession, url: str) -> Tuple[Dict, List[Tuple[str, str, str]]]:
        """
        Crawl a single page to extract internal links and page metadata.
        
        Args:
            session: aiohttp session
            url: URL to crawl
            
        Returns:
            Tuple of (page_data, links_list)
        """
        page_data = {
            'url': url,
            'title': '',
            'status_code': 0,
            'content_length': 0
        }
        links = []

        try:
            async with session.get(url) as response:
                page_data['status_code'] = response.status
                page_data['content_length'] = int(response.headers.get('content-length', 0))

                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Extract page title
                    title_tag = soup.find('title')
                    if title_tag:
                        page_data['title'] = title_tag.get_text().strip()

                    # Extract all links
                    link_position = 0
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        anchor_text = link.get_text().strip()

                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, href)
                        normalized_url = self.normalize_url(absolute_url)

                        if self.is_internal_url(normalized_url) and normalized_url != url:
                            links.append((url, normalized_url, anchor_text))
                            link_position += 1

                    logger.debug(f"Crawled {url} → {len(links)} internal links")
                else:
                    logger.warning(f"Failed to crawl {url}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")

        return page_data, links

    async def build_link_graph_from_sitemap(self, progress_callback: Optional[callable] = None) -> Dict:
        """
        Build complete link graph from sitemap.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            # Fetch sitemap
            pages = await self.fetch_sitemap()
            if not pages:
                return {"error": "No pages found in sitemap"}

            logger.info(f"Building link graph for {len(pages)} pages")

            # Initialize progress
            if progress_callback:
                progress_callback(0, len(pages), "Starting to crawl pages...")

            all_pages_data = []
            all_links = []

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'MCP-SEO-PageRank-Analyzer/1.0'},
                connector=aiohttp.TCPConnector(limit=10)
            ) as session:

                # Limit concurrent requests
                semaphore = asyncio.Semaphore(5)

                async def crawl_with_semaphore(url: str, index: int) -> Tuple[Dict, List]:
                    async with semaphore:
                        page_data, links = await self.crawl_page_links(session, url)
                        if progress_callback:
                            progress_callback(index + 1, len(pages), f"Crawled {url}")
                        return page_data, links

                # Crawl all pages concurrently
                tasks = [crawl_with_semaphore(url, i) for i, url in enumerate(pages)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {pages[i]}: {result}")
                        continue

                    page_data, links = result
                    all_pages_data.append(page_data)
                    all_links.extend(links)

            # Add pages to database
            logger.info("Adding pages to graph database...")
            self.kuzu_manager.add_pages_batch(all_pages_data)

            # Add links to database
            logger.info("Adding links to graph database...")
            self.kuzu_manager.add_links_batch(all_links)

            # Calculate degree centrality
            logger.info("Calculating degree centrality...")
            self.kuzu_manager.calculate_degree_centrality()

            # Get final statistics
            stats = self.kuzu_manager.get_graph_stats()
            stats.update({
                'crawled_pages': len(all_pages_data),
                'discovered_links': len(all_links),
                'successful_crawls': len([p for p in all_pages_data if p['status_code'] == 200])
            })

            logger.info(f"Link graph built successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error building link graph: {e}")
            return {"error": str(e)}

    async def build_link_graph_from_urls(self, urls: List[str], 
                                       progress_callback: Optional[callable] = None) -> Dict:
        """
        Build link graph from a list of URLs.
        
        Args:
            urls: List of URLs to crawl
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            # Limit URLs to max_pages
            urls = urls[:self.max_pages]
            logger.info(f"Building link graph for {len(urls)} URLs")

            if progress_callback:
                progress_callback(0, len(urls), "Starting to crawl URLs...")

            all_pages_data = []
            all_links = []

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'MCP-SEO-PageRank-Analyzer/1.0'},
                connector=aiohttp.TCPConnector(limit=10)
            ) as session:

                semaphore = asyncio.Semaphore(5)

                async def crawl_with_semaphore(url: str, index: int) -> Tuple[Dict, List]:
                    async with semaphore:
                        page_data, links = await self.crawl_page_links(session, url)
                        if progress_callback:
                            progress_callback(index + 1, len(urls), f"Crawled {url}")
                        return page_data, links

                tasks = [crawl_with_semaphore(url, i) for i, url in enumerate(urls)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {urls[i]}: {result}")
                        continue

                    page_data, links = result
                    all_pages_data.append(page_data)
                    all_links.extend(links)

            # Add to database
            self.kuzu_manager.add_pages_batch(all_pages_data)
            self.kuzu_manager.add_links_batch(all_links)
            self.kuzu_manager.calculate_degree_centrality()

            stats = self.kuzu_manager.get_graph_stats()
            stats.update({
                'crawled_pages': len(all_pages_data),
                'discovered_links': len(all_links),
                'successful_crawls': len([p for p in all_pages_data if p['status_code'] == 200])
            })

            return stats

        except Exception as e:
            logger.error(f"Error building link graph from URLs: {e}")
            return {"error": str(e)}

    def get_link_opportunities(self) -> Dict:
        """
        Identify internal linking opportunities.
        
        Returns:
            Dictionary with linking opportunities and suggestions
        """
        try:
            pages_data = self.kuzu_manager.get_page_data()
            if not pages_data:
                return {"error": "No page data available"}

            opportunities = {
                'orphaned_pages': [p for p in pages_data if p['in_degree'] == 0],
                'low_outlink_pages': sorted(pages_data, key=lambda x: x['out_degree'])[:10],
                'high_authority_pages': sorted(pages_data, key=lambda x: x['pagerank'], reverse=True)[:10],
                'potential_hubs': sorted(pages_data, key=lambda x: x['out_degree'], reverse=True)[:10]
            }

            # Generate specific suggestions
            suggestions = []
            
            if opportunities['orphaned_pages']:
                suggestions.append({
                    'type': 'orphaned_pages',
                    'priority': 'high',
                    'description': f"Link to {len(opportunities['orphaned_pages'])} orphaned pages from relevant content",
                    'pages': [{'url': p['url'], 'title': p['title']} for p in opportunities['orphaned_pages'][:5]]
                })

            if opportunities['low_outlink_pages']:
                suggestions.append({
                    'type': 'internal_linking',
                    'priority': 'medium',
                    'description': f"Add more internal links from {len(opportunities['low_outlink_pages'])} pages with few outgoing links",
                    'pages': [{'url': p['url'], 'title': p['title'], 'out_degree': p['out_degree']} 
                             for p in opportunities['low_outlink_pages'][:5]]
                })

            opportunities['suggestions'] = suggestions
            return opportunities

        except Exception as e:
            logger.error(f"Error identifying link opportunities: {e}")
            return {"error": str(e)}

    async def expand_graph_from_discovered_pages(self, max_new_pages: int = 50) -> Dict:
        """
        Expand the graph by crawling newly discovered pages.
        
        Args:
            max_new_pages: Maximum number of new pages to add
            
        Returns:
            Dictionary with expansion statistics
        """
        try:
            # Get all unique URLs from links that aren't in pages yet
            links_data = self.kuzu_manager.get_links_data()
            pages_data = self.kuzu_manager.get_page_data()
            
            existing_urls = {p['url'] for p in pages_data}
            discovered_urls = set()
            
            for link in links_data:
                if link['target_url'] not in existing_urls:
                    discovered_urls.add(link['target_url'])

            new_urls = list(discovered_urls)[:max_new_pages]
            
            if not new_urls:
                return {"message": "No new pages to discover"}

            logger.info(f"Expanding graph with {len(new_urls)} newly discovered pages")
            
            # Crawl new pages
            stats = await self.build_link_graph_from_urls(new_urls)
            stats['expansion_pages'] = len(new_urls)
            
            return stats

        except Exception as e:
            logger.error(f"Error expanding graph: {e}")
            return {"error": str(e)}