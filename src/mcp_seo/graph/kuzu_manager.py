"""
Kuzu Graph Database Manager for SEO Analysis

Manages Kuzu database connections, schema setup, and graph operations
for SEO link structure analysis and PageRank calculations.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import kuzu

logger = logging.getLogger(__name__)


class KuzuManager:
    """Manages Kuzu graph database for SEO link analysis."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Kuzu database manager.
        
        Args:
            db_path: Path to database file. If None, creates temporary database.
        """
        if db_path:
            self.db_path = db_path
            self.is_temp = False
        else:
            # Create temporary database
            temp_dir = tempfile.mkdtemp(prefix="kuzu_seo_")
            self.db_path = str(Path(temp_dir) / "seo_graph.db")
            self.is_temp = True
            
        self.database: Optional[kuzu.Database] = None
        self.connection: Optional[kuzu.Connection] = None
        self._schema_initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def connect(self) -> None:
        """Establish connection to Kuzu database."""
        try:
            self.database = kuzu.Database(self.db_path)
            self.connection = kuzu.Connection(self.database)
            logger.info(f"Connected to Kuzu database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to Kuzu database: {e}")
            raise

    def close(self) -> None:
        """Close database connection and cleanup."""
        if self.connection:
            self.connection.close()
            self.connection = None
            
        if self.database:
            self.database = None
            
        # Clean up temporary database
        if self.is_temp and Path(self.db_path).exists():
            try:
                Path(self.db_path).unlink()
                # Remove temporary directory
                Path(self.db_path).parent.rmdir()
                logger.info("Cleaned up temporary database")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary database: {e}")

    def initialize_schema(self) -> None:
        """Initialize database schema for SEO analysis."""
        if self._schema_initialized:
            return

        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            # Create Page node table
            self.connection.execute("""
                CREATE NODE TABLE IF NOT EXISTS Page(
                    url STRING,
                    title STRING,
                    domain STRING,
                    path STRING,
                    pagerank DOUBLE DEFAULT 0.0,
                    in_degree INT64 DEFAULT 0,
                    out_degree INT64 DEFAULT 0,
                    crawled BOOLEAN DEFAULT false,
                    status_code INT64 DEFAULT 0,
                    content_length INT64 DEFAULT 0,
                    PRIMARY KEY (url)
                )
            """)

            # Create Links relationship table
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS Links(
                    FROM Page TO Page,
                    anchor_text STRING DEFAULT "",
                    link_type STRING DEFAULT "internal",
                    position INT64 DEFAULT 0
                )
            """)

            self._schema_initialized = True
            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def add_page(self, url: str, title: str = "", status_code: int = 0, 
                content_length: int = 0) -> None:
        """
        Add a page to the graph database.
        
        Args:
            url: Page URL
            title: Page title
            status_code: HTTP status code
            content_length: Content length in bytes
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        parsed = urlparse(url)
        
        try:
            self.connection.execute("""
                MERGE (p:Page {url: $url})
                SET p.title = $title,
                    p.domain = $domain,
                    p.path = $path,
                    p.status_code = $status_code,
                    p.content_length = $content_length,
                    p.crawled = true
            """, parameters={
                "url": url,
                "title": title or parsed.path.replace('/', ' ').strip() or "Home",
                "domain": parsed.netloc,
                "path": parsed.path,
                "status_code": status_code,
                "content_length": content_length
            })
            
        except Exception as e:
            logger.error(f"Failed to add page {url}: {e}")
            raise

    def add_pages_batch(self, pages: List[Dict]) -> None:
        """
        Add multiple pages in batch for better performance.
        
        Args:
            pages: List of page dictionaries with url, title, etc.
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        for page in pages:
            try:
                self.add_page(
                    url=page['url'],
                    title=page.get('title', ''),
                    status_code=page.get('status_code', 0),
                    content_length=page.get('content_length', 0)
                )
            except Exception as e:
                logger.warning(f"Failed to add page {page.get('url', 'unknown')}: {e}")
                continue

    def add_link(self, source_url: str, target_url: str, anchor_text: str = "", 
                position: int = 0) -> None:
        """
        Add a link between two pages.
        
        Args:
            source_url: Source page URL
            target_url: Target page URL
            anchor_text: Link anchor text
            position: Link position on page
        """
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            # Ensure both pages exist
            self.connection.execute("MERGE (p:Page {url: $url})", parameters={"url": source_url})
            self.connection.execute("MERGE (p:Page {url: $url})", parameters={"url": target_url})
            
            # Create link
            self.connection.execute("""
                MATCH (source:Page {url: $source}), (target:Page {url: $target})
                MERGE (source)-[l:Links]->(target)
                SET l.anchor_text = $anchor_text,
                    l.position = $position
            """, parameters={
                "source": source_url,
                "target": target_url,
                "anchor_text": anchor_text,
                "position": position
            })
            
        except Exception as e:
            logger.error(f"Failed to add link {source_url} -> {target_url}: {e}")
            raise

    def add_links_batch(self, links: List[Tuple[str, str, str]]) -> None:
        """
        Add multiple links in batch.
        
        Args:
            links: List of (source_url, target_url, anchor_text) tuples
        """
        for source, target, anchor in links:
            try:
                self.add_link(source, target, anchor)
            except Exception as e:
                logger.warning(f"Failed to add link {source} -> {target}: {e}")
                continue

    def calculate_degree_centrality(self) -> None:
        """Calculate and update in-degree and out-degree for all pages."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            # Calculate and update in-degree
            self.connection.execute("""
                MATCH (p:Page)
                OPTIONAL MATCH (p)<-[:Links]-(other:Page)
                WITH p, COUNT(other) as in_deg
                SET p.in_degree = in_deg
            """)

            # Calculate and update out-degree
            self.connection.execute("""
                MATCH (p:Page)
                OPTIONAL MATCH (p)-[:Links]->(other:Page)
                WITH p, COUNT(other) as out_deg
                SET p.out_degree = out_deg
            """)

            logger.info("Degree centrality calculation completed")

        except Exception as e:
            logger.error(f"Failed to calculate degree centrality: {e}")
            raise

    def get_page_data(self) -> List[Dict]:
        """Get all page data with metrics."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            result = self.connection.execute("""
                MATCH (p:Page)
                RETURN p.url as url, p.title as title, p.pagerank as pagerank,
                       p.in_degree as in_degree, p.out_degree as out_degree,
                       p.domain as domain, p.path as path, p.status_code as status_code
                ORDER BY p.pagerank DESC
            """)

            pages = []
            for row in result:
                pages.append({
                    'url': row[0],
                    'title': row[1],
                    'pagerank': row[2] or 0.0,
                    'in_degree': row[3] or 0,
                    'out_degree': row[4] or 0,
                    'domain': row[5],
                    'path': row[6],
                    'status_code': row[7] or 0
                })

            return pages

        except Exception as e:
            logger.error(f"Failed to get page data: {e}")
            return []

    def get_links_data(self) -> List[Dict]:
        """Get all links data."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            result = self.connection.execute("""
                MATCH (source:Page)-[l:Links]->(target:Page)
                RETURN source.url as source_url, target.url as target_url,
                       l.anchor_text as anchor_text, l.position as position
            """)

            links = []
            for row in result:
                links.append({
                    'source_url': row[0],
                    'target_url': row[1],
                    'anchor_text': row[2] or "",
                    'position': row[3] or 0
                })

            return links

        except Exception as e:
            logger.error(f"Failed to get links data: {e}")
            return []

    def get_incoming_links(self, url: str) -> List[Dict]:
        """Get incoming links for a specific page."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            result = self.connection.execute("""
                MATCH (source:Page)-[l:Links]->(target:Page {url: $url})
                RETURN source.url as source_url, source.pagerank as source_pagerank,
                       source.out_degree as source_out_degree, l.anchor_text as anchor_text
            """, parameters={"url": url})

            links = []
            for row in result:
                links.append({
                    'source_url': row[0],
                    'source_pagerank': row[1] or 0.0,
                    'source_out_degree': row[2] or 0,
                    'anchor_text': row[3] or ""
                })

            return links

        except Exception as e:
            logger.error(f"Failed to get incoming links for {url}: {e}")
            return []

    def update_pagerank_scores(self, scores: Dict[str, float]) -> None:
        """Update PageRank scores for all pages."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            for url, score in scores.items():
                self.connection.execute("""
                    MATCH (p:Page {url: $url})
                    SET p.pagerank = $score
                """, parameters={"url": url, "score": score})

        except Exception as e:
            logger.error(f"Failed to update PageRank scores: {e}")
            raise

    def get_graph_stats(self) -> Dict:
        """Get basic graph statistics."""
        if not self.connection:
            raise RuntimeError("Database connection not established")

        try:
            # Get page count
            page_result = self.connection.execute("MATCH (p:Page) RETURN COUNT(p) as count")
            page_count = next(page_result)[0]

            # Get link count
            link_result = self.connection.execute("MATCH ()-[l:Links]->() RETURN COUNT(l) as count")
            link_count = next(link_result)[0]

            # Get domain count
            domain_result = self.connection.execute("MATCH (p:Page) RETURN COUNT(DISTINCT p.domain) as count")
            domain_count = next(domain_result)[0]

            return {
                'total_pages': page_count,
                'total_links': link_count,
                'total_domains': domain_count
            }

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {'total_pages': 0, 'total_links': 0, 'total_domains': 0}