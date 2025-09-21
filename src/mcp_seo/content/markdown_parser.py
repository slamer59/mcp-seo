#!/usr/bin/env python3
"""
Markdown Parser for Blog Content Analysis
========================================

A flexible markdown parser that extracts metadata, content, and internal links
from markdown files. Designed to be generic and work with any blog structure.

Features:
- Parse frontmatter metadata (title, keywords, dates, etc.)
- Extract internal links using various patterns
- Calculate content quality metrics
- Keyword extraction from content and headers
- Content structure analysis

Author: Extracted from GitAlchemy Kuzu PageRank Analyzer
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import frontmatter
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Parse markdown files and extract metadata, content, and internal links."""

    def __init__(
        self,
        content_dir: Union[str, Path],
        link_patterns: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the markdown parser.

        Args:
            content_dir: Directory containing markdown files
            link_patterns: Custom regex patterns for internal links
            file_extensions: File extensions to parse (default: ['.md', '.mdx'])
        """
        self.content_dir = Path(content_dir)
        self.posts = {}

        # Default link patterns - can be customized for different systems
        self.link_patterns = link_patterns or [
            r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",  # WikiLinks: [[filename|anchor]]
            r"\[([^\]]+)\]\(([^)]+\.md[^)]*)\)",  # Markdown links to .md files
            r"\[([^\]]+)\]\(([^)]+)\)",  # General markdown links
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.link_patterns]

        # File extensions to process
        self.file_extensions = file_extensions or [".md", ".mdx"]

    def parse_all_posts(
        self, recursive: bool = False, filter_published: bool = True
    ) -> Dict[str, Dict]:
        """
        Parse all markdown files in the content directory.

        Args:
            recursive: Whether to search subdirectories recursively
            filter_published: Whether to skip unpublished posts

        Returns:
            Dictionary of post data keyed by slug
        """
        self.posts = {}

        # Find all markdown files
        if recursive:
            markdown_files = []
            for ext in self.file_extensions:
                markdown_files.extend(self.content_dir.rglob(f"*{ext}"))
        else:
            markdown_files = []
            for ext in self.file_extensions:
                markdown_files.extend(self.content_dir.glob(f"*{ext}"))

        logger.info(f"Found {len(markdown_files)} markdown files to parse")

        for md_file in markdown_files:
            try:
                post_data = self._parse_single_post(md_file)
                if post_data:
                    # Filter published posts if requested
                    if filter_published and not post_data.get("published", True):
                        continue

                    self.posts[post_data["slug"]] = post_data

            except Exception as e:
                logger.error(f"Error parsing {md_file}: {e}")
                continue

        logger.info(f"Successfully parsed {len(self.posts)} blog posts")
        return self.posts

    def parse_single_file(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """Parse a single markdown file."""
        return self._parse_single_post(Path(file_path))

    def _parse_single_post(self, file_path: Path) -> Optional[Dict]:
        """Parse a single markdown file and extract metadata."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # Extract frontmatter
            metadata = post.metadata
            content = post.content

            # Extract basic metadata with fallbacks
            slug = self._extract_slug(metadata, file_path)
            title = self._extract_title(metadata, slug)

            # Calculate word count
            word_count = len(content.split())

            # Extract keywords
            keywords = self._extract_keywords(metadata, content, title)

            # Extract internal links
            internal_links = self._extract_internal_links(content)

            # Calculate content quality metrics
            quality_metrics = self._calculate_quality_metrics(content)

            # Extract additional metadata
            published = metadata.get("published", True)
            date = self._extract_date(metadata)

            return {
                "slug": slug,
                "title": title,
                "file_path": str(file_path),
                "date": date,
                "author": metadata.get("author", ""),
                "description": metadata.get("description", ""),
                "keywords": keywords,
                "word_count": word_count,
                "internal_links": internal_links,
                "published": published,
                "frontmatter": metadata,
                "content": content,
                "quality_metrics": quality_metrics,
                "relative_path": str(file_path.relative_to(self.content_dir)),
            }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_slug(self, metadata: Dict, file_path: Path) -> str:
        """Extract slug from metadata or filename."""
        # Try multiple common slug field names
        for field in ["slug", "id", "permalink", "url"]:
            if field in metadata and metadata[field]:
                slug = str(metadata[field])
                # Clean up slug
                return slug.replace("/", "-").replace(".md", "").strip("/")

        # Fallback to filename
        return file_path.stem

    def _extract_title(self, metadata: Dict, slug: str) -> str:
        """Extract title from metadata with fallback to slug."""
        # Try multiple common title field names
        for field in ["title", "name", "heading"]:
            if field in metadata and metadata[field]:
                return str(metadata[field])

        # Fallback to formatted slug
        return slug.replace("-", " ").replace("_", " ").title()

    def _extract_date(self, metadata: Dict) -> str:
        """Extract date from metadata."""
        # Try multiple common date field names
        for field in ["date", "published", "created", "publishDate"]:
            if field in metadata and metadata[field]:
                return str(metadata[field])
        return ""

    def _extract_keywords(self, metadata: Dict, content: str, title: str) -> List[str]:
        """Extract keywords from metadata and content."""
        keywords = set()

        # Keywords from frontmatter
        for field in ["keywords", "tags", "categories", "topics"]:
            if field in metadata:
                field_value = metadata[field]
                if isinstance(field_value, str):
                    # Handle comma-separated strings
                    keywords.update(
                        [k.strip().lower() for k in field_value.split(",") if k.strip()]
                    )
                elif isinstance(field_value, list):
                    keywords.update([str(k).strip().lower() for k in field_value if k])

        # Extract keywords from title
        title_words = re.findall(r"\b[a-zA-Z]{3,}\b", title.lower())
        keywords.update(title_words[:5])  # Take first 5 words from title

        # Extract from headers
        headers = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
        for header in headers:
            # Simple keyword extraction from headers
            header_words = re.findall(r"\b[a-zA-Z]{3,}\b", header.lower())
            keywords.update(header_words[:3])  # Take first 3 words from each header

        # Filter out common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "she",
            "use",
            "way",
            "will",
            "with",
            "this",
            "that",
            "have",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "like",
            "long",
            "make",
            "many",
            "over",
            "such",
            "take",
            "than",
            "them",
            "well",
            "were",
        }

        filtered_keywords = [
            kw for kw in keywords if kw not in stop_words and len(kw) > 2
        ]
        return sorted(list(set(filtered_keywords)))

    def _extract_internal_links(self, content: str) -> List[Dict[str, str]]:
        """Extract internal links using configured patterns."""
        links = []

        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)

            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        # Different patterns have different group arrangements
                        if "[[" in pattern.pattern:
                            # WikiLink pattern: [[target|anchor]]
                            target, anchor = match
                        else:
                            # Markdown pattern: [anchor](target)
                            anchor, target = match
                    else:
                        target = match[0] if match else ""
                        anchor = target
                else:
                    target = match
                    anchor = target

                # Clean up the target
                target_slug = self._clean_link_target(target)
                if not target_slug:
                    continue

                # Clean up anchor text
                if not anchor or anchor == target:
                    anchor = target_slug.replace("-", " ").replace("_", " ").title()

                # Skip external links
                if self._is_external_link(target):
                    continue

                links.append(
                    {
                        "target_slug": target_slug,
                        "anchor_text": anchor.strip(),
                        "original_target": target,
                    }
                )

        return links

    def _clean_link_target(self, target: str) -> str:
        """Clean and normalize link target."""
        if not target:
            return ""

        # Remove file extensions
        target = re.sub(r"\.(md|mdx)$", "", target)

        # Remove URL fragments and query parameters
        target = target.split("#")[0].split("?")[0]

        # Handle relative paths
        target = target.strip("./").strip("../")

        # Convert to slug format
        target = target.replace("/", "-").replace(" ", "-").lower()

        return target

    def _is_external_link(self, target: str) -> bool:
        """Check if a link target is external."""
        external_indicators = [
            "http://",
            "https://",
            "ftp://",
            "mailto:",
            "www.",
            ".com",
            ".org",
            ".net",
            ".io",
            ".dev",
        ]

        target_lower = target.lower()
        return any(indicator in target_lower for indicator in external_indicators)

    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Calculate content quality metrics."""
        # Remove HTML and get clean text
        soup = BeautifulSoup(content, "html.parser")
        clean_text = soup.get_text()

        word_count = len(clean_text.split())
        char_count = len(clean_text)

        # Count structural elements
        header_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))
        image_count = len(re.findall(r"!\[.*?\]\(.*?\)", content))
        code_block_count = len(re.findall(r"```", content)) // 2
        link_count = len(re.findall(r"\[.*?\]\(.*?\)", content))

        # Calculate readability metrics
        sentences = len(re.findall(r"[.!?]+", clean_text))
        avg_sentence_length = word_count / max(sentences, 1)

        # Simple readability score (inverse of complexity)
        readability_score = max(0, 100 - (avg_sentence_length * 2))

        # Content structure score
        structure_score = min(
            100, (header_count * 10) + (image_count * 5) + (code_block_count * 5)
        )

        return {
            "word_count": word_count,
            "char_count": char_count,
            "header_count": header_count,
            "image_count": image_count,
            "code_block_count": code_block_count,
            "link_count": link_count,
            "sentence_count": sentences,
            "avg_sentence_length": avg_sentence_length,
            "readability_score": readability_score,
            "structure_score": structure_score,
        }

    def get_content_statistics(self) -> Dict[str, any]:
        """Get overall statistics for parsed content."""
        if not self.posts:
            return {}

        total_posts = len(self.posts)
        total_words = sum(post["word_count"] for post in self.posts.values())
        total_links = sum(len(post["internal_links"]) for post in self.posts.values())

        # Keyword frequency analysis
        all_keywords = []
        for post in self.posts.values():
            all_keywords.extend(post["keywords"])

        keyword_freq = defaultdict(int)
        for keyword in all_keywords:
            keyword_freq[keyword] += 1

        return {
            "total_posts": total_posts,
            "total_words": total_words,
            "total_internal_links": total_links,
            "avg_words_per_post": total_words / total_posts,
            "avg_links_per_post": total_links / total_posts,
            "unique_keywords": len(keyword_freq),
            "most_common_keywords": sorted(
                keyword_freq.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "link_density": total_links / (total_posts * (total_posts - 1))
            if total_posts > 1
            else 0,
        }
