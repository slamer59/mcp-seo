"""
Performance test configuration for MCP SEO system.

This module contains configuration constants and utilities for performance testing.
Adjust these values based on your testing environment and requirements.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PerformanceConfig:
    """Configuration for performance tests."""

    # Dataset sizes for testing
    SMALL_DATASET_SIZE: int = 100
    MEDIUM_DATASET_SIZE: int = 500
    LARGE_DATASET_SIZE: int = 1000
    XLARGE_DATASET_SIZE: int = 5000

    # Link density (links per page ratio)
    LOW_LINK_DENSITY: float = 0.5
    MEDIUM_LINK_DENSITY: float = 2.0
    HIGH_LINK_DENSITY: float = 5.0

    # Performance thresholds (in seconds)
    FAST_OPERATION_THRESHOLD: float = 1.0
    MEDIUM_OPERATION_THRESHOLD: float = 5.0
    SLOW_OPERATION_THRESHOLD: float = 15.0

    # Memory thresholds (in MB)
    LOW_MEMORY_THRESHOLD: float = 50.0
    MEDIUM_MEMORY_THRESHOLD: float = 200.0
    HIGH_MEMORY_THRESHOLD: float = 500.0

    # Concurrency settings
    LOW_CONCURRENCY: int = 2
    MEDIUM_CONCURRENCY: int = 5
    HIGH_CONCURRENCY: int = 10

    # API rate limiting
    API_REQUESTS_PER_MINUTE: int = 100
    API_BURST_SIZE: int = 10

    # Database performance
    DB_INSERT_BATCH_SIZE: int = 100
    DB_QUERY_TIMEOUT: float = 30.0

    # Benchmark settings
    BENCHMARK_ROUNDS: int = 3
    BENCHMARK_ITERATIONS: int = 1
    BENCHMARK_WARMUP_ROUNDS: int = 1

    # Test timeouts
    QUICK_TEST_TIMEOUT: float = 30.0
    STANDARD_TEST_TIMEOUT: float = 120.0
    LONG_TEST_TIMEOUT: float = 300.0

    @classmethod
    def from_environment(cls) -> 'PerformanceConfig':
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        env_mappings = {
            'PERF_LARGE_DATASET_SIZE': 'LARGE_DATASET_SIZE',
            'PERF_XLARGE_DATASET_SIZE': 'XLARGE_DATASET_SIZE',
            'PERF_HIGH_CONCURRENCY': 'HIGH_CONCURRENCY',
            'PERF_API_REQUESTS_PER_MINUTE': 'API_REQUESTS_PER_MINUTE',
            'PERF_BENCHMARK_ROUNDS': 'BENCHMARK_ROUNDS',
            'PERF_LONG_TEST_TIMEOUT': 'LONG_TEST_TIMEOUT',
        }

        for env_var, attr_name in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                if hasattr(config, attr_name):
                    current_value = getattr(config, attr_name)
                    if isinstance(current_value, int):
                        setattr(config, attr_name, int(value))
                    elif isinstance(current_value, float):
                        setattr(config, attr_name, float(value))

        return config

    def get_dataset_size(self, size_category: str) -> int:
        """Get dataset size for a given category."""
        sizes = {
            'small': self.SMALL_DATASET_SIZE,
            'medium': self.MEDIUM_DATASET_SIZE,
            'large': self.LARGE_DATASET_SIZE,
            'xlarge': self.XLARGE_DATASET_SIZE,
        }
        return sizes.get(size_category.lower(), self.MEDIUM_DATASET_SIZE)

    def get_link_density(self, density_category: str) -> float:
        """Get link density for a given category."""
        densities = {
            'low': self.LOW_LINK_DENSITY,
            'medium': self.MEDIUM_LINK_DENSITY,
            'high': self.HIGH_LINK_DENSITY,
        }
        return densities.get(density_category.lower(), self.MEDIUM_LINK_DENSITY)

    def get_concurrency_level(self, level_category: str) -> int:
        """Get concurrency level for a given category."""
        levels = {
            'low': self.LOW_CONCURRENCY,
            'medium': self.MEDIUM_CONCURRENCY,
            'high': self.HIGH_CONCURRENCY,
        }
        return levels.get(level_category.lower(), self.MEDIUM_CONCURRENCY)

    def is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        return any(key in os.environ for key in ['CI', 'GITHUB_ACTIONS', 'JENKINS_URL'])

    def get_ci_adjusted_config(self) -> 'PerformanceConfig':
        """Get configuration adjusted for CI environment limitations."""
        if not self.is_ci_environment():
            return self

        # Create a copy with reduced requirements for CI
        ci_config = PerformanceConfig()
        ci_config.LARGE_DATASET_SIZE = min(self.LARGE_DATASET_SIZE, 500)
        ci_config.XLARGE_DATASET_SIZE = min(self.XLARGE_DATASET_SIZE, 1000)
        ci_config.HIGH_CONCURRENCY = min(self.HIGH_CONCURRENCY, 3)
        ci_config.LONG_TEST_TIMEOUT = min(self.LONG_TEST_TIMEOUT, 180.0)
        ci_config.BENCHMARK_ROUNDS = min(self.BENCHMARK_ROUNDS, 2)

        return ci_config


class PerformanceDataGenerator:
    """Utility class for generating performance test data."""

    @staticmethod
    def generate_pages(count: int, domain_count: int = 3) -> list:
        """Generate page data for performance testing."""
        import random

        domains = [f"test-domain-{i}.com" for i in range(domain_count)]
        pages = []

        for i in range(count):
            domain = random.choice(domains)
            page_type = random.choice(['page', 'post', 'category', 'tag'])
            pages.append({
                'url': f"https://{domain}/{page_type}-{i}",
                'title': f"{domain.replace('-', ' ').title()} {page_type.title()} {i}",
                'status_code': 200 if random.random() > 0.05 else random.choice([404, 500, 301]),
                'content_length': random.randint(500, 10000),
                'domain': domain,
                'path': f"/{page_type}-{i}"
            })

        return pages

    @staticmethod
    def generate_links(pages: list, density: float = 2.0) -> list:
        """Generate link data for performance testing."""
        import random

        links = []
        total_links = int(len(pages) * density)

        for _ in range(total_links):
            source = random.choice(pages)
            target = random.choice(pages)

            # Avoid self-links
            if source['url'] != target['url']:
                # Bias towards internal links
                if (source['domain'] == target['domain'] or
                    random.random() < 0.8):  # 80% internal links
                    links.append((
                        source['url'],
                        target['url'],
                        f"Link to {target['title'][:20]}..."
                    ))

        return links

    @staticmethod
    def generate_keywords(count: int) -> list:
        """Generate keyword data for performance testing."""
        import random

        base_keywords = [
            'seo', 'optimization', 'ranking', 'keywords', 'content',
            'marketing', 'search', 'engine', 'analytics', 'performance',
            'website', 'traffic', 'conversion', 'organic', 'backlinks'
        ]

        keywords = []
        for i in range(count):
            # Create compound keywords
            if random.random() < 0.3:  # 30% compound keywords
                keyword = f"{random.choice(base_keywords)} {random.choice(base_keywords)}"
            else:
                keyword = f"{random.choice(base_keywords)} {i}"

            keywords.append({
                'keyword': keyword,
                'search_volume': random.randint(10, 10000),
                'competition': random.uniform(0.1, 1.0),
                'cpc': random.uniform(0.1, 5.0)
            })

        return keywords


class PerformanceMetrics:
    """Class for tracking and analyzing performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}

    def start_timing(self, operation: str):
        """Start timing an operation."""
        import time
        self.start_times[operation] = time.time()

    def end_timing(self, operation: str):
        """End timing an operation and record the duration."""
        import time
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_duration"] = duration
            del self.start_times[operation]
            return duration
        return None

    def record_metric(self, name: str, value: Any):
        """Record a performance metric."""
        self.metrics[name] = value

    def get_metric(self, name: str) -> Any:
        """Get a recorded metric."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()

    def calculate_throughput(self, operation: str, count: int) -> float:
        """Calculate throughput for an operation."""
        duration = self.get_metric(f"{operation}_duration")
        if duration and duration > 0:
            return count / duration
        return 0.0

    def analyze_performance(self, config: PerformanceConfig) -> Dict[str, str]:
        """Analyze performance against configuration thresholds."""
        analysis = {}

        for metric_name, value in self.metrics.items():
            if metric_name.endswith('_duration'):
                operation = metric_name.replace('_duration', '')
                if value <= config.FAST_OPERATION_THRESHOLD:
                    analysis[operation] = "FAST"
                elif value <= config.MEDIUM_OPERATION_THRESHOLD:
                    analysis[operation] = "MEDIUM"
                elif value <= config.SLOW_OPERATION_THRESHOLD:
                    analysis[operation] = "SLOW"
                else:
                    analysis[operation] = "TOO_SLOW"

        return analysis


# Global configuration instance
perf_config = PerformanceConfig.from_environment()

# CI-adjusted configuration
ci_config = perf_config.get_ci_adjusted_config()