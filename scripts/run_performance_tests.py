#!/usr/bin/env python3
"""
Performance test runner for MCP SEO system.

This script provides different configurations for running performance tests:
- Quick performance tests (for CI/development)
- Full performance benchmarks (for release testing)
- Memory profiling tests
- Load testing scenarios

Usage:
    python scripts/run_performance_tests.py [--quick|--full|--memory|--load]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def run_quick_tests():
    """Run quick performance tests for development/CI."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py",
        "-v",
        "--benchmark-skip-save",
        "--benchmark-disable-gc",
        "--benchmark-max-time=30",
        "-m", "performance and not slow",
        "--tb=short"
    ]
    return run_command(cmd, "Quick Performance Tests")


def run_full_benchmarks():
    """Run comprehensive performance benchmarks."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py",
        "-v",
        "--benchmark-save=performance_results",
        "--benchmark-save-data",
        "--benchmark-histogram=benchmark_histogram",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds,iterations",
        "--benchmark-sort=mean",
        "-m", "performance or benchmark",
        "--tb=line"
    ]
    return run_command(cmd, "Full Performance Benchmarks")


def run_memory_tests():
    """Run memory usage and leak detection tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py",
        "-v",
        "-s",  # Show print statements for memory monitoring
        "-m", "memory",
        "--tb=short"
    ]
    return run_command(cmd, "Memory Usage Tests")


def run_load_tests():
    """Run load and stress tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py",
        "-v",
        "--benchmark-save=load_test_results",
        "-m", "slow and performance",
        "--tb=line",
        "--maxfail=3"  # Stop after 3 failures
    ]
    return run_command(cmd, "Load and Stress Tests")


def run_scaling_tests():
    """Run scaling tests to understand performance characteristics."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py::TestGraphAnalysisScaling",
        "-v",
        "--benchmark-save=scaling_results",
        "--benchmark-columns=min,max,mean,ops",
        "-s"
    ]
    return run_command(cmd, "Scaling Performance Tests")


def run_concurrent_tests():
    """Run concurrent processing tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_performance.py::TestConcurrentProcessing",
        "-v",
        "--benchmark-save=concurrent_results",
        "-s"
    ]
    return run_command(cmd, "Concurrent Processing Tests")


def generate_performance_report():
    """Generate a performance report from benchmark results."""
    print("\n" + "="*60)
    print("Performance Test Report")
    print("="*60)

    # Check for benchmark results
    benchmark_files = list(Path(".").glob(".benchmarks/**/*.json"))

    if benchmark_files:
        print(f"Found {len(benchmark_files)} benchmark result files:")
        for file in benchmark_files:
            print(f"  - {file}")

        # Try to show latest results
        try:
            cmd = [
                "python", "-m", "pytest",
                "--benchmark-compare",
                "--benchmark-compare-fail=min:5%,max:10%,mean:5%",
                "--tb=no"
            ]
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Could not generate comparison report: {e}")
    else:
        print("No benchmark results found. Run benchmarks first.")

    print("\nTips for interpreting results:")
    print("- Look for operations per second (ops) for throughput")
    print("- Check mean and median times for typical performance")
    print("- Watch for high standard deviation indicating inconsistent performance")
    print("- Memory tests should show stable memory usage without leaks")


def main():
    """Main entry point for performance test runner."""
    parser = argparse.ArgumentParser(description="Run MCP SEO performance tests")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick performance tests (for development)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full performance benchmarks"
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="Run memory usage tests"
    )
    parser.add_argument(
        "--load", action="store_true",
        help="Run load and stress tests"
    )
    parser.add_argument(
        "--scaling", action="store_true",
        help="Run scaling performance tests"
    )
    parser.add_argument(
        "--concurrent", action="store_true",
        help="Run concurrent processing tests"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate performance report from existing results"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all performance tests"
    )

    args = parser.parse_args()

    # Check if pytest-benchmark is available
    try:
        import pytest_benchmark
    except ImportError:
        print("❌ pytest-benchmark not found. Install with:")
        print("   pip install pytest-benchmark")
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("tests/test_performance.py").exists():
        print("❌ test_performance.py not found. Run from project root.")
        sys.exit(1)

    success = True

    if args.report:
        generate_performance_report()
        return

    if args.quick or args.all:
        success &= run_quick_tests()

    if args.memory or args.all:
        success &= run_memory_tests()

    if args.scaling or args.all:
        success &= run_scaling_tests()

    if args.concurrent or args.all:
        success &= run_concurrent_tests()

    if args.load or args.all:
        success &= run_load_tests()

    if args.full or args.all:
        success &= run_full_benchmarks()

    # If no specific test type was requested, run quick tests
    if not any([args.quick, args.full, args.memory, args.load,
                args.scaling, args.concurrent, args.all]):
        print("No test type specified. Running quick performance tests...")
        success = run_quick_tests()

    # Generate report if we ran any tests
    if not args.report:
        generate_performance_report()

    if success:
        print("\n✅ All performance tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some performance tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()