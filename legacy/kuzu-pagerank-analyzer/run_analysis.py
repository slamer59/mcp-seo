#!/usr/bin/env python3
"""
Simple runner script for the Kuzu PageRank SEO Analyzer
"""

import os
import sys
import subprocess

def main():
    """Run the analysis tool"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(script_dir, 'src')
    
    # Default arguments
    args = [
        sys.executable, '-m', 'kuzu_pagerank_analyzer.main',
        '--blog-dir', '../../src/content/blog',
        '--output-dir', './analysis-results',
        '--export-format', 'both'
    ]
    
    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])
    
    # Activate virtual environment
    venv_python = os.path.join(script_dir, '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        args[0] = venv_python
    
    # Run the analysis
    try:
        result = subprocess.run(args, env=env, check=True)
        print("\n✅ Analysis completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 130

if __name__ == '__main__':
    sys.exit(main())