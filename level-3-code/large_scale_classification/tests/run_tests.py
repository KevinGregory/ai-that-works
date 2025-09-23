"""Test runner script for the large-scale classification system.

This script provides an easy way to run different types of tests:
- Integration tests (narrowing accuracy, selection accuracy, full pipeline)
- Unit tests (individual components)
- Performance benchmarks

Usage:
    python tests/run_tests.py --narrowing-accuracy
    python tests/run_tests.py --selection-accuracy
    python tests/run_tests.py --all
"""

import argparse
import sys
from pathlib import Path

import dotenv

# Add the project root and src directory to the Python path so we can import from tests package
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))  # Add src first so baml_client imports from local
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
dotenv.load_dotenv(project_root / ".env")



def run_narrowing_accuracy_test():
    """Run the narrowing accuracy integration test."""
    print("Running Narrowing Accuracy Test...")
    from tests.integration.test_narrowing_accuracy import main as narrowing_test_main

    narrowing_test_main()


def run_selection_accuracy_test():
    """Run the selection accuracy integration test."""
    print("Running Selection Accuracy Test...")
    from tests.integration.test_selection_accuracy import main as selection_test_main

    selection_test_main()


def run_pipeline_accuracy_test():
    """Run the complete pipeline accuracy integration test."""
    print("Running Pipeline Accuracy Test...")
    from tests.integration.test_pipeline_accuracy import main as pipeline_test_main

    pipeline_test_main()


def run_all_tests():
    """Run all available tests."""
    print("Running All Tests")
    print("=" * 60)

    # Run narrowing accuracy test
    run_narrowing_accuracy_test()

    print("\n" + "=" * 60)

    # Run selection accuracy test
    run_selection_accuracy_test()

    print("\n" + "=" * 60)

    # Run pipeline accuracy test
    run_pipeline_accuracy_test()

    print("\n" + "=" * 60)
    print("All test results have been saved to JSON files in tests/results/")
    print("   - Narrowing results: tests/results/narrowing/")
    print("   - Selection results: tests/results/selection/")
    print("   - Pipeline results: tests/results/pipeline/")
    print("   Use these files for detailed analysis and comparison across test runs.")


def main():
    """Run the tests."""
    parser = argparse.ArgumentParser(description="Test runner for large-scale classification system")

    parser.add_argument(
        "--narrowing-accuracy",
        action="store_true",
        help="Run narrowing strategy accuracy tests",
    )

    parser.add_argument("--selection-accuracy", action="store_true", help="Run selection accuracy tests")

    parser.add_argument("--pipeline-accuracy", action="store_true", help="Run complete pipeline accuracy tests")

    parser.add_argument("--all", action="store_true", help="Run all available tests")

    args = parser.parse_args()

    if args.narrowing_accuracy:
        run_narrowing_accuracy_test()
    elif args.selection_accuracy:
        run_selection_accuracy_test()
    elif args.pipeline_accuracy:
        run_pipeline_accuracy_test()
    elif args.all:
        run_all_tests()
    else:
        # Default: run all tests
        print("No specific test specified. Running all tests...")
        run_all_tests()


if __name__ == "__main__":
    main()
