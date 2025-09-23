"""Test script to evaluate the accuracy of category selection.

This script tests how often the correct category is selected by the LLM
from the narrowed candidate categories. It provides detailed metrics
and analysis to help optimize the selection process.
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dotenv

from src.classification.embeddings import EmbeddingService
from src.classification.narrowing import HybridNarrowing
from src.classification.selection import CategorySelector
from src.data.category_loader import CategoryLoader
from src.data.models import Category
from src.shared import constants as C
from tests.data.test_cases import TestCase, tests

dotenv.load_dotenv()

@dataclass
class SelectionResult:
    """Result of a single selection test."""

    test_case: TestCase
    candidate_categories: list[Category]
    selected_category: Category
    correct_selection: bool
    processing_time_ms: float
    candidate_count: int


@dataclass
class SelectionResults:
    """Aggregated results for selection testing."""

    total_tests: int
    correct_selections: int
    accuracy_percent: float
    avg_candidate_count: float
    avg_processing_time_ms: float
    results: list[SelectionResult]


class SelectionAccuracyTester:
    """Test harness for evaluating selection accuracy."""

    def __init__(self):
        """Initialize the tester with required components."""
        self.category_loader = CategoryLoader()
        self.selector = CategorySelector()
        self.categories = self.category_loader.load_categories()
        
        # Initialize narrowing strategy to generate candidate categories
        embedding_service = EmbeddingService()
        self.narrower = HybridNarrowing(embedding_service=embedding_service)

        # Create category lookup for validation
        self.category_lookup = {cat.path: cat for cat in self.categories}

        print(f"Loaded {len(self.categories)} categories for testing")
        print("-" * 60)


    def test_selection(self) -> SelectionResults:
        """Test selection accuracy against all test cases.

        Returns:
            Aggregated results for selection testing
        """
        results = []

        print("\n🎯 Testing Category Selection Accuracy")
        print("=" * 50)

        for i, test_case in enumerate(tests, 1):
            # First, use narrowing to generate candidate categories
            try:
                narrowing_start_time = time.time()
                candidate_categories = self.narrower.narrow(test_case["text"], self.categories)
                narrowing_time_ms = (time.time() - narrowing_start_time) * 1000
            except Exception as e:
                print(f"❌ Narrowing failed for test case {i}: {e}")
                continue

            if not candidate_categories:
                print(f"❌ No candidate categories found for test case {i}")
                continue

            # Run selection on the narrowed candidates
            try:
                selection_start_time = time.time()
                selected_category = self.selector.select_best_category(test_case["text"], candidate_categories)
                selection_time_ms = (time.time() - selection_start_time) * 1000
                processing_time_ms = narrowing_time_ms + selection_time_ms
            except Exception as e:
                print(f"❌ Selection failed for test case {i}: {e}")
                continue

            # Check if correct category was selected
            expected_category_path = test_case["category"]
            correct_selection = selected_category.path == expected_category_path

            result = SelectionResult(
                test_case=test_case,
                candidate_categories=candidate_categories,
                selected_category=selected_category,
                correct_selection=correct_selection,
                processing_time_ms=processing_time_ms,
                candidate_count=len(candidate_categories),
            )
            results.append(result)

            # Print progress
            status = "✅" if correct_selection else "❌"
            print(f"{i:2d}. {status} {test_case['text'][:60]}...")
            print(f"    Expected: {expected_category_path}")
            print(f"    Selected: {selected_category.path}")
            print(f"    Candidates: {len(candidate_categories)} ({processing_time_ms:.1f}ms)")
            if not correct_selection:
                print("    ⚠️  Incorrect selection!")
                candidate_paths = [cat.path for cat in candidate_categories]
                print(f"    Available: {candidate_paths}")
            print()

        if not results:
            raise ValueError("No valid test results generated")

        # Calculate aggregate metrics
        correct_selections = sum(1 for r in results if r.correct_selection)
        accuracy_percent = (correct_selections / len(results)) * 100
        avg_candidate_count = sum(r.candidate_count for r in results) / len(results)
        avg_processing_time_ms = sum(r.processing_time_ms for r in results) / len(results)

        return SelectionResults(
            total_tests=len(results),
            correct_selections=correct_selections,
            accuracy_percent=accuracy_percent,
            avg_candidate_count=avg_candidate_count,
            avg_processing_time_ms=avg_processing_time_ms,
            results=results,
        )

    def analyze_failures(self, selection_results: SelectionResults) -> None:
        """Analyze and report on failed selections.

        Args:
            selection_results: Results to analyze
        """
        failures = [r for r in selection_results.results if not r.correct_selection]

        if not failures:
            print("No selection failures to analyze!")
            return

        print("\nSelection Failure Analysis")
        print("=" * 50)

        print(f"Failed {len(failures)} out of {selection_results.total_tests} selections:")
        print()

        # Analyze failure patterns
        for i, failure in enumerate(failures, 1):
            print(f"{i}. Expected: {failure.test_case['category']}")
            print(f"   Selected: {failure.selected_category.path}")
            print(f"   Text: {failure.test_case['text'][:50]}...")
            print(f"   Candidates ({len(failure.candidate_categories)}):")
            for j, cat in enumerate(failure.candidate_categories, 1):
                marker = "👈 SELECTED" if cat.path == failure.selected_category.path else ""
                marker += " ✓ CORRECT" if cat.path == failure.test_case["category"] else ""
                print(f"     {j}. {cat.path} {marker}")
            print()

    def print_summary(self, selection_results: SelectionResults) -> None:
        """Print a summary of selection results.

        Args:
            selection_results: Results to summarize
        """
        print("\nSelection Accuracy Summary")
        print("=" * 50)

        print(f"Total Tests: {selection_results.total_tests}")
        print(f"Correct Selections: {selection_results.correct_selections}")
        print(f"Accuracy: {selection_results.accuracy_percent:.1f}%")
        print(f"Average Candidates per Test: {selection_results.avg_candidate_count:.1f}")
        print(f"Average Processing Time: {selection_results.avg_processing_time_ms:.1f}ms")

        # Distribution of candidate counts
        candidate_counts = [r.candidate_count for r in selection_results.results]
        from collections import Counter

        count_distribution = Counter(candidate_counts)

        print("\nCandidate Count Distribution:")
        for count in sorted(count_distribution.keys()):
            freq = count_distribution[count]
            print(f"  {count} candidates: {freq} tests")

    def run_test(self) -> SelectionResults:
        """Run the complete selection accuracy test.

        Returns:
            Selection test results
        """
        results = self.test_selection()
        self.analyze_failures(results)
        self.print_summary(results)
        return results

    def save_results_to_json(self, results: SelectionResults) -> Path:
        """Save test results to a JSON file with timestamp.

        Args:
            results: Selection test results to save

        Returns:
            Path to the saved JSON file
        """
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parents[1] / C.RESULTS / C.SELECTION
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selection_accuracy_{timestamp}.json"
        filepath = results_dir / filename

        # Prepare data for JSON serialization
        json_data = {
            "test_info": {
                "test_type": "selection_accuracy",
                "timestamp": datetime.now().isoformat(),
                "total_categories": len(self.categories),
                "total_test_cases": len(tests),
            },
            "results": {
                "total_tests": results.total_tests,
                "correct_selections": results.correct_selections,
                "accuracy_percent": results.accuracy_percent,
                "avg_candidate_count": results.avg_candidate_count,
                "avg_processing_time_ms": results.avg_processing_time_ms,
                "individual_results": [],
            },
        }

        # Convert individual results to serializable format
        for result in results.results:
            result_dict = {
                "test_case": result.test_case,
                "candidate_categories": [
                    {"path": cat.path, "name": cat.name, "description": cat.llm_description}
                    for cat in result.candidate_categories
                ],
                "selected_category": {
                    "path": result.selected_category.path,
                    "name": result.selected_category.name,
                    "description": result.selected_category.llm_description,
                },
                "correct_selection": result.correct_selection,
                "processing_time_ms": result.processing_time_ms,
                "candidate_count": result.candidate_count,
            }
            json_data["results"]["individual_results"].append(result_dict)

        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return filepath


def main():
    """Execute the test."""
    print("Category Selection Accuracy Test")
    print("=" * 60)
    print("This test evaluates how often the correct category is selected")
    print("by the LLM from the narrowed candidate categories.")
    print()

    tester = SelectionAccuracyTester()
    results = tester.run_test()

    # Save results to JSON file
    json_filepath = tester.save_results_to_json(results)

    print("\nSelection Testing Complete!")
    print("=" * 60)

    # Print final summary
    print(
        f"Selection Accuracy: {results.correct_selections}/{results.total_tests} "
        f"({results.accuracy_percent:.1f}%) correct selections"
    )

    print(f"\nResults saved to: {json_filepath}")
    print("   Use this file for detailed analysis and comparison with other test runs.")


if __name__ == "__main__":
    main()
