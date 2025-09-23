"""Test script to evaluate the accuracy of the complete classification pipeline.

This script tests the full end-to-end classification pipeline by running
the complete classify() method on all test cases. It provides comprehensive
metrics including accuracy, timing, and detailed failure analysis.

Usage:
    python test_pipeline_accuracy.py [--save-as RUN_NAME] [--description "Description"]
    
Examples:
    python test_pipeline_accuracy.py --save-as v1 --description "Baseline hybrid strategy"
    python test_pipeline_accuracy.py --save-as embedding_only --description "Embedding-only strategy test"
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dotenv

from src.classification.pipeline import ClassificationPipeline
from src.data.models import ClassificationResult
from src.shared import constants as C
from tests.data.test_cases import TestCase, tests

dotenv.load_dotenv()

@dataclass
class PipelineResult:
    """Result of a single pipeline test."""

    test_case: TestCase
    classification_result: ClassificationResult
    correct_classification: bool
    processing_time_ms: float
    narrowed_count: int
    narrowing_time_ms: float
    selection_time_ms: float
    narrowing_strategy: str
    vector_store_enabled: bool


@dataclass
class PipelineResults:
    """Aggregated results for pipeline testing."""

    total_tests: int
    correct_classifications: int
    accuracy_percent: float
    avg_narrowed_count: float
    avg_processing_time_ms: float
    avg_narrowing_time_ms: float
    avg_selection_time_ms: float
    results: list[PipelineResult]


class PipelineAccuracyTester:
    """Test harness for evaluating complete pipeline accuracy."""

    def __init__(self, use_vector_store: bool = True):
        """Initialize the tester with required components.
        
        Args:
            use_vector_store: Whether to use vector store for caching embeddings
        """
        self.pipeline = ClassificationPipeline(use_vector_store=use_vector_store)
        self.use_vector_store = use_vector_store

        print(f"Initialized Classification Pipeline (vector_store={'enabled' if use_vector_store else 'disabled'})")
        print(f"Running tests on {len(tests)} test cases")
        print("-" * 60)

    def test_pipeline(self) -> PipelineResults:
        """Test complete pipeline accuracy against all test cases.

        Returns:
            Aggregated results for pipeline testing
        """
        results = []

        print("\n🚀 Testing Complete Classification Pipeline")
        print("=" * 60)

        for i, test_case in enumerate(tests, 1):
            try:
                start_time = time.time()
                classification_result = self.pipeline.classify(test_case["text"])
                processing_time_ms = (time.time() - start_time) * 1000

                # Extract metadata
                metadata = classification_result.metadata
                narrowed_count = metadata.get(C.NARROWED_TO, 0)
                narrowing_time_ms = metadata.get(C.NARROWING_TIME_MS, 0)
                selection_time_ms = metadata.get(C.SELECTION_TIME_MS, 0)
                narrowing_strategy = metadata.get(C.NARROWING_STRATEGY, "unknown")
                vector_store_enabled = metadata.get(C.VECTOR_STORE_ENABLED, False)

                # Check if correct category was selected
                expected_category_path = test_case["category"]
                correct_classification = classification_result.category.path == expected_category_path

                result = PipelineResult(
                    test_case=test_case,
                    classification_result=classification_result,
                    correct_classification=correct_classification,
                    processing_time_ms=processing_time_ms,
                    narrowed_count=narrowed_count,
                    narrowing_time_ms=narrowing_time_ms,
                    selection_time_ms=selection_time_ms,
                    narrowing_strategy=narrowing_strategy,
                    vector_store_enabled=vector_store_enabled,
                )
                results.append(result)

                # Print progress
                status = "✅" if correct_classification else "❌"
                print(f"{i:2d}. {status} {test_case['text'][:60]}...")
                print(f"    Expected: {expected_category_path}")
                print(f"    Selected: {classification_result.category.path}")
                print(f"    Pipeline: {narrowed_count} candidates → {processing_time_ms:.1f}ms total")
                print(f"             (narrowing: {narrowing_time_ms:.1f}ms, selection: {selection_time_ms:.1f}ms)")
                
                if not correct_classification:
                    print("    ⚠️  Incorrect classification!")
                    candidate_paths = [cat.path for cat in classification_result.candidates]
                    correct_in_candidates = expected_category_path in candidate_paths
                    if correct_in_candidates:
                        print("    📍 Correct category WAS in candidates (selection error)")
                    else:
                        print("    📍 Correct category NOT in candidates (narrowing error)")
                    print(f"    Available: {candidate_paths}")
                print()

            except Exception as e:
                print(f"❌ Pipeline failed for test case {i}: {e}")
                continue

        if not results:
            raise ValueError("No valid test results generated")

        # Calculate aggregate metrics
        correct_classifications = sum(1 for r in results if r.correct_classification)
        accuracy_percent = (correct_classifications / len(results)) * 100
        avg_narrowed_count = sum(r.narrowed_count for r in results) / len(results)
        avg_processing_time_ms = sum(r.processing_time_ms for r in results) / len(results)
        avg_narrowing_time_ms = sum(r.narrowing_time_ms for r in results) / len(results)
        avg_selection_time_ms = sum(r.selection_time_ms for r in results) / len(results)

        return PipelineResults(
            total_tests=len(results),
            correct_classifications=correct_classifications,
            accuracy_percent=accuracy_percent,
            avg_narrowed_count=avg_narrowed_count,
            avg_processing_time_ms=avg_processing_time_ms,
            avg_narrowing_time_ms=avg_narrowing_time_ms,
            avg_selection_time_ms=avg_selection_time_ms,
            results=results,
        )

    def analyze_failures(self, pipeline_results: PipelineResults) -> None:
        """Analyze and report on failed classifications.

        Args:
            pipeline_results: Results to analyze
        """
        failures = [r for r in pipeline_results.results if not r.correct_classification]

        if not failures:
            print("No classification failures to analyze!")
            return

        print("\nPipeline Failure Analysis")
        print("=" * 60)

        # Categorize failures by type
        narrowing_failures = []  # Correct category not in candidates
        selection_failures = []  # Correct category in candidates but not selected

        for failure in failures:
            expected_path = failure.test_case["category"]
            candidate_paths = [cat.path for cat in failure.classification_result.candidates]
            
            if expected_path in candidate_paths:
                selection_failures.append(failure)
            else:
                narrowing_failures.append(failure)

        print(f"Total Failures: {len(failures)} out of {pipeline_results.total_tests} tests")
        print(f"  • Narrowing Failures: {len(narrowing_failures)} (correct category not found in candidates)")
        print(f"  • Selection Failures: {len(selection_failures)} (correct category in candidates but not selected)")
        print()

        # Analyze narrowing failures by category hierarchy level
        if narrowing_failures:
            print("Narrowing Failures by Category Level:")
            level_failures = defaultdict(list)
            for failure in narrowing_failures:
                expected_path = failure.test_case["category"]
                level = expected_path.count("/") - 1
                level_failures[level].append(failure)

            for level in sorted(level_failures.keys()):
                count = len(level_failures[level])
                print(f"  Level {level}: {count} failures")
                
                # Show examples
                for failure in level_failures[level][:2]:  # Show first 2 examples
                    print(f"    - {failure.test_case['category']}")
                    print(f"      Text: {failure.test_case['text'][:50]}...")
            print()

        # Analyze selection failures
        if selection_failures:
            print("Selection Failures (correct category was available):")
            for i, failure in enumerate(selection_failures[:5], 1):  # Show first 5
                print(f"  {i}. Expected: {failure.test_case['category']}")
                print(f"     Selected: {failure.classification_result.category.path}")
                print(f"     Text: {failure.test_case['text'][:50]}...")
                print(f"     Candidates ({len(failure.classification_result.candidates)}):")
                for j, cat in enumerate(failure.classification_result.candidates, 1):
                    marker = "👈 SELECTED" if cat.path == failure.classification_result.category.path else ""
                    marker += " ✓ CORRECT" if cat.path == failure.test_case["category"] else ""
                    print(f"       {j}. {cat.path} {marker}")
                print()

    def print_summary(self, pipeline_results: PipelineResults) -> None:
        """Print a comprehensive summary of pipeline results.

        Args:
            pipeline_results: Results to summarize
        """
        print("\nPipeline Accuracy Summary")
        print("=" * 60)

        print(f"Total Tests: {pipeline_results.total_tests}")
        print(f"Correct Classifications: {pipeline_results.correct_classifications}")
        print(f"Overall Accuracy: {pipeline_results.accuracy_percent:.1f}%")
        print()

        print("Performance Metrics:")
        print(f"  Average Candidates per Test: {pipeline_results.avg_narrowed_count:.1f}")
        print(f"  Average Total Time: {pipeline_results.avg_processing_time_ms:.1f}ms")
        print(f"    - Narrowing: {pipeline_results.avg_narrowing_time_ms:.1f}ms ({pipeline_results.avg_narrowing_time_ms/pipeline_results.avg_processing_time_ms*100:.1f}%)")
        print(f"    - Selection: {pipeline_results.avg_selection_time_ms:.1f}ms ({pipeline_results.avg_selection_time_ms/pipeline_results.avg_processing_time_ms*100:.1f}%)")
        print()

        # Test type breakdown
        llm_generated = [r for r in pipeline_results.results if r.test_case["test_type"] == "llm_generated"]
        human_generated = [r for r in pipeline_results.results if r.test_case["test_type"] == "human_generated"]

        if llm_generated:
            llm_accuracy = sum(1 for r in llm_generated if r.correct_classification) / len(llm_generated) * 100
            print(f"LLM Generated Tests: {len(llm_generated)} tests, {llm_accuracy:.1f}% accuracy")

        if human_generated:
            human_accuracy = sum(1 for r in human_generated if r.correct_classification) / len(human_generated) * 100
            print(f"Human Generated Tests: {len(human_generated)} tests, {human_accuracy:.1f}% accuracy")

        # Configuration info
        sample_result = pipeline_results.results[0] if pipeline_results.results else None
        if sample_result:
            print(f"\nConfiguration:")
            print(f"  Narrowing Strategy: {sample_result.narrowing_strategy}")
            print(f"  Vector Store: {'Enabled' if sample_result.vector_store_enabled else 'Disabled'}")

    def run_test(self) -> PipelineResults:
        """Run the complete pipeline accuracy test.

        Returns:
            Pipeline test results
        """
        results = self.test_pipeline()
        self.analyze_failures(results)
        self.print_summary(results)
        return results

    def save_results_to_json(self, results: PipelineResults) -> Path:
        """Save test results to a JSON file with timestamp.

        Args:
            results: Pipeline test results to save

        Returns:
            Path to the saved JSON file
        """
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parents[1] / C.RESULTS / "pipeline"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_accuracy_{timestamp}.json"
        filepath = results_dir / filename

        # Get configuration info from first result
        sample_result = results.results[0] if results.results else None
        narrowing_strategy = sample_result.narrowing_strategy if sample_result else "unknown"
        vector_store_enabled = sample_result.vector_store_enabled if sample_result else False

        # Prepare data for JSON serialization
        json_data = {
            "test_info": {
                "test_type": "pipeline_accuracy",
                "timestamp": datetime.now().isoformat(),
                "total_test_cases": len(tests),
                "narrowing_strategy": narrowing_strategy,
                "vector_store_enabled": vector_store_enabled,
            },
            "results": {
                "total_tests": results.total_tests,
                "correct_classifications": results.correct_classifications,
                "accuracy_percent": results.accuracy_percent,
                "avg_narrowed_count": results.avg_narrowed_count,
                "avg_processing_time_ms": results.avg_processing_time_ms,
                "avg_narrowing_time_ms": results.avg_narrowing_time_ms,
                "avg_selection_time_ms": results.avg_selection_time_ms,
                "individual_results": [],
            },
        }

        # Convert individual results to serializable format
        for result in results.results:
            result_dict = {
                "test_case": result.test_case,
                "selected_category": {
                    "path": result.classification_result.category.path,
                    "name": result.classification_result.category.name,
                    "description": result.classification_result.category.llm_description,
                },
                "candidate_categories": [
                    {"path": cat.path, "name": cat.name, "description": cat.llm_description}
                    for cat in result.classification_result.candidates
                ],
                "embedding_candidates": [
                    {"path": cat.path, "name": cat.name, "description": cat.llm_description}
                    for cat in result.classification_result.embedding_candidates
                ],
                "llm_candidates": [
                    {"path": cat.path, "name": cat.name, "description": cat.llm_description}
                    for cat in result.classification_result.llm_candidates
                ],
                "correct_classification": result.correct_classification,
                "processing_time_ms": result.processing_time_ms,
                "narrowed_count": result.narrowed_count,
                "narrowing_time_ms": result.narrowing_time_ms,
                "selection_time_ms": result.selection_time_ms,
                "narrowing_strategy": result.narrowing_strategy,
                "vector_store_enabled": result.vector_store_enabled,
            }
            json_data["results"]["individual_results"].append(result_dict)

        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return filepath


def main():
    """Execute the test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run pipeline accuracy test with optional saved run creation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--save-as", 
        type=str, 
        help="Save results as a named run (e.g., 'v1', 'baseline')"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        help="Description for the saved run"
    )
    
    args = parser.parse_args()
    
    # Also check environment variables for save parameters
    save_as = args.save_as or os.environ.get('SAVE_AS')
    description = args.description or os.environ.get('SAVE_DESCRIPTION')
    
    print("🚀 Classification Pipeline Accuracy Test")
    print("=" * 60)
    print("This test evaluates the complete end-to-end classification pipeline")
    print("by running the full classify() method on all test cases.")
    
    if save_as:
        print(f"Will save results as: '{save_as}'")
        if description:
            print(f"Description: {description}")
    print()

    tester = PipelineAccuracyTester()
    results = tester.run_test()

    # Save results to JSON file
    json_filepath = tester.save_results_to_json(results)

    print("\nPipeline Testing Complete!")
    print("=" * 60)

    # Print final summary
    print(
        f"Pipeline Accuracy: {results.correct_classifications}/{results.total_tests} "
        f"({results.accuracy_percent:.1f}%) correct classifications"
    )
    print(f"Average Processing Time: {results.avg_processing_time_ms:.1f}ms per classification")

    print(f"\nResults saved to: {json_filepath}")
    
    # Save as named run if requested
    if save_as:
        from ui.data_operations import save_current_results_as_run
        
        # Load the just-saved results
        with open(json_filepath, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        
        final_description = description or f"Pipeline test run saved as {save_as}"
        
        if save_current_results_as_run(save_as, final_description, pipeline_data):
            print(f"✅ Successfully saved as run '{save_as}'")
        else:
            print(f"❌ Failed to save as run '{save_as}'")
    
    print("   Use this file for detailed analysis and comparison with other test runs.")


if __name__ == "__main__":
    main()
