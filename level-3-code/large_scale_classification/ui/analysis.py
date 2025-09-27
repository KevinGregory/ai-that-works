"""Analysis and visualization functions for the Streamlit UI.

This module handles error analysis, performance metrics, and chart generation
for the classification system results.
"""
# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.data.models import Category
from src.shared.correctness import CorrectnessDefinition, CorrectnessEvaluator

TEST_CASE_DESCRIPTION_DISPLAY_LENGTH = 100


def analyze_pipeline_errors(
    ui_data: List[Dict[str, Any]],
    correctness_definition: CorrectnessDefinition = CorrectnessDefinition.EXACT,
    all_categories: List[Category] | None = None,
) -> Dict[str, Any]:
    """Analyze test results to categorize pipeline failures.

    Args:
        ui_data: List of test case data
        correctness_definition: Definition of correctness to use
        all_categories: Complete list of categories for hierarchy navigation

    Returns:
        Analysis results with flexible correctness evaluation
    """
    analysis = {
        "total_cases": 0,
        "successful_cases": 0,
        "failed_cases": 0,
        "embedding_filtering_failures": [],
        "llm_filtering_failures": [],
        "final_selection_failures": [],
        "success_cases": [],
        "correctness_definition": correctness_definition.value,
    }

    # Initialize correctness evaluator if using flexible definitions
    evaluator = None
    if correctness_definition != CorrectnessDefinition.EXACT and all_categories:
        evaluator = CorrectnessEvaluator(all_categories)

    for test_case in ui_data:
        analysis["total_cases"] += 1

        ground_truth = test_case["ground_truth"]

        # Get candidates from each stage
        embedding_candidates = test_case["stages"]["embedding"]["candidates"]
        llm_candidates = test_case["stages"]["llm"]["candidates"]
        final_selection = test_case["stages"]["selection"]["final_choice"]

        # Get category paths for easier comparison
        embedding_paths = [cat["path"] for cat in embedding_candidates]
        llm_paths = [cat["path"] for cat in llm_candidates]
        final_path = final_selection.get("path", "") if final_selection else ""

        # Determine failure point
        failure_info = {
            "test_case": test_case,
            "ground_truth": ground_truth,
            "description": test_case["description"][:TEST_CASE_DESCRIPTION_DISPLAY_LENGTH] + "..."
            if len(test_case["description"]) > TEST_CASE_DESCRIPTION_DISPLAY_LENGTH
            else test_case["description"],
            "selected_instead": final_path,
        }

        # Determine correctness using flexible definition
        is_correct = False
        if evaluator and correctness_definition != CorrectnessDefinition.EXACT:
            is_correct = evaluator.is_correct(final_path, ground_truth, correctness_definition)
        else:
            is_correct = ground_truth == final_path

        if is_correct:
            # Success case
            analysis["successful_cases"] += 1
            analysis["success_cases"].append(failure_info)
        else:
            # Failed case - determine where it failed
            analysis["failed_cases"] += 1

            if ground_truth not in embedding_paths:
                # Failed at embedding filtering stage
                failure_info["failure_type"] = "embedding_filtering_failure"
                failure_info["failure_description"] = "Correct category not found in embedding filtering stage"
                analysis["embedding_filtering_failures"].append(failure_info)
            elif ground_truth not in llm_paths:
                # Failed at LLM filtering stage (was in embedding but not in LLM)
                failure_info["failure_type"] = "llm_filtering_failure"
                failure_info["failure_description"] = "Correct category filtered out by LLM narrowing stage"
                analysis["llm_filtering_failures"].append(failure_info)
            else:
                # Failed at final selection stage (was in LLM candidates but not selected)
                failure_info["failure_type"] = "final_selection_failure"
                failure_info["failure_description"] = "Correct category available but not selected as final choice"
                analysis["final_selection_failures"].append(failure_info)

    return analysis


def create_waffle_chart(values, labels, colors, title):
    """Create a true waffle chart where each square represents one item."""
    total_items = sum(values)

    if total_items == 0:
        return None

    # Calculate optimal grid dimensions (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(total_items)))
    rows = int(np.ceil(total_items / cols))

    # Adjust figure size based on grid size (even smaller squares)
    fig_width = max(3, cols * 0.15)
    fig_height = max(2, rows * 0.15)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create the waffle data - each item gets exactly one square
    waffle_data = []
    for i, count in enumerate(values):
        waffle_data.extend([i] * count)

    # Create the plot - one square per item
    square_idx = 0
    for i in range(rows):
        for j in range(cols):
            if square_idx < len(waffle_data):
                category = waffle_data[square_idx]
                color = colors[category] if category < len(colors) else colors[0]

                # Draw square
                rect = patches.Rectangle((j, rows - i - 1), 1, 1, linewidth=1, edgecolor="white", facecolor=color)
                ax.add_patch(rect)
                square_idx += 1

    # Set up the plot
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    # Create legend
    legend_elements = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        if i < len(values) and values[i] > 0:
            percentage = (values[i] / total_items) * 100
            legend_elements.append(patches.Patch(color=color, label=f"{label}: {values[i]} ({percentage:.1f}%)"))

    ax.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(len(legend_elements), 3),
        fontsize=6,
    )

    plt.tight_layout()
    return fig
