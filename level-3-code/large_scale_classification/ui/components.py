"""UI rendering components for the Streamlit application.

This module contains all the Streamlit rendering functions for different
parts of the user interface including error analysis, test case analysis,
and custom testing components.
"""
# ruff: noqa: E402

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.classification.pipeline import ClassificationPipeline
from src.data.category_loader import CategoryLoader
from src.shared.correctness import CorrectnessDefinition
from ui.analysis import analyze_pipeline_errors, create_waffle_chart


def render_error_overview(analysis):
    """Render high-level error metrics with waffle chart."""
    st.markdown("### 📊 Pipeline Performance Overview")

    total = analysis["total_cases"]
    successful = analysis["successful_cases"]
    failed = analysis["failed_cases"]

    if total == 0:
        st.warning("No test cases to analyze.")
        return

    # Create waffle chart for success/failure overview
    col1, col2 = st.columns([2, 1])

    with col1:
        # Performance waffle chart
        values = [successful, failed]
        labels = ["Successful", "Failed"]
        colors = ["#22C55E", "#EF4444"]  # Green for success, red for failure

        fig = create_waffle_chart(
            values=values, labels=labels, colors=colors, title=f"Classification Performance ({total} test cases)"
        )
        if fig:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No data to display in waffle chart.")

    with col2:
        st.markdown("#### Key Metrics")
        success_rate = (successful / total * 100) if total > 0 else 0

        st.metric("Total Cases", total)
        st.metric("Success Rate", f"{success_rate:.1f}%")

        if failed > 0:
            most_common_failure = max(
                [
                    ("Embedding Filtering", len(analysis["embedding_filtering_failures"])),
                    ("LLM Filtering", len(analysis["llm_filtering_failures"])),
                    ("Final Selection", len(analysis["final_selection_failures"])),
                ],
                key=lambda x: x[1],
            )
            st.metric("Top Failure Type", most_common_failure[0])
        else:
            st.success("🎉 Perfect Performance!")


def render_failure_breakdown(analysis):
    """Render detailed failure breakdown with waffle chart."""
    st.markdown("### 🔍 Failure Point Analysis")

    if analysis["failed_cases"] == 0:
        st.success("🎉 **Perfect Performance!** All test cases were classified correctly.")
        return

    embedding_filtering_failures = len(analysis["embedding_filtering_failures"])
    llm_filtering_failures = len(analysis["llm_filtering_failures"])
    final_selection_failures = len(analysis["final_selection_failures"])
    total_failures = analysis["failed_cases"]

    # Create waffle chart for failure breakdown
    col1, col2 = st.columns([2, 1])

    with col1:
        # Only include failure types that have actual failures
        values = []
        labels = []
        colors = []

        if embedding_filtering_failures > 0:
            values.append(embedding_filtering_failures)
            labels.append("Embedding Filtering")
            colors.append("#F97316")  # Orange

        if llm_filtering_failures > 0:
            values.append(llm_filtering_failures)
            labels.append("LLM Filtering")
            colors.append("#EAB308")  # Yellow

        if final_selection_failures > 0:
            values.append(final_selection_failures)
            labels.append("Final Selection")
            colors.append("#EF4444")  # Red

        if values:  # Only create chart if there are failures
            fig = create_waffle_chart(
                values=values,
                labels=labels,
                colors=colors,
                title=f"Failure Point Distribution ({total_failures} failed cases)",
            )
            if fig:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No failure data to display in waffle chart.")

    with col2:
        st.markdown("#### Failure Breakdown")

        if embedding_filtering_failures > 0:
            embedding_pct = embedding_filtering_failures / total_failures * 100
            st.markdown("### 🔍 Embedding Filtering:")
            st.markdown(f"#### Count: {embedding_filtering_failures}")
            st.markdown(f"#### Percentage: {embedding_pct:.1f}%")

        if llm_filtering_failures > 0:
            llm_pct = llm_filtering_failures / total_failures * 100
            st.markdown("### 🧠 LLM Filtering:")
            st.markdown(f"#### Count: {llm_filtering_failures}")
            st.markdown(f"#### Percentage: {llm_pct:.1f}%")

        if final_selection_failures > 0:
            final_selection_pct = final_selection_failures / total_failures * 100
            st.markdown("### 🎯 Final Selection:")
            st.markdown(f"#### Count: {final_selection_failures}")
            st.markdown(f"#### Percentage: {final_selection_pct:.1f}%")


def render_failed_cases_table(analysis):
    """Render table of failed test cases."""
    st.markdown("### 📋 Failed Test Cases Details")

    if analysis["failed_cases"] == 0:
        st.success("No failed cases to display!")
        return

    # Combine all failures into one list
    all_failures = []
    all_failures.extend(analysis["embedding_filtering_failures"])
    all_failures.extend(analysis["llm_filtering_failures"])
    all_failures.extend(analysis["final_selection_failures"])

    if not all_failures:
        return

    # Create DataFrame for display with reordered columns
    failure_data = []
    for failure in all_failures:
        failure_data.append(
            {
                "Description": failure["description"],
                "Ground Truth": failure["ground_truth"],
                "Predicted": failure.get("selected_instead", "Unknown"),
                "Failure Point": failure["failure_type"].replace("_", " ").title(),
            }
        )

    df = pd.DataFrame(failure_data)

    # Add filtering options
    failure_types = df["Failure Point"].unique()
    selected_failure_type = st.selectbox("Filter by failure type:", ["All"] + list(failure_types), key="failure_filter")

    if selected_failure_type != "All":
        df = df[df["Failure Point"] == selected_failure_type]

    st.dataframe(df, width="stretch", hide_index=True)


def render_error_analysis(ui_data, correctness_definition: CorrectnessDefinition = CorrectnessDefinition.EXACT):
    """Render the error analysis tab showing pipeline failure patterns."""
    if not ui_data:
        st.warning("⚠️ No test results available. Please load a saved run first.")
        return

    # Get unique test types from the data
    test_types = set()
    for test_case in ui_data:
        test_types.add(test_case.get("test_type", "unknown"))
    test_types = sorted(list(test_types))

    # Add filter dropdown
    st.markdown("### 🔍 Filter Results")
    filter_options = ["All"] + test_types
    selected_filter = st.selectbox("Select test case type to analyze:", filter_options, key="error_analysis_filter")

    # Filter data based on selection
    if selected_filter == "All":
        filtered_data = ui_data
        filter_description = "all test cases"
    else:
        filtered_data = [tc for tc in ui_data if tc.get("test_type") == selected_filter]
        filter_description = f"{selected_filter} test cases"

    st.markdown(f"**Analyzing {len(filtered_data)} {filter_description} out of {len(ui_data)} total test cases**")
    st.markdown("---")

    # Load all categories for hierarchy analysis if using flexible correctness
    all_categories = None
    if correctness_definition != CorrectnessDefinition.EXACT:
        try:
            category_loader = CategoryLoader()
            all_categories = category_loader.load_categories()
        except Exception as e:
            st.error(f"Could not load categories for flexible correctness: {e}")
            return

    # Analyze errors for filtered data with flexible correctness
    error_analysis = analyze_pipeline_errors(filtered_data, correctness_definition, all_categories)

    # Display high-level metrics
    render_error_overview(error_analysis)

    # Display detailed breakdowns
    st.markdown("---")
    render_failure_breakdown(error_analysis)

    # Display failed test cases table
    st.markdown("---")
    render_failed_cases_table(error_analysis)


def _evaluate_flexible_correctness(final_path: str, ground_truth: str, correctness_definition: CorrectnessDefinition):
    """Evaluate correctness using flexible definition and return results."""
    is_flexible_correct = final_path == ground_truth  # Default to exact match
    explanation = ""

    if correctness_definition != CorrectnessDefinition.EXACT:
        try:
            category_loader = CategoryLoader()
            all_categories = category_loader.load_categories()
            from src.shared.correctness import CorrectnessEvaluator

            evaluator = CorrectnessEvaluator(all_categories)
            is_flexible_correct = evaluator.is_correct(final_path, ground_truth, correctness_definition)
            explanation = evaluator.get_correctness_explanation(final_path, ground_truth, correctness_definition)
        except Exception as e:
            st.error(f"Error evaluating flexible correctness: {e}")
            explanation = "Error in evaluation"

    return is_flexible_correct, explanation


def _render_test_case_header(case_data: dict[str, Any], correctness_definition: CorrectnessDefinition):
    """Render the test case header with description, ground truth, and prediction."""
    st.markdown("### 📝 Test Case Details")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"#### Description: {case_data['description']}")

        # Ground truth and model prediction
        ground_truth = case_data["ground_truth"]
        final_selection = case_data["stages"]["selection"]["final_choice"]
        final_path = final_selection.get("path", "") if final_selection else ""

        st.markdown(f"#### 🎯 Ground Truth: {ground_truth}")

        # Evaluate and display correctness
        is_exact_correct = case_data["is_correct"]
        is_flexible_correct, explanation = _evaluate_flexible_correctness(
            final_path, ground_truth, correctness_definition
        )

        # Display result based on correctness definition
        if correctness_definition == CorrectnessDefinition.EXACT:
            icon = "✅" if is_exact_correct else "❌"
            st.markdown(f"#### {icon} Model Guess: {final_path}")
        else:
            icon = "✅" if is_flexible_correct else "❌"
            st.markdown(f"#### {icon} Model Guess: {final_path}")
            if explanation:
                st.info(f"**Note**: {explanation}")

    with col2:
        st.metric("Processing Time", f"{case_data['processing_time_ms']:.1f}ms")
        st.metric("Narrowing Time", f"{case_data['narrowing_time_ms']:.1f}ms")
        st.metric("Selection Time", f"{case_data['selection_time_ms']:.1f}ms")


def _collect_pipeline_categories(case_data: dict[str, Any]) -> set:
    """Collect all categories involved in the pipeline analysis."""
    categories_to_analyze = set()

    # Add all embedding candidates
    for candidate in case_data["stages"]["embedding"]["candidates"]:
        categories_to_analyze.add(candidate["path"])

    # Add all LLM candidates
    for candidate in case_data["stages"]["llm"]["candidates"]:
        categories_to_analyze.add(candidate["path"])

    # Add all final candidates (for backward compatibility)
    for candidate in case_data["stages"]["narrowing"]["candidates"]:
        categories_to_analyze.add(candidate["path"])

    # Always add ground truth
    categories_to_analyze.add(case_data["ground_truth"])

    return categories_to_analyze


def _create_pipeline_table_data(case_data: dict[str, Any], categories_to_analyze: set) -> list[dict[str, Any]]:
    """Create the data for the pipeline analysis table."""
    table_data = []

    # Sort categories by path for consistent display
    sorted_categories = sorted(categories_to_analyze)

    # Get stage-specific candidate paths
    embedding_candidate_paths = [cat["path"] for cat in case_data["stages"]["embedding"]["candidates"]]
    llm_candidate_paths = [cat["path"] for cat in case_data["stages"]["llm"]["candidates"]]

    ground_truth = case_data["ground_truth"]
    final_selection = case_data["stages"]["selection"]["final_choice"]
    final_path = final_selection.get("path", "") if final_selection else ""

    for category_path in sorted_categories:
        # Check pipeline stages
        made_it_through_embedding = "✅" if category_path in embedding_candidate_paths else ""
        made_it_through_llm = "✅" if category_path in llm_candidate_paths else ""
        finally_selected = "✅" if category_path == final_path else ""

        # Determine row styling
        is_ground_truth = category_path == ground_truth
        is_correctly_selected = case_data["is_correct"] and category_path == final_path

        table_data.append(
            {
                "Category Path": category_path,
                "Embedding Filter": made_it_through_embedding,
                "LLM Filter": made_it_through_llm,
                "Finally Selected": finally_selected,
                "_is_ground_truth": is_ground_truth,
                "_is_correctly_selected": is_correctly_selected,
            }
        )

    return table_data


def _render_pipeline_table(table_data: list[dict[str, Any]]):
    """Render the styled pipeline analysis table."""
    # Create the display dataframe without helper columns
    display_data = []
    for item in table_data:
        display_data.append(
            {
                "Category Path": item["Category Path"],
                "Embedding Filter": item["Embedding Filter"],
                "LLM Filter": item["LLM Filter"],
                "Finally Selected": item["Finally Selected"],
            }
        )

    display_df = pd.DataFrame(display_data)

    # Apply styling based on the original table_data
    def highlight_row(row):
        row_index = row.name
        original_item = table_data[row_index]

        if original_item["_is_correctly_selected"]:
            # Green background for correct selection
            return ["background-color: #d4edda"] * 4
        elif original_item["_is_ground_truth"]:
            # Red background for missed ground truth
            return ["background-color: #f8d7da"] * 4
        else:
            return [""] * 4

    styled_df = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, width="stretch", hide_index=True)


def render_test_case_analysis(
    ui_data, selected_case_index, correctness_definition: CorrectnessDefinition = CorrectnessDefinition.EXACT
):
    """Render analysis for a specific test case."""
    if not ui_data or selected_case_index >= len(ui_data):
        st.warning("⚠️ No test case selected or data available.")
        return

    case_data = ui_data[selected_case_index]

    # Render test case header with details and metrics
    _render_test_case_header(case_data, correctness_definition)

    # Display pipeline analysis table
    st.markdown("### 🔍 Pipeline Analysis")

    categories_to_analyze = _collect_pipeline_categories(case_data)

    if categories_to_analyze:
        table_data = _create_pipeline_table_data(case_data, categories_to_analyze)
        _render_pipeline_table(table_data)
    else:
        st.warning("No categories found for this test case.")


def render_custom_testing():
    """Render the custom testing interface."""
    st.markdown("### 🧪 Custom Test Case")

    with st.form("custom_test_form"):
        test_text = st.text_area(
            "Enter text to classify:", placeholder="e.g., 'French door refrigerator with ice maker'", height=100
        )

        submit_button = st.form_submit_button("Classify Text", type="primary")

        if submit_button and test_text.strip():
            with st.spinner("🔄 Classifying text..."):
                try:
                    # Initialize pipeline and run classification
                    pipeline = ClassificationPipeline()
                    result = pipeline.classify(test_text)

                    # Display results
                    st.success("✅ Classification Complete!")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Selected Category:** {result.category.path}")
                        st.markdown(f"**Category Name:** {result.category.name}")
                        if result.category.llm_description:
                            st.markdown(f"**Description:** {result.category.llm_description}")

                    with col2:
                        metadata = result.metadata
                        st.metric("Candidates Found", metadata.get("narrowed_to", "Unknown"))
                        st.metric("Total Time", f"{metadata.get('total_time_ms', 0):.1f}ms")
                        st.metric("Narrowing Strategy", metadata.get("narrowing_strategy", "Unknown"))

                    # Show all candidates
                    st.markdown("#### 🔍 All Candidates")
                    if result.candidates:
                        candidate_data = []
                        for i, candidate in enumerate(result.candidates, 1):
                            is_selected = candidate.path == result.category.path

                            candidate_data.append(
                                {
                                    "Rank": i,
                                    "Category Path": candidate.path,
                                    "Category Name": candidate.name,
                                    "Selected": "✅" if is_selected else "",
                                }
                            )

                        df = pd.DataFrame(candidate_data)
                        st.dataframe(df, width="stretch", hide_index=True)

                except Exception as e:
                    error_str = str(e)
                    if "ConnectTimeout" in error_str or "APITimeoutError" in error_str:
                        st.error(
                            "🌐 **API Timeout Error**\n\n"
                            "The classification failed due to OpenAI API timeout. "
                            "Please check your network connection and try again."
                        )
                    elif "OPENAI_API_KEY" in error_str or "Incorrect API key provided" in error_str:
                        st.error("🔑 **API Key Configuration Error**")
                        st.markdown("""
                        **The `.env` file is missing or incorrectly configured.**

                        **To fix this:**
                        1. Create a file named `.env` in the project root directory
                        2. Add your OpenAI API key:
                        ```
                        OPENAI_API_KEY=sk-your-actual-api-key-here
                        ```
                        3. Replace `sk-your-actual-api-key-here` with your real API key from https://platform.openai.com/account/api-keys

                        **Important:**
                        - Do NOT include quotes around the API key
                        - The API key should start with `sk-`
                        - Make sure the `.env` file is in the same directory as `streamlit_app.py`
                        """)
                    else:
                        st.error(f"**Classification Error:** {error_str}")
