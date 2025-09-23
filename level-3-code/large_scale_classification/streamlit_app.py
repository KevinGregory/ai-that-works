"""Streamlit GUI for Large Scale Classification System.

This application provides an interactive interface for analyzing
pipeline classification results and comparing different test runs.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the src directory to Python path
import sys
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Ensure environment variables are loaded before importing settings
import os
from dotenv import load_dotenv
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

from src.classification.pipeline import ClassificationPipeline
from src.config.settings import settings
from src.data.category_loader import CategoryLoader
from src.shared.enums import NarrowingStrategy


# Page configuration
st.set_page_config(
    page_title="Classification System GUI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 22px;
    font-weight: bold;
}
.stDataFrame thead th {
    font-size: 22px !important;
    font-weight: bold !important;
    background-color: #f0f2f6 !important;
    padding: 14px 10px !important;
}
.stDataFrame tbody td {
    font-size: 16px !important;
    padding: 12px 8px !important;
}
.stDataFrame {
    font-size: 16px !important;
}
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #1f77b4;
}
.product-description {
    font-size: 1.4rem !important;
    line-height: 1.6;
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


def load_latest_pipeline_results() -> Optional[Dict[str, Any]]:
    """Load the most recent pipeline test results from JSON files.
    
    Returns:
        Dictionary containing the latest pipeline test results, or None if no results found
    """
    project_root = Path(__file__).parent
    
    # Look for the most recent pipeline results
    pipeline_dir = project_root / "tests" / "results" / "pipeline"
    
    if not pipeline_dir.exists():
        return None
    
    pipeline_files = list(pipeline_dir.glob("*.json"))
    
    if not pipeline_files:
        return None
    
    # Sort by modification time to get the latest
    pipeline_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_pipeline_file = pipeline_files[0]
    
    try:
        with open(latest_pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
            return pipeline_data
    except Exception as e:
        st.error(f"Error loading pipeline results: {e}")
        return None


def get_available_saved_runs() -> List[Dict[str, Any]]:
    """Get metadata for all available saved test runs.
    
    Returns:
        List of dictionaries containing saved run metadata
    """
    project_root = Path(__file__).parent
    saved_runs_dir = project_root / "tests" / "results" / "saved_runs"
    
    if not saved_runs_dir.exists():
        return []
    
    saved_runs = []
    
    for metadata_file in saved_runs_dir.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                saved_runs.append(metadata)
        except Exception as e:
            st.warning(f"Error loading saved run metadata from {metadata_file.name}: {e}")
    
    # Sort by timestamp, most recent first
    saved_runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return saved_runs


def load_saved_run(run_name: str) -> Optional[Dict[str, Any]]:
    """Load a specific saved test run by name.
    
    Args:
        run_name: Name of the saved run to load
    
    Returns:
        Dictionary containing the saved run data, or None if not found
    """
    project_root = Path(__file__).parent
    saved_runs_dir = project_root / "tests" / "results" / "saved_runs"
    
    if not saved_runs_dir.exists():
        return None
    
    # Find the metadata file for this run
    metadata_file = saved_runs_dir / f"{run_name}_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load the actual pipeline results
        pipeline_file = Path(metadata['pipeline_results_path'])
        
        if not pipeline_file.exists():
            st.error(f"Pipeline results file not found: {pipeline_file}")
            return None
        
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
        
        return {
            'metadata': metadata,
            'pipeline_data': pipeline_data
        }
    
    except Exception as e:
        st.error(f"Error loading saved run '{run_name}': {e}")
        return None


def save_current_results_as_run(run_name: str, description: str, pipeline_data: Dict[str, Any]) -> bool:
    """Save the current test results as a named run.
    
    Args:
        run_name: Name for the saved run
        description: Description of the run
        pipeline_data: Pipeline test results to save
    
    Returns:
        True if successful, False otherwise
    """
    project_root = Path(__file__).parent
    saved_runs_dir = project_root / "tests" / "results" / "saved_runs"
    saved_runs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save the pipeline results with a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_filename = f"pipeline_{run_name}_{timestamp}.json"
        pipeline_filepath = saved_runs_dir / pipeline_filename
        
        with open(pipeline_filepath, 'w', encoding='utf-8') as f:
            json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
        
        # Create metadata for this saved run
        metadata = {
            'run_name': run_name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'pipeline_results_path': str(pipeline_filepath),
            'config': {
                'narrowing_strategy': pipeline_data.get('test_info', {}).get('narrowing_strategy', 'unknown'),
                'vector_store_enabled': pipeline_data.get('test_info', {}).get('vector_store_enabled', False),
                'total_test_cases': pipeline_data.get('test_info', {}).get('total_test_cases', 0)
            },
            'results_summary': {
                'total_tests': pipeline_data.get('results', {}).get('total_tests', 0),
                'correct_classifications': pipeline_data.get('results', {}).get('correct_classifications', 0),
                'accuracy_percent': pipeline_data.get('results', {}).get('accuracy_percent', 0.0)
            }
        }
        
        # Save metadata
        metadata_filename = f"{run_name}_metadata.json"
        metadata_filepath = saved_runs_dir / metadata_filename
        
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving run '{run_name}': {e}")
        return False


def transform_pipeline_results_for_ui(pipeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform pipeline results into UI-friendly format.
    
    Args:
        pipeline_data: Raw pipeline results from JSON
        
    Returns:
        List of test case data for UI display
    """
    ui_data = []
    
    if not pipeline_data or 'results' not in pipeline_data:
        return ui_data
    
    individual_results = pipeline_data['results'].get('individual_results', [])
    
    for result in individual_results:
        test_case = result.get('test_case', {})
        selected_category = result.get('selected_category', {})
        candidate_categories = result.get('candidate_categories', [])
        
        # Transform for UI
        test_case_data = {
            'description': test_case.get('text', 'Unknown'),
            'ground_truth': test_case.get('category', 'Unknown'),
            'test_type': test_case.get('test_type', 'unknown'),
            'stages': {
                'narrowing': {
                    'candidates': candidate_categories
                },
                'selection': {
                    'final_choice': selected_category if selected_category.get('path') else None
                }
            },
            'is_correct': result.get('correct_classification', False),
            'processing_time_ms': result.get('processing_time_ms', 0),
            'narrowing_time_ms': result.get('narrowing_time_ms', 0),
            'selection_time_ms': result.get('selection_time_ms', 0)
        }
        
        ui_data.append(test_case_data)
    
    return ui_data


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
                rect = patches.Rectangle((j, rows - i - 1), 1, 1, 
                                       linewidth=1, edgecolor='white', 
                                       facecolor=color)
                ax.add_patch(rect)
                square_idx += 1
    
    # Set up the plot
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    # Create legend
    legend_elements = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        if i < len(values) and values[i] > 0:
            percentage = (values[i] / total_items) * 100
            legend_elements.append(patches.Patch(color=color, 
                                               label=f'{label}: {values[i]} ({percentage:.1f}%)'))
    
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.25), 
              ncol=min(len(legend_elements), 3), fontsize=6)
    
    plt.tight_layout()
    return fig


def analyze_pipeline_errors(ui_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze test results to categorize pipeline failures."""
    analysis = {
        'total_cases': 0,
        'successful_cases': 0,
        'failed_cases': 0,
        'embedding_failures': [],
        'final_selection_failures': [],
        'success_cases': []
    }
    
    for test_case in ui_data:
        analysis['total_cases'] += 1
        
        ground_truth = test_case['ground_truth']
        narrowed_cats = test_case['stages']['narrowing']['candidates']
        final_selection = test_case['stages']['selection']['final_choice']
        
        # Get category paths for easier comparison
        narrowed_paths = [cat['path'] for cat in narrowed_cats]
        final_path = final_selection.get('path', '') if final_selection else ''
        
        # Determine failure point
        failure_info = {
            'test_case': test_case,
            'ground_truth': ground_truth,
            'description': test_case['description'][:100] + "..." if len(test_case['description']) > 100 else test_case['description'],
            'selected_instead': final_path
        }
        
        if ground_truth == final_path:
            # Success case
            analysis['successful_cases'] += 1
            analysis['success_cases'].append(failure_info)
        else:
            # Failed case - determine where it failed
            analysis['failed_cases'] += 1
            
            if ground_truth not in narrowed_paths:
                failure_info['failure_type'] = 'embedding_failure'
                failure_info['failure_description'] = 'Correct category not found in narrowing stage'
                analysis['embedding_failures'].append(failure_info)
            else:
                failure_info['failure_type'] = 'final_selection_failure'
                failure_info['failure_description'] = 'Correct category available but not selected'
                analysis['final_selection_failures'].append(failure_info)
    
    return analysis


def render_error_overview(analysis):
    """Render high-level error metrics with waffle chart."""
    st.markdown("### 📊 Pipeline Performance Overview")
    
    total = analysis['total_cases']
    successful = analysis['successful_cases']
    failed = analysis['failed_cases']
    
    if total == 0:
        st.warning("No test cases to analyze.")
        return
    
    # Create waffle chart for success/failure overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance waffle chart
        values = [successful, failed]
        labels = ['Successful', 'Failed']
        colors = ['#22C55E', '#EF4444']  # Green for success, red for failure
        
        fig = create_waffle_chart(
            values=values,
            labels=labels, 
            colors=colors,
            title=f"Classification Performance ({total} test cases)"
        )
        if fig:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No data to display in waffle chart.")
    
    with col2:
        st.markdown("#### Key Metrics")
        success_rate = (successful / total * 100) if total > 0 else 0
        failure_rate = (failed / total * 100) if total > 0 else 0
        
        st.metric("Total Cases", total)
        st.metric("Success Rate", f"{success_rate:.1f}%")
        st.metric("Failure Rate", f"{failure_rate:.1f}%")
        
        if failed > 0:
            most_common_failure = max([
                ('Embedding', len(analysis['embedding_failures'])),
                ('Final Selection', len(analysis['final_selection_failures']))
            ], key=lambda x: x[1])
            st.metric("Top Failure Type", most_common_failure[0])
        else:
            st.success("🎉 Perfect Performance!")


def render_failure_breakdown(analysis):
    """Render detailed failure breakdown with waffle chart."""
    st.markdown("### 🔍 Failure Point Analysis")
    
    if analysis['failed_cases'] == 0:
        st.success("🎉 **Perfect Performance!** All test cases were classified correctly.")
        return
    
    embedding_failures = len(analysis['embedding_failures'])
    final_selection_failures = len(analysis['final_selection_failures'])
    total_failures = analysis['failed_cases']
    
    # Create waffle chart for failure breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Only include failure types that have actual failures
        values = []
        labels = []
        colors = []
        
        if embedding_failures > 0:
            values.append(embedding_failures)
            labels.append('Narrowing Stage')
            colors.append('#F97316')  # Orange
        
        if final_selection_failures > 0:
            values.append(final_selection_failures)
            labels.append('Final Selection')
            colors.append('#EF4444')  # Red
        
        if values:  # Only create chart if there are failures
            fig = create_waffle_chart(
                values=values,
                labels=labels,
                colors=colors,
                title=f"Failure Point Distribution ({total_failures} failed cases)"
            )
            if fig:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No failure data to display in waffle chart.")
    
    with col2:
        st.markdown("#### Failure Breakdown")
        
        if embedding_failures > 0:
            embedding_pct = (embedding_failures / total_failures * 100)
            st.metric("🔍 Narrowing Stage", f"{embedding_failures}", delta=f"{embedding_pct:.1f}%")
        
        if final_selection_failures > 0:
            final_selection_pct = (final_selection_failures / total_failures * 100)
            st.metric("🎯 Final Selection", f"{final_selection_failures}", delta=f"{final_selection_pct:.1f}%")


def render_failed_cases_table(analysis):
    """Render table of failed test cases."""
    st.markdown("### 📋 Failed Test Cases Details")
    
    if analysis['failed_cases'] == 0:
        st.success("No failed cases to display!")
        return
    
    # Combine all failures into one list
    all_failures = []
    all_failures.extend(analysis['embedding_failures'])
    all_failures.extend(analysis['final_selection_failures'])
    
    if not all_failures:
        return
    
    # Create DataFrame for display with reordered columns
    failure_data = []
    for failure in all_failures:
        failure_data.append({
            'Description': failure['description'],
            'Ground Truth': failure['ground_truth'],
            'Predicted': failure.get('selected_instead', 'Unknown'),
            'Failure Point': failure['failure_type'].replace('_', ' ').title()
        })
    
    df = pd.DataFrame(failure_data)
    
    # Add filtering options
    failure_types = df['Failure Point'].unique()
    selected_failure_type = st.selectbox(
        "Filter by failure type:",
        ['All'] + list(failure_types),
        key="failure_filter"
    )
    
    if selected_failure_type != 'All':
        df = df[df['Failure Point'] == selected_failure_type]
    
    st.dataframe(
        df,
        width='stretch',
        hide_index=True
    )


def render_error_analysis(ui_data):
    """Render the error analysis tab showing pipeline failure patterns."""
    
    if not ui_data:
        st.warning("⚠️ No test results available. Please load a saved run first.")
        return
    
    # Get unique test types from the data
    test_types = set()
    for test_case in ui_data:
        test_types.add(test_case.get('test_type', 'unknown'))
    test_types = sorted(list(test_types))
    
    # Add filter dropdown
    st.markdown("### 🔍 Filter Results")
    filter_options = ['All'] + test_types
    selected_filter = st.selectbox(
        "Select test case type to analyze:",
        filter_options,
        key="error_analysis_filter"
    )
    
    # Filter data based on selection
    if selected_filter == 'All':
        filtered_data = ui_data
        filter_description = "all test cases"
    else:
        filtered_data = [tc for tc in ui_data if tc.get('test_type') == selected_filter]
        filter_description = f"{selected_filter} test cases"
    
    st.markdown(f"**Analyzing {len(filtered_data)} {filter_description} out of {len(ui_data)} total test cases**")
    st.markdown("---")
    
    # Analyze errors for filtered data
    error_analysis = analyze_pipeline_errors(filtered_data)
    
    # Display high-level metrics
    render_error_overview(error_analysis)
    
    # Display detailed breakdowns
    st.markdown("---")
    render_failure_breakdown(error_analysis)
    
    # Display failed test cases table
    st.markdown("---")
    render_failed_cases_table(error_analysis)


def render_test_case_analysis(ui_data, selected_case_index):
    """Render analysis for a specific test case."""
    if not ui_data or selected_case_index >= len(ui_data):
        st.warning("⚠️ No test case selected or data available.")
        return
    
    case_data = ui_data[selected_case_index]
    
    # Display test case details
    st.markdown("### 📝 Test Case Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Description:** {case_data['description']}")
        
        # Ground truth and model prediction
        ground_truth = case_data['ground_truth']
        final_selection = case_data['stages']['selection']['final_choice']
        final_path = final_selection.get('path', '') if final_selection else ''
        
        st.markdown(f"#### 🎯 Ground Truth: {ground_truth}")
        
        if case_data['is_correct']:
            st.markdown(f"#### ✅ Model Guess: {final_path}")
        else:
            st.markdown(f"#### ❌ Model Guess: {final_path}")
    
    with col2:
        st.metric("Processing Time", f"{case_data['processing_time_ms']:.1f}ms")
        st.metric("Narrowing Time", f"{case_data['narrowing_time_ms']:.1f}ms") 
        st.metric("Selection Time", f"{case_data['selection_time_ms']:.1f}ms")
    
    # Display candidates
    st.markdown("### 🔍 Classification Candidates")
    
    candidates = case_data['stages']['narrowing']['candidates']
    
    if candidates:
        candidate_data = []
        for i, candidate in enumerate(candidates, 1):
            is_selected = final_path == candidate['path']
            is_correct = ground_truth == candidate['path']
            
            status = ""
            if is_selected and is_correct:
                status = "✅ SELECTED & CORRECT"
            elif is_selected:
                status = "🔵 SELECTED"
            elif is_correct:
                status = "🎯 CORRECT"
            
            candidate_data.append({
                "Rank": i,
                "Category Path": candidate['path'],
                "Category Name": candidate['name'],
                "Status": status
            })
        
        df = pd.DataFrame(candidate_data)
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.warning("No candidates found for this test case.")


def render_custom_testing():
    """Render the custom testing interface."""
    st.markdown("### 🧪 Custom Test Case")
    
    with st.form("custom_test_form"):
        test_text = st.text_area(
            "Enter text to classify:",
            placeholder="e.g., 'French door refrigerator with ice maker'",
            height=100
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
                            
                            candidate_data.append({
                                "Rank": i,
                                "Category Path": candidate.path,
                                "Category Name": candidate.name,
                                "Selected": "✅" if is_selected else ""
                            })
                        
                        df = pd.DataFrame(candidate_data)
                        st.dataframe(df, width='stretch', hide_index=True)
                    
                except Exception as e:
                    error_str = str(e)
                    if "ConnectTimeout" in error_str or "APITimeoutError" in error_str:
                        st.error("🌐 **API Timeout Error**\n\n"
                                "The classification failed due to OpenAI API timeout. "
                                "Please check your network connection and try again.")
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


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Large Scale Classification System</h1>', unsafe_allow_html=True)
    
    
    # Sidebar for saved runs
    st.sidebar.markdown("## 📁 Saved Test Runs")
    
    # Get available saved runs
    saved_runs = get_available_saved_runs()
    
    if not saved_runs:
        st.sidebar.warning("No saved runs available. Run a pipeline test to create saved results.")
        current_data = None
        current_metadata = None
    else:
        # Create dropdown options
        run_options = ["Latest Results"] + [run['run_name'] for run in saved_runs]
        
        # Initialize session state
        if 'selected_run' not in st.session_state:
            st.session_state.selected_run = "Latest Results"
        
        selected_run = st.sidebar.selectbox(
            "Select test run to analyze:",
            run_options,
            index=run_options.index(st.session_state.selected_run) if st.session_state.selected_run in run_options else 0,
            key="run_selector"
        )
        
        # Load the selected run
        if selected_run == "Latest Results":
            pipeline_data = load_latest_pipeline_results()
            current_metadata = None
        else:
            saved_data = load_saved_run(selected_run)
            if saved_data:
                pipeline_data = saved_data['pipeline_data']
                current_metadata = saved_data['metadata']
            else:
                pipeline_data = None
                current_metadata = None
        
        # Transform data for UI
        if pipeline_data:
            current_data = transform_pipeline_results_for_ui(pipeline_data)
            print(f"DEBUG: current_data created with {len(current_data) if current_data else 0} items")
            if current_data and len(current_data) > 0:
                print(f"DEBUG: First item keys: {list(current_data[0].keys())}")
                if 'stages' in current_data[0]:
                    print(f"DEBUG: Stages keys: {list(current_data[0]['stages'].keys())}")
        else:
            current_data = None
            print("DEBUG: No pipeline_data, current_data is None")
        
        # Display run info
        if current_metadata:
            st.sidebar.markdown("### 📊 Run Details")
            st.sidebar.markdown(f"**Name:** {current_metadata['run_name']}")
            st.sidebar.markdown(f"**Description:** {current_metadata['description']}")
            st.sidebar.markdown(f"**Strategy:** {current_metadata['config']['narrowing_strategy']}")
            st.sidebar.markdown(f"**Accuracy:** {current_metadata['results_summary']['accuracy_percent']:.1f}%")
            st.sidebar.markdown(f"**Test Cases:** {current_metadata['results_summary']['total_tests']}")
            
            # Calculate average narrowed categories from saved run data
            if current_data:
                embedding_counts = []
                llm_counts = []
                try:
                    for case in current_data:
                        # Debug: Check if the structure exists
                        if 'stages' in case and 'embedding' in case['stages'] and 'llm' in case['stages']:
                            embedding_counts.append(len(case['stages']['embedding']['candidates']))
                            llm_counts.append(len(case['stages']['llm']['candidates']))
                        else:
                            # Debug: Show what structure actually exists
                            st.sidebar.markdown(f"**Debug:** Missing stages structure in case")
                            if 'stages' in case:
                                st.sidebar.markdown(f"**Available stages:** {list(case['stages'].keys())}")
                            break
                    
                    if embedding_counts and llm_counts:
                        avg_embedding = sum(embedding_counts) / len(embedding_counts)
                        avg_llm = sum(llm_counts) / len(llm_counts)
                        st.sidebar.markdown(f"**Avg Embedding Candidates:** {avg_embedding:.1f}")
                        st.sidebar.markdown(f"**Avg LLM Candidates:** {avg_llm:.1f}")
                except Exception as e:
                    st.sidebar.markdown(f"**Debug Error:** {str(e)}")
            else:
                st.sidebar.markdown(f"**Debug:** current_data is None")
        elif pipeline_data:
            st.sidebar.markdown("### 📊 Latest Results")
            results = pipeline_data.get('results', {})
            st.sidebar.markdown(f"**Strategy:** {pipeline_data.get('test_info', {}).get('narrowing_strategy', 'unknown')}")
            st.sidebar.markdown(f"**Accuracy:** {results.get('accuracy_percent', 0):.1f}%")
            st.sidebar.markdown(f"**Test Cases:** {results.get('total_tests', 0)}")
            
            # Calculate average narrowed categories from latest results
            if current_data:
                embedding_counts = []
                llm_counts = []
                for case in current_data:
                    embedding_counts.append(len(case['stages']['embedding']['candidates']))
                    llm_counts.append(len(case['stages']['llm']['candidates']))
                
                if embedding_counts and llm_counts:
                    avg_embedding = sum(embedding_counts) / len(embedding_counts)
                    avg_llm = sum(llm_counts) / len(llm_counts)
                    st.sidebar.markdown(f"**Avg Embedding Candidates:** {avg_embedding:.1f}")
                    st.sidebar.markdown(f"**Avg LLM Candidates:** {avg_llm:.1f}")
    
    # Save current results form
    if current_data and not current_metadata:  # Only show for latest results
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 Save Current Results")
        
        with st.sidebar.form("save_run_form"):
            run_name = st.text_input("Run Name:", placeholder="e.g., baseline_v2")
            description = st.text_area("Description:", placeholder="Brief description of this test run")
            save_button = st.form_submit_button("Save Run")
            
            if save_button and run_name.strip() and description.strip():
                if save_current_results_as_run(run_name.strip(), description.strip(), pipeline_data):
                    st.sidebar.success(f"✅ Saved run '{run_name}'")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to save run")
    
    # Main content tabs
    if current_data:
        tab1, tab2, tab3 = st.tabs(["🔍 Error Analysis", "📊 Test Case Analysis", "🧪 Custom Test Case"])
        
        with tab1:
            render_error_analysis(current_data)
        
        with tab2:
            # Test case selector
            test_case_options = [
                f"{'✅' if case['is_correct'] else '❌'} {case['description'][:60]}..." 
                for case in current_data
            ]
            
            selected_case_index = st.selectbox(
                "Select a test case to analyze:",
                range(len(test_case_options)),
                format_func=lambda x: test_case_options[x],
                key="test_case_selector"
            )
            
            render_test_case_analysis(current_data, selected_case_index)
        
        with tab3:
            render_custom_testing()
    else:
        st.warning("⚠️ No test results available. Please load a saved run or run a pipeline test.")
        
        # Still show custom testing tab
        st.markdown("---")
        render_custom_testing()


if __name__ == "__main__":
    main()
