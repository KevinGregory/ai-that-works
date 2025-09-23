"""Main Streamlit application for Large Scale Classification System.

This is the main entry point for the Streamlit GUI that provides an interactive
interface for analyzing pipeline classification results and comparing different test runs.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add the src directory to Python path
import sys
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Ensure environment variables are loaded before importing settings
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Import UI modules
from ui.data_operations import (
    get_available_saved_runs, 
    load_latest_pipeline_results, 
    load_saved_run, 
    save_current_results_as_run, 
    transform_pipeline_results_for_ui
)
from ui.components import render_error_analysis, render_test_case_analysis, render_custom_testing


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


def render_sidebar():
    """Render the sidebar with saved runs and controls."""
    st.sidebar.markdown("## 📁 Saved Test Runs")
    
    # Get available saved runs
    saved_runs = get_available_saved_runs()
    
    if not saved_runs:
        st.sidebar.warning("No saved runs available. Run a pipeline test to create saved results.")
        return None, None, None
    
    # Create dropdown options
    run_options = [run['run_name'] for run in saved_runs]
    # run_options = ["Latest Results"] + [run['run_name'] for run in saved_runs]
    
    # Initialize session state
    if 'selected_run' not in st.session_state:
        st.session_state.selected_run = run_options[-1]
    
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
    else:
        current_data = None
    
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
                    # Check if the structure exists
                    if 'stages' in case and 'embedding' in case['stages'] and 'llm' in case['stages']:
                        embedding_counts.append(len(case['stages']['embedding']['candidates']))
                        llm_counts.append(len(case['stages']['llm']['candidates']))
                
                if embedding_counts and llm_counts:
                    avg_embedding = sum(embedding_counts) / len(embedding_counts)
                    avg_llm = sum(llm_counts) / len(llm_counts)
                    st.sidebar.markdown(f"**Avg Embedding Candidates:** {round(avg_embedding)}")
                    st.sidebar.markdown(f"**Avg LLM Candidates:** {round(avg_llm)}")
            except Exception as e:
                st.sidebar.markdown(f"**Debug Error:** {str(e)}")
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
            try:
                for case in current_data:
                    # Check if the structure exists
                    if 'stages' in case and 'embedding' in case['stages'] and 'llm' in case['stages']:
                        embedding_counts.append(len(case['stages']['embedding']['candidates']))
                        llm_counts.append(len(case['stages']['llm']['candidates']))
                
                if embedding_counts and llm_counts:
                    avg_embedding = sum(embedding_counts) / len(embedding_counts)
                    avg_llm = sum(llm_counts) / len(llm_counts)
                    st.sidebar.markdown(f"**Avg Embedding Candidates:** {int(avg_embedding)}")
                    st.sidebar.markdown(f"**Avg LLM Candidates:** {int(avg_llm)}")
            except Exception as e:
                st.sidebar.markdown(f"**Debug Error:** {str(e)}")
    
    return current_data


def render_main_content(current_data):
    """Render the main content area with tabs."""
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


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Large Scale Classification System</h1>', unsafe_allow_html=True)
    
    # Render sidebar and get data
    current_data = render_sidebar()
    
    # Render main content
    render_main_content(current_data)


if __name__ == "__main__":
    main()
