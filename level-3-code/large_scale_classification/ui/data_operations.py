"""Data loading and transformation operations for the Streamlit UI.

This module handles all data operations including loading pipeline results,
managing saved runs, and transforming data for UI display.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


def load_latest_pipeline_results() -> Optional[Dict[str, Any]]:
    """Load the most recent pipeline test results from JSON files.
    
    Returns:
        Dictionary containing the latest pipeline test results, or None if no results found
    """
    project_root = Path(__file__).parent.parent
    
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
    project_root = Path(__file__).parent.parent
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
    project_root = Path(__file__).parent.parent
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
    project_root = Path(__file__).parent.parent
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
        
        # Get stage-specific candidates with backward compatibility
        embedding_candidates = result.get('embedding_candidates', [])
        llm_candidates = result.get('llm_candidates', [])
        
        # Backward compatibility: if no stage data, fall back gracefully
        if not embedding_candidates and not llm_candidates:
            # For older results without stage data, we can only approximate
            # Embedding stage: We don't have this data, so leave empty
            embedding_candidates = []
            # LLM stage: For hybrid strategy, final candidates are LLM output; for embedding-only, use final candidates
            if result.get('narrowing_strategy') == 'hybrid':
                llm_candidates = candidate_categories  # Final candidates came from LLM stage
            else:
                # For embedding-only strategy, there's no LLM stage
                embedding_candidates = candidate_categories
                llm_candidates = []
        elif not embedding_candidates:
            # If only embedding candidates are missing, leave empty (we can't infer this)
            embedding_candidates = []
        elif not llm_candidates:
            # If only LLM candidates are missing, use final for LLM stage
            llm_candidates = candidate_categories
        
        # Transform for UI
        test_case_data = {
            'description': test_case.get('text', 'Unknown'),
            'ground_truth': test_case.get('category', 'Unknown'),
            'test_type': test_case.get('test_type', 'unknown'),
            'stages': {
                'embedding': {
                    'candidates': embedding_candidates
                },
                'llm': {
                    'candidates': llm_candidates
                },
                'narrowing': {
                    'candidates': candidate_categories  # Final candidates (for backward compatibility)
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
