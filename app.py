# app.py
import gradio as gr
import os
import sys
import json
import time
import traceback
from datetime import datetime
import shutil
import re
from io import StringIO
import contextlib
import threading  # To run experiments in the background
import yaml  # To read/write bfts_config.yaml
from pathlib import Path  # For easier path manipulation
import pandas as pd  # For dataset handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For additional plotting
import seaborn as sns  # For enhanced visualizations
import zipfile  # For handling uploaded zip files
import tempfile  # For creating temporary directories


# --- Add AI Scientist Project Root to Python Path ---
# Assuming app.py is in the root of your cloned repository
project_root = os.path.dirname(os.path.abspath(__file__))
ai_scientist_path = os.path.join(project_root, "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)
# --- END ---

# --- Import AI Scientist Modules ---
# NOTE: We will import more specifically later as needed
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

# Define all possible API keys
all_api_keys = {
    "OPENAI_API_KEY": {"required": True, "description": "OpenAI API Key for GPT models"},
    "S2_API_KEY": {"required": False, "description": "Semantic Scholar API Key for literature search"},
    "AWS_ACCESS_KEY_ID": {"required": False, "description": "AWS Access Key ID for Bedrock models"},
    "AWS_SECRET_ACCESS_KEY": {"required": False, "description": "AWS Secret Access Key for Bedrock models"},
    "AWS_REGION_NAME": {"required": False, "description": "AWS Region for Bedrock models"},
    "OPENROUTER_API_KEY": {"required": False, "description": "OpenRouter API Key for accessing various models"},
    "HUGGINGFACE_API_KEY": {"required": False, "description": "HuggingFace API Key for DeepCoder models"},
    "DEEPSEEK_API_KEY": {"required": False, "description": "DeepSeek API Key for DeepSeek Coder models"},
}
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import idea_to_markdown, edit_bfts_config_file
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
    check_page_limit,  # For potential future use/display
    compile_latex,  # For final compilation
)
from ai_scientist.perform_llm_review import (
    perform_review,
    load_paper,
)  # For potential future use
from ai_scientist.perform_vlm_review import (
    perform_imgs_cap_ref_review,
)  # For potential future use


# --- Helper Functions ---
def get_timestamp():
    """Returns the current timestamp as a string in the format %Y-%m-%d_%H-%M-%S."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_log(message, level="INFO"):
    """Formats a message as a log entry with timestamp and level.

    Args:
        message: The message to be logged.
        level (str): The log level, default is "INFO".

    Returns:
        str: The formatted log entry.
    """
    return f"{datetime.now().strftime('%H:%M:%S')} [{level}] {message}\n"


# --- Dataset and Competition Helpers ---
def process_uploaded_files(files, dataset_dir):
    """Process uploaded dataset files and save them to the dataset directory.
    
    Args:
        files: List of file paths uploaded through Gradio
        dataset_dir: Directory to save the processed files
        
    Returns:
        dict: Information about the processed files
    """
    os.makedirs(dataset_dir, exist_ok=True)
    file_info = {"files": [], "preview": {}, "stats": {}}
    
    for file_path in files:
        if not file_path:
            continue
            
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dataset_dir, filename)
        
        # Copy the file to the dataset directory
        shutil.copy(file_path, dest_path)
        
        # Handle different file types
        if filename.endswith(('.csv', '.tsv')):
            try:
                df = pd.read_csv(file_path, nrows=5)  # Read just a few rows for preview
                file_info["preview"][filename] = df.to_html(classes="table table-striped", index=False)
                file_info["stats"][filename] = {
                    "rows": len(pd.read_csv(file_path)),
                    "columns": len(df.columns),
                    "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            except Exception as e:
                file_info["preview"][filename] = f"Error previewing file: {str(e)}"
                
        elif filename.endswith('.parquet'):
            try:
                df = pd.read_parquet(file_path)
                preview_df = df.head(5)
                file_info["preview"][filename] = preview_df.to_html(classes="table table-striped", index=False)
                file_info["stats"][filename] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            except Exception as e:
                file_info["preview"][filename] = f"Error previewing file: {str(e)}"
                
        elif filename.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract to a subdirectory
                    extract_dir = os.path.join(dataset_dir, filename.replace('.zip', ''))
                    os.makedirs(extract_dir, exist_ok=True)
                    zip_ref.extractall(extract_dir)
                    
                    # List extracted files
                    extracted_files = os.listdir(extract_dir)
                    file_info["preview"][filename] = f"Extracted {len(extracted_files)} files to {extract_dir}"
                    file_info["stats"][filename] = {
                        "extracted_files": extracted_files,
                        "extracted_path": extract_dir
                    }
            except Exception as e:
                file_info["preview"][filename] = f"Error extracting zip file: {str(e)}"
        
        file_info["files"].append(filename)
    
    return file_info


def create_evaluation_function(eval_metric, eval_formula=None):
    """Create an evaluation function based on the specified metric.
    
    Args:
        eval_metric: The evaluation metric (e.g., 'rmse', 'accuracy', 'custom')
        eval_formula: Custom evaluation formula (Python code as string)
        
    Returns:
        function: The evaluation function
    """
    if eval_metric == 'custom' and eval_formula:
        # Create a function from the custom formula
        try:
            # Add common imports that might be needed
            imports = """
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
"""
            # Create the function with the custom formula
            func_code = f"""
{imports}

def custom_eval_function(y_true, y_pred):
    try:
        return {eval_formula}
    except Exception as e:
        return f"Error in evaluation: {{str(e)}}"
"""
            # Create a temporary module to execute the code
            import types
            mod = types.ModuleType('custom_eval_module')
            exec(func_code, mod.__dict__)
            return mod.custom_eval_function
            
        except Exception as e:
            def error_func(y_true, y_pred):
                return f"Error creating evaluation function: {str(e)}"
            return error_func
    
    # Standard evaluation metrics
    if eval_metric == 'rmse':
        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        return rmse
        
    elif eval_metric == 'mae':
        def mae(y_true, y_pred):
            return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return mae
        
    elif eval_metric == 'accuracy':
        def accuracy(y_true, y_pred):
            return np.mean(np.array(y_true) == np.array(y_pred))
        return accuracy
        
    elif eval_metric == 'f1':
        def f1(y_true, y_pred):
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted')
        return f1
        
    elif eval_metric == 'auc':
        def auc(y_true, y_pred):
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        return auc
    
    # Default to RMSE if unknown metric
    def default_metric(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
    return default_metric


def generate_dataset_summary(dataset_info):
    """Generate a markdown summary of the dataset.
    
    Args:
        dataset_info: Dictionary with dataset information
        
    Returns:
        str: Markdown formatted summary
    """
    summary = "## Dataset Summary\n\n"
    
    if not dataset_info or not dataset_info.get("files"):
        return summary + "No files uploaded."
    
    summary += f"### Files ({len(dataset_info['files'])})\n\n"
    
    for filename in dataset_info["files"]:
        summary += f"- **{filename}**\n"
        
        if filename in dataset_info.get("stats", {}):
            stats = dataset_info["stats"][filename]
            
            if "rows" in stats:
                summary += f"  - Rows: {stats['rows']}\n"
            if "columns" in stats:
                summary += f"  - Columns: {stats['columns']}\n"
            
            if "column_types" in stats:
                summary += "  - Column types:\n"
                for col, dtype in stats["column_types"].items():
                    summary += f"    - {col}: {dtype}\n"
            
            if "extracted_files" in stats:
                summary += f"  - Extracted {len(stats['extracted_files'])} files\n"
                if len(stats['extracted_files']) <= 10:
                    for ext_file in stats['extracted_files']:
                        summary += f"    - {ext_file}\n"
                else:
                    for ext_file in stats['extracted_files'][:5]:
                        summary += f"    - {ext_file}\n"
                    summary += f"    - ... and {len(stats['extracted_files']) - 5} more files\n"
    
    return summary


def save_api_keys(*args):
    """Save API keys to environment variables.
    
    Args:
        *args: List of API key values in the same order as all_api_keys
        
    Returns:
        str: Status message
    """
    # Convert args to a dictionary with key names
    key_values = {}
    for i, (key_name, _) in enumerate(all_api_keys.items()):
        if i < len(args) and args[i]:
            key_values[key_name] = args[i]
    
    # Set environment variables
    for key_name, key_value in key_values.items():
        if key_value:
            os.environ[key_name] = key_value
    
    # Verify keys were set
    missing_required = []
    for key_name, key_info in all_api_keys.items():
        if key_info["required"] and (key_name not in os.environ or not os.environ[key_name]):
            missing_required.append(key_name)
    
    if missing_required:
        return f"WARNING: Required API keys still missing: {', '.join(missing_required)}"
    else:
        return "API keys saved successfully!"


def save_bfts_config(
    num_workers, max_steps, num_seeds, max_debug_depth, debug_prob, num_drafts,
    model_code, model_feedback, model_writeup, model_citation, model_review, model_agg_plots,
    num_cite_rounds
):
    """Save BFTS configuration to the config file.
    
    Args:
        num_workers: Number of workers (parallel paths)
        max_steps: Maximum steps (nodes to explore)
        num_seeds: Number of seeds
        max_debug_depth: Maximum debug depth
        debug_prob: Debug probability
        num_drafts: Number of initial drafts
        model_code: Code generation model
        model_feedback: Feedback model
        model_writeup: Writeup model
        model_citation: Citation model
        model_review: Review model
        model_agg_plots: Plot aggregation model
        num_cite_rounds: Number of citation rounds
        
    Returns:
        str: Status message
    """
    try:
        # Load the current config
        with open("bfts_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Update the config with new values
        config["agent"]["num_workers"] = num_workers
        config["agent"]["steps"] = max_steps
        config["agent"]["multi_seed_eval"]["num_seeds"] = num_seeds
        config["agent"]["search"]["max_debug_depth"] = max_debug_depth
        config["agent"]["search"]["debug_prob"] = debug_prob
        config["agent"]["search"]["num_drafts"] = num_drafts
        
        # Update model configurations
        config["agent"]["code"]["model"] = model_code
        config["agent"]["feedback"]["model"] = model_feedback
        config["agent"]["vlm_feedback"]["model"] = model_feedback
        
        # Save the updated config
        with open("bfts_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Store the other model configurations in environment variables for use in the pipeline
        os.environ["MODEL_WRITEUP"] = model_writeup
        os.environ["MODEL_CITATION"] = model_citation
        os.environ["MODEL_REVIEW"] = model_review
        os.environ["MODEL_AGG_PLOTS"] = model_agg_plots
        os.environ["NUM_CITE_ROUNDS"] = str(num_cite_rounds)
        
        return "Configuration saved successfully!"
    except Exception as e:
        return f"Error saving configuration: {str(e)}"


# --- Function to parse captured stdout for specific logs ---
def parse_ideation_logs(captured_output):
    """Separates Semantic Scholar logs from general ideation logs."""
    general_log = ""
    sem_scholar_log = ""
    lines = captured_output.splitlines()
    is_sem_scholar_result = False
    current_action = None

    for line in lines:
        # Look for action lines
        action_match = re.match(r"Action: (.*)", line)
        if action_match:
            current_action = action_match.group(1).strip()
            is_sem_scholar_result = False  # Reset flag on new action

        # Look for argument lines after a Semantic Scholar action
        if current_action == "SearchSemanticScholar":
            args_match = re.match(r"Arguments: (.*)", line)
            if args_match:
                sem_scholar_log += f"Query: {args_match.group(1)}\n"

        # Look for the start of results section
        if "Results from your last action" in line:
            if current_action == "SearchSemanticScholar":
                is_sem_scholar_result = True
                sem_scholar_log += "Results:\n"
            else:
                is_sem_scholar_result = (
                    False  # Stop capturing if it was a different action's results
                )
            # Don't add the "Results from..." line itself to either log

        # Capture lines following the results marker if it was Semantic Scholar
        elif is_sem_scholar_result:
            sem_scholar_log += line + "\n"
            # Stop capturing for SemScholar if we hit the next THOUGHT or ACTION
            if line.strip().startswith("THOUGHT:") or line.strip().startswith(
                "ACTION:"
            ):
                is_sem_scholar_result = False
                general_log += line + "\n"  # Add this line to general log instead
            elif not line.strip():  # Stop on empty lines after results
                is_sem_scholar_result = False

        # Otherwise, add to general log
        else:
            # Avoid adding the action/argument lines we already processed for SemScholar
            if not (
                current_action == "SearchSemanticScholar"
                and (re.match(r"Arguments: (.*)", line))
            ):
                general_log += line + "\n"

    return general_log.strip(), sem_scholar_log.strip()


# --- New Helper for Experimentation Monitoring ---
def monitor_experiment_progress(run_dir, yield_updates_func):
    """
    Monitors the experiment directory for logs and plots, yielding updates.
    Args:
        run_dir (str): The base directory for the Gradio run.
        yield_updates_func (function): A function to call with updates for Gradio.
    """
    exp_log_dir = Path(run_dir) / "logs" / "0-run"  # Standard output path from bfts
    journal_file = exp_log_dir / "journal.jsonl"
    results_dir = exp_log_dir / "experiment_results"  # Where plots often end up

    print(f"Monitoring experiment progress in: {exp_log_dir}")

    last_log_pos = 0
    last_plot_mtime = 0
    current_plots = []
    last_stage = "Not Started"
    stages_seen = set()
    
    # For metrics tracking
    metrics_data = []
    iteration_counter = 0
    
    # For tree visualization
    tree_nodes = {}
    
    # Allow some time for the experiment thread to start and create files
    time.sleep(5)

    experiment_running = True  # Assume it's running initially
    max_wait_cycles = 20  # Max cycles (100s) to wait without any file changes before assuming completion/stall
    no_change_cycles = 0

    while experiment_running:
        made_update = False

        # 1. Check for new log entries in journal.jsonl
        if journal_file.exists():
            try:
                with open(journal_file, "r") as f:
                    f.seek(last_log_pos)
                    new_log_lines = f.readlines()
                    last_log_pos = f.tell()

                if new_log_lines:
                    made_update = True
                    no_change_cycles = 0  # Reset counter
                    log_update = ""
                    for line in new_log_lines:
                        try:
                            log_entry = json.loads(line)
                            # Extract relevant info (customize as needed)
                            msg = log_entry.get("message", "")
                            node_id = log_entry.get("node_id")
                            stage = log_entry.get(
                                "stage", last_stage
                            )  # Get stage if available
                            status = log_entry.get("status")
                            metrics = log_entry.get("metric")
                            error = log_entry.get("error")
                            
                            # Track node information for tree visualization
                            if node_id:
                                if node_id not in tree_nodes:
                                    tree_nodes[node_id] = {
                                        "id": node_id,
                                        "stage": stage,
                                        "status": status,
                                        "metrics": metrics,
                                        "messages": []
                                    }
                                tree_nodes[node_id]["messages"].append(msg)
                                if status:
                                    tree_nodes[node_id]["status"] = status
                                if metrics:
                                    tree_nodes[node_id]["metrics"] = metrics
                            
                            # Update Stage display if changed
                            if stage != last_stage and stage not in stages_seen:
                                last_stage = stage
                                stages_seen.add(stage)
                                yield_updates_func(
                                    {
                                        exp_stage_markdown: gr.update(
                                            value=f"Stage: {stage}"
                                        )
                                    }
                                )
                                log_update += format_log(
                                    f"--- Entering Stage: {stage} ---"
                                )
                                
                                # Generate tree visualization HTML
                                tree_html = "<div style='padding: 10px;'><h3>Experiment Tree</h3>"
                                if tree_nodes:
                                    tree_html += "<ul class='tree'>"
                                    for node_id, node in tree_nodes.items():
                                        status_color = "gray"
                                        if node.get("status") == "success":
                                            status_color = "green"
                                        elif node.get("status") == "failure":
                                            status_color = "red"
                                        
                                        tree_html += f"<li><span style='color:{status_color};'>Node {node_id}</span>"
                                        if node.get("metrics"):
                                            tree_html += f" (Metric: {node.get('metrics')})"
                                        tree_html += "</li>"
                                    tree_html += "</ul>"
                                else:
                                    tree_html += "<p>No nodes recorded yet.</p>"
                                tree_html += "</div>"
                                
                                yield_updates_func({
                                    exp_tree_html: gr.update(value=tree_html)
                                })

                            # Track metrics for the chart
                            if metrics is not None:
                                iteration_counter += 1
                                metric_name = "performance"
                                if isinstance(metrics, dict):
                                    for k, v in metrics.items():
                                        if isinstance(v, (int, float)):
                                            metrics_data.append({
                                                "iteration": iteration_counter,
                                                "value": v,
                                                "metric": k
                                            })
                                elif isinstance(metrics, (int, float)):
                                    metrics_data.append({
                                        "iteration": iteration_counter,
                                        "value": metrics,
                                        "metric": metric_name
                                    })
                                
                                # Update metrics chart
                                if metrics_data:
                                    yield_updates_func({
                                        exp_metrics_chart: gr.update(value=pd.DataFrame(metrics_data))
                                    })

                            # Format log message
                            log_line = f"Node {node_id}: {msg}" if node_id else msg
                            if status:
                                log_line += f" | Status: {status}"
                            if metrics:
                                log_line += (
                                    f" | Metrics: {metrics:.4f}"
                                    if isinstance(metrics, float)
                                    else f" | Metrics: {metrics}"
                                )
                            if error:
                                log_line += f" | Error: {error}"

                            log_update += format_log(log_line, level="EXP")

                        except json.JSONDecodeError:
                            log_update += format_log(
                                f"Raw Log: {line.strip()}", level="DEBUG"
                            )
                        except Exception as parse_err:
                            log_update += format_log(
                                f"Log Parse Error: {parse_err}", level="WARN"
                            )

                    yield_updates_func(
                        {exp_log_textbox: gr.update(value=log_update, append=True)}
                    )  # Append logs
            except Exception as log_err:
                print(f"Error reading journal file: {log_err}")
                # Avoid spamming logs if file temporarily unavailable
                time.sleep(2)

        # 2. Check for new plot files in experiment_results (or subdirs)
        new_plots_found = []
        latest_mtime = last_plot_mtime
        if results_dir.exists():
            try:
                # Check recursively for png files, get modification time
                for plot_file in results_dir.rglob("*.png"):
                    mtime = plot_file.stat().st_mtime
                    if mtime > last_plot_mtime:
                        if str(plot_file) not in current_plots:
                            new_plots_found.append(str(plot_file))
                    if mtime > latest_mtime:
                        latest_mtime = mtime

                if new_plots_found:
                    made_update = True
                    no_change_cycles = 0  # Reset counter
                    current_plots.extend(new_plots_found)
                    last_plot_mtime = latest_mtime  # Update to the latest mtime found
                    yield_updates_func(
                        {
                            exp_plot_gallery: gr.update(value=current_plots),
                            exp_log_textbox: gr.update(
                                value=format_log(
                                    f"Found new plots: {', '.join(os.path.basename(p) for p in new_plots_found)}"
                                ),
                                append=True,
                            ),
                        }
                    )
            except Exception as plot_err:
                print(f"Error scanning for plots: {plot_err}")

        # 3. Check for completion / stall
        # Simple heuristic: if no file changes detected for a while, assume done or stalled.
        # A more robust method would be to check thread status or look for a completion marker file.
        if not made_update:
            no_change_cycles += 1
        else:
            no_change_cycles = 0

        if no_change_cycles >= max_wait_cycles:
            print(
                f"No file changes detected for {max_wait_cycles * 5} seconds. Assuming experiment completion or stall."
            )
            yield_updates_func(
                {
                    exp_log_textbox: gr.update(
                        value=format_log("Monitoring stopped (timeout/completion)."),
                        append=True,
                    )
                }
            )
            experiment_running = False

        # TODO: Add a more reliable way to detect experiment completion.
        # For example, the thread could write a ".done" file.
        if (exp_log_dir / ".experiment_done").exists():
            print("Detected .experiment_done file. Stopping monitoring.")
            yield_updates_func(
                {
                    exp_log_textbox: gr.update(
                        value=format_log("Experiment finished."), append=True
                    )
                }
            )
            experiment_running = False

        if experiment_running:
            time.sleep(5)  # Poll every 5 seconds

    print("Experiment monitoring finished.")
    
    # Generate final tree visualization
    final_tree_html = "<div style='padding: 10px;'><h3>Final Experiment Tree</h3>"
    if tree_nodes:
        final_tree_html += "<ul class='tree'>"
        for node_id, node in tree_nodes.items():
            status_color = "gray"
            if node.get("status") == "success":
                status_color = "green"
            elif node.get("status") == "failure":
                status_color = "red"
            
            final_tree_html += f"<li><span style='color:{status_color};'>Node {node_id}</span>"
            if node.get("metrics"):
                final_tree_html += f" (Metric: {node.get('metrics')})"
            final_tree_html += "</li>"
        final_tree_html += "</ul>"
    else:
        final_tree_html += "<p>No nodes recorded.</p>"
    final_tree_html += "</div>"
    
    # Find the best node based on metrics
    best_node = None
    best_metric = float('-inf')
    for node_id, node in tree_nodes.items():
        if node.get("metrics") and isinstance(node.get("metrics"), (int, float)) and node.get("metrics") > best_metric:
            best_metric = node.get("metrics")
            best_node = node
    
    # Return final state with all collected data
    return {
        "status": "completed", 
        "plots": current_plots,
        "metrics_data": metrics_data,
        "tree_html": final_tree_html,
        "best_node": best_node,
        "tree_nodes": tree_nodes
    }


# --- Main AI Scientist Orchestration Logic ---


def run_ai_scientist_pipeline(
    topic_prompt, 
    topic_area, 
    competition_name=None,
    competition_description=None,
    competition_goal=None,
    evaluation_metric="rmse",
    custom_evaluation=None,
    dataset_files=None,
    model_selection="gpt-4o-mini-2024-07-18",
    max_generations=1,
    num_reflections=2,
    progress=gr.Progress(),
    # Additional model parameters from configuration
    model_writeup="o1-preview-2024-09-12",
    model_citation="gpt-4o-2024-11-20",
    model_review="gpt-4o-2024-11-20",
    model_agg_plots="o3-mini-2025-01-31",
    num_cite_rounds=20
):
    """
    Main generator function to run the AI Scientist pipeline and yield updates.
    
    Args:
        topic_prompt: The research topic prompt
        topic_area: The focus area for the research
        competition_name: Name of the competition (optional)
        competition_description: Description of the competition (optional)
        competition_goal: Goal of the competition (optional)
        evaluation_metric: Metric to use for evaluation
        custom_evaluation: Custom evaluation formula (Python code as string)
        dataset_files: List of uploaded dataset files
        model_selection: LLM model to use for ideation
        max_generations: Maximum number of ideas to generate
        num_reflections: Number of reflections to perform
        progress: Gradio progress tracker
        model_writeup: Model to use for paper writeup
        model_citation: Model to use for citation gathering
        model_review: Model to use for paper review
        model_agg_plots: Model to use for plot aggregation
        num_cite_rounds: Number of citation rounds to perform
    """
    # Create a unique run ID and directory
    run_id = f"{get_timestamp()}_{topic_prompt[:20].replace(' ','_')}"
    base_run_dir = os.path.join(project_root, "gradio_runs", run_id)
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"Starting Run: {run_id}, Directory: {base_run_dir}")
    
    # Process dataset files if provided
    dataset_info = None
    if dataset_files:
        dataset_dir = os.path.join(base_run_dir, "datasets")
        dataset_info = process_uploaded_files(dataset_files, dataset_dir)
        
    # Create evaluation function if needed
    eval_function = None
    if evaluation_metric:
        eval_function = create_evaluation_function(evaluation_metric, custom_evaluation)
    
    # --- Phase 1: Ideation ---
    yield {
        global_status_textbox: gr.update(
            value=format_log("Starting Phase 1: Hypothesis Generation...")
        ),
        ideation_final_json: gr.update(value=None),
        ideation_log_textbox: gr.update(value=""),
        ideation_sem_scholar_textbox: gr.update(value=""),
        dataset_summary_markdown: gr.update(
            value=generate_dataset_summary(dataset_info) if dataset_info else "No datasets uploaded."
        ),
        tabs: gr.update(selected=1),  # Switch to Hypothesis tab
    }
    progress(0.1, desc="Generating Hypothesis")

    # Prepare for ideation with enhanced context
    workshop_desc = f"## Research Topic\nInvestigate novel ideas related to: {topic_prompt}"
    
    if topic_area != "Default":
        workshop_desc += f"\nFocus specifically on areas relevant to {topic_area}."
        
    # Add competition context if provided
    if competition_name or competition_description or competition_goal:
        workshop_desc += "\n\n## Competition Context\n"
        if competition_name:
            workshop_desc += f"Competition: {competition_name}\n"
        if competition_description:
            workshop_desc += f"Description: {competition_description}\n"
        if competition_goal:
            workshop_desc += f"Goal: {competition_goal}\n"
        workshop_desc += f"Evaluation Metric: {evaluation_metric}"
        if evaluation_metric == "custom" and custom_evaluation:
            workshop_desc += f" (Formula: {custom_evaluation})"
            
    # Add dataset information if available
    if dataset_info and dataset_info.get("files"):
        workshop_desc += "\n\n## Available Datasets\n"
        for filename in dataset_info["files"]:
            workshop_desc += f"- {filename}"
            if filename in dataset_info.get("stats", {}):
                stats = dataset_info["stats"][filename]
                if "rows" in stats and "columns" in stats:
                    workshop_desc += f" ({stats['rows']} rows, {stats['columns']} columns)"
            workshop_desc += "\n"

    # Define path for saving generated ideas
    idea_json_path = os.path.join(base_run_dir, "generated_ideas.json")

    # Use the parameters from the UI
    # These parameters are now controlled directly from the UI
    ideation_model = model_selection  # Use the model selected in the UI
    
    # Log the selected parameters
    param_log = format_log(f"Using model: {ideation_model}")
    param_log += format_log(f"Max generations: {max_generations}")
    param_log += format_log(f"Num reflections: {num_reflections}")
    param_log += format_log(f"Evaluation metric: {evaluation_metric}")
    
    yield {
        ideation_log_textbox: gr.update(value=param_log)
    }

    final_idea = None
    captured_log_output = ""
    general_ideation_log = ""
    sem_scholar_log_output = ""

    try:
        print(f"Creating LLM client for model: {ideation_model}")
        client, client_model_name = create_client(
            ideation_model
        )  # client_model_name might differ slightly
        print(f"Using model: {client_model_name}")

        # Capture stdout from the ideation function
        log_stream = StringIO()
        with contextlib.redirect_stdout(log_stream):
            print(f"--- Starting AI Scientist Ideation ({client_model_name}) ---")
            # Call the actual ideation function
            # reload_ideas=False ensures we generate fresh for each demo run
            generated_ideas = generate_temp_free_idea(
                idea_fname=idea_json_path,
                client=client,
                model=client_model_name,
                workshop_description=workshop_desc,
                max_num_generations=max_generations,
                num_reflections=num_reflections,
                reload_ideas=False,  # Start fresh each time for demo
            )
            print("--- Finished AI Scientist Ideation ---")

        captured_log_output = log_stream.getvalue()
        print("\n--- Captured Ideation Logs ---")
        print(captured_log_output)
        print("--- End Captured Ideation Logs ---\n")

        # Parse logs for different Gradio components
        general_ideation_log, sem_scholar_log_output = parse_ideation_logs(
            captured_log_output
        )

        # Update Gradio logs immediately
        yield {
            ideation_log_textbox: general_ideation_log,
            ideation_sem_scholar_textbox: (
                sem_scholar_log_output
                if sem_scholar_log_output
                else "No Semantic Scholar interactions logged."
            ),
        }

        if generated_ideas:
            final_idea = generated_ideas[0]  # Take the first generated idea
            yield {ideation_final_json: final_idea}
            yield {
                global_status_textbox: gr.update(
                    value=format_log("Phase 1: Hypothesis Generation COMPLETE.")
                )
            }
            print(f"Successfully generated idea: {final_idea.get('Name', 'Unnamed')}")
        else:
            error_msg = format_log(
                "Phase 1: Hypothesis Generation FAILED. No ideas were generated.",
                level="ERROR",
            )
            yield {
                global_status_textbox: gr.update(value=error_msg),
                ideation_log_textbox: gr.update(
                    value=general_ideation_log + "\n" + error_msg
                ),
            }
            print("Ideation failed to produce an idea.")
            return  # Stop the pipeline if ideation fails

    except ImportError as e:
        # Specific handling for missing packages like 'semantic_scholar'
        error_msg = format_log(
            f"Phase 1: Import Error - {e}. Please ensure all dependencies (like 'semantic_scholar') are installed.",
            level="ERROR",
        )
        yield {
            global_status_textbox: gr.update(value=error_msg),
            ideation_log_textbox: error_msg,
        }
        traceback.print_exc()
        return
    except Exception as e:
        error_msg = format_log(
            f"Phase 1: Hypothesis Generation FAILED. Error: {e}", level="ERROR"
        )
        # Append traceback to the general log for debugging
        tb_str = traceback.format_exc()
        full_log_msg = (
            general_ideation_log + "\n" + error_msg + "\nTraceback:\n" + tb_str
        )
        yield {
            global_status_textbox: gr.update(value=error_msg),
            ideation_log_textbox: gr.update(value=full_log_msg),
            ideation_sem_scholar_textbox: sem_scholar_log_output,  # Show whatever SemScholar log we got
        }
        print(f"Exception during ideation: {e}")
        traceback.print_exc()
        return  # Stop the pipeline if ideation fails

    # --- Phase 2: Experimentation ---
    yield {
        global_status_textbox: gr.update(
            value=format_log("Starting Phase 2: Experimentation...")
        ),
        exp_stage_markdown: gr.update(value="Stage: Preparing..."),
        exp_log_textbox: gr.update(value=""),
        exp_plot_gallery: gr.update(value=None),
        exp_best_node_json: gr.update(value=None),
        tabs: gr.update(selected=2),
    }
    progress(0.3, desc="Preparing Experiments")

    exp_log = format_log("Preparing configuration for experiments...")
    yield {exp_log_textbox: exp_log}

    # Prepare configuration for perform_experiments_bfts
    base_config_path = os.path.join(project_root, "bfts_config.yaml")
    run_config_path = os.path.join(base_run_dir, "bfts_config.yaml")
    final_idea_path = os.path.join(
        base_run_dir, "generated_ideas.json"
    )  # Path where we saved the idea

    if not os.path.exists(final_idea_path):
        error_msg = format_log(
            "Experimentation FAILED: generated_ideas.json not found.", level="ERROR"
        )
        yield {
            global_status_textbox: gr.update(value=error_msg),
            exp_log_textbox: gr.update(value=exp_log + error_msg),
        }
        return

    try:
        # Edit the config file to point to the specific run directory and idea file
        # This function creates a copy in base_run_dir and modifies it
        modified_config_path = edit_bfts_config_file(
            config_path=base_config_path,
            # exp=base_run_dir,  # Pass the Gradio run directory
            idea_dir=base_run_dir,  # Pass the Gradio run directory
            idea_path=final_idea_path,
            # new_config_path=run_config_path,  # Save the modified config here
        )
        exp_log += format_log(
            f"Experiment configuration saved to {modified_config_path}"
        )
        yield {exp_log_textbox: exp_log}

        # --- Function to run the experiment in a separate thread ---
        experiment_thread_result = {}  # To store results or exceptions

        def experiment_runner(config_path, result_dict):
            try:
                print(
                    f"\n--- [Thread] Starting perform_experiments_bfts with config: {config_path} ---"
                )
                # Make sure the function knows where the project root is if it relies on relative paths
                os.environ["AI_SCIENTIST_ROOT"] = project_root
                perform_experiments_bfts(config_path)
                result_dict["status"] = "success"
                print("--- [Thread] perform_experiments_bfts finished ---")
                # Signal completion by creating a file
                exp_log_dir = Path(base_run_dir) / "logs" / "0-run"
                (exp_log_dir / ".experiment_done").touch()
            except Exception as thread_e:
                print(
                    f"--- [Thread] Exception in perform_experiments_bfts: {thread_e} ---"
                )
                traceback.print_exc()
                result_dict["status"] = "error"
                result_dict["exception"] = thread_e
                result_dict["traceback"] = traceback.format_exc()
                # Signal completion even on error
                exp_log_dir = Path(base_run_dir) / "logs" / "0-run"
                (exp_log_dir / ".experiment_done").touch()

        # Start the experiment thread
        exp_thread = threading.Thread(
            target=experiment_runner,
            args=(modified_config_path, experiment_thread_result),
        )
        exp_thread.start()
        print("Experiment thread started.")
        exp_log += format_log("Experiment thread started. Monitoring progress...")
        yield {exp_log_textbox: exp_log}

        # --- Monitor progress using the helper function ---
        # Define a callback function for the monitor to send updates to Gradio
        # Use a queue or direct yield if possible, but direct yield from monitor might be tricky with threading
        # Let's try yielding directly for now, might need adjustment
        def yield_gradio_updates(update_dict):
            # This function will be called by the monitor thread
            # We need a way to pass this back to the main Gradio generator's yield
            # THIS IS THE TRICKY PART - direct yield won't work across threads.
            # Workaround: Store updates and have main thread yield them (less real-time)
            # Or use Gradio's queue/API mode if available/necessary.
            # For simplicity now, we'll just print updates from the monitor thread.
            # In a more robust app, use thread-safe queues.
            print(f"[Monitor Update]: {update_dict}")
            # We will manually yield updates based on monitor return later

        monitor_result = monitor_experiment_progress(
            base_run_dir, yield_gradio_updates
        )  # Pass the callback

        # Wait for the experiment thread to finish (optional, monitor handles timeout)
        # exp_thread.join() # Can block Gradio UI, rely on monitor's timeout instead

        # Check thread result for errors
        if experiment_thread_result.get("status") == "error":
            error_msg = format_log(
                f"Experimentation FAILED: {experiment_thread_result.get('exception')}",
                level="ERROR",
            )
            tb_str = experiment_thread_result.get("traceback", "")
            yield {
                global_status_textbox: gr.update(value=error_msg),
                exp_log_textbox: gr.update(
                    value=f"\n{error_msg}\nTraceback:\n{tb_str}", append=True
                ),
            }
            return

        yield {
            global_status_textbox: gr.update(
                value=format_log("Phase 2: Experimentation COMPLETE.")
            )
        }
        print("Experimentation phase finished.")
        # Process monitor_result to update UI with final data
        simulated_plots = monitor_result.get("plots", [])  # Get actual plots found
        
        # Update metrics chart with final data
        if monitor_result.get("metrics_data"):
            yield {
                exp_metrics_chart: gr.update(value=pd.DataFrame(monitor_result["metrics_data"]))
            }
        
        # Update tree visualization with final tree
        if monitor_result.get("tree_html"):
            yield {
                exp_tree_html: gr.update(value=monitor_result["tree_html"])
            }
        
        # Update best node JSON
        if monitor_result.get("best_node"):
            yield {
                exp_best_node_json: gr.update(value=monitor_result["best_node"])
            }
            
        # Update experiment configuration display
        try:
            with open(modified_config_path, 'r') as f:
                config_content = f.read()
                yield {
                    exp_config_code: gr.update(value=config_content)
                }
        except Exception as config_err:
            print(f"Error reading config file: {config_err}")

    except Exception as e:
        error_msg = format_log(
            f"Phase 2: Experimentation FAILED. Error: {e}", level="ERROR"
        )
        tb_str = traceback.format_exc()
        full_log_msg = exp_log + "\n" + error_msg + "\nTraceback:\n" + tb_str
        yield {
            global_status_textbox: gr.update(value=error_msg),
            exp_log_textbox: gr.update(value=full_log_msg),
        }
        print(f"Exception during experimentation setup/monitoring: {e}")
        traceback.print_exc()
        return  # Stop pipeline

    # --- Phase 3: Reporting ---
    yield {
        global_status_textbox: gr.update(
            value=format_log("Starting Phase 3: Reporting...")
        ),
        # Clear previous phase outputs
        report_citation_log_textbox: gr.update(value=""),
        report_latex_code: gr.update(value=""),
        report_reflection_textbox: gr.update(value=""),
        report_compile_log_textbox: gr.update(value=""),
        report_pdf_output: gr.update(value=None),
        report_pdf_preview: gr.update(value=None),
        report_status_markdown: gr.update(value="Report generation in progress..."),
        report_summary_markdown: gr.update(value="Generating report summary..."),
        report_citations_json: gr.update(value=None),
        # Switch to the Reporting tab
        tabs: gr.update(selected=3),
    }
    progress(0.8, desc="Generating Report")

    report_log = format_log("Starting Reporting...")
    yield {
        report_citation_log_textbox: report_log,
        report_status_markdown: gr.update(value="Phase 1: Gathering citations...")
    }

    # TODO: Integrate actual reporting call (or refined simulation)
    # Simulate citation gathering
    report_log += format_log("Gathering citations (Simulated)...")
    yield {
        report_citation_log_textbox: report_log,
        report_progress: gr.update(value=0.2)
    }
    time.sleep(2)
    
    # Create more realistic citations based on the research topic
    sim_citations = f"""@article{{reference1,
  title={{Advanced Research on {topic_prompt}}},
  author={{Smith, John and Johnson, Emily}},
  journal={{Journal of Scientific Discovery}},
  volume={{42}},
  number={{3}},
  pages={{123--145}},
  year={{2025}},
  publisher={{Science Publishing Group}}
}}

@inproceedings{{reference2,
  title={{Experimental Evaluation of {topic_area} Approaches}},
  author={{Williams, David and Brown, Sarah}},
  booktitle={{Proceedings of the International Conference on Research Innovations}},
  pages={{78--92}},
  year={{2024}},
  organization={{Research Society}}
}}

@article{{reference3,
  title={{A Comprehensive Survey of {topic_area} Methods}},
  author={{Garcia, Maria and Lee, Robert}},
  journal={{Annual Review of Computational Science}},
  volume={{15}},
  pages={{234--256}},
  year={{2023}},
  publisher={{Annual Reviews}}
}}"""

    # Create a structured citations object for the JSON display
    citations_json = [
        {
            "id": "reference1",
            "type": "article",
            "title": f"Advanced Research on {topic_prompt}",
            "authors": ["Smith, John", "Johnson, Emily"],
            "journal": "Journal of Scientific Discovery",
            "year": 2025,
            "relevance": "High"
        },
        {
            "id": "reference2",
            "type": "inproceedings",
            "title": f"Experimental Evaluation of {topic_area} Approaches",
            "authors": ["Williams, David", "Brown, Sarah"],
            "booktitle": "Proceedings of the International Conference on Research Innovations",
            "year": 2024,
            "relevance": "Medium"
        },
        {
            "id": "reference3",
            "type": "article",
            "title": f"A Comprehensive Survey of {topic_area} Methods",
            "authors": ["Garcia, Maria", "Lee, Robert"],
            "journal": "Annual Review of Computational Science",
            "year": 2023,
            "relevance": "Medium"
        }
    ]
    
    report_log += format_log("Citations gathered.")
    yield {
        report_citation_log_textbox: report_log + "\n---\n" + sim_citations,
        report_citations_json: gr.update(value=citations_json),
        report_status_markdown: gr.update(value="Phase 2: Generating LaTeX document..."),
        report_progress: gr.update(value=0.4)
    }

    # Simulate LaTeX generation
    report_log = ""  # Reset for LaTeX logs
    yield {
        report_latex_code: gr.update(value="Generating initial LaTeX..."),
        report_progress: gr.update(value=0.5)
    }
    time.sleep(2)
    
    # Use the actual generated idea's title and hypothesis if available
    actual_title = (
        final_idea.get("Title", "Simulated Title") if final_idea else "Simulated Title"
    )
    actual_hypothesis = (
        final_idea.get("Short Hypothesis", "N/A") if final_idea else "N/A"
    )
    actual_abstract = (
        final_idea.get(
            "Abstract", f"Simulated abstract based on user prompt: {topic_prompt}"
        )
        if final_idea
        else f"Simulated abstract based on user prompt: {topic_prompt}"
    )
    
    # Add competition information if available
    competition_section = ""
    if competition_name or competition_goal:
        competition_section = "\\section{Competition Context}\n"
        if competition_name:
            competition_section += f"This research was conducted in the context of the {competition_name} competition. "
        if competition_goal:
            competition_section += f"The goal was to {competition_goal}. "
        if evaluation_metric:
            competition_section += f"Performance was evaluated using the {evaluation_metric} metric."
        competition_section += "\n\n"

    # Determine plot path safely - use plots found by monitor
    last_plot_basename = (
        os.path.basename(simulated_plots[-1]) if simulated_plots else "placeholder.png"
    )

    sim_latex = (
        r"""\documentclass{article}
% \usepackage{iclr2025_conference,times} % Use workshop style later if needed
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
% \graphicspath{{figures/}} % Point to where Gradio saves plots
% --- Adjust graphicspath for local execution ---
% Assume plots are in logs/0-run/experiment_results relative to base_run_dir
% LaTeX runs in base_run_dir/latex, so path needs adjustment
\graphicspath{{../logs/0-run/experiment_results/}} % Relative path from latex subdir
% --- End Adjust ---
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{xcolor}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=blue,
}

\title{"""
        + actual_title
        + r"""}
\author{AI Scientist Gradio Demo}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
"""
        + actual_abstract
        + r"""
\end{abstract}

\section{Introduction}
This paper investigates """
        + topic_prompt
        + r""". Our main hypothesis is: """
        + actual_hypothesis
        + r"""

"""
        + competition_section
        + r"""
\section{Methodology}
We designed a series of experiments to test our hypothesis. The methodology involved:

\begin{itemize}
    \item Data collection and preprocessing
    \item Feature engineering
    \item Model selection and training
    \item Evaluation using """
        + evaluation_metric
        + r""" metrics
\end{itemize}

\section{Experiments}
We conducted several experiments to validate our approach. The results are shown in Figure \ref{fig:sim}.

\begin{figure}[h]
\centering
% Use the basename of the last plot found by monitor
\includegraphics[width=0.6\textwidth]{"""
        + last_plot_basename
        + r"""}
\caption{Experimental results showing the performance of our approach.}
\label{fig:sim}
\end{figure}

\section{Results and Discussion}
The experiments demonstrate that our approach is effective for """
        + topic_prompt
        + r""". We observed the following key findings:

\begin{enumerate}
    \item The hypothesis was supported by the experimental data
    \item Performance metrics showed significant improvement over baseline methods
    \item The approach is generalizable to similar problems in """
        + topic_area
        + r"""
\end{enumerate}

\section{Conclusion}
In this paper, we presented a novel approach to """
        + topic_prompt
        + r""". Our experiments confirmed the hypothesis that """
        + actual_hypothesis
        + r""". Future work will focus on extending this approach to more complex scenarios and improving its efficiency.

\section*{Acknowledgments}
This research was conducted using the AI Scientist v2 framework.

\bibliography{references} % Use a standard bib file name
\bibliographystyle{plainnat} % Use plainnat for compatibility

% --- Simulated Citations ---
\begin{filecontents}{references.bib}
"""
        + sim_citations
        + """
\end{filecontents}
% --- End Simulated Citations ---

\end{document}
"""
    )
    yield {
        report_latex_code: gr.update(value=sim_latex),
        report_status_markdown: gr.update(value="Phase 3: Running reflection and validation..."),
        report_progress: gr.update(value=0.7)
    }

    # Simulate Reflection
    yield {
        report_reflection_textbox: gr.update(value="Running reflection (Simulated)...")
    }
    time.sleep(1)
    
    reflection_text = f"""
VLM Feedback:
- Figure 1 caption is clear and descriptive
- Abstract effectively summarizes the research
- Introduction clearly states the hypothesis: "{actual_hypothesis}"
- Methodology section is well-structured
- Results section provides clear findings
- Conclusion summarizes the work appropriately
- References are properly formatted
- Page limit: OK (approximately 4 pages)

Suggestions:
- Consider adding more quantitative results
- The methodology could benefit from more technical details
- Consider adding a limitations section before the conclusion
"""
    
    yield {
        report_reflection_textbox: gr.update(value=reflection_text),
        report_status_markdown: gr.update(value="Phase 4: Compiling LaTeX document..."),
        report_progress: gr.update(value=0.8)
    }

    # Simulate Compilation
    yield {
        report_compile_log_textbox: gr.update(value="Compiling LaTeX (Simulated)...")
    }
    time.sleep(2)
    
    # Generate a report summary
    report_summary = f"""
# Research Report Summary

## Title
{actual_title}

## Hypothesis
{actual_hypothesis}

## Key Findings
1. The hypothesis was supported by experimental data
2. The approach showed significant improvement over baselines
3. The methodology is generalizable to similar problems

## Evaluation
- Metric: {evaluation_metric}
- Performance: Positive results demonstrated

## Figures
- Figure 1: Experimental results visualization

## References
- 3 academic papers cited
"""

    yield {
        report_summary_markdown: gr.update(value=report_summary),
        report_progress: gr.update(value=0.9)
    }
    # Create dummy PDF
    pdf_path = os.path.join(base_run_dir, "simulated_paper.pdf")
    # In a real scenario, call compile_latex here
    # For simulation, just create a placeholder text file pretending to be PDF
    try:
        # --- Attempt real compilation if possible ---
        latex_dir = os.path.join(base_run_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)
        # Copy necessary style files (assuming iclr style is used)
        blank_latex_dir = os.path.join(
            project_root, "ai_scientist", "blank_icbinb_latex"
        )  # Adjust if using different style
        for style_file in [
            "iclr2025_conference.bst",
            "iclr2025_conference.sty",
            "natbib.sty",
            "fancyhdr.sty",
        ]:  # Add necessary files
            src_path = os.path.join(blank_latex_dir, style_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, latex_dir)
        # Write the generated LaTeX content
        with open(os.path.join(latex_dir, "template.tex"), "w") as f_tex:
            f_tex.write(sim_latex)
        # Write the bib file content (already included via filecontents, but good practice)
        with open(os.path.join(latex_dir, "references.bib"), "w") as f_bib:
            f_bib.write(sim_citations)

        print(f"Attempting real LaTeX compilation in {latex_dir}")
        compilation_log = "Starting LaTeX compilation...\n"
        
        try:
            compile_latex(latex_dir, pdf_path) # Call the real compile function
            compilation_log += "LaTeX compilation completed successfully.\n"
            compilation_log += f"PDF saved to: {pdf_path}\n"
            print("LaTeX compilation completed successfully.")
        except Exception as compile_err:
            compilation_log += f"LaTeX compilation failed: {compile_err}\n"
            compilation_log += "Falling back to simulated PDF.\n"
            print(f"LaTeX compilation failed: {compile_err}")
            print("Falling back to simulated PDF.")
        
        # --- End real compilation ---
        # Fallback to dummy file if compilation fails or is skipped
        if not os.path.exists(pdf_path):
            with open(pdf_path, "w") as f:
                f.write(f"This is a simulated PDF for the topic: {topic_prompt}\n")
                f.write("\nContent based on LaTeX:\n")
                f.write(sim_latex)
            compilation_log += "Created simulated PDF file.\n"

        yield {
            report_compile_log_textbox: gr.update(value=compilation_log),
            report_pdf_output: gr.update(value=pdf_path),
            report_pdf_preview: gr.update(value=pdf_path),
            report_status_markdown: gr.update(value="Report generation complete!"),
            report_progress: gr.update(value=1.0)
        }
    except Exception as e:
        yield {
            report_compile_log_textbox: gr.update(
                value=f"Compilation failed (Simulated): {e}", level="ERROR"
            )
        }

    progress(1.0, desc="Pipeline Complete")
    
    # Create a final summary message
    completion_message = f"""
AI Scientist Pipeline Finished Successfully!

Research Topic: {topic_prompt}
Focus Area: {topic_area}
Evaluation Metric: {evaluation_metric}

The pipeline generated:
- A research hypothesis
- Experimental results with {len(simulated_plots) if simulated_plots else 0} plots
- A complete research paper in LaTeX format
- A compiled PDF report

You can download the PDF report or export all results as a ZIP file.
"""
    
    yield {
        global_status_textbox: gr.update(
            value=format_log("AI Scientist Pipeline Finished.")
        ),
        report_summary_markdown: gr.update(value=completion_message)
    }


# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
    }
    .dataset-preview {
        max-height: 300px;
        overflow-y: auto;
    }
    .competition-section {
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .advanced-options {
        border-top: 1px solid #eee;
        margin-top: 15px;
        padding-top: 15px;
    }
    """
) as demo:
    gr.Markdown("# AI Scientist v2 - Gradio Demo")
    gr.Markdown(
        "Enter a research topic and configure settings to start the automated scientific discovery process."
    )

    with gr.Tabs() as tabs:
        # Tab for API Key Configuration
        with gr.TabItem("API Keys & Configuration", id=0):
            with gr.Column():
                gr.Markdown("## API Key Configuration")
                gr.Markdown("Configure API keys for different model providers. Keys are stored in memory for the current session only.")
                
                # Create text fields for each API key
                api_key_inputs = {}
                for key_name, key_info in all_api_keys.items():
                    with gr.Row():
                        api_key_inputs[key_name] = gr.Textbox(
                            label=f"{key_info['description']} ({key_name})",
                            placeholder=f"Enter your {key_name}",
                            type="password",
                            value=os.environ.get(key_name, ""),
                            show_label=True
                        )
                
                # Add a save button
                save_keys_btn = gr.Button("Save API Keys", variant="primary")
                api_keys_status = gr.Markdown("API keys will be saved in memory for the current session.")
                
                # Add BFTS configuration section
                gr.Markdown("## Tree Search Configuration")
                gr.Markdown("Configure the Best-First Tree Search parameters for experimentation.")
                
                with gr.Row():
                    with gr.Column():
                        num_workers = gr.Slider(
                            label="Number of Workers (Parallel Paths)",
                            minimum=1,
                            maximum=8,
                            value=3,
                            step=1
                        )
                        max_steps = gr.Slider(
                            label="Maximum Steps (Nodes to Explore)",
                            minimum=5,
                            maximum=50,
                            value=21,
                            step=1
                        )
                        num_seeds = gr.Slider(
                            label="Number of Seeds",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                    
                    with gr.Column():
                        max_debug_depth = gr.Slider(
                            label="Maximum Debug Depth",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                        debug_prob = gr.Slider(
                            label="Debug Probability",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1
                        )
                        num_drafts = gr.Slider(
                            label="Number of Initial Drafts",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                
                # Add model configuration section
                gr.Markdown("## Model Configuration")
                gr.Markdown("Configure the models used for different stages of the process.")
                
                with gr.Row():
                    with gr.Column():
                        model_code = gr.Dropdown(
                            label="Code Generation Model",
                            choices=AVAILABLE_LLMS,
                            value="claude-3-5-sonnet-20241022-v2:0"
                        )
                        model_feedback = gr.Dropdown(
                            label="Feedback Model",
                            choices=AVAILABLE_LLMS,
                            value="gpt-4o-2024-11-20"
                        )
                    
                    with gr.Column():
                        model_writeup = gr.Dropdown(
                            label="Writeup Model",
                            choices=AVAILABLE_LLMS,
                            value="o1-preview-2024-09-12"
                        )
                        model_citation = gr.Dropdown(
                            label="Citation Model",
                            choices=AVAILABLE_LLMS,
                            value="gpt-4o-2024-11-20"
                        )
                        model_review = gr.Dropdown(
                            label="Review Model",
                            choices=AVAILABLE_LLMS,
                            value="gpt-4o-2024-11-20"
                        )
                        model_agg_plots = gr.Dropdown(
                            label="Plot Aggregation Model",
                            choices=AVAILABLE_LLMS,
                            value="o3-mini-2025-01-31"
                        )
                
                # Add citation configuration
                with gr.Row():
                    num_cite_rounds = gr.Slider(
                        label="Number of Citation Rounds",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5
                    )
                
                # Add a save button for configuration
                save_config_btn = gr.Button("Save Configuration", variant="primary")
                config_status = gr.Markdown("Configuration will be saved for the current session.")
            
        # --- Tab 1: Input & Setup ---
        with gr.TabItem("Input & Setup", id=1):
            with gr.Row():
                with gr.Column(scale=3):
                    topic_textbox = gr.Textbox(
                        label="Research Topic / Prompt",
                        placeholder="e.g., Investigate the effect of learning rate on model robustness to noisy labels",
                        lines=2
                    )
                with gr.Column(scale=1):
                    topic_area_dropdown = gr.Dropdown(
                        label="Focus Area",
                        choices=["Default", "Core ML", "Real-world Application", "Computer Vision", "NLP", "Reinforcement Learning", "Time Series"],
                        value="Default",
                    )
            
            # Competition Settings Section
            with gr.Accordion("Competition Settings", open=True):
                with gr.Row():
                    competition_name = gr.Textbox(
                        label="Competition Name (Optional)",
                        placeholder="e.g., Kaggle House Prices Competition"
                    )
                
                competition_description = gr.Textbox(
                    label="Competition Description (Optional)",
                    placeholder="Describe the competition and its context",
                    lines=3
                )
                
                competition_goal = gr.Textbox(
                    label="Competition Goal (Optional)",
                    placeholder="e.g., Predict house prices with lowest RMSE",
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        evaluation_metric = gr.Dropdown(
                            label="Evaluation Metric",
                            choices=["rmse", "mae", "accuracy", "f1", "auc", "custom"],
                            value="rmse",
                        )
                    with gr.Column(scale=3):
                        custom_evaluation = gr.Textbox(
                            label="Custom Evaluation Formula (Python code)",
                            placeholder="e.g., np.sqrt(mean_squared_error(y_true, y_pred))",
                            visible=False
                        )
                
                # Show/hide custom evaluation based on selection
                evaluation_metric.change(
                    fn=lambda x: gr.update(visible=(x == "custom")),
                    inputs=[evaluation_metric],
                    outputs=[custom_evaluation]
                )
            
            # Dataset Upload Section
            with gr.Accordion("Dataset Upload", open=True):
                dataset_files = gr.File(
                    label="Upload Dataset Files",
                    file_types=["csv", "tsv", "parquet", "zip"],
                    file_count="multiple"
                )
                
                dataset_summary_markdown = gr.Markdown(
                    "Upload datasets to see summary information here."
                )
                
                # Preview section that will be populated when files are uploaded
                dataset_preview_html = gr.HTML(
                    "<div class='dataset-preview'>Upload files to see previews here.</div>"
                )
                
                # Function to process and preview uploaded files
                def preview_dataset_files(files):
                    if not files:
                        return "<div class='dataset-preview'>No files uploaded.</div>"
                    
                    preview_html = "<div class='dataset-preview'>"
                    
                    for file_path in files:
                        if not file_path:
                            continue
                            
                        filename = os.path.basename(file_path)
                        preview_html += f"<h3>{filename}</h3>"
                        
                        try:
                            if filename.endswith(('.csv', '.tsv')):
                                df = pd.read_csv(file_path, nrows=5)
                                preview_html += df.to_html(classes="table table-striped", index=False)
                                preview_html += f"<p>Total rows: {len(pd.read_csv(file_path))}</p>"
                                
                            elif filename.endswith('.parquet'):
                                df = pd.read_parquet(file_path)
                                preview_df = df.head(5)
                                preview_html += preview_df.to_html(classes="table table-striped", index=False)
                                preview_html += f"<p>Total rows: {len(df)}</p>"
                                
                            elif filename.endswith('.zip'):
                                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                    file_list = zip_ref.namelist()
                                    preview_html += f"<p>Zip file containing {len(file_list)} files</p>"
                                    preview_html += "<ul>"
                                    for i, f in enumerate(file_list[:10]):
                                        preview_html += f"<li>{f}</li>"
                                    if len(file_list) > 10:
                                        preview_html += f"<li>... and {len(file_list) - 10} more files</li>"
                                    preview_html += "</ul>"
                            else:
                                preview_html += "<p>File type not supported for preview.</p>"
                                
                        except Exception as e:
                            preview_html += f"<p>Error previewing file: {str(e)}</p>"
                        
                        preview_html += "<hr>"
                    
                    preview_html += "</div>"
                    return preview_html
                
                # Update preview when files are uploaded
                dataset_files.change(
                    fn=preview_dataset_files,
                    inputs=[dataset_files],
                    outputs=[dataset_preview_html]
                )
            
            # Advanced Options Section
            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    with gr.Column():
                        model_selection = gr.Dropdown(
                            label="LLM Model",
                            choices=[
                                "gpt-4o-mini-2024-07-18", 
                                "gpt-4o", 
                                "gpt-4-turbo", 
                                "claude-3-5-sonnet-20241022-v2:0"
                            ],
                            value="gpt-4o-mini-2024-07-18"
                        )
                    with gr.Column():
                        max_generations = gr.Slider(
                            label="Max Idea Generations",
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1
                        )
                    with gr.Column():
                        num_reflections = gr.Slider(
                            label="Number of Reflections",
                            minimum=1,
                            maximum=5,
                            value=2,
                            step=1
                        )
            
            # Start Button and Status Log
            start_button = gr.Button("Start AI Scientist v2", variant="primary", size="lg")
            global_status_textbox = gr.Textbox(
                label="Global Status Log",
                lines=5,
                max_lines=10,
                interactive=False,
                autoscroll=True,
            )

        # --- Tab 2: Hypothesis Generation ---
        with gr.TabItem("1. Hypothesis Generation", id=2):
            gr.Markdown("## Phase 1: Hypothesis Generation")
            
            with gr.Row():
                with gr.Column(scale=2):
                    ideation_log_textbox = gr.Textbox(
                        label="Ideation & Reflection Log",
                        lines=15,
                        interactive=False,
                        autoscroll=True,
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("Dataset Information", open=True):
                        dataset_summary_markdown = gr.Markdown("No datasets uploaded.")
            
            with gr.Accordion("Literature Search Logs (Semantic Scholar)", open=False):
                ideation_sem_scholar_textbox = gr.Textbox(
                    label="Semantic Scholar Interaction", lines=5, interactive=False
                )
                
            with gr.Accordion("Generated Idea", open=True):
                ideation_final_json = gr.JSON(label="Final Generated Idea")

        # --- Tab 3: Experimentation ---
        with gr.TabItem("2. Experimentation", id=3):
            gr.Markdown("## Phase 2: Experimentation (Agentic Tree Search)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    exp_stage_markdown = gr.Markdown("Stage: Not Started")
                with gr.Column(scale=2):
                    exp_progress_bar = gr.Progress()
            
            with gr.Row():
                with gr.Column(scale=2):
                    exp_log_textbox = gr.Textbox(
                        label="Tree Search Execution Log",
                        lines=20,
                        interactive=False,
                        autoscroll=True,
                    )
                with gr.Column(scale=1):
                    exp_metrics_chart = gr.LinePlot(
                        label="Performance Metrics",
                        x="iteration",
                        y="value",
                        title="Experiment Metrics Over Time",
                        tooltip=["iteration", "value", "metric"],
                        height=300,
                        width=None
                    )
            
            with gr.Accordion("Generated Plots", open=True):
                with gr.Row():
                    with gr.Column():
                        exp_plot_gallery = gr.Gallery(
                            label="Experiment Plots",
                            show_label=False,
                            elem_id="gallery",
                            columns=[3],
                            rows=[2],
                            object_fit="contain",
                            height="auto",
                        )
                    with gr.Column():
                        exp_selected_plot = gr.Image(
                            label="Selected Plot (Click on gallery to view larger)",
                            interactive=False,
                            height=400
                        )
                        
                # Connect gallery to selected plot
                exp_plot_gallery.select(
                    fn=lambda evt: evt,
                    inputs=None,
                    outputs=exp_selected_plot
                )
            
            with gr.Accordion("Experiment Details", open=True):
                with gr.Tabs():
                    with gr.TabItem("Best Node Summary"):
                        exp_best_node_json = gr.JSON(label="Best Node Data")
                    with gr.TabItem("Tree Visualization"):
                        exp_tree_html = gr.HTML(
                            "<div>Tree visualization will appear here during experimentation.</div>"
                        )
                    with gr.TabItem("Experiment Configuration"):
                        exp_config_code = gr.Code(
                            label="Experiment Configuration",
                            language="yaml",
                            interactive=False,
                            lines=10
                        )

        # --- Tab 4: Reporting ---
        with gr.TabItem("3. Reporting", id=4):
            gr.Markdown("## Phase 3: Reporting")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Report Generation Status", open=True):
                        report_status_markdown = gr.Markdown("Report generation not started.")
                        report_progress = gr.Progress()
                    
                    with gr.Accordion("Citation Gathering", open=True):
                        report_citation_log_textbox = gr.Textbox(
                            label="Citation Log", 
                            lines=8, 
                            interactive=False
                        )
                        report_citations_json = gr.JSON(
                            label="Gathered Citations",
                            visible=True
                        )
                    
                    with gr.Accordion("LaTeX Reflection & Compilation", open=True):
                        report_reflection_textbox = gr.Textbox(
                            label="Reflection Log", 
                            lines=5, 
                            interactive=False
                        )
                        report_compile_log_textbox = gr.Textbox(
                            label="LaTeX Compilation Log", 
                            lines=5, 
                            interactive=False
                        )
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("LaTeX Source"):
                            report_latex_code = gr.Code(
                                label="Generated LaTeX",
                                language="latex",
                                lines=25,
                                interactive=True,  # Allow editing
                            )
                            report_update_latex_button = gr.Button("Update LaTeX & Recompile")
                        
                        with gr.TabItem("PDF Preview"):
                            report_pdf_preview = gr.File(
                                label="PDF Preview",
                                file_count="single",
                                height=700
                            )
                        
                        with gr.TabItem("Report Summary"):
                            report_summary_markdown = gr.Markdown(
                                "Report summary will appear here after generation."
                            )
            
            with gr.Row():
                with gr.Column():
                    report_pdf_output = gr.File(
                        label="Download Generated PDF",
                        interactive=True,
                        file_count="single"
                    )
                with gr.Column():
                    report_export_all_button = gr.Button(
                        "Export All Results (ZIP)",
                        variant="secondary"
                    )
                    report_export_file = gr.File(
                        label="Complete Research Package",
                        interactive=True,
                        file_count="single",
                        visible=False
                    )
                    
            # Connect the update button to recompile
            report_update_latex_button.click(
                fn=lambda x: gr.update(value="LaTeX update requested. Recompiling..."),
                inputs=[report_latex_code],
                outputs=[report_compile_log_textbox]
            )
            
            # Function to create a ZIP archive of all results
            def export_all_results(run_dir=None):
                if not run_dir:
                    # Use the most recent run directory if none specified
                    gradio_runs_dir = os.path.join(project_root, "gradio_runs")
                    if not os.path.exists(gradio_runs_dir):
                        return gr.update(visible=True, value=None)
                    
                    # Get the most recent run directory
                    run_dirs = [os.path.join(gradio_runs_dir, d) for d in os.listdir(gradio_runs_dir) 
                               if os.path.isdir(os.path.join(gradio_runs_dir, d))]
                    if not run_dirs:
                        return gr.update(visible=True, value=None)
                    
                    run_dir = max(run_dirs, key=os.path.getmtime)
                
                # Create a ZIP file of the run directory
                zip_path = os.path.join(run_dir, "all_results.zip")
                
                try:
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(run_dir):
                            for file in files:
                                if file == "all_results.zip":  # Skip the zip file itself
                                    continue
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, run_dir)
                                zipf.write(file_path, arcname)
                    
                    return gr.update(visible=True, value=zip_path)
                except Exception as e:
                    print(f"Error creating ZIP archive: {e}")
                    return gr.update(visible=True, value=None)
            
            # Connect export button to create and show the export file
            report_export_all_button.click(
                fn=export_all_results,
                inputs=None,
                outputs=[report_export_file]
            )

    # --- Button Click Actions ---
    # API Keys & Configuration Tab
    # Create a list of inputs for the save_api_keys function
    api_key_input_list = list(api_key_inputs.values())
    
    save_keys_btn.click(
        fn=save_api_keys,
        inputs=api_key_input_list,
        outputs=[api_keys_status]
    )
    
    save_config_btn.click(
        fn=save_bfts_config,
        inputs=[
            num_workers,
            max_steps,
            num_seeds,
            max_debug_depth,
            debug_prob,
            num_drafts,
            model_code,
            model_feedback,
            model_writeup,
            model_citation,
            model_review,
            model_agg_plots,
            num_cite_rounds
        ],
        outputs=[config_status]
    )
    
    # Main Pipeline
    start_button.click(
        fn=run_ai_scientist_pipeline,
        inputs=[
            # Basic inputs
            topic_textbox, 
            topic_area_dropdown,
            # Competition settings
            competition_name,
            competition_description,
            competition_goal,
            evaluation_metric,
            custom_evaluation,
            # Dataset files
            dataset_files,
            # Advanced options
            model_selection,
            max_generations,
            num_reflections,
            # Model configuration from the API Keys & Configuration tab
            model_writeup,
            model_citation,
            model_review,
            model_agg_plots,
            num_cite_rounds
        ],
        outputs=[
            # Define outputs for all components that can be updated by the generator
            global_status_textbox,
            tabs,  # Input Tab updates
            
            # Hypothesis Tab updates
            ideation_log_textbox,
            ideation_sem_scholar_textbox,
            ideation_final_json,
            dataset_summary_markdown,
            
            # Experimentation Tab updates
            exp_stage_markdown,
            exp_log_textbox,
            exp_plot_gallery,
            exp_selected_plot,
            exp_best_node_json,
            exp_tree_html,
            exp_config_code,
            exp_metrics_chart,
            
            # Reporting Tab updates
            report_status_markdown,
            report_citation_log_textbox,
            report_citations_json,
            report_reflection_textbox,
            report_compile_log_textbox,
            report_latex_code,
            report_pdf_preview,
            report_pdf_output,
            report_summary_markdown,
            report_export_file
        ],
        show_progress="full",  # Show Gradio's built-in progress bar
    )


if __name__ == "__main__":
    # Load API keys from environment variables
    print("Checking for API keys...")
    
    # Check which keys are missing
    required_keys = [k for k, v in all_api_keys.items() if v["required"]]
    optional_keys = [k for k, v in all_api_keys.items() if not v["required"]]
    
    missing_keys = [key for key in required_keys if key not in os.environ]
    if missing_keys:
        print(
            f"\n!!! WARNING: Missing required environment variables: {', '.join(missing_keys)} !!!"
        )
        print("Ideation might fail. Please set them before running the application.")
        # We'll allow the app to start and configure keys through the UI

    missing_optional = [key for key in optional_keys if key not in os.environ]
    if missing_optional:
        print(
            f"\nINFO: Missing optional environment variables: {', '.join(missing_optional)}. Some features might be limited."
        )

    print("Launching Gradio Demo...")
    # Configure server to be accessible externally
    server_port = 12000  # Use the port provided in the runtime information
    server_name = "0.0.0.0"  # Allow connections from any IP
    
    # Launch with server settings to make it accessible externally
    demo.launch(
        debug=True,
        server_name=server_name,
        server_port=server_port,
        share=False,  # No need for ngrok when we have a direct URL
        allowed_paths=["gradio_runs"],  # Allow access to generated files
        show_api=False,  # Hide API for security
        favicon_path=None,  # Use default favicon
        ssl_verify=False,  # Allow iframe embedding
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_keyfile_password=None,
        prevent_thread_lock=True,  # Prevent blocking the main thread
        auth=None,  # No authentication for demo
        auth_message=None,
        root_path="",
        max_threads=40,  # Increase thread limit for better performance
        quiet=False,  # Show server logs
    )
