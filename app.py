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


# --- Add AI Scientist Project Root to Python Path ---
# Assuming app.py is in the root of your cloned repository
project_root = os.path.dirname(os.path.abspath(__file__))
ai_scientist_path = os.path.join(project_root, "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)
# --- END ---

# --- Import AI Scientist Modules ---
# NOTE: We will import more specifically later as needed
from ai_scientist.llm import create_client
from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea
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
    # Return final state or collected results if needed (e.g., best node summary)
    # For now, just signal completion
    return {"status": "completed", "plots": current_plots}


# --- Main AI Scientist Orchestration Logic ---


def run_ai_scientist_pipeline(topic_prompt, topic_area, progress=gr.Progress()):
    """
    Main generator function to run the AI Scientist pipeline and yield updates.
    """
    run_id = f"{get_timestamp()}_{topic_prompt[:20].replace(' ','_')}"  # Include part of prompt in run ID
    base_run_dir = os.path.join(project_root, "gradio_runs", run_id)
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"Starting Run: {run_id}, Directory: {base_run_dir}")

    # --- Phase 1: Ideation ---
    yield {
        global_status_textbox: gr.update(
            value=format_log("Starting Phase 1: Hypothesis Generation...")
        ),
        ideation_final_json: gr.update(value=None),
        ideation_log_textbox: gr.update(value=""),
        ideation_sem_scholar_textbox: gr.update(value=""),
        tabs: gr.update(selected=1),  # Switch to Hypothesis tab
    }
    progress(0.1, desc="Generating Hypothesis")

    # Prepare for ideation
    workshop_desc = (
        f"## Research Topic\nInvestigate novel ideas related to: {topic_prompt}"
    )
    if topic_area != "Default":
        workshop_desc += f"\nFocus specifically on areas relevant to {topic_area}."

    # Define path for saving generated ideas
    idea_json_path = os.path.join(base_run_dir, "generated_ideas.json")

    # Configure ideation parameters (can be made Gradio inputs later)
    # Reduce generations/reflections for faster demo turnaround
    max_generations = 1  # Generate only 1 idea for the demo
    num_reflections = 2  # Use fewer reflections

    # Select the model (can be made a Gradio input later)
    # Using a capable model is important for good ideation
    ideation_model = "gpt-4o-mini"  # Cheaper/faster GPT-4o-mini
    # ideation_model = "gpt-4o" # Or full GPT-4o

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
        base_run_dir, "idea.json"
    )  # Path where we saved the idea

    if not os.path.exists(final_idea_path):
        error_msg = format_log(
            "Experimentation FAILED: idea.json not found.", level="ERROR"
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
            exp_dir=base_run_dir,  # Pass the Gradio run directory
            idea_path=final_idea_path,
            new_config_path=run_config_path,  # Save the modified config here
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
        # Process monitor_result if needed (e.g., get final plot list)
        simulated_plots = monitor_result.get("plots", [])  # Get actual plots found

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
        # Switch to the Reporting tab
        tabs: gr.update(selected=3),
    }
    progress(0.8, desc="Generating Report")

    report_log = format_log("Starting Reporting...")
    yield {report_citation_log_textbox: report_log}  # Use citation log for now

    # TODO: Integrate actual reporting call (or refined simulation)
    # Simulate citation gathering
    report_log += format_log("Gathering citations (Simulated)...")
    yield {report_citation_log_textbox: report_log}
    time.sleep(2)
    sim_citations = """@article{simulated1, title="Simulated Paper 1", author="AI", year=2025}
@inproceedings{simulated2, title="Another Simulated Paper", booktitle="SimConf", year=2024}"""
    report_log += format_log("Citations gathered.")
    yield {report_citation_log_textbox: report_log + "\n---\n" + sim_citations}

    # Simulate LaTeX generation
    report_log = ""  # Reset for LaTeX logs
    yield {report_latex_code: gr.update(value="Generating initial LaTeX...")}
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

    # Determine plot path safely - use plots found by monitor
    last_plot_basename = (
        os.path.basename(simulated_plots[-1]) if simulated_plots else "placeholder.png"
    )

    sim_latex = (
        r"""\documentclass{article}
% \usepackage{iclr2025_conference,times} % Use workshop style later if needed
\usepackage{graphicx}
% \graphicspath{{figures/}} % Point to where Gradio saves plots
% --- Adjust graphicspath for local execution ---
% Assume plots are in logs/0-run/experiment_results relative to base_run_dir
% LaTeX runs in base_run_dir/latex, so path needs adjustment
\graphicspath{{../logs/0-run/experiment_results/}} % Relative path from latex subdir
% --- End Adjust ---
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\title{"""
        + actual_title
        + r"""}
\author{AI Scientist Gradio Demo}
\begin{document}
\maketitle
\begin{abstract}
"""
        + actual_abstract
        + r"""
\end{abstract}
\section{Introduction}
Simulated introduction. Hypothesis: """
        + actual_hypothesis
        + r"""
\section{Experiments}
We ran experiments. See Figure \ref{fig:sim}. % Changed text slightly
\begin{figure}[h]
\centering
% Use the basename of the last plot found by monitor
\includegraphics[width=0.5\textwidth]{"""
        + last_plot_basename
        + r"""}
\caption{Experiment results.} % Changed caption
\label{fig:sim}
\end{figure}
\section{Conclusion}
Simulated conclusion.
\bibliography{references} % Use a standard bib file name
% \bibliographystyle{iclr2025_conference} % Use standard style for now
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
    yield {report_latex_code: gr.update(value=sim_latex)}

    # Simulate Reflection
    yield {
        report_reflection_textbox: gr.update(value="Running reflection (Simulated)...")
    }
    time.sleep(1)
    yield {
        report_reflection_textbox: gr.update(
            value="VLM Feedback: Figure 1 caption okay. Page limit: OK."
        )
    }

    # Simulate Compilation
    yield {
        report_compile_log_textbox: gr.update(value="Compiling LaTeX (Simulated)...")
    }
    time.sleep(2)
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
        # compile_latex(latex_dir, pdf_path) # Call the real compile function
        # For now, skip actual compilation in simulation phase
        print("Skipping actual compilation during simulation phase.")
        # --- End real compilation ---
        # Fallback to dummy file if compilation fails or is skipped
        if not os.path.exists(pdf_path):
            with open(pdf_path, "w") as f:
                f.write(f"This is a simulated PDF for the topic: {topic_prompt}\n")
                f.write("\nContent based on LaTeX:\n")
                f.write(sim_latex)

        yield {
            report_compile_log_textbox: gr.update(
                value="Compilation successful (Simulated)."
            ),
            report_pdf_output: gr.update(value=pdf_path),
        }
    except Exception as e:
        yield {
            report_compile_log_textbox: gr.update(
                value=f"Compilation failed (Simulated): {e}", level="ERROR"
            )
        }

    progress(1.0, desc="Pipeline Complete")
    yield {
        global_status_textbox: gr.update(
            value=format_log("AI Scientist Pipeline Finished.")
        )
    }


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Scientist v2 - Gradio Demo")
    gr.Markdown(
        "Enter a research topic to start the automated scientific discovery process."
    )

    with gr.Tabs() as tabs:
        # --- Tab 1: Input & Setup ---
        with gr.TabItem("Input & Setup", id=0):
            with gr.Row():
                with gr.Column(scale=3):
                    topic_textbox = gr.Textbox(
                        label="Research Topic / Prompt",
                        placeholder="e.g., Investigate the effect of learning rate on model robustness to noisy labels",
                    )
                with gr.Column(scale=1):
                    topic_area_dropdown = gr.Dropdown(
                        label="Focus Area (Optional)",
                        choices=["Default", "Core ML", "Real-world Application"],
                        value="Default",
                    )
            start_button = gr.Button("Start AI Scientist v2")
            global_status_textbox = gr.Textbox(
                label="Global Status Log",
                lines=5,
                max_lines=10,
                interactive=False,
                autoscroll=True,
            )

        # --- Tab 2: Hypothesis Generation ---
        with gr.TabItem("1. Hypothesis Generation", id=1):
            gr.Markdown("## Phase 1: Hypothesis Generation")
            ideation_log_textbox = gr.Textbox(
                label="Ideation & Reflection Log",
                lines=10,
                interactive=False,
                autoscroll=True,
            )
            with gr.Accordion("Literature Search Logs (Semantic Scholar)", open=False):
                ideation_sem_scholar_textbox = gr.Textbox(
                    label="Semantic Scholar Interaction", lines=5, interactive=False
                )
            ideation_final_json = gr.JSON(label="Final Generated Idea")

        # --- Tab 3: Experimentation ---
        with gr.TabItem("2. Experimentation", id=2):
            gr.Markdown("## Phase 2: Experimentation (Agentic Tree Search)")
            exp_stage_markdown = gr.Markdown("Stage: Not Started")
            exp_log_textbox = gr.Textbox(
                label="Tree Search Execution Log",
                lines=15,
                interactive=False,
                autoscroll=True,
            )
            with gr.Accordion("Generated Plots", open=True):
                exp_plot_gallery = gr.Gallery(
                    label="Experiment Plots",
                    show_label=False,
                    elem_id="gallery",
                    columns=[4],
                    rows=[2],
                    object_fit="contain",
                    height="auto",
                )
            with gr.Accordion("Best Node Summary (End of Stage)", open=False):
                exp_best_node_json = gr.JSON(label="Best Node Data")

        # --- Tab 4: Reporting ---
        with gr.TabItem("3. Reporting", id=3):
            gr.Markdown("## Phase 3: Reporting")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion(
                        "Citation Gathering Logs (Semantic Scholar)", open=False
                    ):
                        report_citation_log_textbox = gr.Textbox(
                            label="Citation Log", lines=10, interactive=False
                        )
                    with gr.Accordion("LaTeX Reflection & Compilation", open=True):
                        report_reflection_textbox = gr.Textbox(
                            label="Reflection Log", lines=5, interactive=False
                        )
                        report_compile_log_textbox = gr.Textbox(
                            label="LaTeX Compilation Log", lines=5, interactive=False
                        )
                with gr.Column(scale=2):
                    report_latex_code = gr.Code(
                        label="Generated LaTeX",
                        language="latex",
                        lines=20,
                        interactive=False,
                    )
            report_pdf_output = gr.File(label="Generated PDF Output")

    # --- Button Click Action ---
    start_button.click(
        fn=run_ai_scientist_pipeline,
        inputs=[topic_textbox, topic_area_dropdown],
        outputs=[
            # Define outputs for all components that can be updated by the generator
            global_status_textbox,
            tabs,  # Input Tab updates
            ideation_log_textbox,
            ideation_sem_scholar_textbox,
            ideation_final_json,  # Hypothesis Tab updates
            exp_stage_markdown,
            exp_log_textbox,
            exp_plot_gallery,
            exp_best_node_json,  # Experimentation Tab updates
            report_citation_log_textbox,
            report_reflection_textbox,
            report_compile_log_textbox,
            report_latex_code,
            report_pdf_output,  # Reporting Tab updates
        ],
        show_progress="full",  # Show Gradio's built-in progress bar
    )


if __name__ == "__main__":
    # Load API keys from environment variables
    print("Checking for API keys...")
    # Add any other keys your chosen ideation model might need (e.g., Anthropic, Bedrock)
    required_keys = ["OPENAI_API_KEY"]
    # Optional key for literature search
    optional_keys = ["S2_API_KEY"]

    missing_keys = [key for key in required_keys if key not in os.environ]
    if missing_keys:
        print(
            f"\n!!! WARNING: Missing required environment variables: {', '.join(missing_keys)} !!!"
        )
        print("Ideation might fail. Please set them before running the application.")
        # sys.exit(1) # Exit if keys are essential

    missing_optional = [key for key in optional_keys if key not in os.environ]
    if missing_optional:
        print(
            f"\nINFO: Missing optional environment variables: {', '.join(missing_optional)}. Semantic Scholar search might be limited."
        )

    print("Launching Gradio Demo...")
    # Share=True creates a public link, requires login tunnel if run locally without easy public IP.
    # Set debug=True for more detailed Gradio errors during development.
    demo.launch(debug=True, share=True)  # Set share=True to try ngrok tunneling
