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
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_log(message, level="INFO"):
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


# --- Main AI Scientist Orchestration Logic ---


# Placeholder for the actual logic - we will fill this in
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
        # Clear previous phase outputs
        exp_stage_markdown: gr.update(value="Stage: Not Started"),
        exp_log_textbox: gr.update(value=""),
        exp_plot_gallery: gr.update(value=None),
        exp_best_node_json: gr.update(value=None),
        # Switch to the Experimentation tab
        tabs: gr.update(selected=2),
    }
    progress(0.3, desc="Running Experiments (Simulated Tree Search)")

    exp_log = format_log("Starting Experimentation Tree Search...")
    yield {exp_log_textbox: exp_log}

    # TODO: Integrate actual experimentation call (or refined simulation)
    # Simulate stages and node execution
    stages = [
        "Preliminary Investigation",
        "Hyperparameter Tuning",
        "Research Agenda Execution",
        "Ablation Studies",
    ]
    simulated_plots = []
    for i, stage in enumerate(stages):
        yield {exp_stage_markdown: gr.update(value=f"Stage: {stage}")}
        exp_log += format_log(f"--- Entering Stage: {stage} ---")
        yield {exp_log_textbox: exp_log}
        for node_idx in range(3):  # Simulate a few nodes per stage
            progress(
                0.3 + (i * 3 + node_idx + 1) * (0.5 / (len(stages) * 3)),
                desc=f"Running {stage} - Node {node_idx+1}",
            )
            exp_log += format_log(f"Generating Node {i+1}-{node_idx+1}...")
            yield {exp_log_textbox: exp_log}
            time.sleep(0.5)
            exp_log += format_log(f"Executing Node {i+1}-{node_idx+1}...")
            yield {exp_log_textbox: exp_log}
            time.sleep(1)
            is_buggy = (i + node_idx) % 4 == 1  # Simulate some bugs
            if is_buggy:
                exp_log += format_log(
                    f"Node {i+1}-{node_idx+1} FAILED. Error: Simulated ValueError.",
                    level="ERROR",
                )
                yield {exp_log_textbox: exp_log}
            else:
                exp_log += format_log(
                    f"Node {i+1}-{node_idx+1} SUCCESS. Metric: {0.8 - i*0.05 - node_idx*0.02:.2f}"
                )
                # Simulate plot generation
                plot_path = os.path.join(
                    base_run_dir, f"sim_plot_s{i+1}_n{node_idx+1}.png"
                )
                # Create a dummy plot file (replace with actual plot generation if needed)
                from PIL import Image, ImageDraw

                img = Image.new("RGB", (300, 200), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text(
                    (10, 10),
                    f"Simulated Plot\nStage {i+1}, Node {node_idx+1}",
                    fill=(255, 255, 0),
                )
                img.save(plot_path)
                simulated_plots.append(plot_path)
                exp_log += format_log(f"Generated plot: {os.path.basename(plot_path)}")
                exp_log += format_log(
                    f"VLM Feedback: Plot looks reasonable (Simulated)"
                )
                yield {exp_log_textbox: exp_log, exp_plot_gallery: simulated_plots}
            time.sleep(0.5)
        # Simulate selecting best node
        exp_log += format_log(f"Selecting best node for stage {stage}...")
        yield {exp_log_textbox: exp_log}
        time.sleep(0.5)
        best_node_sim = {
            "stage": stage,
            "best_metric": 0.8 - i * 0.05,
            "selected_node": f"{i+1}-0",
        }
        yield {exp_best_node_json: best_node_sim}

    # Simulate final plot aggregation step within experimentation phase (as per original code)
    exp_log += format_log("Aggregating final plots...")
    yield {exp_log_textbox: exp_log}
    time.sleep(1)
    # Add a dummy aggregated plot
    agg_plot_path = os.path.join(base_run_dir, "sim_plot_aggregated.png")
    img = Image.new("RGB", (400, 300), color=(137, 73, 109))
    d = ImageDraw.Draw(img)
    d.text((10, 10), f"Simulated Aggregated Plot", fill=(255, 255, 0))
    img.save(agg_plot_path)
    simulated_plots.append(agg_plot_path)
    yield {exp_plot_gallery: simulated_plots}
    exp_log += format_log("Plot aggregation complete.")
    yield {exp_log_textbox: exp_log}

    experiment_results = {
        "summary": "Simulated experiment results",
        "plot_paths": simulated_plots,
    }  # Pass info

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
    sim_latex = (
        r"""\documentclass{article}
\usepackage{iclr2025_conference,times} % Use workshop style later if needed
\usepackage{graphicx}
\graphicspath{{figures/}} % Point to where Gradio saves plots
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\title{"""
        + final_idea.get("Title", "Simulated Title")
        + r"""}
\author{AI Scientist Gradio Demo}
\begin{document}
\maketitle
\begin{abstract}
Simulated abstract based on user prompt: """
        + topic_prompt
        + r"""
\end{abstract}
\section{Introduction}
Simulated introduction. Hypothesis: """
        + final_idea.get("Short Hypothesis", "N/A")
        + r"""
\section{Experiments}
We ran simulated experiments. See Figure \ref{fig:sim}.
\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{"""
        + os.path.basename(
            simulated_plots[-1] if simulated_plots else "placeholder.png"
        )
        + r"""}
\caption{Simulated aggregated results.}
\label{fig:sim}
\end{{figure}
\section{Conclusion}
Simulated conclusion.
\bibliography{iclr2025_conference} % Use a dummy bib file name
\bibliographystyle{iclr2025_conference}
% --- Simulated Citations ---
\begin{filecontents}{iclr2025_conference.bib}
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
