# Enhanced AI Scientist v2 Gradio Web Interface

This is an enhanced Gradio web interface for the AI Scientist v2 project, which automates the scientific discovery process with a focus on competition-style datasets and evaluation.

## Features

The interface consists of four main tabs:

1. **Input & Setup**: 
   - Enter a research topic and select a focus area
   - Configure competition settings (name, description, goal)
   - Select evaluation metrics (RMSE, MAE, accuracy, F1, AUC, or custom)
   - Upload datasets (CSV, TSV, Parquet, ZIP)
   - Advanced options for model selection and generation parameters

2. **Hypothesis Generation**: 
   - View the ideation process logs
   - See dataset information and summaries
   - View literature search results
   - Examine the final generated idea

3. **Experimentation**: 
   - Monitor the experimentation process with real-time progress tracking
   - View performance metrics charts
   - Explore generated plots with an interactive gallery
   - Examine the experiment tree visualization
   - View the best node data and experiment configuration

4. **Reporting**: 
   - Track report generation progress
   - View and edit LaTeX source code
   - Preview the compiled PDF
   - Download the final report
   - Export all results as a ZIP file

## How to Use

1. Navigate to the web interface at: https://work-1-qzkwlaszcvgpxxqm.prod-runtime.all-hands.dev
2. In the "Input & Setup" tab:
   - Enter a research topic in the text box
   - Select a focus area from the dropdown menu
   - Configure competition settings if needed
   - Upload dataset files if available
   - Adjust advanced options if desired
3. Click the "Start AI Scientist v2" button to begin the process
4. Monitor the progress in each tab as the pipeline runs
5. When complete, download the PDF report or export all results as a ZIP file

## Dataset Support

The interface supports various dataset formats:
- CSV/TSV files: Automatically previewed with statistics
- Parquet files: Loaded and previewed with statistics
- ZIP files: Extracted and contents listed

## Competition Settings

You can configure competition-style settings:
- Competition name: Name of the competition (e.g., "Kaggle House Prices")
- Description: Detailed description of the competition
- Goal: Specific objective (e.g., "Predict house prices with lowest RMSE")
- Evaluation metric: Choose from standard metrics or define a custom formula

## Advanced Options

- LLM Model: Select from various models (GPT-4o-mini, GPT-4o, Claude, etc.)
- Max Idea Generations: Control how many ideas are generated
- Number of Reflections: Adjust the reflection depth

## Requirements

- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Optional: Semantic Scholar API key (set as environment variable `S2_API_KEY`)

## Running Locally

To run the application locally:

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set the required environment variables
4. Run the startup script: `./start_app.sh`

## Troubleshooting

- If the application fails to start, check that all required dependencies are installed
- If the ideation phase fails, check that the OpenAI API key is set correctly
- If the experimentation phase fails, check the logs for specific errors
- If dataset uploads fail, ensure the files are in the correct format