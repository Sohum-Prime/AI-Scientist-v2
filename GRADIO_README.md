# AI Scientist v2 Gradio Web Interface

This is a Gradio web interface for the AI Scientist v2 project, which automates the scientific discovery process.

## Features

The interface consists of four main tabs:

1. **Input & Setup**: Enter a research topic and optionally select a focus area to start the AI Scientist pipeline.
2. **Hypothesis Generation**: View the ideation process, including logs and the final generated idea.
3. **Experimentation**: Monitor the experimentation process, view logs, and see generated plots.
4. **Reporting**: View the final report, including LaTeX code and the compiled PDF.

## How to Use

1. Navigate to the web interface at: https://work-1-qzkwlaszcvgpxxqm.prod-runtime.all-hands.dev
2. In the "Input & Setup" tab, enter a research topic in the text box.
3. Optionally, select a focus area from the dropdown menu.
4. Click the "Start AI Scientist v2" button to begin the process.
5. Monitor the progress in the status log and navigate through the tabs to see the results of each phase.

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

- If the application fails to start, check that all required dependencies are installed.
- If the ideation phase fails, check that the OpenAI API key is set correctly.
- If the experimentation phase fails, check the logs for specific errors.

## Notes

- The application uses a dummy OpenAI API key for demonstration purposes. Replace it with your own key for actual use.
- The application may take some time to complete each phase, especially the experimentation phase.
- The reporting phase may fail if LaTeX is not installed on the system. In this case, a simulated PDF will be generated.