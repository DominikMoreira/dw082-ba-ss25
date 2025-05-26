# Aspect-Based Sentiment Analysis Project

This project implements an aspect-based sentiment analysis (ABSA) system that processes review data, extracts aspects and their corresponding sentiments using different OpenAI model-based pipelines, and evaluates the performance of these pipelines.

## Project Architecture

The system is designed with a modular architecture, where different Python scripts handle specific parts of the workflow, from data loading to evaluation and visualization.

### Overall Workflow

1.  **Data Input**: Raw review data is ingested.
2.  **Data Loading**: The data is loaded and initially prepared.
3.  **Processing Pipelines**: Reviews are processed by one or more pipelines (Zero-Shot, Few-Shot, Fine-Tuned GPT) to extract aspects and sentiments.
4.  **Results Storage**: The outputs from each pipeline are saved.
5.  **Evaluation**: The stored results are evaluated against true labels to measure performance.
6.  **Visualization**: Sentiment data can be visualized.

### Core Components and Data Flow

*   **Data Input**:
    *   The primary input is typically a CSV file containing reviews, such as `data/raw/test_split.csv`.

*   **Data Loading (`src/data_loader.py`)**:
    *   The [`load_data`](src/data_loader.py) function in [`src/data_loader.py`](src/data_loader.py) is responsible for reading the input CSV into a pandas DataFrame.
    *   This module also contains utilities for splitting data (e.g., [`split_data`](src/data_loader.py)).

*   **Main Orchestrator (`src/main.py`)**:
    *   The [`src/main.py`](src/main.py) script acts as the central coordinator.
    *   It initializes necessary components, loads data, invokes the selected processing pipelines, triggers evaluation, and can initiate visualization.

*   **Processing Pipelines**:
    *   These pipelines leverage OpenAI models for aspect extraction and sentiment classification. They interact with [`src/openaiAPI.py`](src/openaiAPI.py) for API calls and use predefined prompt structures from [`src/prompts.py`](src/prompts.py).
    *   **Zero-Shot Pipeline (`src/pipe_synchain_zero_shot.py`)**:
        *   Uses the `process_reviews_with_zero_shot` function.
        *   Employs a chain of prompts (aspect extraction, polarity classification, validation) without specific examples provided in the prompt itself.
        *   Relies on system prompts like `SYSTEM_PROMPT_ASPECT` and user templates like `USER_TEMPLATE_ASPECT_ZEROSHOT` from [`src/prompts.py`](src/prompts.py).
    *   **Few-Shot Pipeline (`src/pipe_synchain_few_shot.py`)**:
        *   Uses the `process_reviews_with_few_shot` function.
        *   Similar to the zero-shot pipeline but provides a few examples within the prompts to guide the model.
        *   Utilizes system prompts like `SYSTEM_PROMPT_ASPECT` and user templates like `USER_TEMPLATE_ASPECT_FEWSHOT` from [`src/prompts.py`](src/prompts.py).
    *   **Fine-Tuned GPT Pipeline (`src/pipe_finetuned_gpt.py`)**:
        *   Uses the `process_reviews_with_finetuned` function.
        *   Interacts with a fine-tuned OpenAI model.
        *   Uses `SYSTEM_PROMPT_FINETUNED` and `USER_TEMPLATE_ABSA_FINETUNED` from [`src/prompts.py`](src/prompts.py).

*   **OpenAI API Interaction (`src/openaiAPI.py`)**:
    *   The `OpenAIClient` class in [`src/openaiAPI.py`](src/openaiAPI.py) handles communication with the OpenAI API. The `request` method sends prompts and receives responses.

*   **Prompt Management (`src/prompts.py`)**:
    *   [`src/prompts.py`](src/prompts.py) stores all system and user prompt templates used by the different pipelines. This allows for easy modification and management of prompts.

*   **Results Storage**:
    *   Each processing pipeline saves its output (including review text, true labels, predicted labels, extracted aspects, and any justifications) to a CSV file in the `data/results/` directory.
    *   Examples: `data/results/zero_shot_results.csv`, `data/results/few_shot_results.csv`, `data/results/finetuned_openai_results.csv`.

*   **Evaluation (`src/evaluate.py`)**:
    *   The `ABSAEvaluator` class in [`src/evaluate.py`](src/evaluate.py) is responsible for assessing the performance of the pipelines.
    *   It reads the result CSVs (e.g., `data/results/finetuned_openai_results.csv`), parses the true and predicted labels, and calculates various metrics such as precision, recall, F1-score for aspect extraction and end-to-end performance, as well as polarity accuracy.
    *   Evaluation metrics are saved as JSON files (e.g., `data/results/eval_synchain_zero_shot.json`, `data/results/absa_evaluation.json`).

*   **Visualization (`src/recommender.py`)**:
    *   The `Recommender` class in [`src/recommender.py`](src/recommender.py) can take a results CSV file as input.
    *   The `create_diagram` method processes the aspect sentiment data and generates bar charts to visualize the frequency of positive, neutral, and negative sentiments for each aspect category.
    *   Diagrams can be displayed or saved to a file (e.g., `data/results/predicted_sentiments_diagram.png`).

### Directory Structure Overview

```
.
├── data/
│   ├── raw/         # Raw input data (e.g., test_split.csv)
│   └── results/     # Output CSVs from pipelines and JSON evaluation files
├── src/             # Source code
│   ├── main.py
│   ├── data_loader.py
│   ├── openaiAPI.py
│   ├── prompts.py
│   ├── pipe_synchain_zero_shot.py
│   ├── pipe_synchain_few_shot.py
│   ├── pipe_finetuned_gpt.py
│   ├── evaluate.py
│   ├── recommender.py
│   ├── aspect_extractor.py
│   └── model_handler.py
├── .gitignore
├── pyproject.toml
└── README.md
```