# Reinforcement Learning Trading Strategy Creator

This project implements and analyzes a trading strategy using Reinforcement Learning (RL).

## Setup

It's assumed you are using [Poetry](https://python-poetry.org/) for dependency management. Install dependencies with:

```bash
poetry install
```

## Usage

### 1. Training the Agent

To train the RL agent and generate the training log:

```bash
poetry run python train.py
```

This will create/overwrite the `training_log.csv` file in the project root.

### 2. Analyzing Results

To analyze the `training_log.csv` and generate performance visualizations:

```bash
poetry run python analyze_results.py
```

This will save plots to the `results_plots/` directory.

### 3. Accessing Reports

Two reports are generated based on the analysis:

-   **Full Comprehensive Analysis:** Detailed findings and methodology can be found in `reinforcement_learning_trading_strategy_analysis.md`.
-   **Presentation Summary:** A concise, presentation-friendly version is available in `reinforcement_learning_trading_strategy_presentation.md`.