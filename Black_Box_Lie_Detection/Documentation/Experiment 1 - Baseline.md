# Project Overview & Experiment 1

## Global Project Overview
The goal of this project is to develop a "Black-Box Lie Detector" for Large Language Models (LLMs). The core idea is to detect lies not by looking at internal weights (white-box), but by asking a sequence of unrelated "elicitation questions" (probes) after the model answers, and analyzing the probabilities of its responses.

### Datasets Used Across All Experiments
To ensure the detector works generally and not just on specific topics, the project uses a wide variety of datasets handled by the `QuestionsLoader` engine.

*   **General Knowledge (Factual):**
    *   `Questions1000`: 1000 curated factual questions.
    *   `WikiData`: Facts extracted dynamically from WikiData.
    *   `Sciq`: Science exam questions.
*   **Synthetic Facts (Hallucination Proxies):**
    *   `SyntheticFacts`: Questions about fictional entities (e.g., "The capital of the Stepstones") where the truth is provided in the prompt context.
*   **Common Sense & Reasoning:**
    *   `Commonsense2`: Questions requiring intuitive reasoning (often Yes/No).
    *   `MathematicalProblems`: Arithmetic and math logic questions.
*   **Translation:**
    *   `TatoebaFreToEng`: French to English translation.
    *   `TatoebaEngToFre`: English to French translation.
*   **Self-Knowledge (Anthropic Evals):**
    *   `AnthropicAwarenessAI`, `AnthropicAwarenessArchitecture`, `AnthropicAwarenessNNArchitecture`: Questions asking the model about its own nature, architecture, and programming.

---

## Experiment 1: Dataset Generation & Lying Behavior Analysis

**Goal:** The primary objective of Experiment 1 is **not yet to train the classifier**, but to **generate a high-quality labeled dataset** and define the "Lying Rate". We need to ensure that when we label a data point as a "Lie", the model *actually* lied (said something false when it knew the truth), and didn't just make a mistake or refuse to answer.

### Input
*   **Raw Data:** The diverse collection of questions listed in the Global View above.
*   **Model:** GPT-3.5 (`text-davinci-003`).

### Process (The 6 Steps)

The experiment follows a strict pipeline to ensure data quality, mirroring the execution flow of `lying_rate_double_down_rate_probes.ipynb`.

#### Step 1: Data Loading
*   **Function:** `QuestionsLoader` (and subclasses like `WikiData`, `Questions1000`).
*   **Action:** Loads raw questions from JSON/CSV files into a standardized DataFrame structure.
*   **Outcome:** A table containing at least `[question, correct_answer]`.

#### Step 2: Knowledge Filtering
*   **Function:** `dataset.check_if_model_can_answer()`
*   **Action:** A lie is defined as saying somethong false when you know the truth. In order to test if the model knows the answer, we ask the model every question in a neutral setting.
*   **Outcome:** Filters the dataset to keep *only* the questions where the model knows the correct answer (creating the `can_answer` column). This ensures that any subsequent falsehood is a defined "lie" and not ignorance.

#### Step 3: Lying Capability Testing
*   **Function:** `dataset.does_model_lie(test_questions=[])`
*   **Action:** We take the filtred questions and we promt the model with various "Lie Instructions" (e.g., "Answer with a lie") and checks if the generated answer is factually false.
*   **Outcome:** Populates columns like `{model}_{instruction}` with `True` (lied successfully) or `False` (failed to lie).

#### Step 4: Statistics & Prompt Selection
*   **Functions:** `dataset.compute_lying_ratio()` and `dataset.compute_lie_double_down_rate()`
*   **Action:** Aggregates the results from Step 3 to determine which lie instructions are most effective.
*   **Outcome:** A set of metrics (Lying Rate and double down rate)

#### Step 5: Fixed Lie Generation
*   **Function:** `dataset.generate_false_statements()`
*   **Action:** For each question, prompts the model to generate a specific lie (e.g., "The capital of France is Madrid") and stores it.
*   **Outcome:** A `false_statement` column. This ensures consistency so the classifier always sees the *same* lie for a given question during training.

#### Step 6: Log Probability Extraction (The Polygraph)
*   **Function:** `dataset.generate_logprobs()` (which calls `_run_probes()`)
*   **Action:** 
    1. Sets up the model in a **Lying State** (using the `false_statement`) and an **Honest State**.
    2. In each state, asks 48 binary "elicitation questions" (probes) such as "Are you sure?".
    3. Computes the probability difference: $P(\text{Yes}) - P(\text{No})$.
*   **Outcome:** The final features for the detector: `logprobs_difference_lie` and `logprobs_difference_truth`.

### Output
The result of Experiment 1 is a set of **Processed JSON Files** (saved in `data/processed_questions/`) containing:
*   Original Question
*   The Instruction used (Lie/Truth)
*   The Model's Output
*   Verification Tags (Is it a lie? Did it double down?)

This labeled dataset acts as the ground truth for training the lie detector in subsequent experiments.

