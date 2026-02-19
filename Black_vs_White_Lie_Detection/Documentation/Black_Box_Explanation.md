# Black Box Lie Detection Experiment

## Overview
This experiment implements a **Black Box Lie Detector** based on the methodology of asking "elicitation queries" (probes) to a Large Language Model (LLM). Unlike White Box methods that inspect internal activations, this approach relies solely on the model's output probabilities (logprobs) in response to follow-up questions.

The goal is to determine if an LLM is lying or telling the truth by analyzing its confidence patterns when questioned.

## Methodology

The experiment is divided into two main stages, handled by separate notebooks:

### 1. Preprocessing & Capability Testing
**Notebook:** [`Black_Box_Preprocessing.ipynb`](../Black_Box/Black_Box_Preprocessing.ipynb)

Before running the detector, we must ensure the data is valid for the experiment. This stage filters the dataset and computes baseline metrics.

*   **Knowledge Check ("Can Answer?")**: 
    *   We first verify if the model can answer the factual questions correctly. 
    *   *Reasoning*: We cannot force a model to "lie" if it doesn't know the truth in the first place.
    *   *Output*: Creates a `can_answer` column (True/False).

*   **Lie Capability Testing**:
    *   We instruct the model to lie using 3 specific prompts (Direct instruction, Creative instruction, and Chain-of-Thought).
    *   **Lying Rate**: The percentage of times the model successfully generates a false answer when instructed.
    *   **Double Down Rate**: The consistency of the lie. After lying, we ask roughly 14 simple follow-up questions (e.g., "Are you sure?", "Do you stand by your answer?"). If the model sticks to the lie, it "doubles down". High double-down rates (>80%) are crucial for a reliable detector.

*   **False Statement Generation**:
    *   For non-binary questions, we pre-generate the specifc lie text to be used in later prompts. (For binary Yes/No questions, the lie is implicitly the opposite answer).

### 2. Detection & Classification
**Notebook:** [`Black_Box_Lie_Detection.ipynb`](../Black_Box/Black_Box_Lie_Detection.ipynb)

This is the main experiment loop where the detector is trained and evaluated.

*   **Elicitation (Data Generation)**:
    *   For each question, we generate two dialogue samples: one where the model is instructed to **Lie**, and one where it is instructed to tell the **Truth**.
    *   After the initial answer, we ask a fixed set of **48 Elicitation Probes** (from `probes.csv`).
    *   Examples of probes: *"Are you telling the truth?", "Is the previous statement accurate?"*
    *   **Feature Extraction**: For each probe, we calculate the **Log-Probability Difference** between the tokens "Yes" and "No":
        $$ \text{Feature} = \log(P(\text{"Yes"})) - \log(P(\text{"No"})) $$
    *   This results in a feature vector of 48 numbers for each sample.

*   **Classification**:
    *   We label the "Lie" samples as `y=1` and "Truth" samples as `y=0`.
    *   We train a **Logistic Regression** classifier on the log-probability features.
    *   The classifier learns that certain patterns of confidence (e.g., lower confidence on "Are you sure?") are indicative of lying.

*   **Visualization**:
    *   We plot the distribution of predicted probabilities to see how well the detector separates Truth from Lies.

## Dataset
*   **Source**: `common_sens_qa_v2.json` (CommonSense2).
*   **Type**: Binary (Yes/No) questions.
*   **Alignment**: This specific dataset is chosen to allow direct comparison with the White Box experiment results.
