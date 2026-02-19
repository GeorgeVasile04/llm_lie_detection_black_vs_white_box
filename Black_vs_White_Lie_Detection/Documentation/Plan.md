## Comprehensive Plan: White-Box vs. Black-Box Lie Detection

This plan details the implementation of both lie detection methods on a common dataset and their subsequent comparison.

### Step 1: Experiment Setup
**Goal**: Initialize a unified environment for running both experiments.
- **Action**: Create a new notebook `Black_vs_White_Lie_Detection/comparison_white_vs_black.ipynb`.
- **Environment**:
    - Configure `sys.path` to include:
        - Root directory `.` (for general imports).
        - `Black_Box_Lie_Detection` (to import `lllm`).
        - `White_Box_Lie_Detection` (to import `repeng`).
    - Import standard libraries (`pandas`, `numpy`, `sklearn`, `matplotlib`).
- **Pedagogical Note**: Setting correct paths ensures we can reuse the specialized tools (`repeng` and `lllm`) already built in the repository without code duplication.

### Step 2: Harmonized Data Generation
**Goal**: Create a single, guaranteed dataset used by both methods to ensure a fair comparison.
- **Source**: Load `Black_Box_Lie_Detection/data/raw_questions/common_sens_qa_v2.json`.
- **Processing**:
    - Select a subset (e.g., 500 questions) to keep runtime manageable.
    - Since these are binary "Yes/No" questions, generate **couples**:
        - **Truth**: (Question + Correct Answer) -> Label: 0 (Honest)
        - **Lie**: (Question + Incorrect Answer) -> Label: 1 (Lying)
    - Save this processed dataset as `Black_vs_White_Lie_Detection/Data/common_dataset.json` (or `.csv`).
- **Pedagogical Note**: Comparing methods requires an identical test bed. By generating the dataset explicitly first, we guarantee both the White Box probe and the Black Box classifier see the exact same examples.

### Step 3: White Box Implementation (Activations)
**Goal**: Detect lies by inspecting the model's internal state.
- **Model**: Load `Llama-2-7b-chat-hf` (or similar available model) using `repeng`.
- **Extraction**:
    - Iterate through the `common_dataset.json`.
    - Feed the prompt: `[INST] {question} [/INST] {answer}`.
    - Use `repeng` to extract hidden state activations at the last token position.
    - Store: `(activation_vector, label)`.
- **Probe**:
    - Split data (Train/Test: 80/20).
    - Train a Logistic Regression probe using `sklearn` on the training activations.
    - Evaluate accuracy on the test set.
- **Pedagogical Note**: This represents the "mind-reading" approach—checking if the model *knows* it is lying even if the output text is coherent.

### Step 4: Black Box Implementation (Behavioral)
**Goal**: Detect lies by inspecting the model's output probabilities.
- **Model**: Use the *same* `Llama-2-7b-chat-hf` model ensuring apples-to-apples comparison.
- **Method**:
    - Use `lllm.questions_loaders.Commonsense2` (or similar loader) to handle the `common_dataset.json`.
    - For each question, prompt the model to Answer (Truth) or Lie.
    - Capture the **log-probabilities** of the "Yes" and "No" tokens using `generate_logprobs` (or equivalent `lllm` function).
    - Calculate the difference: `log(P(Yes)) - log(P(No))` as the feature.
- **Classifier**:
    - Train a Logistic Regression (or simple threshold classifier) on these log-prob differences using the *same* train/test split indices as Step 3.
- **Pedagogical Note**: This represents the "interrogation" approach—checking for hesitation or statistical anomalies in the model's confidence without needing access to internal weights.

### Step 5: Comparison & Visualization
**Goal**: Quantitatively determine which method is more effective.
- **Metrics**: Calculate Accuracy and AUROC (Area Under ROC Curve) for both methods on the test set.
- **Visualization**:
    - Plot ROC curves for both detectors on the same graph.
    - Visualize the distribution of "Lie Scores" (White Box distance outcomes vs. Black Box log-prob differences).
- **Decisions**:
    - **Dataset**: Created a unified `common_dataset.json` to guarantee exact sample matching.
    - **Models**: Used the same `Llama-2-7b-chat-hf` instance for both experiments to isolate method performance from model differences.
