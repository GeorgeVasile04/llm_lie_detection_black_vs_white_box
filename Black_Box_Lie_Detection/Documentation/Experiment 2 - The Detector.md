# Experiment 2: The Lie Detector

## 1. Overview
The second experiment builds upon the data generated in Experiment 1. While the first experiment generated the behaviors (Truth vs. Lie), **Experiment 2 focuses on training a classifier to distinguish between them** using the "elicitation questions" (probes).

The core hypothesis is that a model's internal state when lying is distinguishable from its state when telling the truth, and that this difference can be detected by asking unrelated binary questions.

---

## 2. Where It Starts (Input)
The input for this experiment is the **Processed Data** generated in Experiment 1, stored in `data/processed_questions/*.json`.

Specifically, the code loads all the datasets (WikiData, Questions1000, etc.) and extracts the characteristic features:
*   **X (Features):** The `logprobs_difference` vectors.
    *   Each vector contains ~48 float values.
    *   Value $i$ corresponds to: $\log P(\text{Yes}) - \log P(\text{No})$ for the $i$-th elicitation question.
*   **y (Labels):**
    *   `1`: The model is telling the Truth.
    *   `0`: The model is Lying.

**Key Notebook:** `classification_notebooks/train_classifiers_on_prompted_GPT_3.5.ipynb`

---

## 3. The Process (What it uses)

The experiment follows a standard Machine Learning pipeline:

### Step 1: Data Aggregation & Splitting
*   The code iterates through all available datasets (General Knowledge, Math, Translation, etc.).
*   It combines them into a single massive dataset (`logprobs_dataset`).
*   **Train/Test Split:** The data is split into training and testing sets. Crucially, the split is often done to test **out-of-distribution generalization** (e.g., train on facts, test on math) in generalization experiments, though the baseline trains on a mix.

### Step 2: Feature Engineering
Two types of features are derived from the raw log probabilities:
1.  **Logprobs (Continuous):** The raw scalar difference $\log P(\text{Yes}) - \log P(\text{No})$. This contains the most information and "confidence".
2.  **Binary (Discrete):** A simple True/False check: Is $P(\text{Yes}) > P(No)$? This simulates a setup where we only see the generated text output, not the internal probabilities.

### Step 3: Model Training
*   **Algorithm:** **Logistic Regression**.
*   **Why Logistic Regression?** It is simple, interpretable (we can look at coefficients to see which questions matter), and works well on high-dimensional vectors with limited data.
*   The classifier learns a weight $w_i$ for each elicitation question. If a question is good at catching lies, it gets a high weight.

### Step 4: Analysis & Optimization
The experiment goes beyond training a single model; it analyzes *why* the detection works and *how* to make it efficient.

#### 4.1 Find the best type of elicitation questions (Probe Analysis)
The code trains separate classifiers on different subsets of "elicitation questions" (probes) to determine which are most effective at detecting lies.
*   **"Lie" Probes:** Questions directly asking about the model's validity (e.g., "Are you telling the truth?", "Is the previous answer a lie?").
*   **"Knowable" Probes:** Totally unrelated factual questions (e.g., "Is the sky blue?", "Is 2+2=4?").
*   **"No-Lie" Probes:** A subset that strictly excludes any direct questions about lying, to test if we can detect deception without asking for a confession.

**Results & Intuition:**
*   **Direct "Lie" probes perform best:** Unsurprisingly, asking the model if it is lying is the strongest signal.
*   **"Knowable" probes works surprisingly well:** Even unrelated questions (like math or facts) show a shift in probabilities. **Intuition:** Lying is not just a local output; it induces a global "nervousness" or state shift in the model that affects its confidence on completely unrelated simple facts.

#### 4.2 How much data do I need? (Sample Efficiency)
The notebook runs a sample size sweep, training the likelihood-based classifiers on datasets ranging from 5 to ~1000 examples to generate learning curves.

**Results & Intuition:**
*   **High Efficiency:** The model achieves extremely high accuracy (>90%) with very few samples (often as few as **15-20 pairs** of Truth/Lie examples).
*   **Saturation:** Performance saturates quickly, meaning adding thousands of more examples yields diminishing returns.
*   **Intuition:** The "Lying Signal" in the logprobs space is very strong and distinct (low noise). The vector direction separating "Truth" from "Lie" is consistent across different questions, making it easy for a linear classifier (Logistic Regression) to find the decision boundary with minimal data.

---

## 4. Where It Ends (Output)

The experiment produces trained models and performance metrics.

### Validated Models
The trained classifiers are saved as Python pickle files in:
`results/trained_classifiers/`

Examples:
*   `logistic_logprobs_classifier_all_probes.pkl`: The best model using all questions and continuous logprobs.
*   `logistic_binary_classifier_no_lie.pkl`: A model using only binary answers and excluding direct questions about lying.

### Metric Reports
The notebook outputs:
*   **Accuracy:** How often correctly classifies Lie vs Truth (often >80-90%).
*   **AUC (Area Under Curve):** A robust metric for binary classification.
*   **Calibration Curves:** Checking if the model's confidence matches its accuracy.

These trained classifiers are then the "artifacts" used in future experiments to see if they work on *other* models (like Vicuna or Alpaca) without retraining.
