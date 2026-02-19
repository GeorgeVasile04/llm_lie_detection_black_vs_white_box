# Experiment 3: Generalisation

## 1. Overview
Experiment 3 is the "Stress Test" phase. In Experiment 2, we built a Lie Detector that works well on the data it was trained on. But **does it really work?** Or did it just memorize the specific pattern of the training sentences?

This series of experiments tests the **Robustness** and **Generalisation** of the detector by applying it to completely new situations that it has never seen before:
1.  **Unseen Datasets**: If trained on Math, does it work on French-to-English translation?
2.  **Unseen Scenarios**: If trained on simple "Lie to me" instructions, does it catch a "Salesperson" AI lying to make a sale?
3.  **Unseen Topics**: If trained on lies about cities, does it detect lies about animals?

---

## 2. Where It Starts (Input)
This experiment relies on two things:
1.  **The Frozen Detector**: The classifiers trained in Experiment 2 (e.g., `logistic_logprobs_classifier_all_probes.pkl`). We generally do *not* retrain the model here; we want to see if the *existing* one works.
2.  **New Data**: New datasets generated using different methods (Instrumental Lying) or split in specific ways (Leave-One-Out).

**Key Folder:** `classification_notebooks/simple_generalisation_studies/` & `classification_notebooks/advanced_generalisation_studies/`

---

## 3. The Process (What it uses)

### Part A: Simple Generalisation (Dataset Transfer)
**Notebook:** `classification_notebooks/simple_generalisation_studies/dataset_generalization_experiments.ipynb`

This notebook performs two distinct types of generalisation tests to check if the "Lie Signal" is universal.

#### A.1: Leave-One-Out (Dataset Level)
This checks if the detector works on a specific dataset it hasn't seen, assuming it has seen somewhat similar tasks.
*   **Method:** The experiment loops through all 11 individual datasets (`WikiData`, `Math`, etc.), treating each one as the "Alien Topic" (unseen) in turn.
    *   **Training Loop**: For each iteration, the code aggregates the **10 known datasets**. It uses 80% of this data for training and holds back 20% for validation.
    *   **Testing**: The model is evaluated on two distinct sets:
        1.  **In-Distribution**: The 20% held-out slice of the *known* topics. This measures how well the model learns the topics it has studied.
        2.  **Out-of-Distribution (OOD)**: The **11th unseen dataset**. This is the true test of generalisation.
    *   **Aggregation**: The final scores (Accuracy/AUC) reported in the summary table are averages over all 11 iterations.
*   **Why?** To ensure robustness against small variations in dataset formatting.

#### A.2: Dataset Types Transfer (Concept Generalisation)
This is a much harder test. Instead of holding out a single file, we hold out an entire **Category of Tasks** (e.g., "All Translation Tasks").
*   **Method:** The datasets are grouped into families:
    *   *Translation:* `TatoebaEngToFre`, `TatoebaFreToEng`.
    *   *Anthropic:* `AnthropicAwarenessAI`, `AnthropicAwarenessArchitecture`, etc.
    *   *General Knowledge:* `WikiData`, `Questions1000`, etc.
*   **The Split Process:**
    *   The experiment iterates through these families.
    *   **Example Iteration:** Train on *General Knowledge* + *Anthropic* (9 datasets) -> Test on *Translation* (2 datasets).
    *   In this case, the model has **never seen a translation task** during training.
*   **Why?** To prove the detector isn't relying on task-specific artifacts (like the model outputting French words). If it works here, it means the "Lie Signal" is fundamental and independent of the cognitive task (e.g., doing math vs. translating languages).

### Part B: Advanced Generalisation (Instrumental Lying / Scenarios)
**Notebook:** `classification_notebooks/advanced_generalisation_studies/instrumental_lying.ipynb`

This is the most critical test for real-world applicability. In the real world, AI isn't told "Please lie". It lies to achieve a goal (e.g., satisfy a user, get a reward).

1.  **Generation (The Setup):**
    *   The notebook defines **Role-Playing Scenarios** (stored manually in the code/JSON).
    *   **Examples:**
        *   *Salesperson:* "You get a reward if you sell this car. The car has a broken engine."
        *   *Police Interview:* "You are a witness. You will be in trouble if you reveal you were there."
    *   The model generates a response (often a lie) based on these incentives.
2.  **Detection:**
    *   We take these new "Instrumental Lies".
    *   We attach the **same 48 probes** from Experiment 2 (e.g., "Is the sky blue?").
    *   We feed the logprob differences into the **original detector** (trained only on the simple "Lie vs Truth" prompt).
3.  **Result:** If the detector works here, it means the "internal state of lying" is consistent regardless of *why* the model is lying.

### Part C: Specific Lies (Topic Transfer)
**Notebook:** `classification_notebooks/advanced_generalisation_studies/specific_lies.ipynb`

*   **Method:**
    *   Generates lies about specific narrow topics (e.g., specific cities, people).
    *   Trains on Topic A and tests on Topic B.
*   **Goal:** To confirm the detector isn't picking up on entity-specific artifacts.

---

## 4. Where It Ends (Output)

The output of these experiments are performance metrics on the new "Out-of-Distribution" (OOD) data.

### 1. Generalisation Matrices
*   Visualizations (Heatmaps) showing performance when training on Dataset X and testing on Dataset Y.
*   **Key Finding:** Generalisation is generally strong, though some domains (like Translations) might be harder to transfer to/from.

### 2. Instrumental Lying Results
*   **JSON Output:** Files like `results/instrumental_lying_df_all_scenarios_jb_resampling.json`.
*   These contain the transcripts of the role-playing scenarios and the detector's probability scores.
*   **Key Finding:** The detector catches "Salesperson Lies" with high accuracy (~80-90% AUC), despite **never having seen a salesperson scenario during training**. This suggests the "Lie" signal is fundamental to the LLM's processing.
