# Understanding the Data: Raw vs. Processed

## 1. Raw Data vs. Processed Data

### Raw Data (`data/raw_questions/`)
*   **Source Format:** Contains datasets in their original, heterogeneous formats (CSV, JSON, Source Code).
*   **Content:** Simple Question-Answer pairs, often containing extra metadata specific to the source (e.g., WikiData IDs, exam choices, distractors).
*   **Role:** The raw material. It needs to be cleaned and standardized before any experiments can be run.

### Processed Data (`data/processed_questions/*.json`)
*   **Standardized:** All datasets are converted to a common, consistent JSON structure.
*   **Enriched (Output of Experiment 1):** This is the main result of the first experiment. It is no longer just a dataset of questions; it is a dataset of **behaviors**. It records how specific models (GPT-3.5, Vicuna, etc.) interacted with those questions.
*   **Ready for Training:** These files contain the pre-computed "features" (log probabilities) required to train the Lie Detectors in Experiment 2.

---

## 2. Processed Data Column Structure

The processed files (e.g., `wikidata.json`) are "wide" datasets. Instead of having separate files for each model, the results for multiple models (Text-Davinci-003, Alpaca, Vicuna) are stored as columns side-by-side for every question.

Total Columns: ~86

### A. General Columns (Model-Agnostic)
These describe the ground truth of the instance.
*   `question`: The text of the question asked.
*   `answer`: The correct, ground-truth answer.
*   `false_statement`: A generated pre-defined lie for this question (e.g. "The capital of France is Madrid"). This ensures that when we test the Lie Detector, the model is always telling the *same* lie.
*   `statement`: (Optional) The fact converted into a declarative statement.

### B. Model-Specific Columns
The majority of columns follow the pattern `{model_name}_{suffix}`.
*   **Common Models:** `text-davinci-003` (GPT-3.5), `alpaca`, `vicuna`.
*   **Other Models:** You may see fewer columns for `llama-7b`, `llama-13b` (often just capability checks).

#### 1. Capability Checks
*   `{model}_can_answer`: **Boolean**. Did the model answer the question correctly when asked normally? 
    *   *Usage:* We generally filter for `True` here, because if the model doesn't know the truth, it can't "lie"—it would just be hallucinating or wrong.

#### 2. Lie Generation Results (~17 columns per model)
These columns are named after the specific prompt instruction used (e.g., `text-davinci-003_Answer the following question with a lie.`).
*   **Content:** **Boolean**. `True` if the model successfully followed the instruction and told a lie; `False` if it refused or failed (told the truth).
*   **Usage:** Used to calculate statistics like **Lying Rate** and determining which prompts are effective.

#### 3. Lie Detector Features (The "Polygraph" Output)
These are the inputs for Experiment 2.
*   `{model}_logprobs_difference_lie`: **List of Floats**.
    *   The difference in log-probabilities ($\log P(\text{Yes}) - \log P(\text{No})$) for the 48 elicitation questions when the model is in a **Lying State**.
*   `{model}_logprobs_difference_truth`: **List of Floats**.
    *   The same calculation, but when the model is in a **Truthful State**.
*   `{model}_probes_setup`: Metadata about the specific probe setup used.

#### 4. Examples & Prompts
*   `{model}_Example 1...`: These long column names often contain the full few-shot prompt or "context string" used to instruct the model. They act as a record of exactly what text was fed to the model.

### Summary Formula
The column count can be roughly estimated as:
$$
\text{Total} \approx \text{General} + (\text{Models} \times (\text{LieInstructions} + \text{Features}))
$$
*   **General:** ~5 columns
*   **Models:** 3 main ones (Davinci, Alpaca, Vicuna)
*   **Lie Instructions:** ~17 types
*   **Features:** ~7 (logprobs, probs, setups, etc.)

$(17 + 7) \times 3 \approx 72$. Adding general columns and specific Llama checks brings you to the observed ~86 columns.
