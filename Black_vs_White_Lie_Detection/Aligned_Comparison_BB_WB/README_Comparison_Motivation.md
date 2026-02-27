# Aligned Comparison: Black Box vs. White Box Lie Detection

## 1. Goal
The primary objective of this project is to perform a scientifically rigorous and direct comparison between two distinct paradigms of Lie Detection in Large Language Models (LLMs):

1.  **White Box Lie Detection (Internal States):**
    *   **Mechanism:** Inspects the internal activation vectors (hidden states) of the model during inference.
    *   **Hypothesis:** The model has an internal representation of "truth" vs. "falsehood" that is distinct and linearly separable in the high-dimensional activation space.
    *   **Method:** Training linear probes (classifiers) on the activations of the model at specific layers.

2.  **Black Box Lie Detection (Behavioral/Output-Based):**
    *   **Mechanism:** Interacts with the model purely through text input and output (I/O). It does *not* access weights or activations.
    *   **Hypothesis:** A lying model is in a distinct "cognitive state" that leaks into its behavior on unrelated tasks. Even if the lie text itself is perfect, the model's calibration on simple questions ("Is the sky blue?") shifts.
    *   **Method:** Asking a set of unrelated "elicitation questions" (probes) after a suspect answer and training a classifier on the log-probabilities of the model answering "Yes" or "No".

---

## 2. The Problem: Misalignment in Methodology
In their original formulations, these two approaches are difficult to compare directly:

*   **Original White Box Approach:** Typically uses dataset pairs (Question + True Answer vs. Question + False Answer). The model is **forced** to process a lie (it reads it). It does not "generate" the lie itself.
*   **Original Black Box Approach:** Typically asks the model to "Answer with a lie". The model **generates** the lie itself.

This creates a confounding variable: **Generation vs. Processing.**
*   If the White Box detector works better, is it because internal states are better signals? Or is it because reading a false statement is cleaner than generating one?
*   If the Black Box detector fails, is it because the behavioral signal is weak? Or is it because the model generated a "bad" or inconsistent lie?

---

## 3. The Solution: Unified Alignment Strategy
To isolate the performance of the *detection methods* (Internal Activations vs. External Behavior) from the *generation method*, we align them using a **Forced Lie Scenario**.

### Methodology
We use a common dataset (e.g., **CommonSenseQA**) and enforce the exact same interaction history for both approaches.

1.  **Shared Input (Context):**
    For every question $Q$, we create two deterministic scenarios:
    *   **Truth Scenario:** `User: Q?  Assistant: [Correct Answer]`
    *   **Lie Scenario:** `User: Q?  Assistant: [False Answer]`
    *   *Note:* The model is **forced** to output the False Answer (we pre-fill its response history).

2.  **Parallel Extraction:**
    *   **White Box Track:** We feed this exact history into the model and extract the **Hidden State Activations** at the execution of the final token.
    *   **Black Box Track:** We take this exact history and immediately ask the **Elicitation Questions** (e.g., "Is the sky blue?"). We record the **Log-Probability** of the Yes/No answers.

### Motivation
This alignment is the standard for fair comparison because:
1.  **Identical "Cognitive State":** Both detectors are analyzing the model at the exact same moment (after "saying" the false answer).
2.  **Removal of Generation Noise:** We eliminate the variability of the model refusing to lie, failing to follow instructions, or generating a "lie" that is accidentally true (hallucination).
3.  **Apples-to-Apples:** Any difference in performance can be attributed strictly to the **signal quality**: Is the information about "lie" more accessible in the neurons (White Box) or in the behavioral side-effects (Black Box)?


## Dataset Statistics

| Dataset | Total Samples | Positive (True) | Negative (False) |
| :--- | :---: | :---: | :---: |
| **commonsense_qa** | 6105 | 1221 | 4884 |
| **race** | 5556 | 1389 | 4167 |
| **arc_easy** | 2281 | 570 | 1711 |
| **arc_challenge** | 1194 | 299 | 895 |
| **open_book_qa** | 2000 | 500 | 1500 |
| **boolq*** | 6000 | 3000 | 3000 |
| **copa*** | 200 | 100 | 100 |
| **rte*** | 554 | 277 | 277 |
| **piqa*** | 0 | 0 | 0 |
| **imdb*** | 6000 | 3000 | 3000 |
| **amazon_polarity*** | 6000 | 3000 | 3000 |
| **ag_news*** | 6000 | 3000 | 3000 |
| **dbpedia_14*** | 6000 | 3000 | 3000 |
| **got_cities** | 606 | 303 | 303 |
| **got_sp_en_trans** | 116 | 58 | 58 |
| **got_larger_than** | 826 | 413 | 413 |
| **got_cities_cities_conj** | 322 | 156 | 166 |
| **got_cities_cities_disj** | 292 | 152 | 140 |
| **TOTAL** | **50,052** | **20,438** | **29,614** |
