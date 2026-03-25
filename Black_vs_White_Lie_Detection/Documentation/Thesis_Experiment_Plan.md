# Master's Thesis Execution & Methodology Plan

This document outlines the optimal chronologic execution plan for your four Research Questions (RQs). Because the conclusions of RQ4 (imbalance) and RQ3 (scalability) fundamentally alter *how* we train models, it is scientifically rigorous to execute them sequentially. Each phase's findings will refine the base methodology used in the next.

---

## Phase 1: Robustness to Class Imbalance (RQ4)
**Goal:** Determine if White-Box and Black-Box representations degrade when trained on highly skewed, natural label distributions (e.g., 1 True answer vs 4 False answers). 

### The Scientific Problem
Most multiple-choice QA datasets have a natural imbalance resulting from their structure (one correct option, multiple incorrect options). If an AI lie detector is trained on this skewed data without correction, does it actually learn to detect lies, or does it simply collapse into guessing "False" 80% of the time to artificially inflate its overall accuracy?

### What Data Was Used
We targeted strictly multi-choice formatted datasets that natively feature skewed option distributions. We extracted features using the `meta-llama/Llama-2-7b-chat-hf` model across:
*   `commonsense_qa` (1 True vs 4 False)
*   `arc_challenge` & `arc_easy` (1 True vs 3 False)
*   `open_book_qa` (1 True vs 3 False)
*   `race` (1 True vs 3 False)
*   `ag_news` (1 True vs 3 False)
*   `dbpedia_14` (1 True vs 13 False)

### The Implementation Strategy (How we did it)
1. **Control Variable:** Total training volume is fixed at roughly 1,000 samples for all tests. The only difference is the *ratio* of True/False statements. (Note: For `dbpedia_14`, size is slightly adjusted to $994$ ($71 \text{ qs} \times 14 \text{ opts}$) to prevent truncation of the final question block).
2. **Dynamic Structural Grouping (The Ratio Mechanism):** Instead of manually stripping ratios, we leverage the natural structure of the questions. Every multiple-choice question generates a semantic "block" sharing the same Question `id`. By filtering at the block level, the code forces or adopts ratios organically:
   * **Scenario A (Forced Balanced Extraction):** For every `id` block, the script randomly picks exactly 1 Truth row and exactly 1 Lie row. 
     * **Result:** A globally uniform **1:1** ratio ($50\%$ True / $50\%$ False) regardless of what dataset is being processed.
   * **Scenario B (Natural Imbalanced Extraction):** For every `id` block, the script picks the 1 Truth row and extracts **ALL** associated Lie rows for that question. 
     * **Result:** The ratio perfectly adopts the "natural bias" of the dataset form factor. Processing `ag_news` organically yields a **$1:3$ ($25\%$)** true-rate, while processing `dbpedia_14` yields a violent **$1:13$ ($7.1\%$)** true-rate.
3. **Strict Validation Constraint:** All models are tested against a completely unseen validation set (400 samples) that is strictly forced to use Scenario A's balanced 50/50 logic. This guarantees a perfectly fair "exam" that mathematically penalizes blind guessing.
4. **Feature Extraction & Training Matrix:** 
   * **White-Box:** Extracted hidden states from Layers 1 to 32 (step 2) and trained 8 different algorithms (Logistic Regression, PCA, CCS, LDA, etc.).
   * **Black-Box:** Extracted logprobs based on the 48 custom elicitation probes from the original RepEng paper.
   * We trained the entire set of algorithms on Scenario A, and a competing set on Scenario B.
5. **The Metrics:** We discarded standard "Accuracy" (which is easily tricked by imbalance) and evaluated based on:
   * **Macro F1-Score:** Takes the harmonic mean, violently penalizing models that ignore minority classes.
   * **Class-wise Recall (Recall Truth vs Recall Lie):** Directly exposes if the imbalanced model collapses into only predicting "Lies".

**Outcome:** If models trained under Scenario B suffer a major structural collapse in **Recall (True)** compared to the exact same models trained in Scenario A, we definitively prove that class balance is critical to Lie Detection geometry. We will then mathematically enforce Scenario A's baseline (1:1 ratio) for all remaining Research Questions.

---

## Phase 2: Data Efficiency and Scalability (RQ3)
**Goal:** Discover how much data is actually required to find the "truth geometry," and whether a plateau is reached. 

### The Scientific Problem
Deep learning scales exponentially, but linear probes (PCA, LR) often plateau very quickly. You want to avoid wasting Colab GPU hours extracting 10,000 samples if maximum accuracy is reached at 500 samples.

### The Implementation Strategy
1. **Static Validation Set:** Create a completely frozen validation set of 1,000 samples (balanced, based on Phase 1). This ensures all models are tested on the exact same exam.
2. **Scaling the Training Data:** Scale the training sizes logarithmically. Given your API/Colab constraints, you do not need 10,000 samples to observe a plateau for Logistic Regression. 
   * **Dataset Size Bins:** `[50, 100, 250, 500, 1000, 2000]`.
3. **Visualization:** Plot a line graph: X-axis = *Training Size* (log scale), Y-axis = *AUC/Accuracy*.
   * Plot a line for the Black Box approach.
   * Plot lines for the Best White Box approaches (e.g., LR, PCA, CCS).
   
**Outcome:** You will likely observe that unsupervised methods (PCA, CCS) plateau quickly (e.g., ~250 samples), while Black Box might require more. Once you find the plateau (e.g., 1000 samples), you use exactly that size for all remaining experiments to maximize efficiency.

---

## Phase 3: Performance Ceiling & Access Constraints (RQ1)
**Goal:** Compare BB vs WB using the optimized data settings (Phase 1 & 2), and introduce different API constraints.

### The Scientific Problem
Your current `main_comparison.ipynb` evaluates Black Box utilizing `Logprobs`. However, to fully satisfy RQ1 ("different levels of access"), we must simulate a restricted scenario where a developer *only* gets the final generated text (like the free ChatGPT interface).

### The Implementation Strategy
1. Utilize the optimized dataset size (e.g., 1000 balanced samples).
2. Evaluate 3 tiers of model transparency:
   * **Tier 3 (White-Box):** Full weights/activations access (LR, PCA, LAT, etc.).
   * **Tier 2 (Black-Box Probs):** Access to soft token probabilities (Logprobs). *This is already implemented.*
   * **Tier 1 (Black-Box Discrete):** Hard text outputs. Instead of calculating $P(Yes) - P(No)$, instruct the Black Box classifier to run on binary features: `1` if the string output equals "Yes", `0` if "No" or anything else.
3. Compare the peak accuracy/AUC of Tier 1 vs Tier 2 vs Tier 3.

**Outcome:** Scientifically proves the diminishing returns of transparency. E.g., "Full weight access (WB) yields 85% accuracy, Logprobs access maintains 82%, but limiting to text-only crashes accuracy to 60%."

---

## Phase 4: Out-of-Distribution Generalization (RQ2)
**Goal:** Determine if a lie detector trained on movies can detect lies about science.

### The Scientific Problem
High In-Distribution (ID) accuracy implies the probe memorized topic features, not "truthfulness." We must cross-pollinate the datasets. 

### The Implementation Strategy
1. **The Matrix Setup:** Select ~5 highly distinct datasets: `got_cities` (Factual), `commonsense_qa` (Reasoning), `imdb` (Opinion), `open_book_qa` (Science), `boolq` (Reading Comp).
2. **Train & Save:** Train 1 set of models (8 WB, 1 BB) strictly on Dataset A.
3. **Cross-Evaluation:** Loop through the saved models from Dataset A and use them to `predict()` on the validation data for Datasets B, C, D, and E.
4. **The Metric Matrix:** You suggested Recovered Accuracy (OOD / ID). That is an *excellent* normalized metric. 
   * Formulate an N x N Heatmap.
   * X-axis = `Trained On`
   * Y-axis = `Evaluated On`
   * Cell Value Formula: $\frac{\text{AUC}_{ood}}{\text{AUC}_{id}}$ 

**Outcome:** This will be your thesis's climax. It systematically answers the hardest question in AI alignment: does a universal "lie concept" exist, or is lie detection strictly domain-bound? You will be able to prove whether White Box's interior neural paths or Black Box's psychological probes generalize better.