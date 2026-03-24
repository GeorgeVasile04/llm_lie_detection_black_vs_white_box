# Master's Thesis Execution & Methodology Plan

This document outlines the optimal chronologic execution plan for your four Research Questions (RQs). Because the conclusions of RQ4 (imbalance) and RQ3 (scalability) fundamentally alter *how* we train models, it is scientifically rigorous to execute them sequentially. Each phase's findings will refine the base methodology used in the next.

---

## Phase 1: Robustness to Class Imbalance (RQ4)
**Goal:** Determine if models are sensitive to skewed data and establish the "gold standard" balancing ratio for remaining experiments.

### The Scientific Problem
If you compare 1000 items (balanced) vs 400 items (imbalanced), the balanced model will win simply because it saw *more total data*. To test imbalance fairly, we must isolate the **class ratio** while keeping the **total volume of training data constant**.

### The Implementation Strategy
1. **Control Variable:** Total Training Size = $N$ (e.g., $N=1000$ samples).
2. **Independent Variables (Scenarios):**
   * **Setup A (Balanced):** 500 True / 500 False (requires 500 unique questions).
   * **Setup B (Imbalanced):** 200 True / 800 False (requires 200 unique questions, fully simulating the original RepEng schema).
3. **Strict Validation Constraint:** Evaluate *both* models on a strictly **balanced** validation set (e.g., 500T / 500F). 
4. **The Metrics:** Do not use overall Accuracy. An imbalanced model might just predict "False" 90% of the time. You must report:
   * **Macro F1-Score:** Penalizes models that ignore the minority class.
   * **Class-wise Recall:** Shows if the model is disproportionately good at catching Truths vs. Lies.

**Outcome:** If Setup A significantly outperforms Setup B on the Validation set, you conclusively prove that class balance is critical. You will then *enforce Setup A (1:1 balance)* for Phases 2, 3, and 4. 

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