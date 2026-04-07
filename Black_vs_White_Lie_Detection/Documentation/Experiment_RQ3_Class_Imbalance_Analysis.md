# RQ3: Robustness to Class Imbalance - Experiment Summary

## 1. What I Did (Objective)
Evaluated the impact of **Class Imbalance** on the robustness of Black-Box (Logprobs) and White-Box (Probes) lie detection methods. The core experiment isolates the effect of varying sample sizes (truths vs. lies) to determine how highly skewed data affects performance metrics and decision boundaries.
All related experiment scripts and evaluation logic are located in the `Black_vs_White_Lie_Detection/Class_Balance_Impact_BB_WB/` folder.

## 2. Methodology & Code Organization (How I Did It)
The pipeline is constructed through specific modular scripts:
* **Data Loading** (`load_data_imbalance.py`): Fetches and formats datasets into the correct Imbalanced/Balanced setups.
* **White-Box Extraction** (`wb_probes_imbalance.py`): Extracts internal activations/probes representing truthfulness. *Includes a critical fix to unsupervised probe polarity by determining direction solely on training data, thus preventing data leakage.*
* **Black-Box Extraction** (`bb_classifier_imbalance.py`): Extracts logprobs and trains classifiers on output distributions.
* **Evaluation** (`evaluation_utils_imbalance.py`): Calculates the core metrics (AUC, Accuracy, Recall, Macro F1, MAP, BRP_90).
* **Statistical Compilation** (`summarize_wilcoxon.py` & `Class_Imbalance_Experiment.ipynb`): Runs the paired Wilcoxon tests across all setups and visualizes the statistical significance of the results.

## 3. Data & Setup Construction
The experiment was conducted on **7 diverse datasets** (e.g., `commonsense_qa`, `race`, `arc_challenge`, `open_book_qa`, `dbpedia_14`, `ag_news`). To isolate class imbalance, models were evaluated under two controlled scenarios:

* **Setup A (Balanced)**: The baseline. Each prompt has **1 Truth vs 1 uniformly random Lie** (50/50 split).
* **Setup B (Imbalanced)**: The skewed scenario. Each prompt has **1 Truth vs ALL available Lies** naturally attached to the question (resulting in natural splits like 1:3, 1:4, or 1:13).

## 4. Metrics Evaluated
Instead of relying solely on standard Accuracy (which can be high simply by predicting the majority class), we tracked precise metrics:
* **ROC AUC**: Measures *ranking ability*—does the model successfully score an actual Truth higher than an actual Lie, regardless of quantity?
* **Recall (Truth) & Macro F1 Score**: Measures the health of the *decision boundary*—does the model actually identify the minority class (Truth), or does it stubbornly default to predict "Lie"?
* **MAP (Mean Average Precision) & BRP_90**: Measures ranking precision (how many top predictions are actually relevant) and calibration confidence.

## 5. Statistical Rigor: The Wilcoxon Signed-Rank Test
To scientifically validate performance differences between setups, the **Wilcoxon Signed-Rank Test** was applied:
* **What it is:** A non-parametric paired statistical test that compares two related samples to assess whether their population mean ranks differ.
* **How it was applied:** The performance of Setup A vs Setup B was paired and compared across **63 distinct configurations** (7 Datasets x 9 Extraction Algorithms). It's important to mention that at the biginning there was (2 * 7 * (16 * 8 + 1) = 1806 trained classifiers. In order to simplyfi, the best layer was chose in order to have this 63 pairs for the 3 metrics.
* **The Goal:** To test the null hypothesis that class imbalance does *not* affect performance. Finding a p-value < 0.05 definitively rejects this, proving class imbalance harms the extraction.

## 6. Final Results & Interpretation

### Finding 1: Balanced Training Yields Statistically Better Ranking (AUC & MAP)
While the core "lie signal" remains broadly intact regardless of imbalance, the Wilcoxon test definitively proved (p < 0.05) that **Setup A (Balanced) significantly outperforms Setup B (Imbalanced)** in terms of overall ranking capability. However, the actual performance magnitude drop is small:
* **ROC AUC:** Decreased by a mean difference of `0.0044` when moving to Imbalanced data.
* **MAP:** Decreased by a mean difference of `0.0042` when moving to Imbalanced data.
Even though the difference is minute (less than 1% absolute drop), the degradation is strictly consistent across datasets, proving balanced training mathematically preserves the discriminator's precision.

### Finding 2: Decision Boundaries Collapse (Asymmetric Sensitivity)
Beyond the small drop in ranking metrics, the functional application of the detectors completely breaks under imbalance. Because there are dramatically more lies in Setup B, the classifier shifts its decision boundary drastically to penalize Truths.
* **Result**: While standard *Accuracy* appears high (because it safely labels all the lies correctly), the **Recall (Truth)** and **Macro F1 Scores** plummet catastrophically (e.g., from 96% down to 33% on difficult reasoning tasks). 
* **Exception**: Very easy datasets (like `dbpedia_14`) where the separation is incredibly obvious do not suffer from this decision boundary collapse.

### Finding 3: Calibration Remains Unaffected (BRP_90)
Interestingly, highly-confident predictions are not fundamentally disturbed. The Brier score on the top 90% most confident predictions (BRP_90) showed no statistically significant difference (p = 0.0988) between Setup A and Setup B, meaning the model's calibration on its highest certainty answers is untouched by imbalance.

### Conclusion
Class imbalance causes a total collapse of correct decision boundaries (Recall/F1) while simultaneously inducing a small, but persistent, decay in core extraction quality (AUC/MAP). Because the goal is to extract maximum possible performance and to avoid the model defaulting to assumptions of deception, **we will utilize the Balanced scenario (Setup A) moving forward**, as it is logically and statistically superior, even if the AUC improvement is slight.