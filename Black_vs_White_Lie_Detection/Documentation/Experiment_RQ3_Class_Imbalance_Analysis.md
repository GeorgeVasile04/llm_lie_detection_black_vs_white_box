# RQ3: Robustness to Class Imbalance - Analysis

## Overview of the Experiment
The methodology isolated the impact of class imbalance by controlling sample sizes. For 7 diverse datasets, models were evaluated under two conditions:
- **Setup A (Balanced)**: 1 Truth vs 1 uniformly random Lie per prompt (50/50 split).
- **Setup B (Imbalanced)**: 1 Truth vs *ALL* available Lies per prompt naturally attached to the question (resulting in natural 1:3, 1:4, or 1:13 splits).

The goal was to track metric degradation across both **White-Box (Probes)** and **Black-Box (Logprobs)** extraction strategies when linear boundary classifiers are trained on skewed (unbalanced) data distributions.

---

## Key Findings Simplified

### 1. The Core Signal Survives (AUC remains stable)

**Understanding the Concept:**
In simple terms, **ROC AUC (Area Under the Curve)** measures the *ranking* ability of the detector. If you grab one random true statement and one random lie, AUC is simply the chance that the detector gave a higher "truth score" to the actual truth. 
- An **AUC of 0.50** means the detector is flipping a coin.
- An **AUC of 1.00** means the detector perfectly lines up every truth above every lie.

Crucially, AUC doesn't care if you have 10 truths and 100 lies, or 50 truths and 50 lies. It only measures whether the model *can tell the difference* between the core concepts.

**The Finding:**
When moving from Setup A (Balanced) to Setup B (Imbalanced), the **AUC stays nearly identical**.
- In `commonsense_qa`, the standard White-Box probe hits an AUC of `0.828` when balanced, and stays at `0.827` when flooded with lies.
- In `race`, the AUC goes from `0.800` (Balanced) to a very similar `0.782` (Imbalanced).

**What this means:**
The fundamental "lie signal" inside the Large Language Model does not break when there are way more lies than truths. Whether we look at internal brain states (White-Box) or output probabilities (Black-Box), the LLM still separates the *concept* of truth from the *concept* of a lie just as well. 

### 2. The Decision Boundary Breaks (Recall & Macro F1 Collapse)

**Understanding the Concepts:**
While AUC measures if the model *can* rank truths above lies, we still have to draw a line in the sand (a **threshold**) and say "Everything above this line is labeled Truth, everything below is labeled Lie."
- **Accuracy**: The total percentage of correct guesses. This can be very misleading! If a test is 90% lies, a model that simply screams "LIE" at everything will get 90% Accuracy, even though it's completely useless.
- **Recall (Truth)**: Out of all the actual true statements, how many did the model successfully catch? This is the true test of not ignoring the minority class.
- **Macro F1 Score**: This is the ultimate "fairness" score. It calculates a harsh average of how well the model detects Truths AND how well it detects Lies independently. You cannot get a high Macro F1 score if you are secretly ignoring the smaller group.

**The Finding:**
Because there are so many more lies in Setup B, the detector learns a bad habit: it shifts its decision boundary so high that it practically calls everything a lie. This leads to an **extreme asymmetric sensitivity** where the model is great at catching lies, but terrible at recognizing truths.
- **commonsense_qa**: Recall for Truths plummets from an outstanding **96.2%** (Balanced) down to a bleak **33.4%** (Imbalanced). The **Macro F1 score** similarly crashes from **0.885** down to **0.620**.
- **arc_challenge**: Recall for Truths drops from **72.2%** to **48.8%**.
- **open_book_qa**: Recall for Truths is slashed from **85.2%** to **47.4%**.

**What this means:**
The detector turns extremely "pessimistic." To get the highest simple math score during training, it realizes it's safer to just assume most things are lies. Therefore, true statements are heavily punished. This is why looking only at **Accuracy** is dangerous; the Accuracy often stays high (e.g., 80%) because the model is correctly identifying the massive pile of lies, completely masking the fact that its ability to detect truth has broken down. The **Macro F1 score** exposes this failure clearly.

### 3. "Easy" Tasks Hide the Problem
Datasets that are strictly factual or easy for LLaMA-2-7B to answer completely ignore the imbalance problem.
- **dbpedia_14** (1 Truth vs 13 Lies) and **ag_news** easily kept `Recall (True)` scores above 90% even when unbalanced. 

**What this means:**
If the difference between truth and lie is incredibly obvious to the model (where AUC is near 0.99), the decision boundary can be drawn widely anywhere without making mistakes. The imbalance problem only destroys the detector in highly complex, difficult reasoning tasks (like `commonsense_qa` or `race`).

---

## Conclusion: Answering RQ3

**Research Question 3:** *How robust are black-box and white-box lie detection methods to class imbalance, and do these methods exhibit asymmetric sensitivity to deceptive versus truthful instances under skewed data distributions?*

**Answer:** 
Black-box and white-box lie detection methods display a **deceiving duality** when exposed to class imbalance (like multiple-choice scenarios with many wrong answers and one right answer). 

On a fundamental level, the methods are highly robust: the LLM's internal representation structure successfully preserves the difference between truthful and deceptive concepts, as shown by very stable ROC AUC scores regardless of how unbalanced the data is.

However, the functional application of these methods (the actual classifiers built to say "Truth" or "Lie") exhibits **severe asymmetric sensitivity**. When faced with heavy skews, the decision boundaries drift aggressively toward predicting deception. The detectors become artificially "pessimistic," maintaining high Accuracy by catching almost all the lies, but suffering catastrophic drops in `Recall (Truth)` and `Macro F1`—often losing 30 to 60 percentage points in performance on complex reasoning tasks. 

Ultimately, without explicitly rebalancing the training data or manually adjusting the decision thresholds, standard Black-Box and White-Box lie detectors cannot be trusted in real-world skewed environments, because they will simply default to assuming everything is a lie.

Discussion with the teacher:
1) How the sklearn decides on the treshold of the clasifier? Is it always 0.5? How in the original black box and white box paper what they did?

2) How to make a fair comparison across different classifiers?

3) I have 2 setups how to say that setup A is better than B significance above 0,05?

4) Have a hypothesis test, assume that setup A is better than setup B? How can I see that this is true or false?

5) Do the Wilcoxon Signed-Rank Test