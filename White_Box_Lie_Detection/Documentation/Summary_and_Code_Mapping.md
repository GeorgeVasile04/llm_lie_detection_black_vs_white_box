# Summary: How well do truth probes generalise?

This document summarizes the methodology and findings of the paper "How well do truth probes generalise?" by mishajw. It serves as a guide to understanding the "White Box" approach to lie detection implemented in this repository.

## Core Concept: Representation Engineering & Truth Probes
The central hypothesis is that Large Language Models (LLMs) possess a single, generalized direction in their internal representation space that corresponds to "truth". If this holds, a "probe" trained to detect truth in one domain (e.g., geography) should theoretically function in a completely different domain (e.g., sentiment analysis).

### The Setup
*   **Model:** Llama-2-13b-chat.
*   **Concept:** **Linear Probes**. A probe is simply a vector $v$. To detect a lie, we extract the model's internal activation vector $a$ at a specific layer (usually corresponding to the last token of a prompt). The "truth score" is calculated as the dot product $v^T a$.
*   **Goal:** Find a vector $v$ such that high scores correspond to "Truth" and low scores correspond to "Lie".

---

## Deep Dive: How It Works
Before getting into the experiments, it is crucial to understand the geometry of the system.

### 1. The Space (The "Brain")
*   The LLM does not think in words; it thinks in **vectors**.
*   The "Activation Space" (or Hidden State) is a high-dimensional room. For **Llama-2-13b**, this room has **5,120 dimensions**.
*   Every concept (cat, dog, King, Queen) is a specific point in this room.
*   Similar concepts cluster together.

### 2. The Vector Probe ($v$) and the Extraction Process
The "probe" is just a list of 5,120 numbers (a vector) that points in the direction of "Truth".
*   **How we get activations ($a$):**
    1.  Input a question: *"Paris is in France."*
    2.  The model processes it token-by-token.
    3.  We stop at the **Last Token**.
    4.  *Why the last token?* Because in Autoregressive models (like Llama), each token can only see previous tokens. The last token is the specific point where the model has processed the entire sentence and "understands" the full context.
    5.  We copy the 5,120 numbers at that specific layer. This is our activation vector $a$.

*   **How we build the probe ($v$) - Difference-in-Means method:**
    1.  Collect 500 vectors ($a$) for known **True** statements. Calculate the average "True Point".
    2.  Collect 500 vectors ($a$) for known **False** statements. Calculate the average "False Point".
    3.  Subtract them: $v = \text{AverageTrue} - \text{AverageFalse}$.
    4.  This vector $v$ is an arrow pointing from False to True.

    **Example Scenario:**
    Imagine we just want to train a probe using simple facts.
    1.  **Input:** We have a dataset of pure facts, labeled as False or True (e.g., "Fire is hot" [True], "Ice is hot" [False]). We do not ask questions, we just feed these statements.
    2.  **Extraction:** For every single True statement, we feed it into the model and record the activation state ($a$) at the very last token. We do the exact same for every False statement.
    3.  **Averaging:** We compute the mean (average) vector of all the "True" activations. Then we compute the mean vector of all the "False" activations.
    4.  **Difference:** Finally, we simply subtract: `Mean_True - Mean_False`.
    5.  **Result:** This difference *is* the probe vector $v$. It is nothing more than the difference of means of activation states.

### 3. The Truth Score (The "Lie Detector")
To test a new statement, we get its activation vector $a_{new}$.
We take the **Dot Product**: $Score = v \cdot a_{new}$.
*   **High Score (> 0):** The thought aligns with the Truth arrow.
*   **Low Score (< 0):** The thought aligns with the False direction.

---

## Experiment 1: Methods & Probe Creation
Before testing generalization, the paper defines the "ingredients" for building these probes.

### 1. Algorithms (How to calculate $v$)
The paper implements and compares 8 distinct algorithms. These are categorized by whether they require labeled truth data (**Supervised**) and whether they group related data points locally (**Grouped**).

#### Supervised (Requires True/False labels)
*   **Logistic Regression (LR):** Standard classification to separating true/false activations.
*   **Logistic Regression Grouped (LR-G):** LR, but data is centered relative to its group mean to remove noise.
*   **Difference-in-Means (DIM):** The simplest valid approach. Calculate the mean vector of all "True" examples and the mean vector of all "False" examples. The probe is $Mean_{true} - Mean_{false}$.
*   **Linear Discriminant Analysis (LDA):** Similar to DIM, but skews the boundary based on feature covariance to handle interference between features.

#### Unsupervised (No labels needed)
*   **Principal Component Analysis (PCA):** Takes the direction of maximum variance.
*   **PCA-Grouped (PCA-G):** PCA applied to data centered by question.
*   **Contrast-Consistent Search (CCS):** Searches for a direction that is logically consistent (e.g., $Probability(Yes) + Probability(No) \approx 1$) across contrastive pairs.
*   **Linear Artificial Tomography (LAT):** Computes PCA on differences between random pairs of activations.

### 1.1 Deep Dive: Understanding "Grouped" Data
Why do some algorithms use "Grouped" data? This concept is critical for robustness.

**The Concept:**
An activation vector ($a$) is a mix of *everything* the model is thinking. It contains both the **Topic** (e.g., Geography) and the **Truth Value** (True/False).
*   **Ungrouped:** The probe might get confused by the strong "Topic" signal and learn to detect "Geography" instead of "Truth".
*   **Grouped:** We isolate the "Truth" signal by removing the "Topic" signal using normalization. We group statements that share the exact same topic (e.g., answers to the same question).

**Concrete Example: "The Color of the Sky"**
Imagine we have a group of answers to the question: "What color is the sky?"
We run these through the model and extract the activation vectors ($a$) at layer 21.

*   **Statement A:** "The sky is Blue" (True) $\rightarrow$ Activation $a_1$
*   **Statement B:** "The sky is Green" (False) $\rightarrow$ Activation $a_2$
*   **Statement C:** "The sky is Red" (False) $\rightarrow$ Activation $a_3$

*For illustration, let's pretend these 5,120-dimensional vectors look like this (Topic, Truth):*
*   $a_1 = [100, 5]$ (100 = Sky concept, 5 = True)
*   $a_2 = [100, -5]$ (100 = Sky concept, -5 = False)
*   $a_3 = [100, -5]$ (100 = Sky concept, -5 = False)

**The Grouped Approach (Normalization):**
1.  **Calculate Group Mean:** Average of the vectors.
    *   Topic: Average(100, 100, 100) = 100
    *   Truth: Average(5, -5, -5) = -1.6
    *   **Mean Vector:** $[100, -1.6]$
2.  **Subtract Mean from Each Statement:**
    *   $a_1 - Mean = [0, 6.6]$
    *   $a_2 - Mean = [0, -3.4]$
    *   $a_3 - Mean = [0, -3.4]$

**Result:** The "Sky" concept (100) is cancelled out to **0**. The probe now only sees the pure "Truth" signal.

### 2. Datasets
The paper utilizes 18 diverse datasets from three different sources, each with a unique style.

#### A. DLK Datasets (8 datasets)
*   **Style:** Traditional Natural Language Processing tasks turned into Binary Q&A.
*   **Content:** Sentiment analysis, topic classification, etc.
*   **Prompting:** Uses a complex template asking the model to choose between Choice 1 and Choice 2.
    > *Example:* "Consider the following example: {content}... The sentiment is {choice1 / choice2}"
*   **List:** `imdb`, `amazon_polarity`, `ag_news`, `dbpedia_14`, `rte`, `copa`, `boolq`, `piqa`.

#### B. RepE Datasets (5 datasets)
*   **Style:** Multiple Choice Questions (2-5 options) focusing on reasoning and common sense.
*   **Unique Feature:** Uses a specific prompt suffix to trigger truth evaluation: *"The probability of the answer being plausible is"*.
*   **List:** `openbook_qa`, `common_sense_qa`, `race`, `arc_challenge`, `arc_easy`.

#### C. GoT (Geometry of Truth) Datasets (5 datasets)
*   **Style:** Short, synthetic statements of pure fact.
*   **Content:** Designed to be "uncontroversial, unambiguous, and simple".
*   **Prompting:** No question format, just straight assertions.
    > *Example:* "The city of {city} is in {country}."
*   **List:** `cities`, `sp_en_trans`, `larger_than`, `cities_cities_conj`, `cities_cities_disj`.
*   **Note:** `cities_cities_conj` (Conjunctive cities) proved to be an exceptionally strong dataset for training generalized probes.

### **Link to Code**
In the codebase (likely within `repeng/` and `experiments/`), you should look for:
*   **Activation Extraction:** Scaffolding to run the model and hook into specific hidden layers.
*   **Probe Classes:** Python classes that implement `.fit(activations)` for the algorithms above (especially `DIM` and `CCS`).
*   **Data Loaders:** Scripts that normalize the 18 different datasets into a common format of `(prompt, activation_vector, label)`.

---

## Experiment 2: Measuring Generalization (The Main Result)
**Objective:** Training a probe on Dataset A and testing it on Dataset B.

**Method:** The paper introduces **Recovered Accuracy**.
1.  Train a baseline probe directly on Dataset B (The "Cheating" Probe). This sets the max possible score for that dataset.
2.  Train the experimental probe on Dataset A.
3.  Test the experimental probe on Dataset B.
4.  **Recovered Accuracy** = (Accuracy of A on B) / (Accuracy of Baseline on B).

**Findings:**
*   **Truth Generalizes:** 36% of all probes (in mid-to-late layers) recover >80% accuracy on completely unseen tasks.
*   **The Winner:** A **DIM** probe, trained on the **dbpedia_14** dataset, at **Layer 21**. It recovered **92.8%** of accuracy on average across all other datasets.

---

## Experiment 3: Hyperparameter Analysis
**Objective:** Which choices define a good lie detector?

**Findings:**
*   **Layers:** Layers 0-9 are poor. Mid-to-late layers (>=13) are where the "truth direction" becomes stable.
*   **Algorithms:**
    *   Supervised methods generally beat unsupervised ones.
    *   **Grouped PCA** significantly outperforms standard PCA.
    *   **LDA** performed poorly (noted as an outlier, potentially due to implementation specifics).
*   **Datasets:**
    *   **DLK** datasets act as strong "teachers" (good to train on).
    *   `got_cities_cities_conj` was an outlier that yielded excellent results.

### **Link to Code**
In `experiments/`, there should be logic that:
*   Loops through `[Algorithm] x [Dataset] x [Layer]`.
*   Saves accuracy scores into results files (JSON/CSV).
*   Calculates the "Recovered Accuracy" metric for the plots seen in the paper.

---

## Experiment 4: Truth vs. Likelihood (Control Experiment)
**Objective:** Is the detector just identifying "sentences that look probable"? (e.g., "The earth is flat" might look improbable to a language model regardless of truth).

**Method:** Evaluate the probes on **TruthfulQA**.
*   TruthfulQA contains "imitated falsehoods"—statements that are false but sound plausible (misconceptions).
*   If the probe only detects *likelihood*, it will fail here.
*   If the probe detects *truth*, it will succeed.

**Findings:**
*   Probes that generalized well also scored high on TruthfulQA.
*   This confirms the probes are measuring factual correctness, separating "Truth" from "Likelihood".
