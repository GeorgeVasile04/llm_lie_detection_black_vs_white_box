# Probe Algorithms: Deep Dive

This document explains the logic and mechanics of the 8 probe algorithms used in our experiments.

Each algorithm shares the same goal: input a set of **Activation Vectors** ($a$) and output a single **Probe Vector** ($v$).

---

## 1. Linear Artificial Tomography (LAT)
**Source:** [Representation Engineering: A Top-Down Approach to AI Transparency (RepE)](https://arxiv.org/abs/2310.01405)

### The Concept
LAT is an **unsupervised** method. It discovers a meaningful direction without needing "True" or "False" labels. It assumes that if we look at the *differences* between random thoughts, the most significant variation corresponds to the concept we care about (in this specific dataset context).

### The Algorithm Steps
1.  **Input:** A pile of Activation Vectors ($a_{1}, a_{2}, ..., a_{n}$). These are unlabeled mixed thoughts from the model.
2.  **Pairing:** Randomly pair them up.
3.  **Difference:** Calculate the difference vector for each pair ($d = a_{i} - a_{j}$).
    *   *Why?* Taking the difference removes common information and leaves only the specific way in which they differ.
4.  **PCA (Principal Component Analysis):** Find the "First Principal Component" of these difference vectors.
    *   *Translation:* Find the single direction arrow that explains the *most* variance in these differences.
5.  **Output:** The PCA component is our Probe Vector ($v$).

### Concrete Example
Imagine a 2D world where "Truth" is the X-axis (Left/Right) and "Topic" is the Y-axis (Up/Down).

**Data Points (Input):**
*   $a_1$ (True, Dogs): `[5, 10]`
*   $a_2$ (False, Dogs): `[-5, 10]`
*   $a_3$ (True, Cats): `[5, -10]`
*   $a_4$ (False, Cats): `[-5, -10]`

**Process:**
1.  **Random Pair:** Let's pair $a_1$ and $a_2$.
    *   Diff $d_1 = a_1 - a_2 = [5, 10] - [-5, 10] = [10, 0]$
    *   *Notice:* The Y-value (Topic) cancelled out to 0! We are left with a pure X-vector (Truth).
2.  **Random Pair:** Let's pair $a_3$ and $a_4$.
    *   Diff $d_2 = a_3 - a_4 = [5, -10] - [-5, -10] = [10, 0]$
    *   *Notice:* Again, the Y-value is gone.
3.  **PCA:** The algorithm looks at $d_1$ `[10, 0]` and $d_2$ `[10, 0]`.
    *   It easily sees that **all the variation lies along the X-axis**.
    *   It returns the vector `[1, 0]` as the probe.

**Result:** The probe successfully points to "Truth" (X-axis) without ever being told which statement was true or false.

### Deep Dive: Why does Random Pairing Work?
You might wonder: *If we subtract random pairs (e.g., "Dog" - "Car"), doesn't that create a mess?*

You are right that a single pair creates noise (e.g., "Dogness minus Carness"). However, the algorithm relies on the **Law of Large Numbers**:

1.  **The Noise (Topic):** When we subtract thousands of random pairs, the "Topic" differences point in random directions (Left, Up, Down-Right). Geometrically, this looks like a round, fuzzy ball of data. There is no consistent direction.
2.  **The Signal (Truth):** In many pairs, we will happen to subtract a True statement from a False one (or vice versa). This introduces a consistent "Truth" component.
3.  **PCA's Role:** PCA simply looks for the direction where the differences are the **most extreme** (highest variance).
    *   Since topics are random, their differences tend to cancel out or stay small (a round ball).
    *   Since Truth is consistent, its differences add up to create huge values (e.g., True minus False = +10, False minus True = -10), stretching the data into a long "cigar" shape.
    *   PCA identifies this long axis as the Truth Vector.

---

## 2. Contrast-Consistent Search (CCS)
**Source:** [Discovering Latent Knowledge in Language Models Without Supervision (DLK)](https://arxiv.org/abs/2212.03827)

### The Concept
CCS is an **unsupervised** method (no True/False labels needed), but it is a **Grouped** method. It assumes that "Truth" is the direction that satisfies logical consistency rules.

It works by feeding the model pairs of **contrastive statements** (Yes/No answers to the same question). Even if the model doesn't know the answer, its "belief" should be logically consistent.

**The Logical Rules:**
1.  **Consistency:** The probability of "Yes" plus the probability of "No" should equal 1 (approx).
    *   $P(Yes) + P(No) \approx 1$
2.  **Confidence:** The model should be confident in one of them. It shouldn't just say "Maybe" (0.5) to both.

### The Algorithm Steps
1.  **Input:** Pairs of activations ($x_i^+, x_i^-$).
    *   $x_i^+$: Activation for "Question $i$: Yes"
    *   $x_i^-$: Activation for "Question $i$: No"
2.  **Normalization:** We subtract the mean of each pair from itself.
    *   This is "Group Normalization". It removes the "Topic" of the question, leaving only the "Yes/No" signal.
3.  **Optimization:** We don't just use PCA. We search for a direction $v$ that minimizes a special loss function based on the logical rules above.
    *   We want $v \cdot x^+$ (Probability of Yes) and $v \cdot x^-$ (Probability of No) to sum to 1.
4.  **Output:** The vector $v$ that best satisfies these rules is the probe.

### Concrete Example
**Question:** "Is a cat an animal?"

**Input Vectors:**
*   $x^+$ (Answer: Yes): `[5, 10]`  *(Think: 5=Truth, 10=Animal Topic)*
*   $x^-$ (Answer: No):  `[-5, 10]` *(Think: -5=False, 10=Animal Topic)*

**Step 1: Normalization (Removing Topic)**
*   Average of pair: `[0, 10]`
*   $x_{norm}^+ = [5, 10] - [0, 10] = [5, 0]$
*   $x_{norm}^- = [-5, 10] - [0, 10] = [-5, 0]$

**Step 2: Finding Consistency**
The algorithm looks for a direction $v$.
*   **Hypothesis A (Vertical Probe):** $v=[0, 1]$
    *   Score Yes: $0$
    *   Score No: $0$
    *   Sum: 0. **Fail!** (Target is 1).
*   **Hypothesis B (Horizontal Probe):** $v=[1, 0]$
    *   Score Yes: $5$ (High/True)
    *   Score No: $-5$ (Low/False)
    *   *Note: In practice, these scores are mapped to probabilities (0 to 1).*
    *   If mapped, Yes $\approx 1.0$, No $\approx 0.0$. Sum $\approx 1.0$. **Success!**

**Result:** The algorithm finds the horizontal vector `[1, 0]` because it's the only direction where "Yes" and "No" behave like opposite probabilities.

---

## 3. Difference-in-Means (DIM)
**Source:** [The Geometry of Truth (GoT)](https://arxiv.org/abs/2310.18168)

### The Concept
DIM is a **supervised** method. It requires labels (True/False). It is also the simplest and most intuitive method, often serving as the primary baseline or "winner" in generalization tasks.

It relies on the **Geometry of Truth hypothesis**: that True statements cluster in one region and False statements cluster in another, and the vector connecting them represents the concept of Truth.

### The Algorithm Steps
1.  **Input:** Two piles of activation vectors.
    *   $A_{true}$: All activations labeled "True".
    *   $A_{false}$: All activations labeled "False".
2.  **Centroids:** Calculate the center of mass (mean) for each pile.
    *   $\mu_{true} = \text{average}(A_{true})$
    *   $\mu_{false} = \text{average}(A_{false})$
3.  **Output:** Subtract the two means.
    *   $v = \mu_{true} - \mu_{false}$

### Concrete Example
**Data Points:**
*   True 1 (Cats are animals): `[5, 10]`
*   False 1 (Cats are plants): `[-5, 10]`
*   True 2 (Fire is hot): `[5, -20]`
*   False 2 (Fire is cold): `[-5, -20]`

**Process:**
1.  **Mean True:** Average of `[5, 10]` and `[5, -20]`
    *   X: $(5 + 5)/2 = 5$
    *   Y: $(10 - 20)/2 = -5$
    *   $\mu_{true} = [5, -5]$
2.  **Mean False:** Average of `[-5, 10]` and `[-5, -20]`
    *   X: $(-5 - 5)/2 = -5$
    *   Y: $(10 - 20)/2 = -5$
    *   $\mu_{false} = [-5, -5]$
3.  **Difference ($v$):**
    *   $\mu_{true} - \mu_{false} = [5, -5] - [-5, -5] = [10, 0]$

**Result:** The Y-values (Topics) averaged out to similar noisy values (-5) in both groups, so they disappeared in the subtraction. The X-values (Truth) reinforced each other. We found the truth direction `[10, 0]`.

---

## 4. Linear Discriminant Analysis (LDA)
**Source:** [The Geometry of Truth (GoT)](https://arxiv.org/abs/2310.18168) (referred to as Mass-Mean Probe IID)

### The Concept
LDA is a **supervised** method, very similar to DIM. It is essentially an "Upgraded DIM".

**The Problem with DIM:** DIM draws a line directly between the centers of True and False. This works perfectly if the data clouds are round circles.
**The Reality:** Data clouds often look like stretched ellipses (because some dimensions are correlated). If the clouds are stretched, the best separating line isn't the direct path between centers—it needs to account for the shape of the clouds.

LDA fixes this by "whitening" the data (removing correlations) before drawing the line.

### The Algorithm Steps
1.  **Input:** Two piles of activation vectors (True and False).
2.  **Calculate Means:** Same as DIM ($\mu_{true}, \mu_{false}$).
3.  **Calculate Covariance:** Compute the Covariance Matrix ($\Sigma$) of the data. This matrix describes how the dimensions influence each other (the "shape" of the data cloud).
4.  **Output:** Multiply the difference in means by the *inverse* of the covariance.
    *   $v = \Sigma^{-1} (\mu_{true} - \mu_{false})$

### Concrete Example (The Diagonal Problem)
Imagine 2D data that is stretched diagonally.

**Centers:**
*   $\mu_{true} = [2, 2]$
*   $\mu_{false} = [-2, -2]$

**DIM's Solution:**
*   $v = [4, 4]$ (Points diagonally up-right at 45 degrees).
*   *Issue:* If the data clouds are stretched heavily along that 45-degree line, the "True" and "False" clouds will overlap along that line. The DIM probe will draw a line that cuts through both clouds, mixing them up.

**LDA's Solution:**
*   The Covariance Matrix ($\Sigma$) tells LDA: "Hey, these variables are highly correlated, don't trust the diagonal direction too much."
*   $\Sigma^{-1}$ effectively rotates the space so the clouds look like spheres (decorrelation).
*   The resulting vector $v$ shifts to find a perpendicular angle that slices *between* the diagonal clouds cleanly, rather than crashing through them.

**Note:** In the paper's results, LDA performed poorly out-of-distribution. This suggests that the detailed correlations (Covariance) are specific to one dataset (e.g., "cities") and do not generalize to another (e.g., "sentiment"), effectively overfitting to the training domain.

---

## 5. Principal Component Analysis (PCA)
**Source:** Standard statistical baseline.

### The Concept
This is the simplest **Unsupervised, Ungrouped** method. It assumes that "Truth" is the singule most important concept in the dataset.

It simply asks: "What is the direction of maximum variance in the raw data?"

### The Algorithm Steps
1.  **Input:** The raw pile of activation vectors ($a$). No pairing, no labels.
2.  **PCA:** Calculate the First Principal Component of the raw vectors.
3.  **Output:** That component is the probe $v$.

### Intuition: How PCA works (The Baguette Analogy)
How does the algorithm physically find this vector?

1.  **The Shape:** Imagine the data points form a cloud. Because one signal (like Truth) is very strong, this cloud is stretched out, shaped like a long **French Baguette**.
2.  **The Shadow:** Imagine sticking a line (skewer) through the center of this cloud. If you project every point onto this line (cast a shadow), how long is the total shadow?
    *   If the line goes through the width (short side), the points bunch up. The shadow is short (**Low Variance**).
    *   If the line goes through the length (long axis), the points spread out far. The shadow is long (**High Variance**).
3.  **The Mechanism:** The algorithm effectively **spins the line** until the "shadow" is as long as possible. The direction of the line at that exact moment is the Probe Vector.

### Comparison with LAT
LAT and PCA both use the PCA algorithm, but on **different inputs**. This difference is massive.

*   **LAT Input:** *Differences* between random pairs ($a_i - a_j$).
*   **PCA Input:** *Raw* vectors ($a_i$).

**Why LAT is usually better:**
*   **PCA Failure Mode:** In the raw vectors, the biggest variance is usually **not** Truth. It is often the **sentence length**, the **token position**, or a broad topic like "Science vs. Art". If you just run PCA on raw data, the probe will likely detect "Long Sentence vs Short Sentence" instead of "True vs False".
*   **LAT's Fix:** By subtracting pairs ($a_i - a_j$), LAT cancels out the static properties (like "we are in layer 21" or "embedding space typical magnitude"). It forces the PCA to look only at *what changes* between sentences.

**Example:**
*   **Raw Data Points:** `[1000, 5]`, `[1000, -5]`. (1000 is a huge constant "Layer Bias").
*   **Standard PCA:** Will point towards `[1, 0]` (The 1000 axis). It thinks the 1000 is the important feature.
*   **LAT (Differences):** `[1000, 5] - [1000, -5] = [0, 10]`. The 1000 vanishes. LAT finds `[0, 1]` (Truth).

---

## 6. Grouped Principal Component Analysis (PCA-G)
**Source:** [Discovering Latent Knowledge (DLK)](https://arxiv.org/abs/2212.03827) (referred to as CRC-TPC)

### The Concept
This is the **Unsupervised, Grouped** version of PCA. It combines the logic of "Grouping" (removing topics) with the logic of PCA (finding max variance).

It is significantly stronger than standard PCA because it manually removes the "Topic Noise" before asking PCA to find the "Truth Signal".

### The Algorithm Steps
1.  **Input:** Grouped activation vectors (e.g., pairs of Yes/No answers for the same questions).
2.  **Normalization:** For each group (question), calculate the mean vector and subtract it from all answers in that group.
    *   $x_i' = x_i - \text{mean}(\text{Group}_i)$
    *   *Effect:* This kills the "Question Topic".
3.  **PCA:** Run standard PCA on these new centered vectors ($x'$).
4.  **Output:** The First Principal Component is the probe $v$.

### Comparison: PCA-G vs. LAT
Both use PCA. Both try to remove noise. But they do it differently.

*   **LAT (Random Subtraction):** "If we subtract *random* things enough times, the noise averages out statistically." (Statistical cleaning).
*   **PCA-G (Targeted Subtraction):** "We *know* these two answers are about the same question. Let's subtract them specifically to remove the topic deterministically." (Deterministic cleaning).

**Result:** PCA-G is generally cleaner and more accurate than LAT because it uses the structure of the data (the groups) rather than relying on random pairings.

---

## 7. Logistic Regression (LR)
**Source:** Standard Supervised Classification.

### The Concept
This is the classic Machine Learning approach. It is **Supervised** and **Ungrouped**.
Instead of just looking for the average direction (DIM) or the widest direction (PCA), Logistic Regression tries to find the **Best Separator**.

It asks: *"Draw a line that puts as many Blue Dots (True) on one side and Red Dots (False) on the other side as possible."*

### The Algorithm Steps
1.  **Input:** The raw pile of activation vectors ($a$) with labels ($y \in \{0, 1\}$).
2.  **Training:** Use an optimizer (like Gradient Descent) to find the weight vector $v$ that minimizes the error:
    *   Error = -[y * log(prediction) + (1-y) * log(1-prediction)]
3.  **Output:** The weights of the trained classifier are the probe $v$.

### Comparison with DIM (Why use LR?)
*   **DIM:** Relies on the **centers** of the clouds.
    *   *Weakness:* If the "False" cloud has one massive outlier far away, it pulls the mean (center) drastically, potentially ruining the probe.
*   **LR:** Relies on the **boundary** between clouds.
    *   *Strength:* It can ignore outliers that are far away from the boundary. It focuses intensely on the "difficult cases" right in the middle where True and False mix.
    *   *Trade-off:* It is mathematically more complex and prone to "overfitting" (memorizing the specific training data examples instead of learning the general direction of truth).

---

## 8. Grouped Logistic Regression (LR-G)
**Source:** Combining concepts from DLK and Standard ML.

### The Concept
This is the **Supervised, Grouped** version of LR. It simply combines the "Topic Removal" trick from earlier with the logic of Logistic Regression.

### The Algorithm Steps
1.  **Input:** Grouped activation vectors with labels.
2.  **Normalization (The Fix):** Just like in PCA-G, calculate the mean activation for each Question Group and subtract it from the answers.
    *   $x' = x - \text{GroupMean}$
3.  **Training:** Train a standard Logistic Regression classifier on these *centered* vectors $x'$.
4.  **Output:** The learned weights are the probe $v$.

### Why do this?
Standard LR often fails in high dimensions because it learns shortcuts.
*   *Trap:* If all "Geography" questions in the training set happen to be True, and all "Math" questions happen to be False, LR will learn to detect "Geography vs Math" because it yields 100% accuracy on the Training Data.
*   *Fix:* By grouping and normalizing first, LR-G **forces** the classifier to ignore the topic. It physically cannot see "Geography" anymore (it was subtracted out). It is forced to find the subtle "Truth" signal.

---
**Summary Comparison Table**

| Method | Labels Needed? | Removes Topic? | Logic |
| :--- | :--- | :--- | :--- |
| **DIM** | Yes | No | Connect the centers (Mean). |
| **LDA** | Yes | No | Connect centers + Fix shape (Covariance). |
| **LR** | Yes | No | Find the best boundary line. |
| **LR-G** | Yes | **Yes** | Remove topic $\to$ Find best boundary. |
| **PCA** | No | No | Find biggest variance (often fails). |
| **LAT** | No | **Statistical** | Subtract random pairs $\to$ Find variance. |
| **PCA-G**| No | **Yes** | Remove topic $\to$ Find variance. |
| **CCS** | No | **Yes** | Find logical consistency. |
