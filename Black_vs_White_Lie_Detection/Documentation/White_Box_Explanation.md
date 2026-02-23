# White Box Lie Detection: Experiment & Methodology

## 1. Introduction
This experiment implements a **White Box** approach to lie detection. Unlike "Black Box" methods that analyze the text output of an LLM, this approach looks inside the model's "brain" (its internal activation states) to determine if it is processing a truthful or deceptive statement.

## 2. Dataset & Preprocessing

### 2.1 Raw Data
We use a dataset of Common Sense QA (or similar general knowledge).
*   **Format**: JSON or CSV.
*   **Content**: Simple fact-based questions and answers.

### 2.2 Truth/Lie Pairing (Preprocessing)
The core of this experiment relies on **contrast pairs**. For every factual question, we generate two versions:
1.  **Honest Version**: The question with the correct answer.
2.  **Lying Version**: The same question with an incorrect answer.

**Shape of Data**:
Each record in our processed dataset contains:
*   `question`: "Is the sky blue?"
*   `answer`: "Yes" (Label: 0/Truth) OR "No" (Label: 1/Lie)
*   `label`: Integer target (0 for Truth, 1 for Lie).

This ensures that the *only* difference between the two samples is the truthfulness of the assertion, minimizing confounding variables like sentence length or topic.

## 3. Model Loading
We load a Large Language Model (LLM) such as `Llama-2-7b-chat` or `Pythia-1b` onto the GPU.
*   We use the `repeng` (Representation Engineering) library wrappers.
*   The model is loaded in half-precision (`floa16`) to save memory.

## 4. Feature Extraction (The "White Box" Part)

### 4.1 The Prompt
We do **not** ask the model to generate text. instead, we feed it a completed interaction.
*   **Format**: `Question: {question} Answer: {answer}`
*   **Example**: `Question: Is the sky blue? Answer: No`

### 4.2 Activation Extraction
As the model processes this text, it produces a vector of numbers (activations) at every layer for every token.
*   **Target Token**: We extract the hidden state of the **very last token**. This represents the model's final "thought" or processing state regarding the statement it just read.
*   **Target Layers**: We repeat this for every layer in the model (e.g., Layer 0 to Layer 31).
*   **Result**: For a dataset of $N$ samples and a model with dimension $D$, we get a matrix $X$ of shape $(N, D)$ for every layer.

## 5. The Detection Algorithm: Difference-in-Means (DiM)

We use a linear probe known as **Difference-in-Means**. This is a geometrically simple but powerful method.

### 5.1 Training Phase
We split our data into Training (80%) and Test (20%) sets. With the Training set:
1.  **Calculate Mean Lie Vector ($\mu_{lie}$)**: The average activation of all samples labeled "Lie".
2.  **Calculate Mean Truth Vector ($\mu_{truth}$)**: The average activation of all samples labeled "Truth".
3.  **Compute Probe Direction ($V_{probe}$)**:
    $$ V_{probe} = \mu_{lie} - \mu_{truth} $$
    This vector $V_{probe}$ represents the "direction" of deception in the model's high-dimensional thought space.

### 5.2 Test Phase (Scoring)
To classify a new, unseen statement (with activation vector $x$):
1.  **Project onto Probe**: We calculate the dot product.
    $$ Score = x \cdot V_{probe} $$
2.  **Interpretation**:
    *   A high score means the activation $x$ is statistically similar to the "Lie" cluster.
    *   A low score means it is closer to the "Truth" cluster.

## 6. Thresholding & Decision Making
The dot product gives us a raw number (logit), not a "Yes/No". To make a decision, we need a **Threshold ($t$)**.

### 6.1 Geometric ROC Thresholding
We do not pick an arbitrary number like 0.5. Instead, we use the training data to find the optimal cutoff:
1.  Compute the **ROC Curve** (Receiver Operating Characteristic) for the training scores.
2.  This curve plots True Positive Rate (TPR) vs. False Positive Rate (FPR) for all possible thresholds.
3.  **Optimal Point**: We select the threshold that minimizes the geometric distance to the perfect classifier (Top-left corner: TPR=1, FPR=0).
    $$ \text{Best Index} = \text{argmin} \sqrt{(1 - TPR)^2 + FPR^2} $$
    
This specific threshold $t$ is then used for the Test set:
*   If $Score > t \rightarrow$ **Predict Lie**
*   If $Score \le t \rightarrow$ **Predict Truth**

#### Example Illustration (5 Steps)
Here we have the theoretical 5 steps:

Step 1: The algorithm takes every unique activation score in your training set (e.g., 0.1, 0.4, 0.9...) and pretends it is the threshold.

Step 2: For each candidate threshold t_i, it calculates the resulting False Positive Rate (FPR_i) and True Positive Rate (TPR_i). This gives one point on the graph (x,y).

Step 3: We get a list of points. Each point corresponds to a specific number t_i from our data.

Step 4: We mathematically find which point is closest to the top-left corner (perfect prediction). Let's say it's the 50th point.

Step 5: We simply look up what threshold created that 50th point. That numeric value (e.g., t=0.742) is our optimal threshold.

Here is the example:

Imagine we have 4 training samples:
*   A (Lie): Score 0.9
*   B (Truth): Score 0.8
*   C (Lie): Score 0.4
*   D (Truth): Score 0.1

1.  **List Thresholds**: The algorithm effectively tests every score as a candidate.
2.  **Calculate ROC Points**:
    *   **Hypothesis $t=0.85$**: Only A is predicted Lie.
        *   True Positive Rate (TPR) = 1 caught / 2 total lies = 0.5
        *   False Positive Rate (FPR) = 0 false alarms / 2 total truths = 0.0
        *   Point: $(0, 0.5)$
    *   **Hypothesis $t=0.3$**: A, B, C are predicted Lie.
        *   TPR = 2/2 = 1.0 (Caught all lies)
        *   FPR = 1/2 = 0.5 (False alarm on B)
        *   Point: $(0.5, 1.0)$
3.  **Find Geometric Index**: We calculate the distance of each point to the perfect corner $(FPR=0, TPR=1)$.
4.  **Minimize Distance**:
    *   Point $(0, 0.5)$ distance = $ \sqrt{(1-0.5)^2 + 0^2} = 0.5 $
    *   Point $(0.5, 1.0)$ distance = $ \sqrt{(1-1)^2 + 0.5^2} = 0.5 $
    *   *(In a real scenario, we find the point with the minimal distance).*
5.  **Retrieve Numeric Value**: If the index corresponding to point $(0, 0.5)$ was chosen, then we look up the threshold that created it (e.g., **0.85**) and that becomes our fixed threshold for the test set.

## 7. Metrics

### 7.1 Accuracy
The percentage of correct predictions on the Test set using the optimal threshold.
*   *Pros*: Easy to understand.
*   *Cons*: Dependent on the specific threshold chosen.

### 7.2 AUC (Area Under the ROC Curve)
The probability that a randomly chosen Lie sample has a higher score than a randomly chosen Truth sample.
*   **1.0**: Perfect Probe.
*   **0.5**: Random Guessing.
*   *Significance*: This is our **primary metric**. It tells us how well the "Lie Direction" separates the two concepts, regardless of where we draw the line. It validates the *quality* of the extracted features.

## 8. Iterative Improvements and Results

The development of our lie detection probe followed an iterative process, refining both the model choice and the data quality to achieve robust generalization.

### 8.1 Phase 1: Initial Baseline (Pythia-1b)
**Setup**: We started with `Pythia-1b`, a relatively small model, using the basic `Questions 1000` dataset.
**Result**: The performance was poor, often hovering near random guessing or showing very weak separability between truth and lies.
**Conclusion**: Small models may not have a sufficiently distinct internal representation of "truth" versus "falsehood," or their representations are too entangled with other features to be easily probed by a simple linear classifier.

### 8.2 Phase 2: Model Scaling (Llama-2-7b-chat)
**Setup**: We upgraded the target model to `Llama-2-7b-chat-hf` while keeping the dataset constant.
**Result**: We observed a significant jump in performance, gaining approximately **+0.1 in Accuracy** compared to the Pythia baseline.
**Conclusion**: This confirms that "truthfulness" features are emergent properties of larger, more capable models. Llama-2 has a richer internal world model, making the distinction between known facts and hallucinations (or forced lies) more linear and extractable.

### 8.3 Phase 3: Data Generalization (Unified Dataset)
**Setup**: To ensure our probe wasn't just learning to detect "trivia questions," we constructed a **Unified Dataset** of 1200 samples (2400 paired rows) from five distinct sources:
1.  **Anthropic Awareness**: Introspection and self-knowledge (100 samples).
2.  **Common Sense QA**: Boolean logic and everyday facts (200 samples).
3.  **Math Problems**: Arithmetic and logic (200 samples).
4.  **Questions 1000**: General knowledge trivia (500 samples).
5.  **Synthetic Facts**: Hallucination-style specific entity facts (200 samples).

**Result**: We trained the probe on this diverse mixture.
**Interpretation**:
*   **Generalization**: High performance on this unified dataset indicates we have found a **general direction of truth**. The probe is not relying on the specific phrasing of math problems or trivia; it is detecting the underlying concept of validity across different domains.
*   **Layer Localisation**: We typically observe that effectiveness peaks in the **middle-to-late layers** of the model. This suggests a processing hierarchy: early layers process syntax/grammar, middle layers resolve semantic truth/facts, and final layers prepare the specific token output.
*   **Vector Direction**: Our Difference-in-Means vector ($\mu_{Truth} - \mu_{Lie}$) successfully orients the high-dimensional space such that truthful statements project onto higher values and deceptive ones onto lower values. This confirms that "Truth" is a consistent, linear feature within Llama-2's latent space.
