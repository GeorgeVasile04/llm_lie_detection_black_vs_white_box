# Experiment 3: Advanced Generalisation - Intention to Lie

This document details the "Intention to Lie" experiments found in `classification_notebooks/advanced_generalisation_studies/combined_experiments_on_lie_intention.ipynb`.

## 1. Objective
The core question of this experiment is: **Does the lie detector track the model's *actual intention* to lie, or is it merely overfitting to the presence of "lie instructions" in the prompt?**

To answer this, we separate the "instruction to lie" from the "act of lying" using two advanced scenarios:
1.  **Modified Intentions**: We give a lie instruction but then "cancel" it with a modifier (e.g., "Ignore previous instruction, tell the truth"), or vice versa.
2.  **Refusals**: We give a lie instruction, but the model refuses to comply (e.g., due to safety training), resulting in a truthful answer despite the prompt.

If the detector works correctly, it should predict "Truth" in these cases, even though the word "Lie" appears in the context.

## 2. Methodology

### Notebook
- **Path**: `classification_notebooks/advanced_generalisation_studies/combined_experiments_on_lie_intention.ipynb`
- **Key Data Sources**: 
    - `results/modified_intention/averages_dfs_dict.pkl`
    - `results/refusals/averages_dfs_dict.pkl`

### Experimental Setup
The experiment correlates two distinct probabilities to validate the detector. It is crucial to understand the difference between them:

#### 1. Detector Prediction ($P_{detect}$) - "The Brain Scan"
*   **Definition**: The probability that the model is lying, as estimated by our probes from the model's internal activations *during* the generation of the answer.
*   **What it measures**: The internal state of the model. "Does the model's neural activity look like it is lying?"
*   **Analogy**: This is like a polygraph test running in real-time while the suspect speaks.

#### 2. Behavioral Lying Rate ($P_{behavior}$) - "The Consistency Check"
*   **Definition**: The empirical probability that the model will *continue* to lie if asked a follow-up question (e.g., "Are you sure?"). calculated over repeated trials.
*   **What it measures**: The external commitment to the lie. "Does the model stick to its story (double down) or admit the truth?"
*   **Analogy**: This is like checking if the suspect keeps lying when pressed for details. We treat this as the "Ground Truth" of whether the model really intended to lie.

#### Why compare them?
If $P_{detect}$ (internal state) correlates with $P_{behavior}$ (external action), it proves the detector is measuring the valid *intent* to lie, not just random noise or the presence of specific words.

### The "Speaker" Variable
The analysis distinguishes between different sources of intent modification:
- **Instruction**: The system prompt changes the intent (e.g., "Use the following rules...").
- **Question**: The user prompt changes the intent (e.g., "Question: ... (Answer truthfully)").
- **Answer (Refusal)**: The model itself changes the intent by refusing to lie.

## 3. Results & Observations

### Visual Analysis (Scatter Plots)
The notebook generates scatter plots where:
- **X-axis**: Follow-up lying rate (Actual Behavior).
- **Y-axis**: Lie detector prediction (Detector Signal).

**Key Observations:**
1.  **Strong Linear Correlation**: There is a strong positive correlation between the detector's signal and the actual lying behavior. Points cluster along the diagonal.
2.  **Refusals are Detected as Truth**: In cases where the model refuses to lie (despite the instruction), the data points fall in the low-lying-rate / low-detector-score region. This proves the detector is not simply looking for the token "Lie" in the prompt history.
3.  **Modifiers Successfully Switch State**: When an instruction is negated (e.g., "Lie... actually don't"), the detector accurately tracks the *final* net intent, moving the score appropriately.

### Statistical Validation
The notebook calculates **Pearson** and **Spearman** correlation coefficients for the combined datasets. High correlation values (typically > 0.8 or 0.9 in these setups) confirm that our probes generalize beyond simple prompt-pattern matching to detecting the underlying generative state of "lying".

## 4. Conclusion
This specific advanced generalization study provides strong evidence for the **validity** of the probes. It demonstrates that the detector is measuring a property of the model's computation (the intent/process of lying) rather than a property of the input text (the presence of "lie" keywords). This is crucial for claiming that we have built a true "Lie Detector" rather than a "Lie Instruction Detector".
