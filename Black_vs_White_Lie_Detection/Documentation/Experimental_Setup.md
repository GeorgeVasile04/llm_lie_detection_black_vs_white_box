# Experimental Setup, Methodology & Results

This document consolidates the experimental setup, the shared pipeline, and the results/interpretation for the four research experiments of this Master's Thesis on **Black-Box vs. White-Box Lie Detection in Large Language Models**. Experiments 1 and 2 are fully implemented and analysed; Experiments 3 and 4 are described at the conceptual level.

---

## 1. Computational Environment

All experiments are executed on **Google Colab** with a high-VRAM GPU (an **NVIDIA A100 40 GB** in practice; fall-back T4/L4 instances are insufficient for the 7 B-parameter backbone when extracting every hidden layer). This is mandatory because two operations dominate cost:

1. **Full-model forward passes** with `output_hidden_states=True`, producing a `(num_layers+1, seq_len, hidden_dim)` tensor that is then indexed at the last token of every sample.
2. **Black-box probe evaluation**, which multiplies the cost of each sample by the number of elicitation questions (up to 64 probes per sample in `probes.csv`).

The model is loaded in **bfloat16** (matching the original *Representation Engineering* paper for numerical stability). Activations are immediately cast to **float16 on CPU** after each forward pass to halve memory and avoid accumulating fp32 tensors on the GPU. A custom `robust_get_activations` loop in [load_data_efficiency.py](../Data_Efficiency_BB_WB/load_data_efficiency.py) implements **adaptive batch-sizing with OOM recovery** (halving the batch size on failure, falling back to left-truncated sequences of 2048 → 1024 → 768 tokens at batch size 1 before skipping a sample as last resort). This was critical to make the large-scale Data Efficiency sweep (up to 50 000 training samples per dataset) tractable within Colab session limits.

## 2. Backbone Model

All experiments use the open-weights **`meta-llama/Llama-2-7b-chat-hf`** (33 hidden layers: 1 embedding + 32 transformer blocks, hidden dimension 4096). This model is selected because:

- It is widely studied in the lie-detection literature (original RepEng / Marks & Tegmark work), giving directly comparable baselines.
- Its chat-tuned variant natively understands the `[INST]` / `[/INST]` instruction-tag formatting used to force the persona of the answering entity (see §4).
- 7 B parameters are the upper bound that still allows full-layer activation extraction inside a single Colab A100 session.

Tokeniser padding is mapped to the EOS token and kept right-aligned so that the "last non-padding token" of a batch can be located with `attention_mask.sum(dim=1) - 1`. This is the token whose hidden state is extracted as the white-box representation for that sample.

## 3. Datasets

A diverse, deliberately heterogeneous suite of 18 datasets is used, covering factual knowledge, reasoning, natural language inference and classification. All are loaded through the RepEng dataset adapters (`White_Box_Lie_Detection/repeng/datasets/elk/*`) to guarantee the original splits and "truth / lie" pairings.

| Category | Datasets |
| :--- | :--- |
| Multiple-choice QA / reasoning | `commonsense_qa`, `race`, `arc_easy`, `arc_challenge`, `open_book_qa` |
| NLI / logic / yes-no QA | `boolq`, `copa`, `rte` |
| Topic / sentiment classification | `imdb`, `amazon_polarity`, `ag_news`, `dbpedia_14` |
| Factual & spatial (Geometry of Truth) | `got_cities`, `got_sp_en_trans`, `got_larger_than`, `got_cities_cities_conj`, `got_cities_cities_disj` |

Detailed per-dataset sample counts and True/False distributions are tabulated in [Aligned_Comparison_BB_WB/doc/README_Comparison_Motivation.md](../Aligned_Comparison_BB_WB/doc/README_Comparison_Motivation.md).

### 3.1 Label mapping and balanced pairings

For every source question the RepEng loader yields one row per candidate answer, labelled `TRUE` (1) when the candidate is the gold answer and `FALSE` (0) otherwise. Rows from the same original question share a `group_id`, which is the cornerstone of every balanced sampling scheme used downstream:

* **Setup A – Balanced (used as default in Experiments 1 and 2)**: for every `group_id`, sample exactly 1 True and 1 False row ⇒ strict 1 : 1 ratio, regardless of the native dataset structure.
* **Setup B – Natural imbalance (Experiment 2 only)**: for every `group_id`, keep the 1 True row and *all* its False siblings ⇒ ratios inherit the dataset's native skew (1 : 3 for ARC/RACE/OBQA, 1 : 4 for CommonsenseQA, 1 : 13 for DBPedia 14).

## 4. Aligning White-Box and Black-Box via the `[INST]` Forced-Persona Trick

This is the single most important methodological choice of the thesis. It is also the non-trivial step that makes the comparison between white-box and black-box probing **scientifically fair**.

### The problem

The two paradigms are, by default, *misaligned*:

- **White-box** detectors read the model's hidden states after it has *processed* a statement. The model never actually generates an answer – it only reads text.
- **Black-box** detectors rely on the model's *behaviour* after it has actively *generated* a true or false statement. This introduces a generation confound: did the model really lie, or did it refuse? Did it hallucinate an accidentally-true "lie"?

Comparing a detector that sees processing to one that sees generation conflates **signal quality** with **generation noise**.

### The solution: forced persona via chat tags

Both tracks are fed the **exact same context**, constructed using Llama-style instruction tags. The full implementation lives in [prompt_utils.py](../Aligned_Comparison_BB_WB/prompt_utils.py):

```
[INST] Question: {question} [/INST]
{answer}
```

By wrapping the question in `[INST] … [/INST]` and placing the candidate answer *after* the closing tag, the model is tricked into believing that **it has just finished generating `{answer}` itself**. The candidate answer is either the gold answer (TRUE) or a distractor (FALSE). In both cases the model's internal "cognitive state" at that instant is equivalent to having just *actively asserted* the statement – which is precisely the state we want both detectors to observe.

The two tracks then diverge minimally:

- **White box** (`get_white_box_context`): the context stops at the end of `{answer}` and activations are extracted from the last token.
- **Black box** (`get_black_box_context`): the exact same prefix is followed by an **elicitation probe** as a new user turn: `\n[INST] {probe_text} [/INST]`, and the log-probabilities of "Yes" vs. "No" at the next token position are recorded.

Both detectors therefore analyse **the same model at the same moment**, having just "asserted" the same statement. Any resulting performance gap can be attributed strictly to signal location (activations vs. behaviour), not to generation artefacts.

### Why this matters beyond convenience

Beyond fairness, the `[INST]` alignment avoids two classical failure modes of black-box lie elicitation:

1. **Refusal to lie** – a chat-aligned model asked "please lie about X" will often refuse or moralise, destroying the signal.
2. **Accidental truth** – a freely-generated "lie" may be factually true (hallucination), silently corrupting the label.

Pre-filling the answer sidesteps both: the model never has to *choose* to lie – it is placed in a state where it has already "said" the forced statement, and the detector observes the consequences.

### Black-box elicitation probes

The 64 elicitation questions used by the black-box pipeline are stored in [probes.csv](../Aligned_Comparison_BB_WB/probes.csv) and grouped into 9 semantic categories: `Confession`, `Psychological`, `Confusion`, `Ethics`, `Ambiguous`, `Logic`, `Easily knowable`, `Unknowable`, `Verbose`, `New_unknowable`. Each probe is evaluated **independently** (one separate forward pass per probe) so that probes do not condition on each other's outputs. The black-box feature vector for one sample is the concatenation of the per-probe scores `log P(Yes) − log P(No)`, which is then fed to a standardised logistic regression classifier ([bb_classifier.py](../Aligned_Comparison_BB_WB/bb_classifier.py)).

---

## 5. The Shared 6-Step Pipeline

Every experiment in this thesis instantiates the same high-level pipeline. The implementation is modular: each step is a Python module and every experiment's Jupyter notebook only orchestrates the calls.

### Step 1 — Load the data
`load_data.py` / `load_data_efficiency.py` / `load_data_imbalance.py` wrap the RepEng loaders and enforce the chosen balanced-pairing scheme (Setup A by default). Output: a pandas DataFrame with columns `id`, `question`, `answer`, `label`, `dataset`, `split`.

### Step 2 — Load the model
[wb_activations.py::load_model](../Aligned_Comparison_BB_WB/wb_activations.py) loads `meta-llama/Llama-2-7b-chat-hf` in bfloat16, moves it to CUDA, and configures the tokenizer (right padding, PAD = EOS).

### Step 3 — Extract the white-box activation vectors
For every row the forced-persona prompt is built, the model is run with `output_hidden_states=True`, and the hidden vector at the *last non-padding token* is retained for every layer 0…32. Per-sample activations are stored as a dict `{layer_idx: np.ndarray(hidden_dim,)}`. Batched extraction with dynamic OOM recovery is provided by `robust_get_activations`.

### Step 4 — Extract the black-box elicitation vectors
For every row, each of the 64 probes in `probes.csv` is appended as a separate `[INST] probe [/INST]` turn and a forward pass measures `P(Yes)` and `P(No)` at the next-token position. This yields a 64-dimensional feature vector per sample. Probes are chunked (`probe_batch_size=24`) to bound GPU memory on long sequences such as IMDB reviews.

### Step 5 — Train and evaluate the classifiers
**White-box probes** ([wb_probes.py](../Aligned_Comparison_BB_WB/wb_probes.py)) — nine algorithms operating on the per-layer activation matrix:

| Code | Name | Supervised? | Notes |
| :--- | :--- | :--- | :--- |
| `lr` | Logistic Regression | Yes | baseline linear probe |
| `lr-g` | Group-centred Logistic Regression | Yes | subtracts the mean of each question-group before fitting – removes topic confounds |
| `dim` | Difference-in-Means | Weak | probe = µ(TRUE) − µ(FALSE) |
| `lda` | Linear Discriminant Analysis | Yes | covariance-aware linear probe |
| `pca` / `pca-g` | PCA (first component) / group-centred PCA | Unsupervised | finds principal direction of variance |
| `lat` | Linear Artificial Tomography | Unsupervised | PCA on random activation differences |
| `ccs` | Contrast-Consistent Search | Unsupervised | Burns et al.'s consistency objective |

For unsupervised probes (`dim`, `pca`, `pca-g`, `lat`, `ccs`) the **sign of the probe is resolved on training data only** (picking the polarity that maximises train AUC), then locked on the test set. This fixes a subtle data-leakage bug present in earlier versions: auto-flipping the sign based on test AUC inflates unsupervised scores.

**Layer selection** uses 4-fold Stratified CV on the training set (see `select_best_layer_cv` in [wb_probes_imbalance.py](../Class_Balance_Impact_BB_WB/wb_probes_imbalance.py)) to pick the single best layer per (dataset × algorithm × training size).

**Black-box classifier** ([bb_classifier.py](../Aligned_Comparison_BB_WB/bb_classifier.py)) — a `StandardScaler` + `LogisticRegression` pipeline on the 64-dim probe feature vector. The scaler is fit on the training split only to avoid leakage.

**Metrics.** For every configuration we report:
- `Accuracy` (default 0.5 threshold on scaled scores)
- `AUC` (ROC – ranking ability, threshold-free)
- `MAP` (Mean Average Precision – precision-weighted ranking)
- `BRP_90` (Balanced Recall-Precision at the 90 %-confidence operating point)
- `Macro-F1`, `Recall(True)`, `Recall(False)` for the class-imbalance study

### Step 6 — Plot and summarise the results
Each notebook produces:
- A long-format CSV (one row per dataset × method × algorithm × training size × layer) under `outputs/` or `out/`.
- A compact side-by-side comparison CSV.
- Line plots (data-efficiency curves) or heat-maps (planned cross-dataset generalisation matrix).
- Statistical tests where appropriate (paired Wilcoxon signed-rank for Experiment 2).

---

## 6. The Four Core Experiments

### Experiment 1 — Data Efficiency (Black-Box vs. White-Box)
*Status: Implemented — see [Data_Efficiency_Experiment.ipynb](../Data_Efficiency_BB_WB/Data_Efficiency_Experiment.ipynb)*

**Objective.** How many labelled examples does each paradigm need to learn to detect lies? Labelled lie/hallucination data is expensive in real-world deployments, so the *sample-efficiency slope* is arguably more decision-relevant than the peak in-distribution AUC.

**Design.**
- 8 datasets: `commonsense_qa`, `race`, `open_book_qa`, `boolq`, `imdb`, `amazon_polarity`, `ag_news`, `dbpedia_14`.
- Logarithmic training sizes: **100, 200, 500, 1 000, 2 000, 5 000, 10 000, 20 000, 50 000** (capped by each dataset's native size).
- Fixed test set of ≥ 1 000 held-out balanced pairs per dataset.
- All probes and BB classifiers use exactly the same training pool at every step.

**Hypothesis.** White-box probes benefit from the dense, pre-computed representation of "truthfulness" inside middle layers and should therefore converge with far fewer samples than black-box classifiers, whose features must first be *inferred* from coarse Yes/No log-ratios across 64 probes.

### Experiment 2 — Robustness to Class Imbalance
*Status: Implemented — see [Class_Imbalance_Experiment.ipynb](../Class_Balance_Impact_BB_WB/Class_Imbalance_Experiment.ipynb)*

**Objective.** Real-world LLM outputs contain far more truths than lies. Does training detectors on naturally imbalanced data (as one would in production) break them, and if so, which paradigm collapses first?

**Design.**
- 7 datasets with native imbalance: `commonsense_qa`, `arc_easy`, `arc_challenge`, `open_book_qa`, `race`, `ag_news`, `dbpedia_14`.
- Fixed training budget of ~ 1 000 samples per setup; only the *ratio* changes.
- **Setup A (balanced):** 1 True + 1 False per question (50 / 50).
- **Setup B (imbalanced):** 1 True + *all* Falses per question (natural skew, down to 7 % truth rate on DBPedia 14).
- Test set always balanced 50 / 50 so that "predict the majority class" cannot cheat.
- 9 WB algorithms × 7 datasets × 2 setups = **63 paired configurations** for statistical testing.
- Significance: **paired Wilcoxon signed-rank test** on the 63 (A − B) differences, for each of AUC, MAP, BRP_90.

### Experiment 3 — Ceiling Performance by Access Level
*Status: Conceptualised – WB/BB-logprobs already implemented in [main_comparison.ipynb](../Aligned_Comparison_BB_WB/main_comparison.ipynb); the Level-1 discrete-text track is the remaining work.*

**Objective.** Establish the absolute upper-bound AUC at three increasingly permissive tiers of model access.

- **Level 1 – Strict Black-Box (text only):** parse whether the discrete output token is "Yes" or "No" (binary 0/1 features).
- **Level 2 – Soft Black-Box (logprobs):** continuous `log P(Yes) − log P(No)` per probe – already implemented.
- **Level 3 – White-Box:** full hidden-state access – already implemented.

**Why it matters.** Real API providers offer exactly these three tiers (OpenAI returns logprobs optionally, most models return only text, open-weight models give everything). Quantifying the drop from Level 3 → Level 2 → Level 1 maps directly onto deployment reality.

### Experiment 4 — Cross-Dataset (Out-of-Distribution) Generalisation
*Status: Conceptualised.*

**Objective.** Detectors trained in-distribution often merely memorise topic features. Experiment 4 takes the best classifier per (dataset × paradigm) from Experiment 3 and evaluates it on **every other** dataset without any retraining.

**Output format.** An N × N matrix of *Recovered AUC*: cell (i, j) = AUC of a probe trained on dataset *i* evaluated on dataset *j*, normalised by its in-distribution AUC: `AUC_ood(i,j) / AUC_id(i,i)`. This is the thesis's climax – it isolates whether either paradigm has learned a universal "truth direction" or merely a per-dataset decision boundary.

---

## 7. Results

### 7.1 Experiment 1 — Data Efficiency

The large-scale scan stored in [data_efficiency_LRG_large.csv](../Data_Efficiency_BB_WB/outputs/data_efficiency_LRG_large.csv) compares the strongest white-box probe (**LR-G**, group-centred logistic regression) against the black-box logistic-regression classifier. The composite figure [large_scale_comparison_LRG_vs_BB.png](../Data_Efficiency_BB_WB/outputs/large_scale_comparison_LRG_vs_BB.png) plots AUC, MAP and BRP_90 as a function of training size.

**Key numbers (AUC at 100 / 1 000 / 50 000 training samples, WB LR-G vs. BB):**

| Dataset | WB @ 100 | WB @ 1 000 | WB @ max | BB @ 100 | BB @ 1 000 | BB @ 5 000 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `commonsense_qa` | 0.798 | 0.934 | **0.960 (20 k)** | 0.674 | 0.743 | 0.752 |
| `race` | 0.635 | 0.934 | **0.955 (50 k)** | 0.688 | 0.734 | 0.750 |
| `open_book_qa` | 0.882 | 0.909 | **0.940 (10 k)** | 0.668 | 0.727 | 0.746 |
| `boolq` | 0.771 | 0.873 | **0.928 (20 k)** | 0.647 | 0.723 | 0.741 |
| `amazon_polarity` | 0.975 | 0.978 | **0.985 (50 k)** | 0.740 | 0.788 | 0.802 |
| `ag_news` | 0.897 | 0.980 | **0.990 (50 k)** | 0.860 | 0.931 | 0.938 |
| `dbpedia_14` | **0.999 (100)** | 0.968 | 1.000 (50 k) | 0.966 | 0.987 | 0.989 |
| `imdb` | 0.608 (plateau) | 0.609 | **0.985 (10 k)** | 0.659 | 0.648 | 0.644 |

**Observations.**

1. **White-box reaches near-ceiling performance with only 100–500 examples on most datasets.** The gap to its own 50 k-sample peak is typically ≤ 4 AUC points. Black-box, by contrast, is still climbing at 5 000 samples on reasoning datasets and never catches up on `commonsense_qa`, `open_book_qa`, `race`, `boolq`.
2. **On "easy" classification tasks (`dbpedia_14`, `ag_news`, `amazon_polarity`) black-box is competitive at scale** – the semantic separation is so strong that 64 yes/no probes already encode most of it. On `dbpedia_14` the BB classifier reaches 0.99 AUC with just 500 samples.
3. **IMDB is a striking outlier.** WB plateaus around **0.608 AUC** from 100 up to 5 000 samples, then jumps to **0.985 at 10 000**. Inspection of the "best layer" column shows that at small N the CV-selected layer is Layer 1 (embedding-adjacent); only at 10 k does layer 17 become selectable reliably. This is a statistical artefact of layer selection with noisy small samples, not a failure of the paradigm.
4. **Picked layer is consistently in the 13–22 range** on reasoning datasets, confirming that "truth processing" lives in the middle of Llama-2-7B, not at the output.

### 7.2 Experiment 2 — Class Imbalance

Detailed metrics per dataset × algorithm live in [class_imbalance_metrics_detailed_V2.csv](../Class_Balance_Impact_BB_WB/out/class_imbalance_metrics_detailed_V2.csv). The paired comparison is summarised in [wilcoxon_global_summary_V2.csv](../Class_Balance_Impact_BB_WB/out/wilcoxon_global_summary_V2.csv):

| Metric | Pairs | Mean (A − B) | W-statistic | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| AUC | 63 | +0.0072 | 184.0 | 2.0 × 10⁻⁵ | A > B — imbalance degrades |
| MAP | 63 | +0.0080 | 192.0 | 2.9 × 10⁻⁵ | A > B — imbalance degrades |
| BRP_90 | 63 | +0.0195 | 344.5 | 0.020 | A > B — imbalance degrades |

**Key per-algorithm observations** (from [wilcoxon_pairs_AUC_V2.csv](../Class_Balance_Impact_BB_WB/out/wilcoxon_pairs_AUC_V2.csv)):

- **Worst-hit algorithm: CCS** (unsupervised consistency) – e.g. AUC drops from 0.761 → 0.690 on `open_book_qa`, and its BRP_90 collapses from 0.202 → 0.020 on the same dataset. CCS relies on balanced pairs for its consistency loss, so skew breaks it structurally.
- **Most robust supervised algorithm: LR-G** – mean AUC drop ≈ 0.009, and on several datasets (`commonsense_qa`, `arc_easy`) the imbalanced version is within 0.001 AUC of the balanced one.
- **Black-box is on par with the median white-box algorithm on AUC** but degrades more visibly on MAP (precision of high-confidence predictions).

Alongside the ranking metrics, the detailed CSV records **Macro-F1** and **class-wise recall**: under Setup B the decision-boundary-based classifiers collapse into majority-class prediction, so `Recall(True)` drops dramatically (e.g. from 0.96 to 0.33 on reasoning datasets) while `Recall(False)` stays near 1.0. This is the *asymmetric sensitivity* documented in detail in [Experiment_RQ3_Class_Imbalance_Analysis.md](Experiment_RQ3_Class_Imbalance_Analysis.md).

---

## 8. Interpretation

### 8.1 Experiment 1 — Why is white-box so much more sample-efficient?

The explanation lies in *where information is located* vs. *where it must be decoded from*.

**Inside the model,** the concept of truth/falsity is already partially separated at middle layers: a sizeable body of work (Azaria & Mitchell 2023, Marks & Tegmark 2023, RepEng) has shown that a single linear direction in a 4096-dim activation space captures much of the variance between true and false statements. A logistic regression on 4096 pre-organised features therefore needs O(100) examples to identify this direction – well below the statistical-learning threshold where overfitting usually dominates. The group-centred variant (LR-G) removes the per-question topic bias, which is why it consistently beats plain LR on reasoning datasets.

**Outside the model,** the black-box classifier must reconstruct this signal from 64 scalar log-odds. Each probe has its own noise, its own failure modes (the model may refuse, answer off-topic, or copy the question phrasing) and its own per-dataset systematic bias. Combining them requires learning 64 coefficients under substantial noise – which is why BB's learning curve rises much more slowly and plateaus lower on reasoning tasks.

This directly supports the thesis's working hypothesis: **white-box methods exploit a denser, pre-computed representation and are therefore vastly more sample-efficient**, at the cost of requiring full weight access.

The **IMDB anomaly** at small N is worth highlighting: when CV picks Layer 1 rather than Layer 17 the probe sees essentially un-processed embeddings, and performance stalls at 0.6 AUC until the training size is large enough for the CV procedure to reliably identify the correct middle layer. This underlines that *sample efficiency of the probe is coupled to the sample efficiency of the layer-selection procedure* – a methodological observation worth reporting.

### 8.2 Experiment 2 — What does imbalance actually break?

The Wilcoxon test says the degradation is **statistically significant but numerically small** on ranking metrics (≤ 2 AUC points on average). This tells us the *signal* of truth vs. lie is still present in the activations and in the probe responses when we skew the training data. What collapses is **the decision threshold**, not the underlying direction.

Concretely:

- AUC measures *ranking*. Because the signal direction is preserved, AUC barely moves.
- Macro-F1 and class-wise recall measure the *quality of the 0/1 decision* at a fixed threshold (typically 0.5). Since imbalance shifts class priors, a threshold that was well-calibrated on 50 / 50 data is no longer at the correct location for 1 / 13 data, and the classifier starts predicting "lie" for everything.
- BRP_90 (precision at the top-10 % most confident predictions) shows the largest mean drop (0.0195), confirming that the *calibration* of the high-confidence region is what suffers most.

**Practical takeaway for the thesis.** Setup A (balanced) is the right default for the remaining experiments: it preserves both ranking and threshold calibration, at the cost of only a slightly smaller training pool. This decision is now justified statistically (p < 10⁻⁴ on AUC and MAP) rather than by convention. It also explains why **LR-G is the algorithm of choice for Experiments 3–4**: it is both the best performer at all training sizes *and* the most robust to imbalance.

### 8.3 A unifying reading of the two experiments

Taken together, Experiments 1 and 2 paint a consistent picture of the **structural differences** between paradigms:

- The white-box paradigm trades **access requirements** for **sample efficiency and robustness**. Its performance is limited almost entirely by the quality of the underlying model representation – which, in Llama-2-7B's middle layers, is already very good.
- The black-box paradigm trades **signal richness** for **deployability**. It remains competitive on trivially separable tasks (topic/sentiment classification) but lags on reasoning tasks that require subtle semantic discrimination, and it needs substantially more labelled examples to approach its own plateau.

These structural observations set up Experiment 3 (ceiling at each access tier) and Experiment 4 (OOD generalisation) as the two dimensions along which the gap can either narrow or widen – specifically, Experiment 4 will test the common conjecture that black-box probes, being built on behavioural cues rather than geometric features, may generalise *better* across domains despite their lower in-distribution ceiling.

---

## 9. Code Map (where to look for what)

| Component | Location |
| :--- | :--- |
| Forced-persona prompt templates | [prompt_utils.py](../Aligned_Comparison_BB_WB/prompt_utils.py) |
| Model loader + WB activation extraction | [wb_activations.py](../Aligned_Comparison_BB_WB/wb_activations.py) |
| Robust OOM-resilient extraction loop | [load_data_efficiency.py](../Data_Efficiency_BB_WB/load_data_efficiency.py) |
| BB elicitation probe bank (64 probes) | [probes.csv](../Aligned_Comparison_BB_WB/probes.csv) |
| BB logprob feature extraction | [bb_logprobs.py](../Aligned_Comparison_BB_WB/bb_logprobs.py) |
| BB logistic-regression classifier | [bb_classifier.py](../Aligned_Comparison_BB_WB/bb_classifier.py) |
| 9 WB probe algorithms | [wb_probes.py](../Aligned_Comparison_BB_WB/wb_probes.py) |
| Imbalance-specific variants + CV layer selection | [wb_probes_imbalance.py](../Class_Balance_Impact_BB_WB/wb_probes_imbalance.py) |
| Wilcoxon analysis helper | [summarize_wilcoxon.py](../Class_Balance_Impact_BB_WB/summarize_wilcoxon.py) |
| Experiment 1 notebook | [Data_Efficiency_Experiment.ipynb](../Data_Efficiency_BB_WB/Data_Efficiency_Experiment.ipynb) |
| Experiment 2 notebook | [Class_Imbalance_Experiment.ipynb](../Class_Balance_Impact_BB_WB/Class_Imbalance_Experiment.ipynb) |
| Experiment 3 in-progress notebook | [main_comparison.ipynb](../Aligned_Comparison_BB_WB/main_comparison.ipynb) |
| Companion motivation doc | [README_Comparison_Motivation.md](../Aligned_Comparison_BB_WB/doc/README_Comparison_Motivation.md) |
| Initial aligned-comparison results | [Aligned_Comparison_Result_1.md](../Aligned_Comparison_BB_WB/doc/Aligned_Comparison_Result_1.md) |
| Full imbalance write-up | [Experiment_RQ3_Class_Imbalance_Analysis.md](Experiment_RQ3_Class_Imbalance_Analysis.md) |
| Original thesis plan | [Thesis_Experiment_Plan.md](Thesis_Experiment_Plan.md) |
