# How well do truth probes generalise?
**by mishajw**
*24th Feb 2024*

Representation engineering (RepEng) has emerged as a promising research avenue for model interpretability and control. Recent papers have proposed methods for discovering truth in models with unlabeled data, guiding generation by modifying representations, and building LLM lie detectors. RepEng asks the question: If we treat representations as the central unit, how much power do we have over a model’s behaviour?

Most techniques use linear probes to monitor and control representations. An important question is whether the probes generalise. If we train a probe on the truths and lies about the locations of cities, will it generalise to truths and lies about Amazon review sentiment? This report focuses on truth due to its relevance to safety, and to help narrow the work.

Generalisation is important. Humans typically have one generalised notion of “truth”, and it would be enormously convenient if language models also had just one[1]. This would result in extremely robust model insights: every time the model “lies”, this is reflected in its “truth vector”, so we could detect intentional lies perfectly, and perhaps even steer away from them.

We find that truth probes generalise surprisingly well, with the 36% of methodologies recovering >80% of the accuracy on out-of-distribution datasets compared with training directly on the datasets. The best probe recovers 92% accuracy.

Thanks to Hoagy Cunningham for feedback and advice. Thanks to LISA for hosting me while I did a lot of this work. Code is available at mishajw/repeng, along with steps for reproducing datasets and plots.

## Methods
We run all experiments on Llama-2-13b-chat, for parity with the source papers. Each probe is trained on 400 questions, and evaluated on 2000 different questions, although numbers may be lower for smaller datasets.

### What makes a probe?
A probe is created using a training dataset, a probe algorithm, and a layer.

We pass the training dataset through the model, extracting activations[2] just after a given layer. We then run some statistics over the activations, where the exact technique can vary significantly - this is the probe algorithm - and this creates a linear probe. Probe algorithms and datasets are listed below.

A probe allows us to take the activations, and produce a scalar value where larger values represent “truth” and smaller values represent “lies”. The probe is always linear. It’s defined by a vector ($v$), and we use it by calculating the dot-product against the activations ($a$): $v^T a$. In most cases, we can avoid picking a threshold to distinguish between truth and lies (see appendix for details).

We always take the activations from the last token position in the prompt. For the majority of the datasets, the factuality of the text is only revealed at the last token, for example if saying true/false or A/B/C/D.

For this report, we’ve replicated the probing algorithm and datasets from three papers:

1. Discovering Latent Knowledge in Language Models Without Supervision (DLK).
2. Representation Engineering: A Top-Down Approach to AI Transparency (RepE).
3. The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets (GoT).

We also borrow a lot of terminology from Eliciting Latent Knowledge from Quirky Language Models (QLM), which offers another great comparison between probe algorithms.

### Probe algorithms
The DLK, RepE, GoT, and QLM papers describe eight probe algorithms. For each algorithm, we can ask whether it's supervised and whether it uses grouped data.

Supervised algorithms use the true/false labels to discover probes. This should allow better performance when truth isn’t salient in the activations. However, using supervised data encourages the probes to predict what humans would label as correct rather than what the model believes is correct.

Grouped algorithms utilise “groups” of statements to build the probes. For example, all possible answers to a question (true/false, A/B/C/D) constitute a group. Using this information should allow the probe to remove noise from the representations.


*Sketch of what the data and training process looks like when we have grouped/ungrouped and labelled/unlabelled data. Note that for CCS the “group normalisation” stage consists of taking two contrasting statements and returning the difference between them.*

| | Outline | Supervised[3]? | Grouped? |
| :--- | :--- | :--- | :--- |
| **Linear Artificial Tomography (LAT)**<br>from RepE. | Takes the first principle component of the differences between random pairs of activations. Details. | No. | No. |
| **Contrast-Consistent Search (CCS)**<br>from DLK. | Given contrastive statements (e.g. “is the sky blue? yes” and “is the sky blue? no”), build a linear probe that satisfies:<br><br><ul><li>Consistency: $p(x^+) = 1 - p(x^-)$</li><li>Confidence: $min(p(x^+), p(x^-)) = 0$</li></ul><br>Details. | No. | Yes. |
| **Difference-in-means (DIM)**<br>from GoT (as MMP). | Take the difference in means between true and false statements. Details. | Yes. | No.[4] |
| **Linear discriminant analysis (LDA)**<br>from GoT (as MMP-IID). | Take the difference in means between true and false statements (like DIM), and skew the decision boundary by the inverse of the covariance matrix of the activations. Details.<br><br>Intuitively, this takes the truth direction and then accounts for interference with other features. | Yes. | No. |
| **Principal component analysis (PCA).** | Take the top principal component of the activations. Details. | No. | No. |
| **Grouped principal component analysis (PCA-G)[5]**<br>from DLK (as CRC-TPC) | For each question, calculate the mean activation and subtract it from every answer’s activations.<br><br>Then take the top principal component of the question-normalised activations. Details. | No. | Yes. |
| **Logistic regression (LR).** | Perform a logistic regression, taking in activations and predicting whether they are true. Details. | Yes. | No. |
| **Grouped logistic regression (LR-G).** | As LR, but using activations that have the group means subtracted. Details. | Yes. | Yes. |

### Datasets
The key contribution of the papers above isn’t just the probe algorithms, it’s also the datasets they use to build the probes. Across all papers, we find 18 datasets (eight from DLK, five from RepE, and five from GoT).

The DLK datasets are straight-forward binary questions, used to craft contrasting pairs.

The RepE datasets differ in two important ways. One, they allow more than two answers per question, as the RepE paper doesn’t use contrasting pairs. Two, they add the suffix “The probability of the answer being plausible is” which prompts the model to evaluate the truthfulness of the statement.

The GoT dataset is the most principled of this collection. They take “uncontroversial, unambiguous, and simple enough” statements, attempting to hone in on the obviously true. These statements really are simple, such as “{city} is in {country}” or “{x} is less than {y}”.

| Datasets | Choices? | Style? | Simple? | Approx. prompt template | Datasets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DLK** | Binary[6]. | Q&A. | No. | Consider the following example: `{content}`<br><br>Choice 1: `{choice1}`<br><br>Choice 2: `{choice2}`<br><br>Between choice 1 and choice 2, the sentiment of this passage is `{choice 1 / choice 2}` | imdb, amazon_polarity, ag_news, dbpedia_14, rte, copa, boolq, piqa |
| **RepE** | Multiple (2-5). | Q&A. | No. | Consider the correctness of the following:<br><br>Question: `{question}`<br><br>Answer: `{answer}`<br><br>The probability of the answer being plausible is | openbook_qa, common_sense_qa, race, arc_challenge, arc_easy |
| **GoT** | Binary/none[7]. | Statements. | Yes. | The city of `{city}` is in `{country}`. | cities, sp_en_trans, larger_than, cities_cities_conj, cities_cities_disj |

## Measuring generalisation
We measure generalisation by seeing how well probes trained on one dataset generalise to other out-of-distribution datasets. For example, we train a probe on whether a news article is about business or sports (ag_news) and see how well it performs when detecting if numbers are bigger or smaller than each other (larger_than). We use 18 diverse datasets.

### Recovered accuracy
We measure how well a probe recovers the accuracy on an out-of-distribution dataset, compared with training a probe directly on that dataset.

For each evaluation dataset, we create an accuracy threshold for that dataset. When evaluating whether a probe generalises to the evaluation dataset, we compare its accuracy to this threshold.

To create a threshold for a dataset we:

1. Train a suite of probes (one for every probing algorithm and layer) on the train subset.
2. Take the best performing probe according to accuracy on the validation subset.
3. Take its accuracy on the test subset as the threshold.

To evaluate a probe against a threshold we:

1. Train the probe on the train subset of its dataset (typically different from the evaluation dataset).
2. Evaluate the accuracy of the probe on the evaluation dataset’s test subset.
3. Taking the percent of the threshold accuracy achieved by the probe (accuracy/threshold).

We clip recovered accuracy at 100%.

### Finding the best generalising probe
The obvious way to compare two probes is to compare their average recovered accuracy across all datasets. However, if one probe has been trained on a dataset that is difficult to generalise to but nevertheless still offers good performance, then it has an unfair advantage.

To remedy this, do a pairwise comparison between each probe which looks at only the datasets that neither of them were trained on. We take the best probe as the probe with the highest number of “wins” across all pairwise comparisons.

Note that, as above, we perform the pairwise comparisons on the validation set. All results below are reported from the test set.

## Results
We train ~1.5K probes (8 algorithms * 18 datasets * 10 layers[8]), and evaluate on each dataset, totalling ~25K evaluations (~1.5K probes * 18 datasets).

While probes on early layers (<=9) perform poorly, we find that for mid-to-late layers (>=13) 36% of all probes recover >80% accuracy. This is evidence in favour of a single generalised truth notion in Llama-2-13b-chat. However, the fact that we don’t see 100% recovered accuracy in all cases suggests that either (1) there is interference in how truth is represented in these models, and the interference doesn’t generalise across datasets, or (2) truth is in fact represented in different but highly correlated ways.


*ECDF plots of recovered accuracy, broken down by layer. You can read this as: for a given recovered accuracy, what percent of probes trained achieved it?*

The best generalising probe is a DIM probe trained with dbpedia_14 on layer 21. It recovers, on average, 92.8% of accuracy on all datasets.

### Examining the best probe
Let’s dig a bit deeper into our best probe. We vary each of the three hyperparameters (algorithm, train dataset, and layer) individually. This shows that the probe isn’t too sensitive to hyperparameters, as other choices perform nearly as well.


*How well training on dbpedia_14 and layer 21 generalises to evaluation datasets when varying the probe algorithm. Recovered accuracy metric is shown.*

*How well training with DIM and layer 21 generalises to evaluation datasets when varying training dataset. Recovered accuracy metric is shown.*

*How well training with DIM and dbpedia_14 generalises to evaluation datasets when varying the layer.  Recovered accuracy metric is shown.*

### Examining algorithm performance
Let’s break down by algorithm, and take the average recovered accuracy across all probes created using that algorithm. 

**Takeaways:**

* There is little variation in probe performance.
* Supervised methods appear to outperform unsupervised methods.
* There isn’t a standout pattern in grouped methods: the PCA-G significantly outperforms the PCA version, but there’s less of a difference between LR and LR-G.
* LDA is an outlier, performing significantly worse than other probes. See appendix for further investigation.

### Examining dataset performance
Similarly to above, we break down by what dataset the probe is trained on. 

**Takeaways:**

* The DLK datasets generally perform better than RepE and GoT datasets.
* Surprisingly, the got_cities_cities_conj dataset outperforms all other datasets, and has a huge margin over the other GoT datasets.
* My initial assumption was that the threshold for this dataset was set very low, so all probes managed to meet it. This was not the case, the threshold is set to 99.3%.

We can also plot a “generalisation matrix” showing how well each dataset generalises to each other dataset.


*How well each training dataset generalises to different evaluation datasets. Metric is recovered accuracy averaged over layer and probe algorithm. We only look at layer>=13.*

There seems to be some structure here. We explore further by clustering the datasets, and find that they do form DLK, RepE, and GoT clusters:

*Agglomerative clustering of the datasets, where we use (1 - recovered accuracy) as the distance measure between datasets, and use average distances to merge clusters. We only look at layer>=13.*

Another interesting thing to look at is a comparison between how well a dataset generalises to other datasets, and how well other datasets generalise to that dataset (generalises_to).


*Comparison of how well a dataset generalises to other datasets (generalizes_from) and how well other datasets generalise to that dataset (generalises_to). Mean recovered accuracy shown.*

 **Takeaways:**

* DLK datasets are better than RepE and GoT on both generalising from and to.
* RepE datasets generalise to other datasets better than GoT, but GoT datasets are easier to generalise to than RepE.
* Perhaps the DLK/GoT probes have overfit to the token where truth is established, and the RepE’s prompt suffix breaks this assumption.
* got_cities_cities_conj is once again an outlier, far exceeding the rest of the GoT datasets on both measures.

## How do we know we’re detecting truth, and not just likely statements?
Discovering truth probes is particularly gnarly, as we need to distinguish between what the model believes is true and what the model believes is likely to be said (i.e. likely to be said by humans).

This distinction is hard to make. One good attempt at it is to measure probe performance on TruthfulQA, which is a dataset designed to contain likely-sounding but untrue statements as answers (e.g. common misconceptions).

We measure TruthfulQA performance on all probes, and see how well this correlates with our generalisation scores.


*TruthfulQA accuracy against generalisation score of all probes trained. Dotted green line is the TruthfulQA score for Llama-2 13B chat. Dotted grey line is the same, but prompting for a calibrated score. Results taken from the RepE paper, table 8.*

We find that 94.6% of probes with >80% recovered accuracy measure something more than just statement likelihood. This is evidence for the probes learning more than just how likely some text is. However, it’s worth noting that a lot of the probes fall short of simply prompting the model to be calibrated (see prompt formats in RepE paper, D.3.2).

## Conclusion & future work
We find some probing methods with impressive generalisation, that appear to be measuring something more than truth. This is evidenced for a generalised notion of truth in Llama-2-13B-chat.

The results above ask more questions than they answer:

* Why do probes not generalise perfectly?
* Do they overfit to the interference from other features?
* Are some datasets biased in some way, skewing truth probes?
* Is truth represented in highly-correlated but distinct ways in different datasets?
* What explains the variance in performance between probes?
* Does training on multiple datasets improve performance?
* Do smaller models also have a generalised notion of truth?
* Why does got_cities_cities_conj generalise well?
* Why does LDA generalise poorly?

There’s a lot of work to be done exploring generalisation in RepEng (and even more work to be done in RepEng generally!). Please reach out if you want to discuss further explorations, or have any interest in related experiments in RepEng.

## Appendix

### Validating implementations
We validate the implementations in a few ways.

Do we get 80% accuracy on arc_easy? This should be easily achievable, as it is one of the easiest datasets we investigate.


Do we get ~100% accuracy on GoT datasets? The GoT paper reports >97% accuracy for all of its datasets when trained on that dataset.


Can we reproduce accuracies from the RepE paper? The RepE paper trains on a very limited number of samples, so we’d expect to exceed performance here.


### Validating LDA implementation
Above, the LDA probe performs poorly. This hints at a bug. However, our implementation simply calls out to scikit. We also find that the probe performs better than LDA in-distribution, but worse out-of-distribution:


*Accuracy of DIM and LDA trained on boolq (type=train_boolq) and when trained directly on the dataset (type=train_eval). LDA outperforms DIM in-distribution, DIM outperforms LDA out-of-distribution.*

### Thresholding
To evaluate probe performance, we run the probe over activations from correct and incorrect statements, and check that the probe has higher “truth values” for correct statements. This way, we can avoid having to set thresholds for the truth values that distinguish between truths and lies, as we’re looking at relative values.

Some datasets don’t have groups of statements, so we can’t look at relative values. In this case, we take the best threshold using a ROC curve and report accuracy from there. This is done for got_cities_cities_conj and got_cities_cities_disj.
