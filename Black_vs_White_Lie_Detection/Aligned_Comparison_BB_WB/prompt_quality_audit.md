# Prompt Quality Audit — All 14 Datasets

## How prompts are built

Every sample goes through this pipeline:

1. **Dataset loader** (repeng/DLK) fills `row.text` (full template) and `row.format_args` (individual fields).
2. **`load_data.py`** runs `item.update(safe_args)` (all `format_args` except the binary `label`), then sets
   `item['answer'] = format_args['label']` if no answer is present yet, and normalises
   `question_stem → question`, `choice / statement → answer`.
3. **`get_base_prompt` (prompt_utils.py)** builds:
   ```
   [INST] Question: {item['question']} [/INST]
   {item['answer']}
   ```
   If `item['question']` is empty it **falls back to `item['text']`** (the full template stored by step 2).

This fallback is the source of several structural problems shown below.

---

## Dataset-by-dataset analysis

### 1. commonsense_qa ✅ Good

| Field | Value |
|-------|-------|
| `question` | actual CSQA question, e.g. *"What do people use to escape the cold?"* |
| `answer` | one of the 5 choices, e.g. *"warm house"* |

**Resulting prompt:**
```
[INST] Question: What do people use to escape the cold? [/INST]
warm house
```
Self-contained. No external context needed. Format is correct.

---

### 2. race ⚠️ Article dropped — likely hurts performance

`format_args = {article, question, answer}`. After `item.update`, `item['question']` gets the RACE question and `item['answer']` gets the option. The `article` field is stored in the item but **never used by `get_base_prompt`**.

**Resulting prompt:**
```
[INST] Question: What does the author suggest people do after a long trip? [/INST]
Take a warm bath
```

RACE questions are reading-comprehension questions — they are **unanswerable without the article**. The model is forced to hallucinate based on world knowledge alone. This is almost certainly the primary reason RACE has the worst white-box performance of all MC datasets (best L3 AUC 0.628).

**Recommended fix:** include the article in the prompt:
```
[INST] Article: {article}

Question: {question} [/INST]
{answer}
```
Note: RACE articles can be long; apply `max_length` truncation (e.g. 2048 tokens) to keep them within context.

---

### 3. arc_easy ✅ Good

`format_args = {question, answer}`. Self-contained science questions, no external article needed.

**Resulting prompt:**
```
[INST] Question: Which of the following is a planet in our solar system? [/INST]
Mars
```
Correct format.

---

### 4. arc_challenge ✅ Good

Identical structure to arc_easy. Harder questions, but the prompt is correct.

---

### 5. open_book_qa ✅ Good

`format_args = {question_stem, choice}`. `load_data.py` normalises `question_stem → question` and `choice → answer`.

**Resulting prompt:**
```
[INST] Question: When liquid turns into a solid it undergoes [/INST]
freezing
```
OpenBookQA questions are self-contained (they test elementary science facts, not reading comprehension of a specific passage). Format is correct.

---

### 6. got_cities ⚠️ Structurally odd — no impact in practice

`format_args = {}` (empty). No `question` or `answer` fields are populated. `get_base_prompt` falls back to `item['text']` which is the raw statement (e.g. *"The city of Paris is in France."*) and sets `answer = ''`.

**Resulting prompt:**
```
[INST] Question: The city of Paris is in France. [/INST]

```

The statement is treated as a "question" with an empty answer. Conceptually this misrepresents the forced-persona design (there is nothing for the model to "have just said"), but the model still reads and encodes the statement in its activations. **In practice this works perfectly (best L3 AUC > 0.999).** Not worth fixing.

---

### 7. got_sp_en_trans ⚠️ Same structural oddity as got_cities — no impact

Same pattern. Statement e.g. *"The Spanish word for 'cat' is 'gato'."* as "Question:", empty answer. Performance > 0.99. No fix needed.

---

### 8. got_larger_than ⚠️ Same structural oddity as got_cities — no impact

Statement e.g. *"76 is larger than 49."* as "Question:", empty answer. Performance > 0.99. No fix needed.

---

### 9. imdb ⚠️ Review body as "Question" — functional but structurally odd

`format_args = {text, label1, label2, label}`. Because `'text'` is in `format_args`, `item.update(safe_args)` **overwrites** `item['text']` with the raw review body (not the full template). `item['question']` is never set, so `get_base_prompt` falls back to `item['text']` = raw review.

**Resulting prompt:**
```
[INST] Question: I loved this movie. The acting was superb and... [/INST]
Positive
```

The entire review is jammed into the `[INST] Question: [/INST]` wrapper. Structurally odd (a movie review is not a "question"), but the content is complete and the model reads the full review before seeing the sentiment label. **Performance is near-perfect (best L3 AUC 1.000).** No fix needed.

---

### 10. amazon_polarity ⚠️ Answer leaked into "Question" field — answer appears twice

`format_args = {content, label1, label2, label}`. Because the field is named `'content'` (not `'text'`), `item['text']` is **not overwritten** and stays as `row.text` = the full DLK template, which already includes the answer embedded in it. `get_base_prompt` then falls back to this full template as the "question" and appends the answer again.

**Resulting prompt:**
```
[INST] Question: Consider the following example: "..."
Choice 1: Negative
Choice 2: Positive
Between choice 1 and choice 2, the sentiment of this example is Negative [/INST]
Negative
```

The answer *"Negative"* appears **twice**. Conceptually this breaks the forced-persona idea (the model "reads" its own answer in the question before generating it), but both true and false samples exhibit this symmetrically, so the probe can still learn the truth-relevant signal. **Performance is perfect (best L3 AUC 1.000).** Cosmetic issue only; no fix needed.

---

### 11. ag_news ⚠️ News article as "Question" — functional but structurally odd

Same mechanism as IMDB (`'text'` is in `format_args` → `item['text']` overwritten with raw article).

**Resulting prompt:**
```
[INST] Question: {full news article text} [/INST]
World
```

Large body of text as "Question:", category label as "Answer". Structurally odd but the content is complete. **Performance is high (best L3 AUC 0.981).** No fix needed.

---

### 12. dbpedia_14 ⚠️ Answer leaked into "Question" field — same issue as amazon_polarity

`format_args = {content, label1, label2, label}`. Same `'content'` field name → `item['text']` not overwritten → full template (with category label) becomes the "question" and the label is repeated.

**Resulting prompt:**
```
[INST] Question: Consider the following example: "..."
Choice 1: Company
Choice 2: Artist
Between choice 1 and choice 2, the topic of this example is Company [/INST]
Company
```

Answer appears twice. Same reasoning as amazon_polarity — symmetric across true/false samples, probes still learn. **Performance is perfect (best L3 AUC 1.000).** No fix needed.

---

### 13. rte ⚠️ Answer leaked + "Question:" appears twice

`format_args = {premise, hypothesis, label}`. `safe_args = {premise, hypothesis}`. Neither is named `'text'` so `item['text']` stays as `row.text` = full DLK template, which already contains *"yes"* or *"no"* at the end. `item['question']` is never set (neither key is `'question'`), so `get_base_prompt` falls back to the full template.

**Resulting prompt:**
```
[INST] Question: {premise}
Question: Does this imply that "{hypothesis}", yes or no?
yes [/INST]
yes
```

Two problems: (1) `"Question:"` appears twice, (2) the answer *"yes"/"no"* appears twice. Despite the messy format the content is meaningful and complete. **Performance is decent (best L3 AUC 0.900).** Cosmetic issue only.

If you want to clean this up, a simple fix in `load_data.py` for RTE would be to construct a proper question: set `item['question']` to `"{premise}\nQuestion: Does this imply that \"{hypothesis}\", yes or no?"` and leave `item['answer']` as-is.

---

### 14. boolq ⚠️ Passage dropped — likely hurts performance

`format_args = {passage, question, label}`. `safe_args = {passage, question}`. `item.update(safe_args)` sets `item['question']` = actual BoolQ question (**good!**) and `item['passage']` = reading passage (**stored but never used**). `item['answer']` = *"True"/"False"*.

**Resulting prompt:**
```
[INST] Question: do animals have the same blood type as humans [/INST]
False
```

The reading passage is completely dropped. BoolQ is a reading-comprehension task — the answer to the question depends on a specific passage. Without it the model must rely on world knowledge, which is often sufficient for common facts but fails on passage-specific or misleading questions. **Best L3 AUC is 0.754** — moderate, and almost certainly held back by the missing passage.

**Recommended fix:** include the passage:
```
[INST] Passage: {passage}

Question: {question}? [/INST]
True
```

---

## Summary table

| Dataset | Quality | Issue | Perf. impact | Fix recommended? |
|---|---|---|---|---|
| commonsense_qa | ✅ Good | — | None | No |
| race | ⚠️ Context missing | Article dropped | **High** — worst MC AUC | **Yes** |
| arc_easy | ✅ Good | — | None | No |
| arc_challenge | ✅ Good | — | None | No |
| open_book_qa | ✅ Good | — | None | No |
| got_cities | ⚠️ Odd structure | Statement as "Question:", empty answer | None (AUC 0.999) | No |
| got_sp_en_trans | ⚠️ Odd structure | Same | None (AUC 1.000) | No |
| got_larger_than | ⚠️ Odd structure | Same | None (AUC 1.000) | No |
| imdb | ⚠️ Odd structure | Large text as "Question:" | None (AUC 1.000) | No |
| amazon_polarity | ⚠️ Answer leaked | Answer in "Question:" + repeated | None (AUC 1.000) | No |
| ag_news | ⚠️ Odd structure | Large text as "Question:" | None (AUC 0.981) | No |
| dbpedia_14 | ⚠️ Answer leaked | Answer in "Question:" + repeated | None (AUC 1.000) | No |
| rte | ⚠️ Answer leaked | "Question:" twice, answer twice | Minor | Optional |
| boolq | ⚠️ Context missing | Passage dropped | **Moderate** — AUC 0.754 | **Yes** |

---

## Conclusion

Only two datasets have prompt issues that plausibly harm performance:

- **race**: The reading passage (article) is essential for answering the questions but is completely absent. This is the single most likely explanation for RACE being the worst-performing MC dataset.
- **boolq**: The reading passage is also absent. BoolQ results (0.754) are noticeably below what a full-context prompt would likely achieve.

All other issues are structural/cosmetic: the content is present and the signal is learnable, as confirmed by the high AUC scores. They do not require fixing unless you want cleaner, more principled prompt design.
