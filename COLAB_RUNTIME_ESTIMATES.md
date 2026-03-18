# Colab Pro Runtime Estimates: White Box + Black Box Pipeline

## Hardware Comparison

| Metric | T4 (Free Colab) | A100 (Colab Pro) | H100 (Colab Pro - Rare) |
|--------|---|---|---|
| VRAM | 16 GB | 40 GB | 80 GB |
| Throughput (7B LLM) | ~400 tokens/sec | ~3,200 tokens/sec | ~5,000+ tokens/sec |
| Speedup vs T4 | 1x | ~8x | ~12-15x |
| Inference Speed | 0.5-1 sec/batch | 0.06-0.125 sec/batch | 0.04-0.08 sec/batch |

---

# SCENARIO 1: 8.2K Dataset (6 test datasets)

## Breakdown by Step

### Step 1: Data Loading & Preprocessing
- **Load 6 datasets**: ~1-2 minutes
  - Network I/O: ~30-50 MB total
  - Pandas dataframe creation: negligible
- **A100 Time**: **~2 minutes**
- **H100 Time**: **~2 minutes** (I/O bound, not GPU bound)

---

### Step 2: Model Loading (Llama-2-7B)

**Requirements:**
- Model size: ~13 GB (float16) or ~26 GB (float32)
- With bfloat16: ~13 GB
- LoRA buffers / cache: ~2-3 GB

| GPU | VRAM Available | Load Time | Quantization |
|-----|---|---|---|
| A100 40GB | 40 GB | **1-2 min** | bfloat16 (full precision) |
| H100 80GB | 80 GB | **1-1.5 min** | bfloat16 (full precision) |
| T4 16GB | 16 GB | **3-4 min** | Must use 8-bit quant |

**A100 & H100 Estimate: ~1.5 minutes**

---

### Step 3: White Box - Activation Extraction

**Computation:**
- 8,267 samples × 2 types (truth + lie) = ~16,534 total samples
- Batch size: 16
- **Number of batches**: 16,534 ÷ 16 ≈ **1,033 batches**
- Layers: 10, 15, 20 (extracted simultaneously in one forward pass)

**Per-Batch Timing:**
```
Forward pass breakdown:
- Token generation (avg ~40 tokens per prompt): 
  * T4: 40 tokens × 2.5 ms/token ≈ 100 ms per sample
  * A100: 40 tokens × 0.3 ms/token ≈ 12 ms per sample
  * H100: 40 tokens × 0.2 ms/token ≈ 8 ms per sample

- Activation extraction: ~5 ms (all layers, CPU-GPU copy)
- Total per batch (16 samples):
  * T4: (100 ms × 16) + 5 ms = 1,605 ms = 1.6 sec
  * A100: (12 ms × 16) + 5 ms = 197 ms = 0.2 sec
  * H100: (8 ms × 16) + 5 ms = 133 ms = 0.13 sec
```

**Total WB Time:**
```
T4:  1,033 batches × 1.6 sec = 1,653 sec ≈ 27.5 minutes
A100: 1,033 batches × 0.2 sec = 206.6 sec ≈ 3.5 minutes
H100: 1,033 batches × 0.13 sec = 134.3 sec ≈ 2.2 minutes
```

**With Optimizations (batch_size can be increased on A100/H100):**
```
Batch size 32 on A100 (enough VRAM):
1,033 → 516 batches
- Forward time scales minimally (memory bound, not compute bound)
- Total: ~2 minutes instead of 3.5

Batch size 64 on H100:
1,033 → 258 batches
- Total: ~1 minute
```

**A100 Estimate (batch_size=32): ~2 minutes**
**H100 Estimate (batch_size=64): ~1 minute**

---

### Step 4: Black Box - Logprobs Computation

**Computation:**
- 8,267 samples × 2 types (truth + lie) = 16,534 samples
- 48 elicitation probes
- Batch size: 16
- **Batches needed**: 16,534 ÷ 16 ≈ 1,033 batches
- For each batch × 48 probes

**Key difference from WB**: Each probe requires separate prompt completion

```
Per probe:
- Forward pass + logprob extraction: 0.2 sec (A100) per batch × 48 = 9.6 sec overhead
- Batch: 1,033 batches × 9.6 sec = 9,916 sec ≈ ~2.75 hours

BUT: Probes can be batched! (ask model for logprobs for multiple completions)
With probe batching (group 4 probes per request):
- Effective probes: 48 ÷ 4 = 12 "super-probes"
- BB Time: 1,033 × 0.2 × 12 = ~2,480 sec ≈ 41 minutes (A100)
```

**Realistic scenario with current `compute_bb_features_for_dataset()` function:**
```
If probes are computed sequentially (worst case):
T4:  1,033 × 1.6 × 48 = ~78,000 sec = ~22 hours ❌ TOO LONG
A100: 1,033 × 0.2 × 48 = ~9,914 sec ≈ 2.75 hours ⚠️ Still long
H100: 1,033 × 0.13 × 48 = ~6,470 sec ≈ 1.8 hours ⚠️ Still long

If probes are batched smartly (better case):
A100: 1,033 × 0.2 × 12 = ~2,478 sec ≈ 41 minutes ✅
H100: 1,033 × 0.13 × 12 = ~1,615 sec ≈ 27 minutes ✅
```

**Current implementation check needed**: Does `compute_bb_features_for_dataset()` process probes in parallel?
- If **sequential probes**: **~2.75 hours (A100)** ⚠️
- If **batched probes**: **~40 minutes (A100)** ✅

**Conservative A100 Estimate: ~1.5 hours**
**H100 Estimate (batched): ~1 hour**

---

### Step 5: Classifier Training & Evaluation

**Computations needed:**
```
White Box:
- 6 datasets × 2 methods (LR, DiM) × 3 layers = 36 classifiers
- Per classifier: 80/20 split, fit LR/DiM (~100ms), predict + evaluate (~50ms)
- Total: 36 × 0.15 sec = ~5 seconds

Black Box:
- 6 datasets × 1 method (LR) = 6 classifiers
- Per classifier: fit + evaluate (~0.5 sec with scaling)
- Total: 6 × 0.5 sec = ~3 seconds

Total: ~10 seconds (negligible)
```

**Both A100 & H100: <1 minute** (mostly disk I/O for printing results)

---

### Step 6: Results Aggregation & Saving

- Concatenate 12 dataframes (2 per dataset): ~1 second
- Save to CSV: ~2 seconds
- Plot generation (if added): ~10 seconds

**Both A100 & H100: <1 minute**

---

## TOTAL TIME FOR 8.2K DATASET

### Summary Table

| Step | T4 | A100 | H100 |
|------|----|----|---|
| **1. Data Loading** | 2 min | 2 min | 2 min |
| **2. Model Loading** | 3 min | 1.5 min | 1.5 min |
| **3. WB Extraction** | 27.5 min | **2-3 min** | **1-2 min** |
| **4. BB Logprobs** | 22 hours ❌ | **1-2.75 hrs** ⚠️ | **0.7-1.8 hrs** ⚠️ |
| **5. Classifier Training** | <1 min | <1 min | <1 min |
| **6. Results & Save** | <1 min | <1 min | <1 min |
| **TOTAL** | **~22.5 hours** | **~1.5-3 hours** | **~1-2 hours** |

### Realistic Scenario (if BB probe batching is optimized):

| Step | A100 | H100 |
|------|------|------|
| **1. Data Loading** | 2 min | 2 min |
| **2. Model Loading** | 1.5 min | 1.5 min |
| **3. WB Extraction** | 2-3 min | 1-2 min |
| **4. BB Logprobs (batched)** | 40 min | 25 min |
| **5. Classifier Training** | <1 min | <1 min |
| **6. Results & Save** | <1 min | <1 min |
| **TOTAL** | **~47 minutes** | **~32 minutes** |

---

---

# SCENARIO 2: 50K Dataset (Full Dataset)

Scale factor: **50,000 ÷ 8,267 ≈ 6x**

### Step Breakdown

#### Step 1: Data Loading
- Now loading full dataset (might need to download)
- **A100/H100: ~5 minutes** (network I/O dependent)

#### Step 2: Model Loading
- Same model, already in cache
- **A100/H100: ~1.5 minutes**

#### Step 3: White Box - Activation Extraction

```
50K samples × 2 = 100K total
Batches: 100K ÷ 16 = 6,250 batches
Batch time: 0.2 sec (A100 with batch_size=32)

Total: 6,250 × 0.2 sec = 1,250 sec ≈ 21 minutes

With batch_size=64 on H100:
6,250 ÷ 2 = 3,125 batches
Total: 3,125 × 0.15 sec ≈ 8 minutes
```

**A100 Estimate: ~20 minutes**
**H100 Estimate: ~8 minutes**

---

#### Step 4: Black Box - Logprobs Computation

**With sequential probes (worst):**
```
6,250 batches × 0.2 sec × 48 probes = 60,000 sec ≈ 16.7 hours ❌
```

**With batched probes (optimized):**
```
A100: 6,250 × 0.2 × 12 = 15,000 sec ≈ 4.2 hours ⚠️
H100: 6,250 × 0.15 × 12 = 11,250 sec ≈ 3.1 hours ⚠️
```

**If we can batch more aggressively (group all 48 probes):**
```
A100: 6,250 × 0.5 = 3,125 sec ≈ 52 minutes ✅
H100: 6,250 × 0.3 = 1,875 sec ≈ 31 minutes ✅
```

**Conservative A100 Estimate: ~3-4 hours** (depends on probe batching)
**H100 Estimate: ~2-3 hours** (depends on probe batching)

---

#### Step 5: Classifier Training
- Now 6 datasets with ~8K samples each
- Still <1 minute (sklearn scales well)

**A100/H100: <1 minute**

---

#### Step 6: Results Aggregation
**A100/H100: <1 minute**

---

## TOTAL TIME FOR 50K DATASET

| Step | A100 | H100 |
|------|------|------|
| **1. Data Loading** | 5 min | 5 min |
| **2. Model Loading** | 1.5 min | 1.5 min |
| **3. WB Extraction** | 20 min | 8 min |
| **4. BB Logprobs** | 3-4 hours | 2-3 hours |
| **5. Classifier Training** | <1 min | <1 min |
| **6. Results & Save** | <1 min | <1 min |
| **TOTAL** | **3.5-4.5 hours** | **2.5-3.5 hours** |

### Optimistic Scenario (if BB fully batched):

| Step | A100 | H100 |
|------|------|------|
| **1-3. Loading & WB** | 27 min | 16 min |
| **4. BB Logprobs (all batched)** | 52 min | 31 min |
| **5-6. Training & Results** | 2 min | 2 min |
| **TOTAL** | **~1.5 hours** | **~50 minutes** |

---

# COST COMPARISON

## Colab Pro Pricing (as of 2024-2026)
- **Colab Pro**: $10/month
  - A100: Priority access, ~80% of the time
  - Limited sessions: 24 continuous hours
- **Colab Pro+**: $60/month (not always available)
  - A100 guaranteed
  - H100 access
  - 2,000 compute units/month

## Storage
- **Google Drive**: 100 GB included with Colab Pro
- **Local files**: ~20 GB for datasets + model cache = sufficient

---

# RECOMMENDATIONS

## For 8.2K Dataset Testing
| Goal | Hardware | Time | Cost |
|------|---|---|---|
| Quick validation | A100 | **~45 min** | ~$0.38 (pay-as-you-go) |
| Fast iteration | H100 (rare) | **~30 min** | ~$0.50 (pay-as-you-go) |

✅ **Recommendation**: Use A100, should complete easily within one Colab session (24h limit is not an issue)

---

## For 50K Dataset
| Goal | Hardware | Time | Cost | Notes |
|------|---|---|---|---|
| Quick run (BB sequential) | A100 | **3.5-4.5 hours** | ~$3-4 | Might be slow, but doable |
| Optimized run (BB batched) | A100 | **1.5 hours** | ~$1.50 | Requires BB probe batching fix |
| Ideal (H100, batched) | H100 | **50 min** | ~$1-2 | If H100 available, much better |

⚠️ **Key Bottleneck**: Black Box logprobs computation (40-240 minutes depending on probe batching)

✅ **Recommendation**: 
1. **First**: Test BB probe batching on 8.2K dataset
2. **If batching works**: Run 50K with A100 (~1.5 hours, ~$1.50)
3. **Alternative**: Run without full 50K (use 20K subset = ~45 min)

---

# CRITICAL OPTIMIZATION NEEDED

## Current Weakness: Sequential Probe Processing

Looking at `compute_bb_features_for_dataset()`, if probes are computed **one at a time**:
```
50K × 48 probes = 2,400,000 individual forward passes
With A100: 2.4M × 0.3ms = 12 hours ❌
```

### Optimization Strategy

Instead of:
```python
for probe in probes:  # Loop over 48 probes
    score = compute_logprob(model, prompt, probe)
```

Do:
```python
# Batch 4-8 probes per forward pass
batched_probes = chunk_probes(probes, batch_size=8)
for probe_batch in batched_probes:
    scores = compute_logprobs_batch(model, prompt, probe_batch)
    # Get all 8 scores at once
```

**This could reduce BB time from 4 hours → 30 minutes on A100!**

---

# SUMMARY TABLE: REALISTIC ESTIMATES

## 8.2K Dataset
```
┌─────────────────────────────────────┐
│  A100: 45 minutes (~$0.40)          │
│  H100: 30 minutes (~$0.50)          │
│  Bottleneck: BB logprobs (if unoptimized: 1.5 hours)
└─────────────────────────────────────┘
```

## 50K Dataset
```
Unoptimized BB:
┌─────────────────────────────────────┐
│  A100: 3.5-4.5 hours (~$3-4)        │
│  H100: 2.5-3.5 hours (~$3-4)        │
│  Bottleneck: BB logprobs (3-4 hours)
└─────────────────────────────────────┘

Optimized BB (batched probes):
┌─────────────────────────────────────┐
│  A100: 1.5 hours (~$1.50)           │
│  H100: 50 minutes (~$1-2)           │
│  Bottleneck: None (all balanced)
└─────────────────────────────────────┘
```

---

# NEXT STEPS

1. ✅ Check if `compute_bb_features_for_dataset()` already batches probes
2. ⚠️ If not, optimize probe batching (can reduce time by 6-8x!)
3. 📊 Profile on 8.2K dataset to validate estimates
4. 🚀 Then scale to 50K with confidence

