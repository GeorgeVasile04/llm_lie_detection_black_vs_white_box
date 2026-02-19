# Hardware & Setup Guide: Running Llama-2-13b-chat

This guide details the hardware requirements and setup steps necessary to run the Llama-2-13b-chat model for both **Inference** (sending prompts/extracting activations) and **Training** (Fine-tuning).

## 1. Inference (Sending Prompts & Extracting Activations)
This is the primary operation in Experiment 1 (Probe Creation). You are not modifying the model weights, only running data through them to capture the internal state.

### Hardware Requirements
*   **GPU (Graphics Card):** NVIDIA GPU is **essential**.
    *   **VRAM:** Minimum **24 GB** (e.g., RTX 3090, RTX 4090, or A10G).
    *   *Note:* The 13B model in 16-bit precision requires ~26GB of memory. You can fit it on a 24GB card using 8-bit quantization or by using `bfloat16` with tight overhead.
*   **RAM (System Memory):** 32 GB minimum (64 GB recommended to load the model before moving to GPU).
*   **Storage:** ~30 GB of free disk space for the model weights.

### Setup Steps
1.  **Hugging Face Access:**
    *   Create a Hugging Face account.
    *   Visit the [Meta Llama 2 page](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and request access (requires approving the license).
    *   Create a generic Read-only Access Token in your settings.
    *   Run `huggingface-cli login` in your terminal and paste the token.

2.  **Code Configuration:**
    *   Ensure `dotenv` is installed and a `.env` file exists if the code uses one (though `huggingface-cli` usually handles auth).
    *   In `repeng/models/llms.py`, the code automatically attempts to load `meta-llama/Llama-2-13b-chat-hf`.

3.  **Execution:**
    *   When you run `create_activations_dataset`, the script will:
        1.  Download the model shards (~2-3 min on fast internet).
        2.  Load the shards into RAM.
        3.  Move the weights to GPU VRAM.
        4.  Begin processing your text prompts.

---

## 2. Training (Fine-Tuning)
If you plan to **Fine-Tune** the model (modify its weights) later in the project (e.g., to make it more truthful), the requirements increase significantly.

### Hardware Requirements
*   **GPU:**
    *   **Standard Fine-Tuning:** Requires **~80 GB VRAM** (e.g., A100 80GB). It is impossible on consumer cards.
    *   **LoRA / QLoRA (Efficient Fine-Tuning):** Feasible on **24 GB VRAM** (RTX 3090/4090).
        *   This technique freezes the main model and trains only a tiny adapter layer.
*   **RAM:** 64 GB minimum.
*   **Storage:** 50 GB+ (Model + Checkpoints + Datasets).

### Setup for LoRA (Low-Rank Adaptation)
1.  **Libraries:** Ensure `peft` and `bitsandbytes` are installed (usually in `requirements.txt`).
2.  **Configuration:** You typically need to modify the loading script to load the model in 4-bit or 8-bit precision:
    ```python
    # Example logic (concept only)
    model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)
    ```
    *Note: The current codebase in `repeng` generally loads in `float16` or `bfloat16`, which targets Inference, not Training.*

## Summary Table

| Task | Minimum VRAM | Recommended GPU | System RAM |
| :--- | :--- | :--- | :--- |
| **Inference (16-bit)** | 26 GB | A100 (40GB) or 2x RTX 3090 | 32 GB |
| **Inference (8-bit)** | 14 GB | RTX 3090 / 4090 (24GB) | 32 GB |
| **Inference (4-bit)** | 8 GB | RTX 3080 (10GB+) | 16 GB |
| **Full Fine-Tuning** | >80 GB | A100 (80GB) | 128 GB |
| **LoRA Fine-Tuning** | 24 GB | RTX 3090 / 4090 | 64 GB |

*Warning: Running Llama-2-13b on a CPU (no GPU) is technically possible but will take ~1-5 minutes per prompt, making dataset generation impractical.*

---

## 3. Cloud Execution Guide (RunPod/Lambda)
Since running the model locally on a machine with 8GB RAM is impossible (and APIs do not provide activations), the most cost-effective solution is to rent a GPU server for the duration of the data generation (approx. 1-3 hours).

### Step 1: Choose the Right GPU
From the available list, the **RTX 4090 (24 GB VRAM)** or **L4 (24 GB VRAM)** are the best choices.
*   **Why?** They have exactly enough VRAM (24GB) to fit Llama-2-13b in 8-bit or `bfloat16`. They are significantly cheaper than the A100 or H100 but fast enough for inference.
*   **Cost:** ~$0.40 - $0.60 per hour.
*   **Alternative:** The **A40 (48 GB VRAM)** is a safer bet if you want to run in full precision without worrying about memory errors, costing ~$0.40/hr (cheaper than 4090 in some regions!).

**Recommendation:** Select the **A40 (48GB)** if available (great value, plenty of VRAM). Otherwise, the **RTX 4090** or **L4**.

### Step 2: Rent and Connect
1.  **Create Account:** Sign up on [RunPod.io](https://runpod.io) or [Lambda Labs](https://lambdalabs.com).
2.  **Add Funds:** Deposit small credit ($5-$10 is enough).
3.  **Deploy Pod:**
    *   Select your GPU (e.g., A40 or RTX 4090).
    *   **Template:** Select "RunPod Pytorch 2.1" (or similar standard Pytorch image).
    *   **Disk Space:** Increase the container disk to at least **50 GB** (to hold the model weights).
4.  **Connect via SSH:**
    *   Once the pod is "Running", copy the SSH command provided (e.g., `ssh root@123.456.78.9 -p 12345`).
    *   **VS Code (Recommended):** Use the "Remote - SSH" extension in VS Code to connect directly to the server. This lets you edit files as if they were local.

### Step 3: Setup Environment (On the Cloud Server)
Run these commands in the terminal of your cloud instance:

```bash
# 1. Update and install useful tools
apt-get update && apt-get install -y git nano unzip

# 2. Upload your code
# (Option A) Git Clone your repo: git clone https://github.com/your-repo/llm-lie-detection.git
# (Option B) Drag & Drop your 'White_Box_Lie_Detection' folder into the VS Code Sidebar.

# 3. Install Python Dependencies
pip install -r requirements.txt
pip install bitsandbytes accelerate  # Essential for big model loading

# 4. Login to Hugging Face (Required for Llama-2)
pip install huggingface_hub
huggingface-cli login
# Paste your Read-Only token from https://huggingface.co/settings/tokens
```

### Step 4: Run the Activation Extraction
Navigate to your project folder and run the generation script.

```bash
cd White_Box_Lie_Detection
python experiments/comparison.py 
# Or run your specific script: python repeng/datasets/activations/creation.py
```

*   **Time Estimate:** ~1-2 hours depending on the number of datasets.
*   **Monitor:** Watch the output to ensure it doesn't crash from OOM (Out Of Memory). If it does, switch to 8-bit loading in the code.

### Step 5: Download Results & Destroy
1.  **Locate Output:** The script will create `.pickle` files (likely in `output/` or `results/`).
2.  **Download:** Right-click files in VS Code sidebar -> "Download", or use `scp` to copy them to your local laptop.
3.  **Destroy Pod:** Go back to the RunPod/Lambda dashboard and **Terminate/Destroy** the instance to stop being charged.

**Next Steps:**
Once you have the `.pickle` files on your local laptop (8GB RAM), you **can** run the probe training (Logistic Regression) locally, as that step is CPU-friendly and lightweight.

