# **[Qwen3-TTS-Daggr-UI](https://huggingface.co/spaces/prithivMLmods/Qwen3-TTS-Daggr-UI)**

`Qwen3-TTS-Daggr-UI` is an advanced speech-processing workspace combining next-generation automatic speech recognition (ASR) and neural text-to-speech (TTS) architectures under the Qwen3 framework. The application relies on `daggr`, a directed acyclic graph (DAG) UI execution engine built over Gradio, providing modular, component-driven pipelines for speech synthesis and transcription tasks.

The system features real-time automatic speech-to-text decoding, zero-shot voice cloning with optional acoustic x-vector conditioning, prompt-guided voice designing, and multi-speaker custom text-to-speech. Models dynamically toggle down to bfloat16 optimization targets to ensure optimal speed and memory footprints on modern CUDA hardware.


https://github.com/user-attachments/assets/c7a0ed1e-827c-4ca7-8abc-b24c0bd83cf1


https://github.com/user-attachments/assets/69490c35-a0e5-48c1-9e2c-93add26b0764


https://github.com/user-attachments/assets/90b6ed75-ea5c-4a10-a059-d73431ee3793


https://github.com/user-attachments/assets/8d847e8b-1d0b-4752-912b-470dd3591154

### **Key Features**

* **Prompt-to-Speech (Voice Design):** Generates expressive audio assets based on explicit language criteria combined with natural language style instructions (e.g., emotional tone, pacing, or specific vocal textures).
* **Zero-Shot Voice Cloning:** Clones a target speaker using a short reference audio file. Supports both text-guided alignment transcriptions and pure, textless acoustic x-vector extraction embeddings.
* **Multi-Speaker Custom TTS:** Accesses standard pretrained vocal profiles across a list of distinct built-in speaker targets, featuring explicit style injection metrics and multi-scale weight options (`0.6B` and `1.7B`).
* **Qwen3 ASR & Forced Alignment:** Natively integrates full-context speech transcription pipelines leveraging `Qwen/Qwen3-ASR-1.7B` coupled with a localized `Qwen/Qwen3-ForcedAligner-0.6B` frame processor.
* **Daggr Execution Engine:** Organizes processing logic into independent input/output functional nodes (`FnNode`) to cleanly handle audio preprocessing, format conversion, state transitions, and asynchronous inference loops.

### **Repository Structure**

```text
├── qwen_asr/
│   ├── cli/
│   │   ├── demo.py
│   │   └── serve.py
│   ├── core/
│   │   ├── transformers_backend/
│   │   │   ├── __init__.py
│   │   │   ├── configuration_qwen3_asr.py
│   │   │   ├── modeling_qwen3_asr.py
│   │   │   └── processing_qwen3_asr.py
│   │   └── vllm_backend/
│   │       ├── __init__.py
│   │       └── qwen3_asr.py
│   ├── inference/
│   │   ├── assets/
│   │   │   └── korean_dict_jieba.dict
│   │   ├── qwen3_asr.py
│   │   ├── qwen3_forced_aligner.py
│   │   └── utils.py
│   ├── __init__.py
│   └── __main__.py
├── qwen_tts/
│   ├── cli/
│   │   └── demo.py
│   ├── core/
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── configuration_qwen3_tts.py
│   │   │   ├── modeling_qwen3_tts.py
│   │   │   └── processing_qwen3_tts.py
│   │   ├── tokenizer_12hz/
│   │   │   ├── configuration_qwen3_tts_tokenizer_v2.py
│   │   │   └── modeling_qwen3_tts_tokenizer_v2.py
│   │   ├── tokenizer_25hz/
│   │   │   ├── vq/
│   │   │   │   ├── assets/
│   │   │   │   │   └── mel_filters.npz
│   │   │   │   ├── core_vq.py
│   │   │   │   ├── speech_vq.py
│   │   │   │   └── whisper_encoder.py
│   │   │   ├── configuration_qwen3_tts_tokenizer_v1.py
│   │   │   └── modeling_qwen3_tts_tokenizer_v1.py
│   │   └── __init__.py
│   ├── inference/
│   │   ├── qwen3_tts_model.py
│   │   └── qwen3_tts_tokenizer.py
│   ├── __init__.py
│   └── __main__.py
├── app.py
├── Dockerfile
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

---

### **Installation & Local Setup**

Execution requires Python 3.12+ along with an active CUDA environment to accelerate model parameters. For proper audio file processing, ensure that system-level audio dependencies (such as `ffmpeg` or `sox`) are available on your system path.

#### **Method 1: Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package installer and dependency resolver. It isolates execution contexts instantly and securely.

**1. Install `uv**`

* **Linux / macOS:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows (PowerShell):** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**2. Clone and Synchronize the Workspace**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git
cd Qwen3-TTS-Daggr-UI
uv sync

```

**3. Launch the Web Interface**

```bash
uv run app.py

```

#### **Method 2: Standard Python Virtual Environment (`pip`)**

**1. Prepare Environment and Upgrade Core Installer**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install --upgrade pip>=26.1

```

**2. Install Package Dependencies**

```bash
pip install -r requirements.txt

```

**3. Execute Script**

```bash
python app.py

```

### **Containerized Deployment (Docker)**

A pre-configured production environment using a `python:3.12.8-slim` container base is provided to simplify orchestration. The Dockerfile creates a non-root user account (UID `1000`) and automatically exposes port `7860`.

#### **1. Build the Docker Image**

```bash
docker build -t qwen3-tts-daggr-ui:latest .

```

#### **2. Run the Container**

To utilize local GPU resources within the isolated container context, pass the standard NVIDIA runtime flags:

```bash
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -e HF_TOKEN="your_huggingface_write_token_here" \
  --name qwen3-speech-app \
  qwen3-tts-daggr-ui:latest

```

### **Environment Configuration**

Gated weights on the Hugging Face Hub require structural authentication clearance. To avoid manual logins during execution, export your credentials directly to your local command environment:

```bash
export HF_TOKEN="hx_your_valid_huggingface_access_token"

```

### **License**

Distributed under the Apache License, Version 2.0. For complete terms and restrictions, check out the raw repository layout inside [LICENSE.txt](https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI/blob/main/LICENSE.txt).
