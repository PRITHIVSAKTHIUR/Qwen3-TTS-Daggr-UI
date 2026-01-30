# **Qwen3-TTS-Daggr-UI**

> A Gradio-based demonstration for the Qwen/Qwen3-TTS-12Hz and Qwen3-ASR-1.7B models using Daggr for modular UI nodes. Supports voice design (prompt-to-speech), voice cloning (zero-shot), custom voice synthesis with multiple speakers and languages, and automatic speech recognition (ASR) with transcription. Features lazy model loading to optimize memory, multi-model sizes (0.6B and 1.7B for TTS), and support for various audio inputs.

>[!Note]
HuggingFace Demo: https://huggingface.co/spaces/prithivMLmods/Qwen3-TTS-Daggr-UI

https://github.com/user-attachments/assets/c7a0ed1e-827c-4ca7-8abc-b24c0bd83cf1


https://github.com/user-attachments/assets/69490c35-a0e5-48c1-9e2c-93add26b0764


https://github.com/user-attachments/assets/90b6ed75-ea5c-4a10-a059-d73431ee3793


https://github.com/user-attachments/assets/8d847e8b-1d0b-4752-912b-470dd3591154

## Features
- **Modular UI with Daggr**: Four independent nodes for Voice Design, Voice Clone, Custom Voice, and Qwen3 ASR, allowing parallel or sequential workflows.
- **Lazy Model Loading**: Models (Base, CustomVoice, VoiceDesign for TTS; ASR) load on-demand based on task and size, minimizing VRAM usage.
- **Advanced TTS Tasks**:
  - Voice Design: Generate speech from text and voice description prompts.
  - Voice Clone: Zero-shot cloning from reference audio (with optional transcript).
  - Custom Voice: Synthesis using predefined speakers with optional style instructions.
- **ASR Transcription**: Automatic speech recognition with language detection and support for 28+ languages, including timestamps via forced aligner.
- **Multi-Language Support**:
  - TTS: Auto-detection or manual selection for Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian.
  - ASR: Auto-detection or manual selection for Chinese, Cantonese, English, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.
- **Model Sizes**: 0.6B for faster inference, 1.7B for higher quality (TTS); 1.7B fixed for ASR.
- **Audio Handling**: Supports file paths, base64 data URIs, numpy arrays, microphone/upload sources, and normalization to [-1, 1] float32.
- **Efficient Inference**: Uses bfloat16 on CUDA, with garbage collection and cache clearing for low-memory setups.
- **Speakers**: 9 built-in options including Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian.
- **Queueing**: Up to default Gradio queue limits for concurrent jobs.

## Prerequisites
- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16 and faster inference).
- Stable internet for initial model downloads from Hugging Face.
- Hugging Face token (HF_TOKEN) set as an environment variable for gated models.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git
   cd Qwen3-TTS-Daggr-UI
   ```
2. Install dependencies:
   First, install pre-requirements:
   ```
   pip install -r pre-requirements.txt
   ```
   Then, install main requirements:
   ```
   pip install -r requirements.txt
   ```
   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```
   **requirements.txt content:**
   ```
   daggr==0.5.2
   torch==2.8.0
   transformers==4.57.3
   accelerate==1.12.0
   einops
   gradio #gradio@6
   librosa
   torchaudio
   soundfile
   sox
   onnxruntime
   spaces
   torch
   numpy
   ```
3. Set Hugging Face token (if required):
   ```
   export HF_TOKEN=your_hf_token_here
   ```
4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

### Docker Installation
1. Build the Docker image:
   ```
   docker build -t qwen3-tts-daggr-ui .
   ```
2. Run the container:
   ```
   docker run -p 7860:7860 --env HF_TOKEN=your_hf_token_here qwen3-tts-daggr-ui
   ```
   Access at `http://localhost:7860`.

## Usage
1. **Select a Node**: Use the Daggr UI to interact with Voice Design, Voice Clone, Custom Voice, or Qwen3 ASR sections.
2. **Input Parameters**:
   - For Voice Design: Provide text, language, and voice description.
   - For Voice Clone: Upload reference audio, optional transcript, target text, language, and model size.
   - For Custom Voice: Provide text, language, speaker, optional style instruction, and model size.
   - For Qwen3 ASR: Upload or record audio, select language (or Auto).
3. **Generate Output**: Click the submit button in the respective node to produce audio or transcription.
4. **Output**: For TTS: Audio waveform (numpy) and status message. For ASR: Detected language, transcription text, and status.

### Supported Speakers
| Speaker    | Description |
|------------|-------------|
| Aiden     | General-purpose |
| Dylan     | General-purpose |
| Eric      | General-purpose |
| Ono_anna  | Japanese-style |
| Ryan      | General-purpose |
| Serena    | General-purpose |
| Sohee     | Korean-style |
| Uncle_fu  | General-purpose |
| Vivian    | General-purpose |

### Supported Languages
- **TTS**: Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian.
- **ASR**: Auto, Chinese, Cantonese, English, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian.

## Examples
| Task          | Input Example | Parameters |
|---------------|---------------|------------|
| Voice Design | "It's in the top drawer... wait, it's empty? No way!" | Language: Auto, Description: "Speak in an incredulous tone, but with a hint of panic." |
| Custom Voice | "Hello! Welcome to the Text-to-Speech demo." | Language: English, Speaker: Ryan, Instruction: "Neutral" , Model: 1.7B |
| Voice Clone  | Target: "Cloned voice speaking this." | Ref Audio: Upload WAV/MP3, Ref Text: "Reference speech.", Language: Auto, x-vector only: False, Model: 1.7B |
| Qwen3 ASR    | Upload audio file | Language: Auto |

## Troubleshooting
- **Model Loading**: First use downloads models; monitor console for progress.
- **OOM Errors**: Switch to smaller model (0.6B for TTS), reduce max_new_tokens, or clear cache manually.
- **Audio Input Issues**: Ensure uploads are valid WAV/MP3; check console for processing errors.
- **No Output**: Provide required inputs (e.g., text, audio); ensure HF_TOKEN is set for gated repos.
- **CUDA Fallback**: If no GPU, runs on CPU with float32 (slower).
- **ASR Errors**: Ensure audio is clear; try manual language selection if auto-detection fails.

## Contributing
Contributions welcome! Add new nodes to the Daggr graph, support more models/speakers/languages, or improve audio processing. Submit pull requests via the repository.

Repository: [https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git](https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git)

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
Built by Prithiv Sakthi. Report issues via the repository.
