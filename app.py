import os
import gc
import base64
import io
import numpy as np
import torch
import torchaudio
import soundfile as sf
import gradio as gr
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import snapshot_download, login

from daggr import FnNode, Graph

os.environ["OMP_NUM_THREADS"] = "1"

HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    login(token=HF_TOKEN)

loaded_models = {}

MODEL_SIZES = ["0.6B", "1.7B"]
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]

TTS_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

ASR_SUPPORTED_LANGUAGES = [
    "Chinese", "Cantonese", "English", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
]

def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])

def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping

ASR_LANG_DISPLAY, ASR_LANG_MAP = _build_choices_and_map(ASR_SUPPORTED_LANGUAGES)
ASR_LANG_CHOICES = ["Auto"] + ASR_LANG_DISPLAY

def get_model_path(model_type: str, model_size: str) -> str:
    """Download/Get model path based on type and size."""
    if model_type == "ASR":
        return "Qwen/Qwen3-ASR-1.7B"
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

def get_model(model_type: str, model_size: str):
    """
    Lazy load models. Unloads previous models if VRAM is tight.
    """
    global loaded_models
    key = (model_type, model_size)
    
    if key not in loaded_models:
        print(f"--- Clearing Cache before loading {model_type} ---")
        loaded_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"--- Loading Model: {model_type} {model_size} ---")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if model_type == "ASR":
            from qwen_asr import Qwen3ASRModel
            # Load ASR with Forced Aligner for timestamps
            loaded_models[key] = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B",
                dtype=dtype,
                device_map=device,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
                max_inference_batch_size=4, 
                attn_implementation="sdpa",
            )
        else:
            # Load TTS
            from qwen_tts import Qwen3TTSModel
            model_path = get_model_path(model_type, model_size)
            loaded_models[key] = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=dtype,
                token=HF_TOKEN,
            )
        
    return loaded_models[key]

def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        y = x.astype(np.float32)

    if clip:
        y = np.clip(y, -1.0, 1.0)
        
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
        
    return y

def process_audio_input(audio_input):
    """
    Handles Filepaths, Data URIs (base64), and Numpy arrays.
    Returns (numpy_float32, sample_rate_int)
    """
    if audio_input is None:
        return None

    try:
        if isinstance(audio_input, str):
            if audio_input.startswith("data:"):
                try:
                    header, encoded = audio_input.split(",", 1)
                    data = base64.b64decode(encoded)
                    wav, sr = sf.read(io.BytesIO(data))
                    return _normalize_audio(wav), int(sr)
                except Exception as e:
                    print(f"Failed to decode base64 audio: {e}")
                    return None
            
            if os.path.exists(audio_input):
                wav_tensor, sr = torchaudio.load(audio_input)
                wav = wav_tensor.mean(dim=0).numpy()
                return _normalize_audio(wav), int(sr)
            else:
                print(f"Error: Input string is not a file or valid data URI: {audio_input[:50]}...")
                return None

        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            a0, a1 = audio_input
            if isinstance(a0, int):
                return _normalize_audio(a1), int(a0)
            else:
                return _normalize_audio(a0), int(a1)

        if isinstance(audio_input, dict):
            if "name" in audio_input:
                return process_audio_input(audio_input["name"])
            if "path" in audio_input:
                return process_audio_input(audio_input["path"])
            if "sampling_rate" in audio_input and "data" in audio_input:
                return _normalize_audio(audio_input["data"]), int(audio_input["sampling_rate"])

        return None

    except Exception as e:
        print(f"Audio Processing Error: {e}")
        return None

def run_voice_design(text, language, voice_description):
    """Voice Design (Prompt-to-Speech)"""
    gc.collect()
    torch.cuda.empty_cache()

    if not text: return None, "Text required"
    if not voice_description: return None, "Description required"

    try:
        tts = get_model("VoiceDesign", "1.7B") 
        
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    """Voice Cloning (Zero-Shot)"""
    gc.collect()
    torch.cuda.empty_cache()

    if not target_text: return None, "Target text required"
    
    audio_tuple = process_audio_input(ref_audio)
    
    if audio_tuple is None:
        return None, "Error: Could not process reference audio. Please upload a valid WAV/MP3."

    if not use_xvector_only and not ref_text:
        return None, "Error: Reference text required (or check 'Use x-vector only')"

    try:
        tts = get_model("Base", model_size)
        
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_custom_voice(text, language, speaker, instruct, model_size):
    """Standard TTS"""
    gc.collect()
    torch.cuda.empty_cache()

    if not text: return None, "Text required"

    try:
        tts = get_model("CustomVoice", model_size)
        
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_asr(audio_upload, lang_disp):
    """Automatic Speech Recognition"""
    gc.collect()
    torch.cuda.empty_cache()

    if audio_upload is None:
        return "", "", "No Audio"

    processed_audio = process_audio_input(audio_upload)
    if processed_audio is None:
        return "", "", "Error processing audio"
    
    language = None
    if lang_disp and lang_disp != "Auto":
        language = ASR_LANG_MAP.get(lang_disp, lang_disp)

    try:
        asr_model = get_model("ASR", "1.7B")

        results = asr_model.transcribe(
            audio=processed_audio,
            language=language,
            return_time_stamps=False,
        )

        if not isinstance(results, list) or len(results) != 1:
            return "", "", "Unexpected result format"

        r = results[0]
        detected_lang = getattr(r, "language", "") or ""
        transcribed_text = getattr(r, "text", "") or ""

        return detected_lang, transcribed_text, "Success"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", "", f"Error: {str(e)}"

voice_design_node = FnNode(
    fn=run_voice_design,
    inputs={
        "text": gr.Textbox(
            label="Text to Synthesize (Voice Design)", 
            lines=4, 
            value="It's in the top drawer... wait, it's empty? No way!"
        ),
        "language": gr.Dropdown(
            label="Language (Voice Design)", 
            choices=TTS_LANGUAGES, 
            value="Auto"
        ),
        "voice_description": gr.Textbox(
            label="Voice Description (Voice Design)", 
            lines=3, 
            value="Speak in an incredulous tone, but with a hint of panic."
        ),
    },
    outputs={
        "generated_audio": gr.Audio(label="Generated Audio", type="numpy"),
        "status": gr.Textbox(label="Status", interactive=False),
    },
    name="Voice Design"
)

custom_voice_node = FnNode(
    fn=run_custom_voice,
    inputs={
        "text": gr.Textbox(
            label="Text to Synthesize (Custom Voice)", 
            lines=4, 
            value="Hello! Welcome to the Text-to-Speech demo."
        ),
        "language": gr.Dropdown(label="Language (Custom Voice)", choices=TTS_LANGUAGES, value="English"),
        "speaker": gr.Dropdown(label="Speaker (Custom Voice)", choices=SPEAKERS, value="Ryan"),
        "instruct": gr.Textbox(label="Style Instruction (Custom Voice)", lines=2, placeholder="e.g. Happy, Sad", value="Neutral"),
        "model_size": gr.Dropdown(label="Model Size (Custom Voice)", choices=MODEL_SIZES, value="1.7B"),
    },
    outputs={
        "tts_audio": gr.Audio(label="Generated Audio", type="numpy"),
        "status": gr.Textbox(label="Status", interactive=False),
    },
    name="Custom Voice"
)

voice_clone_node = FnNode(
    fn=run_voice_clone,
    inputs={
        "ref_audio": gr.Audio(label="Reference Audio (Voice Clone)", type="filepath"),
        "ref_text": gr.Textbox(label="Reference Transcript (Voice Clone)", lines=2),
        "target_text": gr.Textbox(label="Target Text (Voice Clone)", lines=4),
        "language": gr.Dropdown(label="Language (Voice Clone)", choices=TTS_LANGUAGES, value="Auto"),
        "use_xvector_only": gr.Checkbox(label="Use x-vector only (Voice Clone)", value=False),
        "model_size": gr.Dropdown(label="Model Size (Voice Clone)", choices=MODEL_SIZES, value="1.7B"),
    },
    outputs={
        "cloned_audio": gr.Audio(label="Cloned Audio", type="numpy"),
        "status": gr.Textbox(label="Status", interactive=False),
    },
    name="Voice Clone"
)

asr_node = FnNode(
    fn=run_asr,
    inputs={
        "audio_upload": gr.Audio(label="Upload Audio (Qwen3 ASR)", type="numpy", sources=["upload", "microphone"]),
        "lang_disp": gr.Dropdown(label="Language (Qwen3 ASR)", choices=ASR_LANG_CHOICES, value="Auto"),
    },
    outputs={
        "detected_lang": gr.Textbox(label="Detected Language", interactive=False),
        "transcription": gr.Textbox(label="Transcription Result", lines=6, interactive=True),
        "status": gr.Textbox(label="Status", interactive=False),
    },
    name="Qwen3 ASR"
)

graph = Graph(
    name="Qwen3-TTS-Daggr-UI",
    nodes=[voice_design_node, custom_voice_node, voice_clone_node, asr_node]
)

if __name__ == "__main__":
    graph.launch(host="0.0.0.0", port=7860)
