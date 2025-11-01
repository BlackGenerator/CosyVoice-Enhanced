# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
# Enhanced by Claude - Professional UI with Model Management and Auto-Transcription
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
import json
import argparse
import gradio as gr
from gradio import SelectData
import numpy as np
import torch
import torchaudio
import random
import librosa
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import threading
import time

# Add project root to sys.path to allow running from 'api' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.i18n import I18n
from utils.spk2info_utils import load_spk2info, extract_spkinfo, append_voice_to_spk2info, delete_voice_from_spk2info

# Configuration and Setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = Path(ROOT_DIR) / "pretrained_models"

# FFmpeg path setup
if sys.platform == 'win32':
    ffmpeg_path = Path(ROOT_DIR) / 'ffmpeg' / 'bin'
    os.environ['PATH'] = f"{ROOT_DIR};{ffmpeg_path};{os.environ['PATH']}"
else:
    ffmpeg_path = Path(ROOT_DIR) / 'ffmpeg'
    os.environ['PATH'] = f"{ROOT_DIR}:{ffmpeg_path}:{os.environ['PATH']}"

sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# Import VLLM utilities for auto-detection
from utils.vllm_utils import check_vllm_availability, should_enable_vllm_for_model, log_vllm_status, register_cosyvoice2_vllm

# Global variables
current_cosyvoice = None
current_model_name = ""
transcription_api_url = ""
transcription_api_key = ""
transcription_model = ""

# Protected system voices that cannot be deleted
PROTECTED_VOICES = [
    "中文女", "中文男", "日语男", "粤语女", 
    "英文女", "英文男", "韩语女"
]

# Initialize i18n manager
i18n = I18n()

def get_available_models() -> List[str]:
    """Get list of available models from pretrained_models directory"""
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        return []
    
    models = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
    return sorted(models)

def get_model_type_name(model) -> str:
    """Get human-readable model type name"""
    if isinstance(model.model, CosyVoice2Model):
        return "CosyVoice2"
    elif isinstance(model.model, CosyVoiceModel):
        return "CosyVoice 1.0"
    else:
        return "Unknown"

def is_cosyvoice2_model(model) -> bool:
    """Check if the model is CosyVoice2"""
    return isinstance(model, CosyVoice2) and isinstance(model.model, CosyVoice2Model)

def get_supported_modes(model) -> list[str]:
    """Get list of supported modes for the current model"""
    if model is None:
        return []
    
    if is_cosyvoice2_model(model):
        # CosyVoice2 supports all modes
        return ["Pretrained Voice", "3s Voice Cloning", "Cross-lingual Cloning", "Natural Language Control"]
    else:
        # CosyVoice1.0 - check if it's an instruct model
        if model.instruct:
            return ["Pretrained Voice", "3s Voice Cloning", "Natural Language Control"]
        else:
            return ["Pretrained Voice", "3s Voice Cloning", "Cross-lingual Cloning"]

def load_model(model_name: str) -> tuple[bool, str]:
    """Load a specific model with better error handling, model type detection, and automatic VLLM integration"""
    global current_cosyvoice, current_model_name
    
    if not model_name:
        return False, "No model selected"
    
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return False, f"Model {model_name} not found"
    
    # Check for model config files to determine model type
    cosyvoice2_config = model_path / "cosyvoice2.yaml"
    cosyvoice_config = model_path / "cosyvoice.yaml"
    
    try:
        if cosyvoice2_config.exists():
            # Load as CosyVoice2 with automatic VLLM detection
            auto_vllm = should_enable_vllm_for_model(str(model_path))
            vllm_status = ""
            
            if auto_vllm:
                try:
                    # Register CosyVoice2 with VLLM
                    register_cosyvoice2_vllm()
                    new_cosyvoice = CosyVoice2(str(model_path), load_vllm=True)
                    vllm_status = " [VLLM: Enabled]"
                except Exception as vllm_error:
                    print(f"VLLM acceleration failed, falling back to standard mode: {vllm_error}")
                    new_cosyvoice = CosyVoice2(str(model_path), load_vllm=False)
                    vllm_status = " [VLLM: Failed, using standard mode]"
            else:
                new_cosyvoice = CosyVoice2(str(model_path), load_vllm=False)
                vllm_status = " [VLLM: Not available]"
            
            current_cosyvoice = new_cosyvoice
            current_model_name = model_name
            model_type = get_model_type_name(new_cosyvoice)
            supported_modes = get_supported_modes(new_cosyvoice)
            return True, f"Successfully loaded {model_name} ({model_type}){vllm_status} - Supports: {', '.join(supported_modes)}"
            
        elif cosyvoice_config.exists():
            # Load as CosyVoice 1.0 (no VLLM support)
            new_cosyvoice = CosyVoice(str(model_path))
            current_cosyvoice = new_cosyvoice
            current_model_name = model_name
            model_type = get_model_type_name(new_cosyvoice)
            supported_modes = get_supported_modes(new_cosyvoice)
            instruct_info = " (Instruct)" if new_cosyvoice.instruct else " (Base)"
            return True, f"Successfully loaded {model_name} ({model_type}{instruct_info}) - Supports: {', '.join(supported_modes)}"
        else:
            return False, f"No valid config file found for {model_name}. Expected cosyvoice.yaml or cosyvoice2.yaml"
    except Exception as e:
        return False, f"Failed to load {model_name}: {str(e)}"

def unload_model() -> tuple[bool, str]:
    """Unload current model"""
    global current_cosyvoice, current_model_name
    
    if current_cosyvoice is None:
        return False, "No model loaded"
    
    current_cosyvoice = None
    current_model_name = ""
    return True, "Model unloaded successfully"

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using external API"""
    if not transcription_api_url or not transcription_api_key or not audio_path:
        return ""
    
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
                'model': (None, transcription_model)
            }
            headers = {
                'Authorization': f'Bearer {transcription_api_key}'
            }
            
            response = requests.post(transcription_api_url, files=files, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                logging.error(f"Transcription API error: {response.status_code}")
                return ""
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return ""

def get_protected_voices() -> List[str]:
    """
    Get list of protected system voices from the current model
    """
    if current_cosyvoice is None:
        return []
    
    # Get all available voices from the current model
    all_voices = current_cosyvoice.list_available_spks()
    
    # Filter out protected voices
    protected = [voice for voice in all_voices if voice in PROTECTED_VOICES]
    
    return protected

def reload_spk2info_file() -> bool:
    """
    Reload the spk2info.pt file for the current model
    Returns True if successful, False otherwise
    """
    if current_cosyvoice is None:
        logging.error("Cannot reload spk2info: No model loaded")
        return False
    
    try:
        # Get the path to the spk2info file
        spk2info_path = Path(MODELS_DIR) / current_model_name / "spk2info.pt"
        
        # Check if the file exists
        if not spk2info_path.exists():
            logging.error(f"spk2info file not found at {spk2info_path}")
            return False
        
        # Load the spk2info file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        spk2info = torch.load(str(spk2info_path), map_location=device)
        
        # Directly update the frontend's spk2info dictionary
        if hasattr(current_cosyvoice, 'frontend') and hasattr(current_cosyvoice.frontend, 'spk2info'):
            current_cosyvoice.frontend.spk2info = spk2info
            logging.info(f"Successfully reloaded {len(spk2info)} voices from {spk2info_path}")
            return True
        else:
            logging.error("Invalid model structure: frontend or spk2info not found")
            return False
            
    except Exception as e:
        logging.error(f"Failed to reload spk2info file: {str(e)}")
        return False

def generate_seed():
    """Generate random seed"""
    return random.randint(1, 100000000)

def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
    """Post-process generated speech"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(current_cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def validate_mode_compatibility(model, mode: str) -> tuple[bool, str]:
    """Validate if the current model supports the requested mode"""
    if model is None:
        return False, "No model loaded"
    
    supported_modes = get_supported_modes(model)
    if mode not in supported_modes:
        return False, f"Mode '{mode}' not supported by this model. Supported modes: {', '.join(supported_modes)}"
    
    return True, ""

def generate_audio(
    tts_text: str,
    mode: str,
    sft_voice: str,
    prompt_text: str,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text: str,
    seed: int,
    stream: bool,
    speed: float,
    auto_transcribe: bool
):
    """Generate audio with improved model detection and validation"""
    if current_cosyvoice is None:
        gr.Warning("No model loaded. Please load a model first.")
        return None
    
    # Map translated mode names to internal mode names
    mode_map = {
        i18n.get_text("mode_pretrained_voice"): "Pretrained Voice",
        i18n.get_text("mode_voice_cloning"): "3s Voice Cloning",
        i18n.get_text("mode_cross_lingual"): "Cross-lingual Cloning",
        i18n.get_text("mode_natural_control"): "Natural Language Control"
    }
    
    # Get the internal mode name
    internal_mode = mode_map.get(mode, "Pretrained Voice")
    
    # Validate mode compatibility
    is_compatible, error_msg = validate_mode_compatibility(current_cosyvoice, internal_mode)
    if not is_compatible:
        gr.Warning(error_msg)
        return None
    
    # Determine prompt audio source
    prompt_wav = None
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    
    # Auto-transcribe if enabled
    if auto_transcribe and prompt_wav and not prompt_text:
        transcribed_text = transcribe_audio(prompt_wav)
        if transcribed_text:
            prompt_text = transcribed_text
            success_msg = i18n.get_text("transcription_success").format(text=transcribed_text[:50])
            gr.Info(success_msg)
    
    # Mode-specific validation
    if internal_mode == "Natural Language Control":
        if not instruct_text:
            gr.Warning("Please enter instruction text for natural language control")
            return None
        # Both CosyVoice 1.0 and CosyVoice2 instruction modes are valid without additional requirements
        # CosyVoice2 can use either prompt audio (zero_shot_spk_id='') or preset speaker (zero_shot_spk_id=sft_voice)
        # CosyVoice 1.0 uses speaker selection
    
    elif internal_mode == "Cross-lingual Cloning":
        if not prompt_wav:
            gr.Warning("Please provide prompt audio for cross-lingual cloning")
            return None
    
    elif internal_mode == "3s Voice Cloning":
        if not prompt_wav:
            gr.Warning("Please provide prompt audio for voice cloning")
            return None
        if not prompt_text:
            gr.Warning("Please provide prompt text for voice cloning")
            return None
        if torchaudio.info(prompt_wav).sample_rate < 16000:
            gr.Warning("Prompt audio sample rate is too low (minimum 16kHz required)")
            return None
    
    elif internal_mode == "Pretrained Voice":
        if not sft_voice:
            gr.Warning("Please select a pretrained voice")
            return None
    
    # Set random seed
    set_all_random_seed(seed)
    
    # Generate audio based on mode and model type
    try:
        if internal_mode == "Pretrained Voice":
            for result in current_cosyvoice.inference_sft(tts_text, sft_voice, stream=stream, speed=speed):
                yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
        
        elif internal_mode == "3s Voice Cloning":
            prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))
            for result in current_cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
                yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
        
        elif internal_mode == "Cross-lingual Cloning":
            prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))
            for result in current_cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
                yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
        
        elif internal_mode == "Natural Language Control":
            if is_cosyvoice2_model(current_cosyvoice):
                # CosyVoice2 uses inference_instruct2 with flexible speaker options
                if prompt_wav:
                    # Use prompt audio for speaker characteristics
                    prompt_speech_16k = postprocess(load_wav(prompt_wav, 16000))
                    for result in current_cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=stream, speed=speed):
                        yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
                else:
                    # Use preset speaker voice (sft_voice) - validate it exists in spk2info
                    if not sft_voice:
                        gr.Warning("Please either upload prompt audio or select a preset speaker for CosyVoice2 instruction mode")
                        return None
                    
                    # Check if the selected voice exists in spk2info
                    available_voices = current_cosyvoice.list_available_spks()
                    if sft_voice not in available_voices:
                        gr.Warning(f"Selected voice '{sft_voice}' not found. Available voices: {', '.join(available_voices)}")
                        return None
                    
                    # Create dummy prompt audio since the method signature requires it, but zero_shot_spk_id will be used
                    dummy_prompt = torch.zeros(1, 16000)  # 1 second of silence
                    for result in current_cosyvoice.inference_instruct2(tts_text, instruct_text, dummy_prompt, zero_shot_spk_id=sft_voice, stream=stream, speed=speed):
                        yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
            else:
                # CosyVoice 1.0 uses inference_instruct with speaker selection
                if not sft_voice:
                    gr.Warning("Please select a speaker for CosyVoice 1.0 instruction mode")
                    return None
                for result in current_cosyvoice.inference_instruct(tts_text, sft_voice, instruct_text, stream=stream, speed=speed):
                    yield (current_cosyvoice.sample_rate, result['tts_speech'].numpy().flatten())
    
    except Exception as e:
        gr.Error(f"Audio generation failed: {str(e)}")
        yield None

def register_voice(voice_name: str, voice_audio, voice_text: str) -> tuple[bool, str, list]:
    """Register a new voice to the current model's spk2info file"""
    if current_cosyvoice is None:
        return False, i18n.get_text("no_model_loaded"), []
    
    if not voice_audio:
        return False, i18n.get_text("no_audio"), []
    
    if not voice_name:
        return False, i18n.get_text("no_voice_name"), []
    
    # Get the path to the spk2info file of the current model
    spk2info_path = Path(MODELS_DIR) / current_model_name / "spk2info.pt"
    
    # Check if the voice already exists
    spk2info = load_spk2info(str(spk2info_path))
    if spk2info is None:
        return False, f"Could not load spk2info file from {spk2info_path}", []
        
    if voice_name in spk2info:
        return False, i18n.get_text("voice_exists"), []
    
    try:
        # Extract speaker info from the audio file
        spk_info = extract_spkinfo(voice_audio, voice_text, current_cosyvoice)
        
        # Append the voice to the spk2info file
        append_voice_to_spk2info(str(spk2info_path), voice_name, spk_info)
        
        # Reload the spk2info file
        if not reload_spk2info_file():
            return False, "Failed to reload voices after registration. Please restart the application or reload the model.", []
        
        # Get updated voices list
        all_voices = current_cosyvoice.list_available_spks()
        
        return True, i18n.get_text("voice_registered"), all_voices
    except Exception as e:
        error_message = str(e)
        logging.error(f"Voice registration error: {error_message}")
        return False, f"Registration failed: {error_message}", []

def delete_voice(voice_name: str) -> tuple[bool, str, list]:
    """Delete a voice from the current model's spk2info file"""
    if current_cosyvoice is None:
        return False, i18n.get_text("no_model_loaded"), []
    
    # Check if the voice is protected
    protected_voices = get_protected_voices()
    if voice_name in protected_voices:
        return False, i18n.get_text("voice_protected"), []
    
    # Get the path to the spk2info file of the current model
    spk2info_path = Path(MODELS_DIR) / current_model_name / "spk2info.pt"
    
    try:
        # Delete the voice from the spk2info file
        delete_voice_from_spk2info(str(spk2info_path), voice_name)
        
        # Reload the spk2info file
        if not reload_spk2info_file():
            return False, "Failed to reload voices after deletion. Please restart the application or reload the model.", []
        
        # Get updated voices list
        all_voices = current_cosyvoice.list_available_spks()
        
        return True, i18n.get_text("voice_deleted"), all_voices
    except Exception as e:
        error_message = str(e)
        logging.error(f"Voice deletion error: {error_message}")
        return False, f"Deletion failed: {error_message}", []

def create_interface(fixed_model=None):
    """Create the main Gradio interface"""
    
    # Custom CSS for professional styling
    custom_css = """
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-loaded {
        background-color: #22c55e;
        box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
    }
    .status-unloaded {
        background-color: #f59e0b;
        box-shadow: 0 0 6px rgba(245, 158, 11, 0.6);
    }
    """
    
    with gr.Blocks(css=custom_css, title="CosyVoice Professional TTS") as demo:
        
        # Session state for model persistence
        model_state = gr.State(value=current_model_name if current_model_name else fixed_model)
        
        # Header
        header = gr.HTML(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">{i18n.get_text('title')}</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">{i18n.get_text('subtitle')}</p>
        </div>
        """)
        
        with gr.Row():
            # Left Panel - Model Management & Settings
            with gr.Column(scale=1):
                with gr.Group():
                    model_management_header = gr.Markdown(f"### 🔧 {i18n.get_text('model_management')}")
                    
                    # model_status = gr.HTML(
                    #    f'<div><span class="status-indicator status-unloaded"></span>{i18n.get_text("model_status")}: {i18n.get_text("not_loaded")}</div>'
                    # )
                    
                    # Show model selection only if not in fixed model mode
                    if fixed_model is None:
                        available_models = gr.Dropdown(
                            choices=get_available_models(),
                            label=i18n.get_text("available_models"),
                            value=current_model_name if current_model_name else None
                        )
                        
                        with gr.Row():
                            load_btn = gr.Button(i18n.get_text("load_model"), variant="primary")
                            unload_btn = gr.Button(i18n.get_text("unload_model"), variant="stop")
                    else:
                        # In fixed model mode, we don't show model selection controls
                        # but we still create them with visible=False for event handlers
                        available_models = gr.Dropdown(
                            choices=get_available_models(),
                            label=i18n.get_text("available_models"),
                            value=fixed_model,
                            visible=False
                        )
                        
                        with gr.Row(visible=False):
                            load_btn = gr.Button(i18n.get_text("load_model"), variant="primary")
                            unload_btn = gr.Button(i18n.get_text("unload_model"), variant="stop")
                    
                    model_info = gr.Textbox(
                        label=i18n.get_text("current_model"),
                        value="No model loaded",
                        interactive=False
                    )
                
                with gr.Group():
                    settings_header = gr.Markdown(f"### ⚙️ {i18n.get_text('settings')}")
                    
                    language_select = gr.Dropdown(
                        choices=[("English", "en"), ("中文", "zh")],
                        label=i18n.get_text("language"),
                        value="en"
                    )
                    
                    with gr.Accordion(i18n.get_text("api_settings"), open=False) as api_accordion:
                        api_url_input = gr.Textbox(
                            label=i18n.get_text("transcription_url"),
                            value="https://api.siliconflow.cn/v1/audio/transcriptions",
                            placeholder=i18n.get_text("placeholder_api_url")
                        )
                        
                        api_key_input = gr.Textbox(
                            label=i18n.get_text("api_key"),
                            type="password",
                            placeholder=i18n.get_text("placeholder_api_key")
                        )
                        
                        save_settings_btn = gr.Button(i18n.get_text("save_settings"))
            
            # Right Panel - TTS Interface
            with gr.Column(scale=2):
                # Create tabs
                with gr.Tabs() as tabs:
                    # Tab 1: TTS Interface
                    with gr.TabItem(i18n.get_text("tts_tab")) as tts_tab:
                        with gr.Group():
                            text_input_header = gr.Markdown(f"### 🎤 {i18n.get_text('text_input')}")
                            
                            tts_text = gr.Textbox(
                                label=i18n.get_text("text_input"),
                                lines=3,
                                value=i18n.get_text("default_tts_text"),
                                placeholder=i18n.get_text("placeholder_tts_text")
                            )
                        
                        with gr.Group():
                            inference_mode_header = gr.Markdown(f"### 🎯 {i18n.get_text('inference_mode')}")
                            
                            with gr.Row():
                                # Define mode choices using i18n translations
                                mode_choices = [
                                    i18n.get_text("mode_pretrained_voice"),
                                    i18n.get_text("mode_voice_cloning"),
                                    i18n.get_text("mode_cross_lingual"),
                                    i18n.get_text("mode_natural_control")
                                ]
                                
                                mode_select = gr.Radio(
                                    choices=mode_choices,
                                    label=i18n.get_text("inference_mode"),
                                    value=i18n.get_text("mode_pretrained_voice")
                                )
                                
                                # Get operation steps for the default mode
                                default_steps = i18n.get_text("steps_pretrained_voice")
                                
                                operation_steps = gr.Textbox(
                                    label=i18n.get_text("operation_steps"),
                                    value=default_steps,
                                    interactive=False,
                                    lines=3
                                )
                            
                            # Model compatibility info
                            model_compatibility_info = gr.HTML(
                                value="<div style='padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;'><b>ℹ️ Model Info:</b> Load a model to see supported modes</div>"
                            )
                        
                        with gr.Row():
                            with gr.Column():
                                sft_voice = gr.Dropdown(
                                    choices=[],
                                    label=i18n.get_text("select_voice"),
                                    value=None
                                )
                                
                                with gr.Row():
                                    stream_mode = gr.Checkbox(
                                        label=i18n.get_text("streaming"),
                                        value=False
                                    )
                                    speed_control = gr.Slider(
                                        minimum=0.5,
                                        maximum=2.0,
                                        step=0.1,
                                        value=1.0,
                                        label=i18n.get_text("speed_control")
                                    )
                            
                            with gr.Column():
                                with gr.Row():
                                    seed_btn = gr.Button("🎲", scale=0)
                                    seed_input = gr.Number(
                                        label=i18n.get_text("random_seed"),
                                        value=42,
                                        scale=1
                                    )
                        
                        with gr.Group():
                            audio_config_header = gr.Markdown(f"### 🎵 {i18n.get_text('audio_configuration')}")
                            
                            with gr.Row():
                                prompt_audio_upload = gr.Audio(
                                    sources=["upload"],
                                    type="filepath",
                                    label=i18n.get_text("prompt_audio_file")
                                )
                                
                                prompt_audio_record = gr.Audio(
                                    sources=["microphone"],
                                    type="filepath",
                                    label=i18n.get_text("record_prompt")
                                )
                            
                            with gr.Row():
                                prompt_text_input = gr.Textbox(
                                    label=i18n.get_text("prompt_text"),
                                    placeholder=i18n.get_text("placeholder_prompt_text"),
                                    lines=2
                                )
                                
                                auto_transcribe_cb = gr.Checkbox(
                                    label=i18n.get_text("auto_transcribe"),
                                    value=True
                                )
                            
                            instruct_text_input = gr.Textbox(
                                label=i18n.get_text("instruct_text"),
                                placeholder=i18n.get_text("placeholder_instruct_text"),
                                lines=2
                            )
                        
                        # Generation Button
                        generate_btn = gr.Button(
                            i18n.get_text("generate_audio"),
                            variant="primary",
                            size="lg"
                        )
                        
                        # Output
                        audio_output = gr.Audio(
                            label=i18n.get_text("generated_audio"),
                            autoplay=True,
                            streaming=True
                        )
                    
                    # Tab 2: Voice Management
                    with gr.TabItem(i18n.get_text("voice_management_tab")) as voice_mgmt_tab:
                        voice_management_header = gr.Markdown(f"### 🎤 {i18n.get_text('voice_management')}")
                        
                        # Different content based on fixed_model
                        if fixed_model is None:
                            # Full voice management functionality when fixed_model is None
                            with gr.Group():
                                gr.Markdown(f"#### {i18n.get_text('register_voice')}")
                                
                                with gr.Row():
                                    vm_voice_name = gr.Textbox(
                                        label=i18n.get_text("voice_name"),
                                        placeholder=i18n.get_text("placeholder_voice_name")
                                    )
                                
                                with gr.Row():
                                    vm_audio_upload = gr.Audio(
                                        sources=["upload", "microphone"],
                                        type="filepath",
                                        label=i18n.get_text("voice_upload")
                                    )
                                
                                with gr.Row():
                                    vm_prompt_text = gr.Textbox(
                                        label=i18n.get_text("prompt_text"),
                                        placeholder=i18n.get_text("placeholder_prompt_text"),
                                        lines=2
                                    )
                                    
                                    vm_auto_transcribe = gr.Checkbox(
                                        label=i18n.get_text("auto_transcribe"),
                                        value=True
                                    )
                                
                                register_btn = gr.Button(
                                    i18n.get_text("register_button"),
                                    variant="primary"
                                )
                                
                                registration_status = gr.Textbox(
                                    label=i18n.get_text("status"),
                                    interactive=False
                                )
                            
                            with gr.Group():
                                gr.Markdown(f"#### {i18n.get_text('voice_list')}")
                                
                                # Create a dataframe to display voices
                                voice_list_df = gr.Dataframe(
                                    headers=[
                                        i18n.get_text("voice_name"),
                                        "Type",
                                        "Action"
                                    ],
                                    datatype=["str", "str", "str"],
                                    col_count=(3, "fixed"),
                                    interactive=False
                                )
                                
                                refresh_voices_btn = gr.Button(i18n.get_text("refresh"))
                        else:
                            # Minimal version with disabled message when fixed_model is specified
                            gr.Markdown(f"""
                            #### {i18n.get_text('voice_management_disabled')}
                            
                            {i18n.get_text('voice_management_disabled_message')}
                            """)
                            
                            # Create hidden components for event handlers
                            vm_voice_name = gr.Textbox(visible=False)
                            vm_audio_upload = gr.Audio(visible=False)
                            vm_prompt_text = gr.Textbox(visible=False)
                            vm_auto_transcribe = gr.Checkbox(visible=False)
                            register_btn = gr.Button(visible=False)
                            registration_status = gr.Textbox(visible=False)
                            voice_list_df = gr.Dataframe(visible=False)
                            refresh_voices_btn = gr.Button(visible=False)
        
        def update_model_compatibility_info():
            """Update model compatibility information display"""
            if current_cosyvoice is None:
                return "<div style='padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;'><b>ℹ️ Model Info:</b> Load a model to see supported modes</div>"
            
            model_type = get_model_type_name(current_cosyvoice)
            supported_modes = get_supported_modes(current_cosyvoice)
            modes_str = ", ".join(supported_modes)
            
            if is_cosyvoice2_model(current_cosyvoice):
                color = "#e3f2fd"
                text_color = "#1565c0"
                icon = "🚀"
            else:
                instruct_info = " (Instruct)" if current_cosyvoice.instruct else " (Base)"
                model_type += instruct_info
                color = "#fff3e0"
                text_color = "#ef6c00"
                icon = "⚡"
            return f"<div></div>"
            # return f"<div style='padding: 10px; background: {color}; border-radius: 5px; margin: 10px 0; color: {text_color};'><b>{icon} {model_type}:</b> Supports {modes_str}</div>"
        
        # Event handlers
        def update_model_status():
            if current_cosyvoice:
                status_html = f'<div><span class="status-indicator status-loaded"></span>{i18n.get_text("model_status")}: {i18n.get_text("loaded")} ({current_model_name})</div>'
                return status_html, current_model_name
            else:
                status_html = f'<div><span class="status-indicator status-unloaded"></span>{i18n.get_text("model_status")}: {i18n.get_text("not_loaded")}</div>'
                return status_html, "No model loaded"
        
        def get_voice_list_data():
            """Generate data for the voice list dataframe"""
            if current_cosyvoice is None:
                return []
            
            # Get voices and protected voices
            all_voices = current_cosyvoice.list_available_spks()
            protected_voices = get_protected_voices()
            
            # Create data for the dataframe
            voice_data = []
            for voice in all_voices:
                if voice in protected_voices:
                    voice_type = i18n.get_text("system_voice")
                    delete_btn = ""  # No delete button for protected voices
                else:
                    voice_type = i18n.get_text("custom_voice")
                    delete_btn = i18n.get_text("delete_voice")
                
                voice_data.append([voice, voice_type, delete_btn])
            
            return voice_data
        
        def handle_load_model(model_name, model_state_value=None):
            success, message = load_model(model_name)
            if success:
                gr.Info(message)
                voices = current_cosyvoice.list_available_spks() if current_cosyvoice else []
                status_html, model_info_text = update_model_status()
                voice_list_data = get_voice_list_data()
                
                # Update mode choices based on loaded model capabilities
                supported_modes = get_supported_modes(current_cosyvoice)
                translated_modes = []
                for mode in supported_modes:
                    if mode == "Pretrained Voice":
                        translated_modes.append(i18n.get_text("mode_pretrained_voice"))
                    elif mode == "3s Voice Cloning":
                        translated_modes.append(i18n.get_text("mode_voice_cloning"))
                    elif mode == "Cross-lingual Cloning":
                        translated_modes.append(i18n.get_text("mode_cross_lingual"))
                    elif mode == "Natural Language Control":
                        translated_modes.append(i18n.get_text("mode_natural_control"))
                
                # Update mode compatibility info
                compatibility_info = update_model_compatibility_info()
                
                # Update mode selection to only show supported modes
                return (
                    status_html, 
                    model_info_text, 
                    gr.update(choices=voices, value=voices[0] if voices else None),
                    voice_list_data, 
                    model_name, 
                    gr.update(value=model_name),
                    gr.update(choices=translated_modes, value=translated_modes[0] if translated_modes else None),  # Update mode choices
                    compatibility_info  # Add compatibility info
                )
            else:
                gr.Error(message)
                return gr.update(), gr.update(), gr.update(), [], model_state_value, gr.update(), gr.update(), gr.update()
        
        def handle_unload_model(model_state_value):
            success, message = unload_model()
            if success:
                gr.Info(message)
            else:
                gr.Warning(message)
            status_html, model_info_text = update_model_status()
            
            # Reset mode choices to default when no model is loaded
            default_mode_choices = [
                i18n.get_text("mode_pretrained_voice"),
                i18n.get_text("mode_voice_cloning"),
                i18n.get_text("mode_cross_lingual"),
                i18n.get_text("mode_natural_control")
            ]
            
            # Clear model state when unloading
            return (
                status_html, 
                model_info_text, 
                gr.update(choices=[], value=None), 
                [], 
                None, 
                gr.update(value=None),
                gr.update(choices=default_mode_choices, value=default_mode_choices[0]),  # Reset mode choices
                "<div style='padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;'><b>ℹ️ Model Info:</b> Load a model to see supported modes</div>"  # Reset compatibility info
            )
            
        def auto_load_model(model_state_value):
            """Auto-load model from session state or fixed model parameter"""
            if not model_state_value or model_state_value == current_model_name:
                # Model is already loaded or no model to load
                status_html, model_info_text = update_model_status()
                voices = current_cosyvoice.list_available_spks() if current_cosyvoice else []
                voice_list_data = get_voice_list_data()
                
                # Update mode choices based on current model
                if current_cosyvoice:
                    supported_modes = get_supported_modes(current_cosyvoice)
                    translated_modes = []
                    for mode in supported_modes:
                        if mode == "Pretrained Voice":
                            translated_modes.append(i18n.get_text("mode_pretrained_voice"))
                        elif mode == "3s Voice Cloning":
                            translated_modes.append(i18n.get_text("mode_voice_cloning"))
                        elif mode == "Cross-lingual Cloning":
                            translated_modes.append(i18n.get_text("mode_cross_lingual"))
                        elif mode == "Natural Language Control":
                            translated_modes.append(i18n.get_text("mode_natural_control"))
                    compatibility_info = update_model_compatibility_info()
                else:
                    # Default mode choices when no model is loaded
                    translated_modes = [
                        i18n.get_text("mode_pretrained_voice"),
                        i18n.get_text("mode_voice_cloning"),
                        i18n.get_text("mode_cross_lingual"),
                        i18n.get_text("mode_natural_control")
                    ]
                    compatibility_info = "<div style='padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;'><b>ℹ️ Model Info:</b> Load a model to see supported modes</div>"
                
                # Update available_models dropdown to show the currently loaded model
                return (
                    status_html, 
                    model_info_text, 
                    gr.update(choices=voices, value=voices[0] if voices else None), 
                    voice_list_data, 
                    model_state_value, 
                    gr.update(value=current_model_name if current_model_name else None),
                    gr.update(choices=translated_modes, value=translated_modes[0] if translated_modes else None),
                    compatibility_info
                )
            
            # Try to load the model from session state
            success, message = load_model(model_state_value)
            if success:
                voices = current_cosyvoice.list_available_spks() if current_cosyvoice else []
                status_html, model_info_text = update_model_status()
                voice_list_data = get_voice_list_data()
                
                # Update mode choices based on loaded model
                supported_modes = get_supported_modes(current_cosyvoice)
                translated_modes = []
                for mode in supported_modes:
                    if mode == "Pretrained Voice":
                        translated_modes.append(i18n.get_text("mode_pretrained_voice"))
                    elif mode == "3s Voice Cloning":
                        translated_modes.append(i18n.get_text("mode_voice_cloning"))
                    elif mode == "Cross-lingual Cloning":
                        translated_modes.append(i18n.get_text("mode_cross_lingual"))
                    elif mode == "Natural Language Control":
                        translated_modes.append(i18n.get_text("mode_natural_control"))
                
                compatibility_info = update_model_compatibility_info()
                
                # Update available_models dropdown to show the newly loaded model
                return (
                    status_html, 
                    model_info_text, 
                    gr.update(choices=voices, value=voices[0] if voices else None), 
                    voice_list_data, 
                    model_state_value, 
                    gr.update(value=model_state_value),
                    gr.update(choices=translated_modes, value=translated_modes[0] if translated_modes else None),
                    compatibility_info
                )
            else:
                # Failed to load model from session state
                default_modes = [
                    i18n.get_text("mode_pretrained_voice"),
                    i18n.get_text("mode_voice_cloning"),
                    i18n.get_text("mode_cross_lingual"),
                    i18n.get_text("mode_natural_control")
                ]
                return gr.update(), gr.update(), gr.update(), [], None, gr.update(), gr.update(choices=default_modes, value=default_modes[0])
        
        def save_api_settings(url, key):
            global transcription_api_url, transcription_api_key
            transcription_api_url = url
            transcription_api_key = key
            gr.Info("API settings saved successfully")
        
        def update_operation_steps(mode):
            # Create a mapping between mode names and their corresponding steps keys
            mode_to_steps = {
                i18n.get_text("mode_pretrained_voice"): "steps_pretrained_voice",
                i18n.get_text("mode_voice_cloning"): "steps_voice_cloning",
                i18n.get_text("mode_cross_lingual"): "steps_cross_lingual",
                i18n.get_text("mode_natural_control"): "steps_natural_control"
            }
            
            # Get the steps key for the selected mode
            steps_key = mode_to_steps.get(mode, "steps_pretrained_voice")
            
            # Return the translated steps
            return i18n.get_text(steps_key)
        
        # Updated change_language function that returns updates for all UI components
        def change_language(lang):
            i18n.set_language(lang)
            
            # Update all text-based components with new language
            updated_header = f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">{i18n.get_text('title')}</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">{i18n.get_text('subtitle')}</p>
            </div>
            """
            
            updated_model_management = f"### 🔧 {i18n.get_text('model_management')}"
            updated_settings = f"### ⚙️ {i18n.get_text('settings')}"
            updated_text_input = f"### 🎤 {i18n.get_text('text_input')}"
            updated_inference_mode = f"### 🎯 {i18n.get_text('inference_mode')}"
            updated_audio_config = f"### 🎵 {i18n.get_text('audio_configuration')}"
            updated_voice_management = f"### 🎤 {i18n.get_text('voice_management')}"
            
            # Update model status
            status_text = "loaded" if current_cosyvoice else "not_loaded"
            model_name_display = f" ({current_model_name})" if current_cosyvoice else ""
            updated_model_status = f'<div><span class="status-indicator status-{status_text}"></span>{i18n.get_text("model_status")}: {i18n.get_text(status_text)}{model_name_display}</div>'
            
            # Define updated mode choices
            updated_mode_choices = [
                i18n.get_text("mode_pretrained_voice"),
                i18n.get_text("mode_voice_cloning"),
                i18n.get_text("mode_cross_lingual"),
                i18n.get_text("mode_natural_control")
            ]
            
            # Create a mapping between old and new mode values to maintain selection
            old_to_new_mode = {
                "Pretrained Voice": i18n.get_text("mode_pretrained_voice"),
                "3s Voice Cloning": i18n.get_text("mode_voice_cloning"),
                "Cross-lingual Cloning": i18n.get_text("mode_cross_lingual"),
                "Natural Language Control": i18n.get_text("mode_natural_control"),
                # Add translations from previous language
                i18n.get_text("mode_pretrained_voice"): i18n.get_text("mode_pretrained_voice"),
                i18n.get_text("mode_voice_cloning"): i18n.get_text("mode_voice_cloning"),
                i18n.get_text("mode_cross_lingual"): i18n.get_text("mode_cross_lingual"),
                i18n.get_text("mode_natural_control"): i18n.get_text("mode_natural_control")
            }
            
            # Get the new mode value based on the current selection
            current_mode = mode_select.value
            new_mode_value = old_to_new_mode.get(current_mode, i18n.get_text("mode_pretrained_voice"))
            
            # Update operation steps based on the new mode value
            current_steps = update_operation_steps(new_mode_value)
            
            # Update voice list dataframe headers
            updated_voice_list_headers = [
                i18n.get_text("voice_name"),
                "Type",
                "Action"
            ]
            
            # Update voice list data to refresh translations
            voice_list_data = get_voice_list_data()
            
            # Get current voices for dropdown
            voices = current_cosyvoice.list_available_spks() if current_cosyvoice else []
            
            # Get current model state (to preserve it during language change)
            current_model = current_model_name if current_cosyvoice else None
            
            # Return updates for all components
            return (
                updated_header,
                updated_model_management,
                updated_model_status,
                gr.update(label=i18n.get_text("available_models")),
                gr.update(value=i18n.get_text("load_model")),
                gr.update(value=i18n.get_text("unload_model")),
                gr.update(label=i18n.get_text("current_model")),
                updated_settings,
                gr.update(label=i18n.get_text("language")),
                gr.update(label=i18n.get_text("api_settings")),
                gr.update(label=i18n.get_text("transcription_url"), placeholder=i18n.get_text("placeholder_api_url")),
                gr.update(label=i18n.get_text("api_key"), placeholder=i18n.get_text("placeholder_api_key")),
                gr.update(value=i18n.get_text("save_settings")),
                updated_text_input,
                gr.update(label=i18n.get_text("text_input"), value=i18n.get_text("default_tts_text"), placeholder=i18n.get_text("placeholder_tts_text")),
                updated_inference_mode,
                gr.update(label=i18n.get_text("inference_mode"), choices=updated_mode_choices, value=new_mode_value),
                gr.update(label=i18n.get_text("operation_steps"), value=current_steps),
                gr.update(label=i18n.get_text("select_voice")),
                gr.update(label=i18n.get_text("streaming")),
                gr.update(label=i18n.get_text("speed_control")),
                gr.update(label=i18n.get_text("random_seed")),
                updated_audio_config,
                gr.update(label=i18n.get_text("prompt_audio_file")),
                gr.update(label=i18n.get_text("record_prompt")),
                gr.update(label=i18n.get_text("prompt_text"), placeholder=i18n.get_text("placeholder_prompt_text")),
                gr.update(label=i18n.get_text("auto_transcribe")),
                gr.update(label=i18n.get_text("instruct_text"), placeholder=i18n.get_text("placeholder_instruct_text")),
                gr.update(value=i18n.get_text("generate_audio")),
                gr.update(label=i18n.get_text("generated_audio")),
                # Tab labels
                gr.update(label=i18n.get_text("tts_tab")),
                gr.update(label=i18n.get_text("voice_management_tab")),
                # Voice management tab components
                updated_voice_management,
                gr.update(label=i18n.get_text("voice_name"), placeholder=i18n.get_text("placeholder_voice_name")),
                gr.update(label=i18n.get_text("voice_upload")),
                gr.update(label=i18n.get_text("prompt_text"), placeholder=i18n.get_text("placeholder_prompt_text")),
                gr.update(label=i18n.get_text("auto_transcribe")),
                gr.update(value=i18n.get_text("register_button")),
                # Voice list dataframe
                gr.update(headers=updated_voice_list_headers, value=voice_list_data),
                gr.update(value=i18n.get_text("refresh")),
                gr.update(choices=voices, value=voices[0] if voices else None),
                # Preserve model state
                current_model
            )
        
        # Voice management event handlers
        def handle_voice_audio_transcription(audio_file, auto_transcribe_enabled):
            """Handle transcription of voice audio file for voice registration"""
            if not audio_file or not auto_transcribe_enabled:
                return gr.update()
            
            if not transcription_api_url or not transcription_api_key:
                gr.Warning(i18n.get_text("api_not_configured"))
                return gr.update()
            
            transcribed_text = transcribe_audio(audio_file)
            if transcribed_text:
                success_msg = i18n.get_text("transcription_success").format(text=transcribed_text[:50])
                gr.Info(success_msg)
                return transcribed_text
            else:
                gr.Warning(i18n.get_text("transcription_failed"))
                return gr.update()
        
        def handle_voice_registration(voice_name, audio_file, prompt_text):
            """Handle voice registration"""
            if current_cosyvoice is None:
                gr.Warning(i18n.get_text("no_model_loaded"))
                return i18n.get_text("no_model_loaded"), [], gr.update()
            
            if not audio_file:
                gr.Warning(i18n.get_text("no_audio"))
                return i18n.get_text("no_audio"), [], gr.update()
            
            if not voice_name:
                gr.Warning(i18n.get_text("no_voice_name"))
                return i18n.get_text("no_voice_name"), [], gr.update()
            
            success, message, voices = register_voice(voice_name, audio_file, prompt_text)
            
            if success:
                gr.Info(message)
            else:
                gr.Warning(message)
            
            # Update voice list data
            voice_list_data = get_voice_list_data()
            
            # Return updated voice list for both the dataframe and the sft_voice dropdown
            return message, voice_list_data, gr.update(choices=voices, value=voices[0] if voices else None)
        
        def handle_voice_deletion(evt: gr.SelectData):
            """Handle voice deletion when delete button is clicked in the dataframe"""
            if current_cosyvoice is None:
                gr.Warning(i18n.get_text("no_model_loaded"))
                return [], gr.update()
            
            # Get the selected row and column
            row_index = evt.index[0]
            col_index = evt.index[1]
            
            # Check if the clicked cell is the delete button (column 2 - Action)
            if col_index == 2:
                # Get current voice list data
                voice_data = get_voice_list_data()
                
                # Get voice name from the clicked row
                if row_index < len(voice_data):
                    voice_name = voice_data[row_index][0]
                    
                    # Delete the voice
                    success, message, voices = delete_voice(voice_name)
                    
                    if success:
                        gr.Info(message)
                    else:
                        gr.Warning(message)
                    
                    # Update voice list data
                    voice_list_data = get_voice_list_data()
                    
                    # Return updated voice list for both the dataframe and the sft_voice dropdown
                    return voice_list_data, gr.update(choices=voices, value=voices[0] if voices else None)
            
            # If not a delete action, return current data without updating sft_voice
            return get_voice_list_data(), gr.update()
        
        def refresh_voice_list():
            """Refresh the voice list dataframe and voice dropdown"""
            if current_cosyvoice is None:
                return [], gr.update()
            
            # Force reload the spk2info file
            reload_spk2info_file()
            
            # Get updated voice data
            voice_list_data = get_voice_list_data()
            voices = current_cosyvoice.list_available_spks()
            
            # Return updates for both the dataframe and the dropdown
            return voice_list_data, gr.update(choices=voices, value=voices[0] if voices else None)
        
        # Connect voice management components
        vm_audio_upload.change(
            handle_voice_audio_transcription,
            inputs=[vm_audio_upload, vm_auto_transcribe],
            outputs=[vm_prompt_text]
        )
        
        register_btn.click(
            handle_voice_registration,
            inputs=[vm_voice_name, vm_audio_upload, vm_prompt_text],
            outputs=[registration_status, voice_list_df, sft_voice]
        )
        
        voice_list_df.select(
            handle_voice_deletion,
            outputs=[voice_list_df, sft_voice]
        )
        
        refresh_voices_btn.click(
            refresh_voice_list,
            outputs=[voice_list_df, sft_voice]
        )
        
        # Connect event handlers
        load_btn.click(
            handle_load_model,
            inputs=[available_models, model_state],
            outputs=[model_status, model_info, sft_voice, voice_list_df, model_state, available_models, mode_select, model_compatibility_info]
        )
        
        unload_btn.click(
            handle_unload_model,
            inputs=[model_state],
            outputs=[model_status, model_info, sft_voice, voice_list_df, model_state, available_models, mode_select, model_compatibility_info]
        )
        
        # Auto-load model on page load
        demo.load(
            auto_load_model,
            inputs=[model_state],
            outputs=[model_status, model_info, sft_voice, voice_list_df, model_state, available_models, mode_select, model_compatibility_info]
        )
        
        # Function to handle audio transcription when an audio is uploaded or recorded
        def handle_audio_transcription(audio_file, auto_transcribe_enabled):
            if not audio_file or not auto_transcribe_enabled:
                return gr.update()
            
            if not transcription_api_url or not transcription_api_key:
                gr.Warning(i18n.get_text("api_not_configured"))
                return gr.update()
            
            transcribed_text = transcribe_audio(audio_file)
            if transcribed_text:
                success_msg = i18n.get_text("transcription_success").format(text=transcribed_text[:50])
                gr.Info(success_msg)
                return transcribed_text
            else:
                gr.Warning(i18n.get_text("transcription_failed"))
                return gr.update()
        
        # Connect audio components to transcription handler
        prompt_audio_upload.change(
            handle_audio_transcription,
            inputs=[prompt_audio_upload, auto_transcribe_cb],
            outputs=[prompt_text_input]
        )
        
        prompt_audio_record.change(
            handle_audio_transcription,
            inputs=[prompt_audio_record, auto_transcribe_cb],
            outputs=[prompt_text_input]
        )
        
        # Connect language_select to the updated change_language function with all components as outputs
        language_select.change(
            fn=change_language,
            inputs=[language_select],
            outputs=[
                header,
                model_management_header,
                model_status,
                available_models,
                load_btn,
                unload_btn,
                model_info,
                settings_header,
                language_select,
                api_accordion,
                api_url_input,
                api_key_input,
                save_settings_btn,
                text_input_header,
                tts_text,
                inference_mode_header,
                mode_select,
                operation_steps,
                sft_voice,
                stream_mode,
                speed_control,
                seed_input,
                audio_config_header,
                prompt_audio_upload,
                prompt_audio_record,
                prompt_text_input,
                auto_transcribe_cb,
                instruct_text_input,
                generate_btn,
                audio_output,
                # New tab components
                tts_tab,
                voice_mgmt_tab,
                voice_management_header,
                vm_voice_name,
                vm_audio_upload,
                vm_prompt_text,
                vm_auto_transcribe,
                register_btn,
                voice_list_df,
                refresh_voices_btn,
                model_state  # Add model_state to preserve it during language changes
            ]
        )
        
        save_settings_btn.click(
            save_api_settings,
            inputs=[api_url_input, api_key_input]
        )
        
        mode_select.change(
            update_operation_steps,
            inputs=[mode_select],
            outputs=[operation_steps]
        )
        
        seed_btn.click(
            lambda: generate_seed(),
            outputs=[seed_input]
        )
        
        generate_btn.click(
            generate_audio,
            inputs=[
                tts_text, mode_select, sft_voice, prompt_text_input,
                prompt_audio_upload, prompt_audio_record, instruct_text_input,
                seed_input, stream_mode, speed_control, auto_transcribe_cb
            ],
            outputs=[audio_output]
        )
    
    return demo

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="CosyVoice TTS Interface")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--transcription_url', type=str, 
                       default='https://api.siliconflow.cn/v1/audio/transcriptions',
                       help='Transcription API URL')
    parser.add_argument('--transcription_key', type=str, default='', help='Transcription API key')
    parser.add_argument('--transcription_model', type=str, default='FunAudioLLM/SenseVoiceSmall', help='Transcription API model')
    parser.add_argument('--language', type=str, default='zh', help='Language')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--hot_reload', action='store_true', help='Enable hot reload mode')
    parser.add_argument('--model', type=str, default=None, help='Specify a model to load at startup')
    
    args = parser.parse_args()
    
    # Set global API configuration
    global transcription_api_url, transcription_api_key, transcription_model
    transcription_api_url = args.transcription_url
    transcription_api_key = args.transcription_key
    transcription_model = args.transcription_model
    
    # Set initial language
    i18n.set_language(args.language)
    
    # Log VLLM availability status
    print("\n" + "="*50)
    print("🎙️  CosyVoice WebUI Starting...")
    print("="*50)
    log_vllm_status()
    print("="*50 + "\n")
    
    # Create and launch interface
    demo = create_interface(fixed_model=args.model)
    
    demo.queue(max_size=10, default_concurrency_limit=4)
  
    # Launch with hot reload if specified
    if args.hot_reload:
        # For hot reload, we need to use a different approach
        import sys
        import subprocess
        
        # Define the file watcher function
        def watch_file():
            import time
            import os
            last_modified = os.path.getmtime(__file__)
            
            while True:
                time.sleep(1)
                try:
                    current_modified = os.path.getmtime(__file__)
                    if current_modified > last_modified:
                        print("File changed, reloading...")
                        os.execv(sys.executable, ['python'] + sys.argv)
                except Exception as e:
                    print(f"Error checking file: {e}")
        
        # Start the file watcher in a separate thread
        import threading
        watcher_thread = threading.Thread(target=watch_file, daemon=True)
        watcher_thread.start()
    
    # Launch the interface
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )



if __name__ == '__main__':
    main()
