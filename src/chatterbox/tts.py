"""
Chatterbox Text-to-Speech (TTS) Module

This module provides the main interface for the Chatterbox TTS system, a production-grade
text-to-speech model with zero-shot voice cloning capabilities and emotion exaggeration control.

The module includes:
- ChatterboxTTS: Main TTS class for generating speech from text with reference audio
- Conditionals: Data structure for managing model conditioning parameters
- punc_norm: Text normalization utility for improving synthesis quality

Key Features:
- Zero-shot voice cloning from audio prompts
- Emotion/intensity exaggeration control
- Built-in Perth watermarking for responsible AI
- Support for classifier-free guidance (CFG)
- Configurable temperature for generation diversity

Example:
    >>> import torchaudio as ta
    >>> from chatterbox.tts import ChatterboxTTS
    >>>
    >>> model = ChatterboxTTS.from_pretrained(device="cuda")
    >>> text = "Hello, this is a test of the Chatterbox TTS system."
    >>> wav = model.generate(text, audio_prompt_path="reference.wav")
    >>> ta.save("output.wav", wav, model.sr)

For more information, see the Chatterbox repository:
https://github.com/resemble-ai/chatterbox
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
    Normalize text punctuation for improved TTS synthesis quality.

    This function performs several text cleanup operations to handle punctuation
    from LLMs or uncommon characters not frequently seen in the training dataset.

    Operations performed:
    - Capitalizes the first letter if lowercase
    - Removes multiple consecutive spaces
    - Replaces uncommon punctuation (ellipsis, em-dash, curly quotes, etc.)
    - Adds a period if no sentence-ending punctuation exists

    Args:
        text: The input text string to normalize

    Returns:
        The normalized text with cleaned punctuation

    Example:
        >>> punc_norm("hello world...")
        'Hello world, '
        >>> punc_norm("this is a test")
        'This is a test.'
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditioning parameters for T3 and S3Gen models.

    This dataclass encapsulates all conditioning information required for the
    Chatterbox TTS pipeline, including both the T3 (text-to-speech tokens) model
    and the S3Gen (speech tokens to waveform) model.

    Attributes:
        t3: T3Cond instance containing T3 model conditioning:
            - speaker_emb: Voice encoder speaker embedding
            - clap_emb: CLAP audio-text embedding (optional)
            - cond_prompt_speech_tokens: Speech tokens from reference audio prompt
            - cond_prompt_speech_emb: Speech embedding from reference audio prompt
            - emotion_adv: Emotion/intensity exaggeration scalar (0.0-1.0)

        gen: Dictionary containing S3Gen model conditioning:
            - prompt_token: Reference audio tokens for vocoder
            - prompt_token_len: Length of prompt tokens
            - prompt_feat: Reference audio features
            - prompt_feat_len: Length of prompt features
            - embedding: Additional embedding information

    Methods:
        to(device): Move all conditioning tensors to specified device
        save(fpath): Save conditioning parameters to file
        load(fpath, map_location): Load conditioning parameters from file

    Example:
        >>> conds = Conditionals(t3_cond, gen_dict)
        >>> conds = conds.to("cuda")
        >>> conds.save(Path("my_voice.pt"))
    """
    t3: T3Cond
    gen: dict

    def to(self, device: Union[str, torch.device]) -> 'Conditionals':
        """
        Move all conditioning tensors to the specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'mps', or torch.device instance)

        Returns:
            Self for method chaining
        """
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path) -> None:
        """
        Save conditioning parameters to a file.

        Args:
            fpath: Path where the conditioning parameters will be saved

        Example:
            >>> conds.save(Path("my_custom_voice.pt"))
        """
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> 'Conditionals':
        """
        Load conditioning parameters from a file.

        Args:
            fpath: Path to the saved conditioning parameters file
            map_location: Device to load tensors to (default: "cpu")

        Returns:
            Conditionals instance with loaded parameters

        Example:
            >>> conds = Conditionals.load("my_custom_voice.pt", map_location="cuda")
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    """
    Main Chatterbox Text-to-Speech synthesis class.

    This class provides the primary interface for the Chatterbox TTS system, enabling
    high-quality zero-shot voice cloning with emotion exaggeration control and built-in
    watermarking for responsible AI usage.

    The synthesis pipeline consists of:
    1. Text normalization and tokenization
    2. T3 model: Text → Speech tokens with voice/emotion conditioning
    3. S3Gen model: Speech tokens → Waveform with reference audio conditioning
    4. Perth watermarking: Adding imperceptible neural watermarks

    Class Attributes:
        ENC_COND_LEN: Length of audio for encoder conditioning (6 seconds at 16kHz)
        DEC_COND_LEN: Length of audio for decoder conditioning (10 seconds at 24kHz)

    Attributes:
        sr: Output audio sample rate (24kHz)
        t3: T3 transformer model for text-to-speech token generation
        s3gen: S3Gen model for speech token to waveform generation
        ve: Voice encoder for extracting speaker embeddings
        tokenizer: Text tokenizer for converting text to tokens
        device: Computation device ('cpu', 'cuda', or 'mps')
        conds: Conditioning parameters (optional, loaded from reference audio)
        watermarker: Perth watermarker for adding neural watermarks

    Example:
        >>> model = ChatterboxTTS.from_pretrained(device="cuda")
        >>> wav = model.generate(
        ...     "Hello, this is a test.",
        ...     audio_prompt_path="reference.wav",
        ...     exaggeration=0.7,
        ...     cfg_weight=0.5
        ... )
    """
    ENC_COND_LEN = 6 * S3_SR  # 6 seconds at 16kHz for encoder conditioning
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds at 24kHz for decoder conditioning

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Optional[Conditionals] = None,
    ):
        """
        Initialize ChatterboxTTS with model components.

        Args:
            t3: Initialized T3 transformer model
            s3gen: Initialized S3Gen vocoder model
            ve: Initialized voice encoder model
            tokenizer: Text tokenizer instance
            device: Target computation device
            conds: Pre-computed conditioning parameters (optional)
        """
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)