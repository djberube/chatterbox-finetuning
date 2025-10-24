"""
Chatterbox: Production-Grade Open Source Text-to-Speech and Voice Conversion

Chatterbox is Resemble AI's first production-grade open source TTS model, featuring:
- State-of-the-art zero-shot voice cloning capabilities
- Unique emotion exaggeration control for expressive speech
- Ultra-stable alignment-informed inference
- Built-in Perth watermarking for responsible AI
- 0.5B parameter Llama backbone trained on 0.5M hours of data

This package provides:
- ChatterboxTTS: Main text-to-speech synthesis class
- ChatterboxVC: Voice conversion functionality
- Fine-tuning utilities for customization

Example:
    Basic text-to-speech usage:

    >>> import torchaudio as ta
    >>> from chatterbox import ChatterboxTTS
    >>>
    >>> model = ChatterboxTTS.from_pretrained(device="cuda")
    >>> wav = model.generate("Hello, this is Chatterbox!")
    >>> ta.save("output.wav", wav, model.sr)

    Voice cloning with reference audio:

    >>> wav = model.generate(
    ...     "Clone this voice!",
    ...     audio_prompt_path="reference.wav",
    ...     exaggeration=0.7
    ... )

For more information:
- Documentation: https://github.com/resemble-ai/chatterbox
- Demo: https://huggingface.co/spaces/ResembleAI/Chatterbox
- Benchmarks: https://podonos.com/resembleai/chatterbox

License: MIT
"""

__version__ = "0.1.2"
__author__ = "Resemble AI"
__email__ = "engineering@resemble.ai"
__license__ = "MIT"
__url__ = "https://github.com/resemble-ai/chatterbox"

# MARK: - Public API
from .tts import ChatterboxTTS
from .vc import ChatterboxVC

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "__version__",
]
