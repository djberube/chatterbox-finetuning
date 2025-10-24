
# Chatterbox TTS Finetuning

### Fine-tuning Models

Fine-tune Chatterbox models on your own datasets to specialize for specific voices, accents, or languages:

#### Fine-tune T3 Model

```bash
cd src

python finetune_t3.py \
  --output_dir ./checkpoints/chatterbox_finetuned \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name YOUR_DATASET_NAME \
  --train_split_name train \
  --eval_split_size 0.0002 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 2000 \
  --save_strategy steps \
  --save_steps 4000 \
  --save_total_limit 4 \
  --fp16 True \
  --report_to tensorboard \
  --dataloader_num_workers 8 \
  --do_train --do_eval \
  --dataloader_pin_memory False \
  --eval_on_start True \
  --label_names labels_speech \
  --text_column_name text_scribe
```

**Example with specific dataset**:
```bash
# Fine-tune on German voice dataset
python finetune_t3.py \
  --output_dir ./checkpoints/chatterbox_finetuned_yodas \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name MrDragonFox/DE_Emilia_Yodas_680h \
  --train_split_name train \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --fp16 True \
  --do_train --do_eval
```

For detailed fine-tuning instructions, see [Fine-tuning Guide](#fine-tuning-guide).

### Voice Conversion

Convert existing audio to a different voice:

```python
from chatterbox.vc import VoiceConverter
import torchaudio as ta

# Load voice converter
converter = VoiceConverter.from_pretrained(device="cuda")

# Convert source audio to target voice
source_audio = "source_voice.wav"
target_voice_reference = "target_voice_sample.wav"

converted = converter.convert(
    source_audio,
    voice_reference=target_voice_reference
)
ta.save("converted_output.wav", converted, converter.sr)
```

See [voice_conversion.py](voice_conversion.py) and [example_vc.py](example_vc.py) for more examples.

### Advanced Configuration

Control speech characteristics with configuration parameters:

```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

# General use - default settings work well
wav = model.generate(
    text="Your text here",
    exaggeration=0.5,  # Emotion intensity (0.0-1.0)
    cfg_weight=0.5     # Classifier-free guidance weight
)

# Expressive/dramatic speech
wav = model.generate(
    text="A dramatic announcement!",
    exaggeration=0.7,   # Higher emotion intensity
    cfg_weight=0.3      # Lower CFG for slower, more deliberate pacing
)

# Fast-speaking reference voice adjustment
wav = model.generate(
    text="Quick paced speech",
    audio_prompt_path="fast_speaker.wav",
    cfg_weight=0.3      # Lower CFG improves pacing
)
```

---

## Configuration

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exaggeration` | float | 0.5 | Emotion/intensity exaggeration (0.0-1.0). Higher values increase expressiveness. |
| `cfg_weight` | float | 0.5 | Classifier-free guidance weight (0.0-1.0). Lower values slow down speech. |
| `temperature` | float | 1.0 | Generation temperature. Higher values increase diversity. |
| `device` | str | "cuda" | Device for inference: "cuda", "cpu", or "mps" (Apple Silicon). |

### Performance Tips

- **General Use**: Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts
- **Fast-Speaking Reference**: Lower `cfg_weight` to ~0.3 to improve pacing
- **Expressive Speech**: Increase `exaggeration` to 0.7+ and lower `cfg_weight` to ~0.3
- **Speed Control**: Higher `exaggeration` speeds up speech; lower `cfg_weight` slows it down

---

## Fine-tuning Guide

### Dataset Preparation

Your dataset should be in one of these formats:

1. **Hugging Face Dataset** with columns:
   - `audio`: Audio file path or bytes
   - `text` or `text_scribe`: Transcription
   - `labels_speech` (optional): Speech labels for training

2. **Local Dataset** with structure:
   ```
   dataset/
   ├── metadata.csv    # columns: filename, transcription
   └── audio/
       ├── file1.wav
       ├── file2.wav
       └── ...
   ```

### Fine-tuning T3 Model

The T3 model handles text-to-speech token prediction:

```bash
python src/finetune_t3.py \
  --output_dir ./checkpoints/my_finetuned_model \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name path/to/dataset \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --fp16 True \
  --do_train --do_eval
```

### Fine-tuning S3Gen Model

The S3Gen model generates speech features:

```bash
python src/finetune_s3gen.py \
  --output_dir ./checkpoints/s3gen_finetuned \
  --model_name_or_path ResembleAI/chatterbox \
  --dataset_name path/to/dataset \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-4
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./checkpoints/your_model_dir
```

### Using Fine-tuned Models

Load your fine-tuned model:

```python
from chatterbox.tts import ChatterboxTTS

# Load fine-tuned model
model = ChatterboxTTS.from_pretrained(
    "path/to/checkpoints/your_model",
    device="cuda"
)

# Generate speech
wav = model.generate("Test with your fine-tuned model")
```
--


