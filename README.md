
# Speech-to-Text Transcription using Whisper Model

This repository contains a Python script for transcribing audio files into text using the Whisper model from the Hugging Face Transformers library. The script is designed to work with local audio files and process them using either CPU or GPU.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- librosa

## Installation

Before running the script, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install torch transformers librosa
```

## Usage

To use the script, simply place your audio file in a directory and update the `audio_file_path` in the script with the path to your audio file. The script will load the audio file, resample it to the required 16 kHz, and transcribe it to text.

## Script

Here is the main script (`transcribe.py`):

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model ID
model_id = "distil-whisper/distil-medium.en"

# Load the model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Setup the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

# Path to your local audio file
audio_file_path = 'path/to/your_audio_file.wav'  # Replace with the path to your audio file

# Load and resample the local audio file to 16 kHz
audio_input, _ = librosa.load(audio_file_path, sr=16000)

# Prepare the input dictionary for the pipeline
inputs = {
    "raw": audio_input,
    "sampling_rate": 16000
}

# Perform the transcription
result = pipe(inputs)
print(result["text"])
```

Replace the `audio_file_path` variable with the path to your local audio file.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your improvements.


