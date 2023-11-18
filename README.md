
# Speech-to-Text Transcription using Distil Whisper Model

This repository contains a Jupyter notebook (`distilwhisper.ipynb`) for transcribing audio files into text using the Whisper model from the Hugging Face Transformers library. The script is designed to work with local audio files and process them using either CPU or GPU.

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

## Using Different Whisper Models

The script is set up to use the `distil-whisper/distil-medium.en` model by default. However, you can use other versions of the Whisper distil models by changing the `model_id` variable. Available models include various sizes and language capabilities. To use a different model, simply replace the `model_id` with the desired model's ID from the [Hugging Face Model Hub](https://huggingface.co/collections/distil-whisper/distil-whisper-models-65411987e6727569748d2eb6).

For example, to use the `distil-large-v2` model, set:

```python
model_id = "distil-whisper/distil-large-v2"
```

## Notebook

Here is the main content of the Jupyter notebook (`distilwhisper.ipynb`):

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

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

audio_file_path = 'ENTER_PATH_HERE'  # Replace with the path to your audio file

audio_input, _ = librosa.load(audio_file_path, sr=16000)

inputs = {
    "raw": audio_input,
    "sampling_rate": 16000
}

result = pipe(inputs)
print(result["text"])
```

Replace the `audio_file_path` variable with the path to your local audio file.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your improvements.

