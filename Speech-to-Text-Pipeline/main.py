import torch
import torchaudio
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

warnings.filterwarnings("ignore")  # Suppress warnings

# -------------------- Load Speech-to-Text Model --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# -------------------- Load Summarization Model --------------------
summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# -------------------- Load Translation Model (Optimized) --------------------
model_name = "facebook/m2m100_418M"  # Smaller, faster model
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

# -------------------- Speech-to-Text Function --------------------
def speech_to_text(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to Mono if Stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz (Whisper requirement)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    inputs = whisper_processor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)

    transcribed_text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcribed_text

# -------------------- Summarization Function --------------------
def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"].to(device), max_length=150, min_length=50, length_penalty=2.0)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -------------------- Translation Function (Batched) --------------------
def translate_text_batched(text, tokenizer, model, src_lang="en", tgt_lang="fr", batch_size=5):
    tokenizer.src_lang = src_lang
    sentences = text.split(". ")  # Split into sentences
    translated_sentences = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs)

        translated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sentences.extend(translated_batch)

    return " ".join(translated_sentences)

# -------------------- Main Pipeline --------------------
if __name__ == "__main__":
    audio_file = "/home/ashnam/LLM-Projects/Speech-to-Text-Pipeline/sample_audio.wav"



    print("\n🔹 Running Speech-to-Text...")
    transcribed_text = speech_to_text(audio_file)
    print("\nRaw Transcription:\n", transcribed_text)

    print("\n🔹 Summarizing Text...")
    summarized_text = summarize_text(transcribed_text)
    print("\nSummarized Text:\n", summarized_text)

    print("\n🔹 Translating to French...")
    translated_text = translate_text_batched(summarized_text, tokenizer, model)
    print("\nTranslated Text (French):\n", translated_text)
