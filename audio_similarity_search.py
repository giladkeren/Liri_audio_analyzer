import boto3
import tempfile
import librosa
import torch as th
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


s3 = boto3.client("s3")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus-sv"
)
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")


def get_audio(s3_bucket, s3_key):
    print(f"Downloading audio from S3: {s3_bucket}/{s3_key}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_file(s3_bucket, s3_key, tmp_file.name)
        audio, sr = librosa.load(tmp_file.name)

    whisper_sr = 16000
    if sr != whisper_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=whisper_sr)

    audio = audio[: whisper_sr * 30]
    return audio


def get_embeddings(aud):
    inputs = feature_extractor(aud, sampling_rate=16000, return_tensors="pt")
    embeddings = model(**inputs).embeddings
    embeddings = th.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings


def compute_sim(emb1, emb2):
    cosine_sim = th.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(emb1, emb2)
    return similarity
