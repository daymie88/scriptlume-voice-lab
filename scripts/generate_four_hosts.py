from pathlib import Path
import json
import subprocess
import time
import urllib.request

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_four_hosts"
SEEDS = ROOT / "seed_refs"
OUT.mkdir(exist_ok=True)
SEEDS.mkdir(exist_ok=True)

HOSTS = {
    "maya": {
        "display_name": "Maya",
        "role": "Host",
        "seed_name": "af_heart",
        "seed_url": "https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-af_heart.mp3",
        "text": "Welcome to ScriptLume. I'm Maya. Let's stay close to the passage, keep the conversation natural, and discover what Genesis is actually saying.",
        "temperature": 0.76,
        "top_p": 0.95,
        "repetition_penalty": 1.20
    },
    "david": {
        "display_name": "David",
        "role": "Scripture Guide",
        "seed_name": "am_michael",
        "seed_url": "https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-am_michael.mp3",
        "text": "I'm David, your Scripture guide. When another passage genuinely clarifies Genesis, I'll bring it in without pulling us away from the chapter.",
        "temperature": 0.72,
        "top_p": 0.94,
        "repetition_penalty": 1.20
    },
    "naomi": {
        "display_name": "Naomi",
        "role": "Context Guide",
        "seed_name": "bf_emma",
        "seed_url": "https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-bf_emma.mp3",
        "text": "I'm Naomi. I look at historical, cultural, and literary context, especially when it helps a difficult detail in Scripture become clearer.",
        "temperature": 0.74,
        "top_p": 0.94,
        "repetition_penalty": 1.20
    },
    "marcus": {
        "display_name": "Marcus",
        "role": "Reflection Voice",
        "seed_name": "am_fenrir",
        "seed_url": "https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-am_fenrir.mp3",
        "text": "I'm Marcus. I'll help us slow down and reflect on what the passage means for life today, while keeping Scripture itself at the center.",
        "temperature": 0.70,
        "top_p": 0.93,
        "repetition_penalty": 1.21
    }
}

def download(url: str, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "ScriptLumeVoiceLab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        path.write_bytes(resp.read())

def trim_trailing_silence(wav: torch.Tensor, sr: int, threshold: float = 0.006, pad_ms: int = 320):
    mono = wav.mean(dim=0) if wav.ndim > 1 else wav
    idx = torch.where(mono.abs() > threshold)[0]
    if idx.numel() == 0:
        return wav
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(wav.shape[-1], int(idx[-1]) + pad)
    return wav[..., start:end]

print(f"Torch {torch.__version__}; CPU threads={torch.get_num_threads()}")
print("Loading Chatterbox Nano on CPU...")
t0 = time.time()
model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
print(f"Model ready in {time.time()-t0:.1f}s; sample rate={model.sr}")

manifest = {
    "engine": "ResembleAI Chatterbox Nano",
    "device": "cpu",
    "sample_rate": model.sr,
    "conditioning": "distinct synthetic Kokoro seed references",
    "hosts": []
}

for key, cfg in HOSTS.items():
    print(f"\nPreparing {cfg['display_name']} ({cfg['seed_name']})")
    seed_mp3 = SEEDS / f"{key}_{cfg['seed_name']}.mp3"
    download(cfg["seed_url"], seed_mp3)
    print(f"Seed bytes: {seed_mp3.stat().st_size}")

    started = time.time()
    wav = model.generate(
        cfg["text"],
        audio_prompt_path=str(seed_mp3),
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        repetition_penalty=cfg["repetition_penalty"],
        norm_loudness=True,
    )
    wav = trim_trailing_silence(wav.cpu(), model.sr)

    wav_path = OUT / f"{key}.wav"
    mp3_path = OUT / f"{key}.mp3"
    ta.save(str(wav_path), wav, model.sr)

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path), "-ac", "1", "-b:a", "64k", str(mp3_path)
    ], check=True)

    secs = wav.shape[-1] / model.sr
    elapsed = time.time() - started
    print(f"{cfg['display_name']}: {secs:.2f}s audio, generated+encoded in {elapsed:.1f}s")

    manifest["hosts"].append({
        "key": key,
        "display_name": cfg["display_name"],
        "role": cfg["role"],
        "seed_voice": cfg["seed_name"],
        "seed_url": cfg["seed_url"],
        "text": cfg["text"],
        "wav_file": wav_path.name,
        "mp3_file": mp3_path.name,
        "audio_seconds": round(secs, 3),
        "generation_seconds": round(elapsed, 3),
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "repetition_penalty": cfg["repetition_penalty"]
    })

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("\nFour-host audition generation complete.")
