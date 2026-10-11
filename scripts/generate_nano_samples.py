from pathlib import Path
import json
import time
import torch
import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

with open(ROOT / "prompts" / "styles.json", "r", encoding="utf-8") as f:
    styles = json.load(f)

print(f"Torch: {torch.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print("Loading Chatterbox Nano on CPU...")
load_started = time.time()
model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
print(f"Model ready in {time.time() - load_started:.1f}s; sample rate={model.sr}")

manifest = {
    "engine": "ResembleAI Chatterbox Nano",
    "device": "cpu",
    "sample_rate": model.sr,
    "note": "Built-in/default conditioning voice; style audition only.",
    "samples": []
}

for key, cfg in styles.items():
    print(f"\nGenerating {key}: {cfg['label']}")
    started = time.time()
    wav = model.generate(
        cfg["text"],
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        repetition_penalty=float(cfg["repetition_penalty"]),
    )
    path = OUT / f"{key}.wav"
    ta.save(str(path), wav.cpu(), model.sr)
    seconds = wav.shape[-1] / model.sr
    elapsed = time.time() - started
    print(f"Saved {path.name}: {seconds:.2f}s audio in {elapsed:.1f}s")
    manifest["samples"].append({
        "key": key,
        "label": cfg["label"],
        "file": path.name,
        "audio_seconds": round(seconds, 3),
        "generation_seconds": round(elapsed, 3),
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "repetition_penalty": cfg["repetition_penalty"],
        "text": cfg["text"],
    })

with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\nAll Chatterbox Nano samples generated successfully.")
