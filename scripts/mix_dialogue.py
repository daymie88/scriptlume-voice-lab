"""Simple WAV mixer for ScriptLume multi-host production.

Input JSON example:
[
  {"file":"maya.wav", "start_ms":0, "gain":1.0},
  {"file":"david.wav", "start_ms":4200, "gain":1.0}
]
"""
from pathlib import Path
import argparse
import json
import torch
import torchaudio as ta

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("output")
args = parser.parse_args()

items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
tracks = []
sr = None
max_samples = 0

for item in items:
    wav, item_sr = ta.load(item["file"])
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr is None:
        sr = item_sr
    if item_sr != sr:
        wav = ta.functional.resample(wav, item_sr, sr)
    start = int(float(item.get("start_ms", 0)) * sr / 1000)
    gain = float(item.get("gain", 1.0))
    wav = wav * gain
    tracks.append((start, wav))
    max_samples = max(max_samples, start + wav.shape[-1])

mix = torch.zeros(1, max_samples)
for start, wav in tracks:
    mix[:, start:start + wav.shape[-1]] += wav

peak = mix.abs().max().item()
if peak > 0.98:
    mix = mix * (0.98 / peak)

ta.save(args.output, mix, sr)
