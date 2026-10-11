# ScriptLume Voice Lab — Chatterbox Nano

Zero-recurring-cost production audition for ScriptLume.

This public repository generates four short Genesis-style speech samples with Resemble AI Chatterbox Nano on a CPU-only GitHub Actions runner. It does **not** call OpenAI, Cloudflare Aura, TinyFish, or a paid TTS API.

## What this first test measures

- Natural Conversation
- Warm & Reflective
- Energetic Discussion with a native `[chuckle]` cue
- Serious / Difficult Passage

The first pass intentionally uses Chatterbox Nano's built-in/default conditioning voice. This isolates **delivery behavior and expressiveness** from the separate question of final ScriptLume host identities.

## Why Nano

Chatterbox Nano is MIT-licensed, 110M parameters, and officially documented to run on CPU. Nano/Turbo support native paralinguistic tags such as `[laugh]`, `[chuckle]`, `[sigh]`, and others.

Nano/Turbo do **not** support the older Chatterbox exaggeration/CFG/min-p style controls; their official code ignores those parameters. This lab therefore uses only controls Nano actually honors: wording, punctuation, paralinguistic tags, temperature, top-p, and repetition penalty.

## Zero-cost architecture

Approved Script → offline TTS generation → audio QA → multi-host timing/overlap mix → finished WAV/Opus/MP3 → upload finished file to ScriptLume storage → listeners stream/download the finished episode.

TTS never runs in the customer playback path.

## Generated artifact

A successful workflow run uploads `scriptlume-chatterbox-nano-samples` containing the four WAV files plus `manifest.json`.