from pathlib import Path
import json, subprocess, time, urllib.request
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"g1_conversation_work"
OUT=ROOT/"output_podcast"
SEEDS=WORK/"seeds"
TURNS=WORK/"turns"
for p in (WORK,OUT,SEEDS,TURNS): p.mkdir(exist_ok=True)

HOSTS={
 "Maya":{"seed":"af_heart","url":"https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-af_heart.mp3","temperature":0.76,"top_p":0.95,"rep":1.20,"seed_num":5101},
 "David":{"seed":"am_michael","url":"https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-am_michael.mp3","temperature":0.72,"top_p":0.94,"rep":1.20,"seed_num":5201},
 "Naomi":{"seed":"bf_emma","url":"https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-bf_emma.mp3","temperature":0.74,"top_p":0.94,"rep":1.20,"seed_num":5301},
 "Marcus":{"seed":"am_fenrir","url":"https://raw.githubusercontent.com/andyart30-web/kokoroTTS/main/examples/kokorotts-am_fenrir.mp3","temperature":0.70,"top_p":0.93,"rep":1.21,"seed_num":5401},
}

TURNS_DATA=[
 {"speaker":"Maya","text":"Okay, Genesis 1 starts huge: in the beginning, God creates the heavens and the earth. Before the chapter tells us anything about us, it tells us who the Creator is."},
 {"speaker":"David","text":"Right—and almost immediately you can hear a rhythm. God speaks. Something happens. God names or orders it. Then the text keeps returning to one word: good.","overlapBeforeMs":150},
 {"speaker":"Maya","text":"Exactly. It doesn't read like chaos slowly figuring itself out. The story keeps showing order arriving through God's word.","overlapBeforeMs":140},
 {"speaker":"Naomi","text":"And notice the lights in the sky. Genesis doesn't treat the sun, moon, and stars as gods. They're created lights with jobs to do—marking days, seasons, and years.","gapBeforeMs":120},
 {"speaker":"David","text":"Then verse 26 slows us down: 'Let's make man in our image.' That's one of the chapter's biggest claims about human identity.","gapBeforeMs":130},
 {"speaker":"Maya","text":"Can I pause you there? Because 'image of God' gets used so often that it can become almost invisible.","overlapBeforeMs":170},
 {"speaker":"David","text":"Yes. And whatever debates we have around the phrase, this chapter at least says human dignity and responsibility are received from God. They aren't earned by status, intelligence, wealth, or power.","overlapBeforeMs":100},
 {"speaker":"Marcus","text":"Which changes how we look at other people. If the person in front of me bears God-given dignity, I don't get to decide whose life matters based on usefulness or similarity to me.","gapBeforeMs":120},
 {"speaker":"Naomi","text":"Psalm 8 later asks why human beings matter in such a vast creation, and it answers with that same combination: dignity and delegated responsibility under God.","gapBeforeMs":120},
 {"speaker":"Maya","text":"And then Genesis 1 lands on 'very good.' So the chapter leaves us with a world that belongs to God, people who carry real responsibility, and creation presented as gift rather than accident.","gapBeforeMs":150},
]

def download(url,path):
    if path.exists() and path.stat().st_size>0:return
    req=urllib.request.Request(url,headers={"User-Agent":"ScriptLumeVoiceLab/1.0"})
    with urllib.request.urlopen(req,timeout=60) as r:path.write_bytes(r.read())

def trim(wav,sr,threshold=0.006,pad_ms=250):
    mono=wav.mean(dim=0) if wav.ndim>1 else wav
    idx=torch.where(mono.abs()>threshold)[0]
    if idx.numel()==0:return wav
    pad=int(sr*pad_ms/1000)
    a=max(0,int(idx[0])-pad); b=min(wav.shape[-1],int(idx[-1])+pad)
    return wav[...,a:b]

print("Loading Chatterbox Nano CPU...")
model=ChatterboxTurboTTS.from_pretrained(device="cpu",nano=True)
sr=model.sr
print("Model ready",sr)

seed_paths={}
for name,h in HOSTS.items():
    p=SEEDS/f"{name.lower()}_{h['seed']}.mp3"
    download(h["url"],p); seed_paths[name]=p
    print(name,"seed bytes",p.stat().st_size)

turn_meta=[]
turn_wavs=[]
for i,t in enumerate(TURNS_DATA):
    h=HOSTS[t["speaker"]]
    print(f"Generating turn {i+1}/10 {t['speaker']}")
    accepted=None; used=None; started=time.time()
    for attempt in range(3):
        seed=h["seed_num"]+i*10+attempt
        torch.manual_seed(seed)
        w=model.generate(
            t["text"],audio_prompt_path=str(seed_paths[t["speaker"]]),
            temperature=h["temperature"],top_p=h["top_p"],
            repetition_penalty=h["rep"],norm_loudness=True
        ).cpu()
        w=trim(w,sr)
        sec=w.shape[-1]/sr
        print("  attempt",attempt+1,"seed",seed,"seconds",round(sec,2))
        if sec>=3.0:
            accepted=w; used=seed; break
    if accepted is None: raise RuntimeError(f"Turn {i+1} failed duration validation")
    turn_wavs.append(accepted)
    fp=TURNS/f"turn_{i+1:02d}_{t['speaker'].lower()}.wav"
    ta.save(str(fp),accepted,sr)
    turn_meta.append({
      "index":i,"speaker":t["speaker"],"text":t["text"],
      "duration_ms":round(accepted.shape[-1]/sr*1000),
      "gapBeforeMs":t.get("gapBeforeMs",None),
      "overlapBeforeMs":t.get("overlapBeforeMs",None),
      "generation_seed":used,"generation_seconds":round(time.time()-started,2)
    })

# Schedule according to ScriptLume handoff metadata.
starts=[0]
for i in range(1,len(turn_wavs)):
    prev_end=starts[i-1]+turn_wavs[i-1].shape[-1]
    t=TURNS_DATA[i]
    if "overlapBeforeMs" in t:
        start=max(0,prev_end-int(sr*t["overlapBeforeMs"]/1000))
    else:
        gap=t.get("gapBeforeMs",130)
        start=prev_end+int(sr*gap/1000)
    starts.append(start)

total=max(starts[i]+turn_wavs[i].shape[-1] for i in range(len(turn_wavs)))
mix=torch.zeros(1,total)

for i,(start,wav) in enumerate(zip(starts,turn_wavs)):
    w=wav.clone()
    # Subtle fade-in/out avoids clicks; duck the outgoing turn during intentional overlap.
    fade=int(sr*0.035)
    if w.shape[-1]>2*fade:
        w[..., :fade]*=torch.linspace(0,1,fade)
        w[..., -fade:]*=torch.linspace(1,0,fade)
    if i+1<len(turn_wavs) and "overlapBeforeMs" in TURNS_DATA[i+1]:
        ov=int(sr*TURNS_DATA[i+1]["overlapBeforeMs"]/1000)
        if ov>0 and w.shape[-1]>ov:
            w[..., -ov:]*=torch.linspace(1.0,0.62,ov)
    mix[:,start:start+w.shape[-1]]+=w
    turn_meta[i]["start_ms"]=round(start/sr*1000)
    turn_meta[i]["end_ms"]=round((start+w.shape[-1])/sr*1000)

peak=mix.abs().max().item()
if peak>0.96: mix*=0.96/peak

wav_out=OUT/"genesis-1-conversation-4hosts.wav"
mp3_out=OUT/"genesis-1-conversation-4hosts.mp3"
ta.save(str(wav_out),mix,sr)
subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(wav_out),"-ac","1","-b:a","64k",str(mp3_out)],check=True)

manifest={
 "chapter":1,"mode":"conversation","hosts":4,"engine":"Chatterbox Nano",
 "duration_ms":round(mix.shape[-1]/sr*1000),
 "sample_rate":sr,"turns":turn_meta,
 "audio_file":mp3_out.name
}
(OUT/"genesis-1-conversation-4hosts.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("DONE",mp3_out,round(mix.shape[-1]/sr,2),"seconds",mp3_out.stat().st_size,"bytes")
