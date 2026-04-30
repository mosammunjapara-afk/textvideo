from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from magic_hour import AsyncClient
import httpx
import os
import uuid
import base64
import asyncio
import json
import subprocess
import shutil

app = FastAPI(title="Caryanams - AI Video Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# CONFIG — apni keys yahan daalo
# ─────────────────────────────────────────
MAGIC_HOUR_KEY = os.getenv("MAGIC_HOUR_KEY")
WAVESPEED_KEY = os.getenv("WAVESPEED_KEY")
IMAGINE_ART_KEY = os.getenv("IMAGINE_ART_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
WAVESPEED_BASE   = "https://api.wavespeed.ai"
IMAGINE_ART_BASE = "https://api.imagine.art"
CREDIT_ERRORS    = ("credit","credits","quota","limit","insufficient","402","payment","balance","exceeded","upgrade","subscription")

OUTPUT_DIR = Path("outputs")
MUSIC_DIR  = Path("outputs/music")
OUTPUT_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

tasks: dict = {}

# ─────────────────────────────────────────
# FFMPEG CHECK
# ─────────────────────────────────────────
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
async def download_video(url: str) -> str:
    fname = f"{uuid.uuid4()}.mp4"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url)
        (OUTPUT_DIR / fname).write_bytes(resp.content)
    return fname

async def poll_video_status(mh_client: AsyncClient, project_id: str) -> str:
    while True:
        await asyncio.sleep(8)
        result = await mh_client.v1.video_projects.get(id=project_id)
        status = result.status
        print(f"[MagicHour] Project {project_id} status: {status}")
        if status == "complete":
            downloads = result.downloads
            if downloads:
                return downloads[0].url
            raise Exception("Video complete but no download URL")
        elif status in ["error", "failed", "canceled"]:
            raise Exception(f"Project failed: {status}")

# ─────────────────────────────────────────
# CREDIT ERROR DETECTOR
# ─────────────────────────────────────────
def is_credit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(kw in msg for kw in CREDIT_ERRORS)

def ws_duration(d: int) -> int:
    """WaveSpeed sirf 5 ya 8 seconds allow karta hai — nearest map karo."""
    return 5 if d <= 6 else 8

async def wavespeed_t2v(prompt: str, duration: int) -> str:
    headers = {"Authorization": f"Bearer {WAVESPEED_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt":   prompt,
        "size":     "832*480",
        "duration": ws_duration(duration),
        "seed":     -1,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{WAVESPEED_BASE}/api/v3/wavespeed-ai/wan-2.2/t2v-480p",
            headers=headers, json=payload
        )
        if resp.status_code not in (200, 201):
            raise Exception(f"WaveSpeed T2V failed: {resp.status_code} — {resp.text[:300]}")
        data = resp.json()
        pid  = data.get("data", {}).get("id") or data.get("id")
        if not pid: raise Exception(f"WaveSpeed T2V: no id in response: {data}")
        print(f"[WaveSpeed T2V] ID: {pid}")
    return await wavespeed_poll(pid)

async def wavespeed_i2v(image_b64: str, prompt: str, duration: int) -> str:
    headers = {"Authorization": f"Bearer {WAVESPEED_KEY}", "Content-Type": "application/json"}
    payload = {
        "image":      f"data:image/jpeg;base64,{image_b64}",
        "prompt":     prompt,
        "resolution": "480p",
        "duration":   ws_duration(duration),
        "seed":       -1,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{WAVESPEED_BASE}/api/v3/wavespeed-ai/wan-2.2/image-to-video",
            headers=headers, json=payload
        )
        if resp.status_code not in (200, 201):
            raise Exception(f"WaveSpeed I2V failed: {resp.status_code} — {resp.text[:300]}")
        data = resp.json()
        pid  = data.get("data", {}).get("id") or data.get("id")
        if not pid: raise Exception(f"WaveSpeed I2V: no id in response: {data}")
        print(f"[WaveSpeed I2V] ID: {pid}")
    return await wavespeed_poll(pid)

async def wavespeed_poll(pid: str) -> str:
    headers  = {"Authorization": f"Bearer {WAVESPEED_KEY}"}
    poll_url = f"{WAVESPEED_BASE}/api/v3/predictions/{pid}/result"
    while True:
        await asyncio.sleep(8)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(poll_url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"WaveSpeed poll error: {resp.status_code} — {resp.text[:200]}")
            body   = resp.json()
            inner  = body.get("data", body)
            status = inner.get("status", "")
            print(f"[WaveSpeed] {pid} → {status}")
            if status == "completed":
                outputs = inner.get("outputs", [])
                if outputs: return outputs[0]
                raise Exception("WaveSpeed: completed but no output URL")
            elif status in ("failed", "canceled", "error"):
                raise Exception(f"WaveSpeed {status}: {inner.get('error', inner.get('message', ''))}")

# ─────────────────────────────────────────
# IMAGINE.ART — 3rd fallback
# ─────────────────────────────────────────
async def imagineart_t2v(prompt: str, duration: int) -> str:
    headers = {
        "Authorization": f"Bearer {IMAGINE_ART_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "duration": min(max(duration, 2), 8),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{IMAGINE_ART_BASE}/api/v1/videos/text-to-video",
            headers=headers, json=payload
        )
        if resp.status_code not in (200, 201, 202):
            raise Exception(f"imagine.art T2V failed: {resp.status_code} — {resp.text[:300]}")
        data = resp.json()
        pid  = data.get("id") or data.get("task_id") or (data.get("data") or {}).get("id")
        if not pid: raise Exception(f"imagine.art T2V: no id in response: {data}")
        print(f"[imagine.art T2V] ID: {pid}")
    return await imagineart_poll(pid)

async def imagineart_i2v(image_b64: str, prompt: str, duration: int) -> str:
    headers = {
        "Authorization": f"Bearer {IMAGINE_ART_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "duration": min(max(duration, 2), 8),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{IMAGINE_ART_BASE}/api/v1/videos/image-to-video",
            headers=headers, json=payload
        )
        if resp.status_code not in (200, 201, 202):
            raise Exception(f"imagine.art I2V failed: {resp.status_code} — {resp.text[:300]}")
        data = resp.json()
        pid  = data.get("id") or data.get("task_id") or (data.get("data") or {}).get("id")
        if not pid: raise Exception(f"imagine.art I2V: no id in response: {data}")
        print(f"[imagine.art I2V] ID: {pid}")
    return await imagineart_poll(pid)

async def imagineart_poll(pid: str) -> str:
    headers  = {"Authorization": f"Bearer {IMAGINE_ART_KEY}"}
    poll_url = f"{IMAGINE_ART_BASE}/api/v1/videos/{pid}"
    while True:
        await asyncio.sleep(8)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(poll_url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"imagine.art poll error: {resp.status_code} — {resp.text[:200]}")
            body   = resp.json()
            inner  = body.get("data", body)
            status = (inner.get("status") or body.get("status") or "").lower()
            print(f"[imagine.art] {pid} → {status}")
            if status in ("completed", "succeeded", "done", "complete"):
                url = (inner.get("video_url") or inner.get("output_url")
                       or inner.get("url") or (inner.get("outputs") or [None])[0])
                if url: return url
                raise Exception("imagine.art: completed but no video URL")
            elif status in ("failed", "canceled", "error"):
                raise Exception(f"imagine.art {status}: {inner.get('error', inner.get('message', ''))}")

# ─────────────────────────────────────────
# WATERMARK — Magic Hour hatao, Caryanams text lagao
# ─────────────────────────────────────────
FONT_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_NORMAL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def add_caryanams_watermark(video_path: Path) -> str:
    """
    Bottom strip ko upar ke clean pixels se replace karo (Magic Hour + koi bhi
    purana box/blur sab wipe) phir sirf Caryanams text overlay karo.
    Koi box, koi blur, koi patch visible nahi hoga.
    """
    if not ffmpeg_available():
        print("[Watermark] FFmpeg nahi mila, skipping")
        return video_path.name

    out_fname = f"wm_{uuid.uuid4()}.mp4"
    out_path  = OUTPUT_DIR / out_fname

    filter_str = (
        "[0:v]split[main][src];"
        "[src]crop=832:70:0:340[cleanpatch];"
        "[main][cleanpatch]overlay=0:410[wiped];"
        f"[wiped]drawtext=fontfile={FONT_BOLD}:text=\'Caryanams\':"
        "fontcolor=0x4A90D9:fontsize=24:x=W-205:y=H-54:"
        "shadowcolor=black:shadowx=2:shadowy=2,"
        f"drawtext=fontfile={FONT_NORMAL}:text=\'Driven by Trust\':"
        "fontcolor=0xE8A020:fontsize=14:x=W-177:y=H-28:"
        "shadowcolor=black:shadowx=2:shadowy=2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f"[Watermark] Error: {result.stderr[-300:]}")
        return video_path.name

    print(f"[Watermark] Done: {out_fname}")
    return out_fname


# ─────────────────────────────────────────
# MUSIC MIXER — FFmpeg se video + audio merge
# ─────────────────────────────────────────
def mix_video_audio(video_path: Path, audio_path: Path, volume: float = 0.8) -> str:
    out_fname = f"mixed_{uuid.uuid4()}.mp4"
    out_path  = OUTPUT_DIR / out_fname

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-af", f"volume={volume}",
        "-shortest",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[FFmpeg] Error: {result.stderr}")
        raise Exception(f"FFmpeg mixing failed: {result.stderr[-200:]}")

    print(f"[FFmpeg] Mixed: {out_fname}")
    return out_fname

# ─────────────────────────────────────────
# AI MUSIC MOOD DETECTOR
# ─────────────────────────────────────────
async def detect_music_mood(prompt: str) -> dict:
    try:
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        system_msg = """You are a music mood analyzer for AI videos.
Given a video description, return a JSON object with:
- mood: one of [epic, calm, dark, happy, romantic, mysterious, action, nature, sad, upbeat]
- tempo: one of [slow, medium, fast]
- genre: one of [orchestral, ambient, electronic, acoustic, jazz, cinematic, rock]
- description: 5-8 word music description
Return ONLY valid JSON, no other text.
Example: {"mood": "epic", "tempo": "fast", "genre": "orchestral", "description": "Epic orchestral battle music with drums"}"""

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                groq_url,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Video description: {prompt}\nDetect music mood:"}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 150,
                }
            )
            if resp.status_code == 200:
                raw   = resp.json()["choices"][0]["message"]["content"].strip()
                start = raw.find('{')
                end   = raw.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(raw[start:end])
    except Exception as e:
        print(f"[MoodDetect] Error: {e}")

    return {"mood": "cinematic", "tempo": "medium", "genre": "orchestral", "description": "Cinematic background music"}

# ─────────────────────────────────────────
# FREE MUSIC — Mood-based tracks
# ─────────────────────────────────────────
async def fetch_dynamic_music(mood_data: dict, duration: int) -> str | None:
    """
    Mood ke basis par free music download karo.
    Pixabay key kaam nahi karti toh direct CDN tracks use karte hain.
    """
    # Free CC0 music tracks — mood ke hisab se
    MOOD_TRACKS = {
        "epic":       "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "action":     "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "cinematic":  "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "calm":       "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
        "mysterious": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        "happy":      "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
        "upbeat":     "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        "romantic":   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "dark":       "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
        "sad":        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
        "nature":     "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3",
    }
    try:
        mood      = mood_data.get("mood", "cinematic")
        music_url = MOOD_TRACKS.get(mood, MOOD_TRACKS["cinematic"])
        print(f"[Music] Downloading track for mood '{mood}': {music_url}")

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(music_url)
            if resp.status_code == 200 and len(resp.content) > 10_000:
                fname = f"ai_music_{uuid.uuid4()}.mp3"
                (MUSIC_DIR / fname).write_bytes(resp.content)
                print(f"[Music] Saved: {fname} ({len(resp.content)//1024}KB)")
                return str(MUSIC_DIR / fname)
            else:
                print(f"[Music] Bad response: {resp.status_code}, size={len(resp.content)}")
    except Exception as e:
        print(f"[Music] Fetch error: {e}")
    return None

# ─────────────────────────────────────────
# BACKGROUND WORKERS
# ─────────────────────────────────────────
async def worker_t2v(task_id: str, prompt: str, duration: int,
                     music_mode: str = "none", music_path: str = None, music_volume: float = 0.8):
    tasks[task_id]["status"] = "processing"
    model_used = "Magic Hour AI"
    try:
        # ── Magic Hour try karo (1st) ─────────────────────────────
        video_url = None
        try:
            mh   = AsyncClient(token=MAGIC_HOUR_KEY)
            resp = await mh.v1.text_to_video.create(
                end_seconds=float(duration),
                style={"prompt": prompt},
                resolution="480p",
                aspect_ratio="16:9",
                name=f"caryanams{task_id[:8]}",
            )
            project_id = resp.id
            print(f"[T2V] Magic Hour project: {project_id}")
            video_url = await poll_video_status(mh, project_id)
        except Exception as mh_err:
            print(f"[T2V] Magic Hour error: {mh_err}")
            if is_credit_error(mh_err):
                # ── WaveSpeed try karo (2nd) ──────────────────────
                print("[T2V] Credits khatam! WaveSpeed.ai pe switch...")
                tasks[task_id]["status"] = "switching_to_wavespeed"
                try:
                    model_used = "WaveSpeed.ai"
                    video_url  = await wavespeed_t2v(prompt, duration)
                except Exception as ws_err:
                    print(f"[T2V] WaveSpeed error: {ws_err}")
                    if is_credit_error(ws_err):
                        # ── imagine.art try karo (3rd) ────────────
                        print("[T2V] WaveSpeed credits khatam! imagine.art pe switch...")
                        tasks[task_id]["status"] = "switching_to_imagineart"
                        model_used = "imagine.art"
                        video_url  = await imagineart_t2v(prompt, duration)
                    else:
                        raise
            else:
                raise

        fname      = await download_video(video_url)
        video_path = OUTPUT_DIR / fname

        tasks[task_id]["status"] = "adding_watermark"
        wm_fname    = add_caryanams_watermark(video_path)
        video_path  = OUTPUT_DIR / wm_fname
        final_fname = wm_fname

        music_info  = None
        audio_url   = None

        if music_mode == "static" and music_path and Path(music_path).exists():
            if ffmpeg_available():
                tasks[task_id]["status"] = "mixing_audio"
                final_fname = mix_video_audio(video_path, Path(music_path), music_volume)
                music_info  = "User uploaded music"
                audio_url   = f"/download-music/{Path(music_path).name}"
            else:
                print("[Music] FFmpeg not found, skipping mix")

        elif music_mode == "dynamic":
            tasks[task_id]["status"] = "detecting_mood"
            mood_data = await detect_music_mood(prompt)
            print(f"[Music] Mood: {mood_data}")
            tasks[task_id]["music_mood"] = mood_data
            tasks[task_id]["status"] = "fetching_music"
            dl_path = await fetch_dynamic_music(mood_data, duration)
            if dl_path and ffmpeg_available():
                tasks[task_id]["status"] = "mixing_audio"
                final_fname = mix_video_audio(video_path, Path(dl_path), music_volume)
                music_info  = f"AI Music: {mood_data.get('description', 'Cinematic')}"
                audio_url   = f"/download-music/{Path(dl_path).name}"
            elif dl_path:
                audio_url  = f"/download-music/{Path(dl_path).name}"
                music_info = f"AI Music (player only): {mood_data.get('description', 'Cinematic')}"

        tasks[task_id].update({
            "status":       "done",
            "filename":     final_fname,
            "model_used":   model_used,
            "download_url": f"/download/{final_fname}",
            "music_info":   music_info,
            "audio_url":    audio_url,
            "music_mood":   tasks[task_id].get("music_mood"),
        })
        print(f"[T2V] Done ({model_used}): {final_fname}")

    except Exception as e:
        print(f"[T2V] Error: {e}")
        tasks[task_id].update({"status": "failed", "error": str(e)})


async def worker_i2v(task_id: str, image_b64: str, prompt: str, duration: int,
                     music_mode: str = "none", music_path: str = None, music_volume: float = 0.8):
    tasks[task_id]["status"] = "processing"
    model_used = "Magic Hour AI"
    try:
        # ── Magic Hour try karo (1st) ─────────────────────────────
        video_url = None
        try:
            mh = AsyncClient(token=MAGIC_HOUR_KEY)
            upload_resp = await mh.v1.files.upload_urls.create(
                items=[{"extension": "jpg", "type_": "image"}]
            )
            upload_url = upload_resp.items[0].upload_url
            file_path  = upload_resp.items[0].file_path
            image_bytes = base64.b64decode(image_b64)
            async with httpx.AsyncClient(timeout=60.0) as http:
                await http.put(upload_url, content=image_bytes, headers={"Content-Type": "image/jpeg"})
            print(f"[I2V] Image uploaded: {file_path}")
            resp = await mh.v1.image_to_video.create(
                end_seconds=float(duration),
                assets={"image_file_path": file_path},
                style={"prompt": prompt} if prompt else None,
                resolution="480p",
                name=f"caryanams_{task_id[:8]}",
            )
            project_id = resp.id
            print(f"[I2V] Magic Hour project: {project_id}")
            video_url  = await poll_video_status(mh, project_id)
        except Exception as mh_err:
            print(f"[I2V] Magic Hour error: {mh_err}")
            if is_credit_error(mh_err):
                # ── WaveSpeed try karo (2nd) ──────────────────────
                print("[I2V] Credits khatam! WaveSpeed.ai pe switch...")
                tasks[task_id]["status"] = "switching_to_wavespeed"
                try:
                    model_used = "WaveSpeed.ai"
                    video_url  = await wavespeed_i2v(image_b64, prompt, duration)
                except Exception as ws_err:
                    print(f"[I2V] WaveSpeed error: {ws_err}")
                    if is_credit_error(ws_err):
                        # ── imagine.art try karo (3rd) ────────────
                        print("[I2V] WaveSpeed credits khatam! imagine.art pe switch...")
                        tasks[task_id]["status"] = "switching_to_imagineart"
                        model_used = "imagine.art"
                        video_url  = await imagineart_i2v(image_b64, prompt, duration)
                    else:
                        raise
            else:
                raise

        fname      = await download_video(video_url)
        video_path = OUTPUT_DIR / fname

        tasks[task_id]["status"] = "adding_watermark"
        wm_fname    = add_caryanams_watermark(video_path)
        video_path  = OUTPUT_DIR / wm_fname
        final_fname = wm_fname

        music_info  = None
        audio_url   = None

        if music_mode == "static" and music_path and Path(music_path).exists():
            if ffmpeg_available():
                tasks[task_id]["status"] = "mixing_audio"
                final_fname = mix_video_audio(video_path, Path(music_path), music_volume)
                music_info  = "User uploaded music"
                audio_url   = f"/download-music/{Path(music_path).name}"

        elif music_mode == "dynamic":
            tasks[task_id]["status"] = "detecting_mood"
            mood_data = await detect_music_mood(prompt)
            tasks[task_id]["music_mood"] = mood_data
            tasks[task_id]["status"] = "fetching_music"
            dl_path = await fetch_dynamic_music(mood_data, duration)
            if dl_path and ffmpeg_available():
                tasks[task_id]["status"] = "mixing_audio"
                final_fname = mix_video_audio(video_path, Path(dl_path), music_volume)
                music_info  = f"AI Music: {mood_data.get('description', 'Cinematic')}"
                audio_url   = f"/download-music/{Path(dl_path).name}"
            elif dl_path:
                audio_url  = f"/download-music/{Path(dl_path).name}"
                music_info = f"AI Music (player only): {mood_data.get('description', 'Cinematic')}"

        tasks[task_id].update({
            "status":       "done",
            "filename":     final_fname,
            "model_used":   model_used,
            "download_url": f"/download/{final_fname}",
            "music_info":   music_info,
            "audio_url":    audio_url,
            "music_mood":   tasks[task_id].get("music_mood"),
        })
        print(f"[I2V] Done ({model_used}): {final_fname}")

    except Exception as e:
        print(f"[I2V] Error: {e}")
        tasks[task_id].update({"status": "failed", "error": str(e)})


# ─────────────────────────────────────────
# CAR PROMPTS — keyword → 4 suggestions
# ─────────────────────────────────────────
CAR_PROMPTS = [
    {"keywords": ["sport","race","racing","fast","speed","drift","track","f1","formula"],
     "suggestions": ["A sleek red Ferrari blazing through a neon-lit city tunnel at night","Formula 1 car drifting on a wet track with sparks flying dramatically","Lamborghini Huracan tearing across an empty desert highway at sunrise","Sports car drifting in slow motion on a rain-soaked mountain road"]},
    {"keywords": ["luxury","fancy","rich","rolls","bentley","mercedes","bmw","audi","premium"],
     "suggestions": ["Rolls Royce gliding silently through golden autumn leaves in slow motion","Black Bentley arriving at a grand palace entrance on a stormy evening","Mercedes S-Class interior glowing with ambient lights driving at dusk","Luxury car parked on a misty mountain peak overlooking a vast valley"]},
    {"keywords": ["offroad","off road","suv","jeep","mud","mountain","adventure","4x4"],
     "suggestions": ["Jeep Wrangler climbing a rocky mountain trail in golden afternoon light","Off-road SUV splashing through a muddy river crossing in dense jungle","4x4 truck racing across sand dunes with dust clouds billowing behind","Land Rover navigating through a foggy forest trail at early morning"]},
    {"keywords": ["electric","ev","tesla","future","futuristic","cyber","tech"],
     "suggestions": ["Tesla Cybertruck glowing under purple neon lights in a futuristic city","Electric supercar silently accelerating on a glass highway above the clouds","Futuristic autonomous car driving through a holographic smart city at night","EV sports car charging with blue electricity arcs in a sleek garage"]},
    {"keywords": ["classic","vintage","old","retro","antique","muscle","mustang","camaro"],
     "suggestions": ["Classic 1967 Ford Mustang cruising along a coastal highway at golden hour","Vintage muscle car parked outside a 1950s diner under neon signs","Retro American car driving through a Route 66 desert landscape at sunset","Old school Camaro roaring down an empty boulevard with smoke trails"]},
    {"keywords": ["night","city","urban","street","neon","dark"],
     "suggestions": ["Sports car speeding through neon-lit city streets with light trails glowing","Black supercar parked in a dark alley with rain reflecting city lights","Car chase through a futuristic cyberpunk city with holographic billboards","Convertible driving on an empty bridge with city skyline glowing at midnight"]},
    {"keywords": [],  # fallback
     "suggestions": ["A sleek supercar racing through a cinematic mountain road at golden hour","Sports car drifting on a rain-soaked track with dramatic motion blur","Luxury car driving through an empty desert highway under a vast starry sky","Red sports car emerging from garage into bright sunlight in slow motion"]},
]

def get_car_suggestions(text: str) -> list:
    t = text.lower()
    for group in CAR_PROMPTS:
        if any(kw in t for kw in group["keywords"]):
            return group["suggestions"]
    return CAR_PROMPTS[-1]["suggestions"]


@app.post("/suggest-prompts")
async def suggest_prompts(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "").strip()
        if not text or len(text) < 2:
            return {"suggestions": []}

        # Groq try karo, fail pe car prompts
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": "You are a creative AI video prompt assistant for car videos. Return ONLY a valid JSON array of exactly 4 cinematic English car video prompts (10-15 words each). No preamble, no markdown."},
                            {"role": "user",   "content": f'User typed: "{text}"\nGive 4 car video suggestions as JSON array:'},
                        ],
                        "temperature": 0.8, "max_tokens": 300,
                    }
                )
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    s, e = raw.find('['), raw.rfind(']') + 1
                    if s != -1 and e > s:
                        parsed = json.loads(raw[s:e])
                        clean = [(x if isinstance(x, str) else (x.get("text") or x.get("prompt") or str(x))) for x in parsed if x]
                        if clean: return {"suggestions": clean[:4]}
        except Exception as groq_err:
            print(f"[Suggest] Groq failed: {groq_err}")

        # Car prompts fallback
        suggestions = get_car_suggestions(text)
        print(f"[Suggest] Car fallback for: '{text}'")
        return {"suggestions": suggestions}

    except Exception as e:
        print(f"[Suggest] Error: {e}")
        return {"suggestions": CAR_PROMPTS[-1]["suggestions"]}


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":             "ok",
        "ffmpeg":             ffmpeg_available(),
        "magic_hour_key_set": bool(MAGIC_HOUR_KEY),
        "groq_key_set":       GROQ_KEY != "APNI_GROQ_KEY_YAHAN",
    }

@app.post("/upload-music")
async def upload_music(music: UploadFile = File(...)):
    allowed = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/ogg", "audio/mp4"]
    if music.content_type not in allowed:
        raise HTTPException(400, "Sirf MP3, WAV, OGG allowed hai")
    raw = await music.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "Music 20MB se chhoti honi chahiye")
    fname = f"user_music_{uuid.uuid4()}{Path(music.filename).suffix}"
    (MUSIC_DIR / fname).write_bytes(raw)
    return {"music_path": str(MUSIC_DIR / fname), "filename": fname, "size_mb": round(len(raw)/1024/1024, 2)}

@app.post("/text-to-video")
async def text_to_video(
    background_tasks: BackgroundTasks,
    prompt:       str   = Form(...),
    duration:     int   = Form(default=4, ge=2, le=8),
    music_mode:   str   = Form(default="none"),
    music_path:   str   = Form(default=""),
    music_volume: float = Form(default=0.8),
):
    if not prompt.strip():
        raise HTTPException(400, "Prompt khali nahi hona chahiye")
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"task_id": task_id, "type": "text_to_video", "prompt": prompt, "status": "queued", "music_mode": music_mode}
    background_tasks.add_task(worker_t2v, task_id, prompt, duration, music_mode, music_path or None, music_volume)
    return {"task_id": task_id, "status": "queued", "message": "Processing shuru ho gayi!"}

@app.post("/image-to-video")
async def image_to_video(
    background_tasks: BackgroundTasks,
    image:        UploadFile = File(...),
    prompt:       str   = Form(default="animate this image with natural motion"),
    duration:     int   = Form(default=4, ge=2, le=8),
    music_mode:   str   = Form(default="none"),
    music_path:   str   = Form(default=""),
    music_volume: float = Form(default=0.8),
):
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Sirf JPG, PNG, WEBP allowed hai")
    raw = await image.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image 10MB se chhoti honi chahiye")
    image_b64 = base64.b64encode(raw).decode()
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"task_id": task_id, "type": "image_to_video", "prompt": prompt, "status": "queued", "music_mode": music_mode}
    background_tasks.add_task(worker_i2v, task_id, image_b64, prompt, duration, music_mode, music_path or None, music_volume)
    return {"task_id": task_id, "status": "queued", "message": "Image processing shuru!"}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task nahi mila")
    return tasks[task_id]

@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File nahi mili")
    return FileResponse(path, media_type="video/mp4", filename=filename)

@app.get("/download-music/{filename}")
def download_music_file(filename: str):
    path = MUSIC_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Music nahi mili")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)

@app.get("/tasks")
def list_tasks():
    return {"total": len(tasks), "tasks": list(tasks.values())}

@app.get("/", response_class=HTMLResponse)
def frontend():
    template_file = Path("templates/index.html")

    if not template_file.exists():
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Caryanams API</title>
            <style>
                body{
                    font-family:Arial,sans-serif;
                    text-align:center;
                    padding:80px;
                    background:linear-gradient(135deg,#0f172a,#1e293b);
                    color:white;
                }
                h1{font-size:3rem;margin-bottom:20px;}
                p{font-size:1.2rem;color:#cbd5e1;}
            </style>
        </head>
        <body>
            <h1>🚀 Caryanams API Running</h1>
            <p>Your Render deployment is successful.</p>
        </body>
        </html>
        """)

    return HTMLResponse(
        content=template_file.read_text(encoding="utf-8")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# ─────────────────────────────────────────
# STARTUP — Clickable URL print karo
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*55)
    print("  ⚡  caryanams Server READY!")
    print("  🌐  Browser mein kholo:")
    print("      http://localhost:8000")
    print("      http://127.0.0.1:8000")
    print("="*55 + "\n")
