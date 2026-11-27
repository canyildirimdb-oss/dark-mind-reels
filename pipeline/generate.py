"""Script, image, expressive Gemini TTS, subtitle, and FFmpeg video generation for Dark Mind Reels."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import edge_tts
import requests
import yaml
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
HISTORY_PATH = ROOT / "state" / "history.json"
OUT_DIR = ROOT / "output"


def load_config() -> dict[str, Any]:
    """Load pipeline configuration from config.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_history() -> dict[str, Any]:
    """Load previously used topics so we never repeat."""
    if not HISTORY_PATH.exists():
        return {"used_topics": []}
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: dict[str, Any]) -> None:
    """Persist used topics back to state/history.json."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _discover_models(api_key: str, keyword: str) -> list[str]:
    """Discover available Gemini models matching a keyword."""
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=30,
        )
        if not resp.ok:
            return []
        return [
            m["name"].removeprefix("models/")
            for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and keyword in m.get("name", "").lower()
        ]
    except Exception:
        return []


def generate_script_gemini(
    api_key: str,
    niche: str,
    scenes_count: int,
    used_topics: list[str],
    custom_topic: str = "",
) -> dict[str, Any]:
    """Ask Gemini for a dual-language (EN + TR) viral Reels script + visual prompts."""
    avoid = "\n".join(f"- {t}" for t in used_topics[-30:]) or "- (none)"
    topic_instruction = (
        f"Use this exact topic: {custom_topic}"
        if custom_topic.strip()
        else "Invent ONE compelling, lesser-known psychological effect, cognitive bias, or Stoic mental trick."
    )

    prompt = f"""You are an elite viral short-form scriptwriter for YouTube Shorts, TikTok, and Instagram Reels.
Niche: {niche}
{topic_instruction}

Do NOT repeat any of these previous topics:
{avoid}

Create a tight {scenes_count}-scene vertical video script (40-55 seconds total).
Rules:
1. Scene 1 MUST be a shocking, curiosity-driven 1-sentence hook (first 2.5 seconds).
2. Scenes 2-{scenes_count-1} deliver punchy, insightful psychological or Stoic explanation with real-life examples.
3. Scene {scenes_count} delivers a memorable closing punchline / call-to-save.
4. Provide BOTH natural English (`narration_en`) and natural, fluent Turkish (`narration_tr` — spoken, dramatic, emotional Turkish; NOT stiff literal translation).
5. Each scene needs an `image_prompt` in English describing a vivid, dramatic, symbolic scene with clear subject lighting and NO text in the image.

Return ONLY valid JSON matching this schema:
{{
  "topic_id": "short-kebab-case-slug",
  "title_en": "Catchy English title",
  "title_tr": "Çarpıcı Türkçe başlık",
  "caption_en": "Short engaging caption with 4 hashtags",
  "caption_tr": "Kısa etkileyici açıklama ve 4 hashtag",
  "scenes": [
    {{
      "index": 1,
      "narration_en": "...",
      "narration_tr": "...",
      "image_prompt": "..."
    }}
  ]
}}"""

    candidates = [
        m
        for m in _discover_models(api_key, "flash")
        if "tts" not in m.lower() and "image" not in m.lower()
    ]
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]:
        if fallback not in candidates:
            candidates.append(fallback)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json",
        },
    }

    data = None
    last_err: Exception | None = None
    for model_name in candidates:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code in (404, 429, 503):
                print(f"[info] Text model {model_name} -> {resp.status_code}, trying next...", file=sys.stderr)
                continue
            resp.raise_for_status()
            data = resp.json()
            print(f"    Using Gemini text model: {model_name}")
            break
        except Exception as exc:
            last_err = exc
    if data is None:
        raise RuntimeError(f"All Gemini text models failed. Last error: {last_err}")

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    script = json.loads(raw_text)
    if len(script.get("scenes", [])) < 2:
        raise ValueError("Gemini returned fewer than 2 scenes.")
    return script


def generate_scene_image(
    account_id: str,
    cf_token: str,
    cf_models: list[str],
    gemini_client: genai.Client,
    prompt: str,
    style_prompt: str,
    out_path: Path,
) -> None:
    """Generate a vivid image via Cloudflare Workers AI (matching CF schemas) or Gemini Image."""
    headers = {"Authorization": f"Bearer {cf_token}"}
    full_prompt = f"{prompt}, {style_prompt}"

    for model in cf_models:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        try:
            if "flux-2" in model:
                fields = {"prompt": (None, full_prompt), "width": (None, "768"), "height": (None, "1344")}
                resp = requests.post(url, headers=headers, files=fields, timeout=120)
                if resp.status_code == 400:
                    resp = requests.post(url, headers=headers, files={"prompt": (None, full_prompt)}, timeout=120)
            elif "flux" in model:
                # flux-1-schnell accepts prompt + steps (rejects width/height/num_steps)
                resp = requests.post(url, headers=headers, json={"prompt": full_prompt, "steps": 6}, timeout=120)
                if resp.status_code == 400:
                    resp = requests.post(url, headers=headers, json={"prompt": full_prompt}, timeout=120)
            else:
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"prompt": full_prompt, "width": 768, "height": 1344},
                    timeout=120,
                )

            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                data = resp.json()
                b64 = (data.get("result") or {}).get("image") or data.get("image")
                if not b64:
                    raise ValueError(f"No image field in response: {str(data)[:200]}")
                out_path.write_bytes(base64.b64decode(b64))
            else:
                if len(resp.content) < 5000:
                    raise ValueError(f"Image payload too small ({len(resp.content)} bytes)")
                out_path.write_bytes(resp.content)
            return
        except Exception as exc:
            print(f"[warn] Cloudflare {model} failed ({exc}), trying next...", file=sys.stderr)

    # Fallback: Gemini native image generation
    for gem_img_model in ["gemini-3.1-flash-lite-image", "gemini-2.5-flash-image", "gemini-2.0-flash-exp-image-generation"]:
        try:
            response = gemini_client.models.generate_content(
                model=gem_img_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="9:16"),
                ),
            )
            for part in response.parts or []:
                if part.inline_data and part.inline_data.data:
                    out_path.write_bytes(part.inline_data.data)
                    print(f"    Generated with Gemini image ({gem_img_model})")
                    return
        except Exception as exc:
            print(f"[warn] Gemini image {gem_img_model} failed ({exc})", file=sys.stderr)

    raise RuntimeError("Both Cloudflare and Gemini image generation failed for scene.")


def _fmt_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cs."""
    cs = max(0, int(round(seconds * 100)))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_ass(chunks: list[tuple[float, float, str]], out_path: Path) -> None:
    """Write styled viral-Reels ASS subtitle file (1080x1920)."""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,70,&H0000E5FF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,6,2,2,80,80,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for start, end, text in chunks:
        clean = text.replace("\n", " ").replace("{", "").replace("}", "").strip()
        if not clean:
            continue
        lines.append(
            f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,0,0,0,,{clean}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _synthesize_gemini_tts(
    client: genai.Client,
    tts_models: list[str],
    text: str,
    voice_name: str,
    lang_code: str,
    out_wav: Path,
) -> bool:
    """Synthesize expressive, emotional speech via Gemini Native TTS."""
    style_dir = (
        "Read the following in a deep, intense, suspenseful, emotionally gripping documentary voice "
        "with natural dramatic pauses and emphasis. Output ONLY the spoken words:"
        if lang_code == "en"
        else "Aşağıdaki metni derin, etkileyici, gizemli ve duygulu bir belgesel anlatıcısı tonuyla, "
        "doğal duraklamalar ve vurgularla Türkçe olarak seslendir. Sadece metni oku:"
    )
    prompt = f"{style_dir}\n\n{text}"

    for model_name in tts_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                        )
                    ),
                ),
            )
            for part in response.parts or []:
                if part.inline_data and part.inline_data.data:
                    raw_bytes = part.inline_data.data
                    raw_path = out_wav.with_suffix(".raw")
                    raw_path.write_bytes(raw_bytes)
                    if raw_bytes[:4] == b"RIFF":
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "44100", "-ac", "2", str(out_wav)],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    else:
                        # Gemini TTS raw PCM is 24kHz 16-bit mono little-endian
                        subprocess.run(
                            [
                                "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                                "-i", str(raw_path), "-ar", "44100", "-ac", "2", str(out_wav),
                            ],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    raw_path.unlink(missing_ok=True)
                    return True
        except Exception as exc:
            print(f"[warn] Gemini TTS {model_name} failed ({exc})", file=sys.stderr)
            time.sleep(1.5)
    return False


async def _synthesize_edge_tts(
    text: str, voice: str, rate: str, mp3_path: Path
) -> list[tuple[float, float, str]]:
    """Fallback synthesizer using Edge-TTS with word boundaries."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, boundary="WordBoundary")
    words: list[tuple[float, float, str]] = []
    with mp3_path.open("wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 10_000_000.0
                dur = chunk["duration"] / 10_000_000.0
                words.append((start, start + dur, chunk["text"]))
    return words


def _probe_duration(media_path: Path) -> float:
    """Get exact media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media_path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def _group_words_into_captions(
    words: list[tuple[float, float, str]],
    offset: float,
    speech_dur: float,
    fallback_text: str,
    max_words: int = 3,
) -> list[tuple[float, float, str]]:
    """Group word boundaries or character-weighted text into punchy 2-3 word Reels captions."""
    if not words:
        parts = fallback_text.split()
        if not parts:
            return []
        groups = [" ".join(parts[i : i + max_words]) for i in range(0, len(parts), max_words)]
        total_chars = sum(max(1, len(g)) for g in groups)
        captions: list[tuple[float, float, str]] = []
        cursor = offset
        for g in groups:
            g_dur = speech_dur * (max(1, len(g)) / total_chars)
            captions.append((cursor, cursor + g_dur, g.upper()))
            cursor += g_dur
        return captions

    captions = []
    for i in range(0, len(words), max_words):
        batch = words[i : i + max_words]
        start = offset + batch[0][0]
        end = offset + batch[-1][1]
        if end <= start:
            end = start + 0.35
        captions.append((start, end, " ".join(w[2] for w in batch).upper()))
    return captions


def render_language_video(
    lang_cfg: dict[str, Any],
    scenes: list[dict[str, Any]],
    scene_images: list[Path],
    gemini_client: genai.Client,
    tts_models: list[str],
    work_dir: Path,
    final_mp4: Path,
    fps: int = 30,
) -> None:
    """Synthesize expressive per-scene voiceovers, build ambient bed, and render 1080x1920 MP4."""
    code = lang_cfg["code"]
    gemini_voice = lang_cfg.get("gemini_voice", "Charon")
    edge_voice = lang_cfg["edge_voice"]
    rate = lang_cfg.get("rate", "+4%")
    lang_dir = work_dir / code
    lang_dir.mkdir(parents=True, exist_ok=True)

    padded_wavs: list[Path] = []
    scene_durations: list[float] = []
    all_captions: list[tuple[float, float, str]] = []
    elapsed = 0.0

    for idx, scene in enumerate(scenes):
        text = scene[f"narration_{code}"].strip()
        raw_wav = lang_dir / f"scene_{idx:02d}.wav"
        words: list[tuple[float, float, str]] = []

        used_gemini = _synthesize_gemini_tts(
            gemini_client, tts_models, text, gemini_voice, code, raw_wav
        )
        if not used_gemini:
            mp3_path = lang_dir / f"scene_{idx:02d}.mp3"
            words = asyncio.run(_synthesize_edge_tts(text, edge_voice, rate, mp3_path))
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "44100", "-ac", "2", str(raw_wav)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        speech_dur = _probe_duration(raw_wav)
        total_scene_dur = speech_dur + 0.22
        all_captions.extend(_group_words_into_captions(words, elapsed, speech_dur, text))

        pad_wav = lang_dir / f"scene_{idx:02d}_pad.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_wav),
                "-af", f"apad=whole_dur={total_scene_dur:.3f}",
                "-ar", "44100", "-ac", "2", str(pad_wav),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        padded_wavs.append(pad_wav)
        scene_durations.append(total_scene_dur)
        elapsed += total_scene_dur

    ass_path = lang_dir / f"subtitles_{code}.ass"
    _write_ass(all_captions, ass_path)

    concat_list = lang_dir / "audio_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{w.as_posix()}'" for w in padded_wavs), encoding="utf-8"
    )
    voice_wav = lang_dir / "voice_full.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(voice_wav),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    scene_clips: list[Path] = []
    for idx, (img_path, dur) in enumerate(zip(scene_images, scene_durations)):
        frames = max(1, int(round(dur * fps)))
        clip_path = lang_dir / f"clip_{idx:02d}.mp4"
        direction = "min(zoom+0.0008,1.10)" if idx % 2 == 0 else "if(eq(on,1),1.10,max(1.0,zoom-0.0008))"
        vf = (
            f"scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,"
            f"zoompan=z='{direction}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s=1080x1920:fps={fps},"
            f"eq=contrast=1.06:brightness=0.02:saturation=1.12"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
                "-vf", vf, "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
                str(clip_path),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        scene_clips.append(clip_path)

    video_concat = lang_dir / "video_concat.txt"
    video_concat.write_text(
        "\n".join(f"file '{c.as_posix()}'" for c in scene_clips), encoding="utf-8"
    )
    raw_video = lang_dir / "video_raw.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(video_concat), "-c", "copy", str(raw_video),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    total_dur = sum(scene_durations)
    ass_escaped = ass_path.as_posix().replace(":", "\\:")
    filter_complex = (
        f"[0:v]ass='{ass_escaped}'[v];"
        f"aevalsrc=0.022*sin(2*PI*55*t)+0.014*sin(2*PI*82.4*t)+0.009*sin(2*PI*110*t):"
        f"s=44100:d={total_dur:.3f}[drone];"
        f"[drone]lowpass=f=220,afade=t=in:st=0:d=1.2,afade=t=out:st={max(0, total_dur-1.2):.2f}:d=1.2[bg];"
        f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[vo];"
        f"[vo][bg]amix=inputs=2:duration=first:dropout_transition=2[a]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-i", str(voice_wav),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(final_mp4),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main() -> None:
    """Run full dual-language Reels production."""
    gemini_key = os.environ["GEMINI_API_KEY"]
    cf_account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    cf_token = os.environ["CLOUDFLARE_API_TOKEN"]
    custom_topic = os.environ.get("CUSTOM_TOPIC", "")

    cfg = load_config()
    history = load_history()
    gemini_client = genai.Client(api_key=gemini_key)

    tts_models = _discover_models(gemini_key, "tts")
    for preferred in ["gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview", "gemini-3.8-flash-tts"]:
        if preferred not in tts_models:
            tts_models.append(preferred)
    print(f"Discovered Gemini TTS candidates: {tts_models}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes_dir = OUT_DIR / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    print("1/4 Generating dual-language script with Gemini...")
    script = generate_script_gemini(
        api_key=gemini_key,
        niche=cfg["channel"]["niche"],
        scenes_count=cfg["production"]["scenes_count"],
        used_topics=history.get("used_topics", []),
        custom_topic=custom_topic,
    )
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    Topic: {script['title_en']} / {script['title_tr']}")

    print("2/4 Generating 9:16 visual frames (Cloudflare FLUX / Gemini Image)...")
    scene_images: list[Path] = []
    for scene in script["scenes"]:
        idx = int(scene["index"])
        img_path = scenes_dir / f"scene_{idx:02d}.png"
        generate_scene_image(
            account_id=cf_account,
            cf_token=cf_token,
            cf_models=cfg["visuals"]["cf_models"],
            gemini_client=gemini_client,
            prompt=scene["image_prompt"],
            style_prompt=cfg["visuals"]["style_prompt"],
            out_path=img_path,
        )
        scene_images.append(img_path)
        print(f"    Frame {idx}/{len(script['scenes'])} ready ({img_path.stat().st_size} bytes)")

    print("3/4 Rendering EN + TR vertical videos with FFmpeg...")
    work_dir = OUT_DIR / "work"
    for lang_cfg in cfg["production"]["languages"]:
        code = lang_cfg["code"]
        out_mp4 = OUT_DIR / f"dark_mind_{code.upper()}.mp4"
        render_language_video(
            lang_cfg=lang_cfg,
            scenes=script["scenes"],
            scene_images=scene_images,
            gemini_client=gemini_client,
            tts_models=tts_models,
            work_dir=work_dir,
            final_mp4=out_mp4,
            fps=cfg["production"]["fps"],
        )
        print(f"    Rendered {out_mp4.name}")

    print("4/4 Updating topic history...")
    topic_entry = f"{script['topic_id']} ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})"
    history.setdefault("used_topics", []).append(topic_entry)
    save_history(history)
    print("Done!")


if __name__ == "__main__":
    main()
