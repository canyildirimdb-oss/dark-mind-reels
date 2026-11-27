"""Script, image, TTS, subtitle, and FFmpeg video generation for Dark Mind Reels."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import edge_tts
import requests
import yaml
from PIL import Image, ImageDraw

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


def generate_script_gemini(
    api_key: str,
    niche: str,
    scenes_count: int,
    used_topics: list[str],
    custom_topic: str = "",
) -> dict[str, Any]:
    """Ask Gemini 2.5 Flash for a dual-language (EN + TR) viral Reels script + visual prompts."""
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
4. Provide BOTH natural English (`narration_en`) and natural, fluent Turkish (`narration_tr` — NOT stiff literal translation; use natural spoken Turkish).
5. Each scene needs an `image_prompt` in English describing a dramatic, symbolic, moody scene with NO text in the image.

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

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    script = json.loads(raw_text)
    if len(script.get("scenes", [])) < 2:
        raise ValueError("Gemini returned fewer than 2 scenes.")
    return script


def generate_scene_image(
    account_id: str,
    cf_token: str,
    model: str,
    prompt: str,
    style_prompt: str,
    out_path: Path,
) -> None:
    """Generate a vertical 9:16 image via Cloudflare Workers AI (with fallback gradient)."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {cf_token}"}
    full_prompt = f"{prompt}, {style_prompt}"

    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"prompt": full_prompt, "num_steps": 6, "width": 768, "height": 1344},
            timeout=120,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            b64 = resp.json()["result"]["image"]
            out_path.write_bytes(base64.b64decode(b64))
        else:
            out_path.write_bytes(resp.content)
        return
    except Exception as exc:
        print(f"[warn] Cloudflare image failed ({exc}), generating fallback frame.", file=sys.stderr)
        img = Image.new("RGB", (768, 1344), (14, 14, 22))
        draw = ImageDraw.Draw(img)
        for y in range(1344):
            shade = int(14 + (y / 1344) * 26)
            draw.line([(0, y), (768, y)], fill=(shade, shade, shade + 12))
        img.save(out_path, "PNG")


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
Style: Default,DejaVu Sans,68,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,80,80,420,1

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


async def _synthesize_scene_audio(
    text: str, voice: str, rate: str, mp3_path: Path
) -> list[tuple[float, float, str]]:
    """Synthesize one scene via Edge-TTS and return word-level (start, end, word) timings."""
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
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _group_words_into_captions(
    words: list[tuple[float, float, str]],
    offset: float,
    scene_dur: float,
    fallback_text: str,
    max_words: int = 3,
) -> list[tuple[float, float, str]]:
    """Group word boundaries into punchy 2-3 word Reels captions."""
    if not words:
        parts = fallback_text.split()
        if not parts:
            return []
        groups = [" ".join(parts[i : i + max_words]) for i in range(0, len(parts), max_words)]
        step = scene_dur / len(groups)
        return [
            (offset + i * step, offset + (i + 1) * step, g.upper())
            for i, g in enumerate(groups)
        ]

    captions: list[tuple[float, float, str]] = []
    for i in range(0, len(words), max_words):
        batch = words[i : i + max_words]
        start = offset + batch[0][0]
        end = offset + batch[-1][1]
        if end <= start:
            end = start + 0.35
        phrase = " ".join(w[2] for w in batch).upper()
        captions.append((start, end, phrase))
    return captions


def render_language_video(
    lang_cfg: dict[str, Any],
    scenes: list[dict[str, Any]],
    scene_images: list[Path],
    work_dir: Path,
    final_mp4: Path,
    fps: int = 30,
) -> None:
    """Synthesize per-scene voiceovers, build ambient bed, and render 1080x1920 MP4."""
    code = lang_cfg["code"]
    voice = lang_cfg["voice"]
    rate = lang_cfg.get("rate", "+0%")
    lang_dir = work_dir / code
    lang_dir.mkdir(parents=True, exist_ok=True)

    scene_audios: list[Path] = []
    scene_durations: list[float] = []
    all_captions: list[tuple[float, float, str]] = []
    elapsed = 0.0

    for idx, scene in enumerate(scenes):
        text = scene[f"narration_{code}"].strip()
        mp3_path = lang_dir / f"scene_{idx:02d}.mp3"
        words = asyncio.run(_synthesize_scene_audio(text, voice, rate, mp3_path))
        dur = _probe_duration(mp3_path) + 0.18  # brief breathing pause between scenes
        all_captions.extend(_group_words_into_captions(words, elapsed, dur, text))
        scene_audios.append(mp3_path)
        scene_durations.append(dur)
        elapsed += dur

    ass_path = lang_dir / f"subtitles_{code}.ass"
    _write_ass(all_captions, ass_path)

    # Concatenate voiceovers with 0.18s silence pad per scene
    concat_list = lang_dir / "audio_concat.txt"
    padded_wavs: list[Path] = []
    for idx, (mp3, dur) in enumerate(zip(scene_audios, scene_durations)):
        wav = lang_dir / f"scene_{idx:02d}_pad.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp3),
                "-af", f"apad=whole_dur={dur:.3f}",
                "-ar", "44100", "-ac", "2", str(wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        padded_wavs.append(wav)
    concat_list.write_text(
        "\n".join(f"file '{w.as_posix()}'" for w in padded_wavs), encoding="utf-8"
    )
    voice_wav = lang_dir / "voice_full.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(voice_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Render each scene image with smooth Ken Burns slow zoom + vignette
    scene_clips: list[Path] = []
    for idx, (img_path, dur) in enumerate(zip(scene_images, scene_durations)):
        frames = max(1, int(round(dur * fps)))
        clip_path = lang_dir / f"clip_{idx:02d}.mp4"
        direction = "min(zoom+0.0009,1.12)" if idx % 2 == 0 else "if(eq(on,1),1.12,max(1.0,zoom-0.0009))"
        vf = (
            f"scale=1200:2133:force_original_aspect_ratio=increase,"
            f"crop=1200:2133,"
            f"zoompan=z='{direction}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s=1080x1920:fps={fps},"
            f"vignette=PI/4"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
                "-vf", vf, "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
                str(clip_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Generate subtle dark-ambient drone pad + burn subtitles + mix audio
    total_dur = sum(scene_durations)
    ass_escaped = ass_path.as_posix().replace(":", "\\:")
    filter_complex = (
        f"[0:v]ass='{ass_escaped}'[v];"
        f"aevalsrc=0.025*sin(2*PI*55*t)+0.015*sin(2*PI*82.4*t)+0.01*sin(2*PI*110*t):"
        f"s=44100:d={total_dur:.3f}[drone];"
        f"[drone]lowpass=f=240,afade=t=in:st=0:d=1.5,afade=t=out:st={max(0, total_dur-1.5):.2f}:d=1.5[bg];"
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
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    """Run full dual-language Reels production."""
    gemini_key = os.environ["GEMINI_API_KEY"]
    cf_account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    cf_token = os.environ["CLOUDFLARE_API_TOKEN"]
    custom_topic = os.environ.get("CUSTOM_TOPIC", "")

    cfg = load_config()
    history = load_history()

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

    print("2/4 Generating 9:16 visual frames with Cloudflare FLUX...")
    scene_images: list[Path] = []
    for scene in script["scenes"]:
        idx = int(scene["index"])
        img_path = scenes_dir / f"scene_{idx:02d}.png"
        generate_scene_image(
            account_id=cf_account,
            cf_token=cf_token,
            model=cfg["visuals"]["cf_model"],
            prompt=scene["image_prompt"],
            style_prompt=cfg["visuals"]["style_prompt"],
            out_path=img_path,
        )
        scene_images.append(img_path)
        print(f"    Frame {idx}/{len(script['scenes'])} ready")

    print("3/4 Rendering EN + TR vertical videos with FFmpeg...")
    work_dir = OUT_DIR / "work"
    for lang_cfg in cfg["production"]["languages"]:
        code = lang_cfg["code"]
        out_mp4 = OUT_DIR / f"dark_mind_{code.upper()}.mp4"
        render_language_video(
            lang_cfg=lang_cfg,
            scenes=script["scenes"],
            scene_images=scene_images,
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
