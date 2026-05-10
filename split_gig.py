#!/usr/bin/env python3
"""
split_gig.py — detect song boundaries in a live gig recording.

Usage:
    python split_gig.py path/to/gig.mov

Dependencies:
    pip install librosa matplotlib numpy
    ffmpeg must be on PATH (brew install ffmpeg)

Phase 1: analyses the first 15 minutes and shows a plot of the energy curve
         so you can tune the threshold before committing to full splits.
"""

import sys
import os
import re
import json
import subprocess
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import whisper

PREVIEW_SECONDS  = None   # None = full file
MIN_SONG_SECONDS = 90    # ignore segments shorter than this
SMOOTH_SECONDS   = 15    # longer window merges quiet moments within songs
THRESHOLD        = 0.38  # songs are ABOVE this; pre-show silence and between-song gaps are below

# Known ground-truth timestamps for calibration (seconds). Edit as needed.
GROUND_TRUTH = {
    "song1_start": 228,  # 3:48
    "song1_end":   404,  # 6:44
    "song2_start": 455,  # 7:35
    "song2_end":   638,  # 10:38
}


def extract_audio(video_path: str, out_path: str, duration) -> None:
    label = f"first {duration//60} min of" if duration else "full"
    print(f"[1/4] Extracting {label} audio via ffmpeg...", flush=True)
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vn",
        "-acodec", "mp3", "-q:a", "5",  # lower quality = faster encode, fine for analysis
        "-ar", "11025",                  # downsample to 11kHz — enough for RMS analysis
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:\n", result.stderr[-2000:])
        sys.exit(1)
    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"    Audio saved: {out_path} ({size_mb:.1f} MB)", flush=True)


def detect_songs(audio_path: str, threshold: float = THRESHOLD):
    print("[2/4] Loading audio...", flush=True)
    y, sr = librosa.load(audio_path, sr=11025, mono=True)
    duration_min = len(y) / sr / 60
    print(f"    Loaded {duration_min:.1f} min of audio at {sr}Hz", flush=True)

    hop = sr // 4  # 0.25s per frame

    print("[3/4] Computing RMS energy...", flush=True)
    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=hop)[0]
    print(f"    RMS computed: {len(rms)} frames", flush=True)

    # Smooth to avoid brief applause spikes triggering false cuts
    smooth_frames = int(SMOOTH_SECONDS * sr / hop)
    smoothed = np.convolve(rms, np.ones(smooth_frames) / smooth_frames, mode="same")

    # Normalize to 0-1
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)

    times = librosa.frames_to_time(np.arange(len(normalized)), sr=sr, hop_length=hop)

    # Print energy at known timestamps + the following 10s to capture build-up
    print("\n    --- Calibration (energy at known timestamps) ---")
    for label, t in GROUND_TRUTH.items():
        samples = []
        for offset in range(0, 11, 2):  # 0s, 2s, 4s, 6s, 8s, 10s after timestamp
            frame_idx = int((t + offset) * sr / hop)
            if frame_idx < len(normalized):
                samples.append(f"+{offset:2d}s={normalized[frame_idx]:.3f}")
        print(f"    {label:15s} ({t//60}:{t%60:02d})  {' | '.join(samples)}", flush=True)
    print(f"\n    Threshold={threshold}  (songs = energy above this)")
    print(f"    Music = energy above {threshold}")
    print("    ---\n")

    # Songs are HIGH energy; pre-show silence and between-song gaps are LOW energy
    is_music = normalized > threshold

    # Remove very short music or crowd blips (< 10s)
    min_frames = int(10 * sr / hop)
    # Simple run-length clean-up
    is_music = _clean_signal(is_music, min_frames)

    transitions = np.diff(is_music.astype(int))
    starts_idx = np.where(transitions == 1)[0] + 1
    ends_idx   = np.where(transitions == -1)[0] + 1

    # Handle edge cases: starts at music / ends at music
    if is_music[0]:
        starts_idx = np.concatenate([[0], starts_idx])
    if is_music[-1]:
        ends_idx = np.concatenate([ends_idx, [len(is_music) - 1]])

    start_times = times[starts_idx]
    end_times   = times[ends_idx]

    # Filter by minimum song length
    songs = [
        (s, e) for s, e in zip(start_times, end_times)
        if (e - s) >= MIN_SONG_SECONDS
    ]

    return normalized, times, songs, threshold


def _clean_signal(signal: np.ndarray, min_run: int) -> np.ndarray:
    """Remove runs shorter than min_run frames by flipping them to their neighbour."""
    out = signal.copy()
    i = 0
    while i < len(out):
        val = out[i]
        j = i
        while j < len(out) and out[j] == val:
            j += 1
        run_len = j - i
        if run_len < min_run:
            fill = not val if i == 0 else out[i - 1]
            out[i:j] = fill
        i = j
    return out


def plot_results(normalized, times, songs, threshold, out_dir=None):
    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(times / 60, normalized, color="#888", linewidth=0.8, label="RMS energy (normalised)")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")

    for i, (s, e) in enumerate(songs):
        ax.axvspan(s / 60, e / 60, alpha=0.25, color="steelblue")
        ax.text((s + e) / 2 / 60, 0.05, f"Song {i+1}", ha="center", fontsize=8, color="steelblue")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Normalised RMS")
    ax.set_title("Gig energy — blue = detected songs (high energy), gaps = between songs / pre-show")
    patch = mpatches.Patch(color="steelblue", alpha=0.4, label="Detected song")
    ax.legend(handles=[ax.lines[0], ax.lines[1], patch])
    plt.tight_layout()
    plot_path = os.path.join(out_dir or os.getcwd(), "gig_energy_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[4/4] Plot saved to {plot_path}", flush=True)
    subprocess.Popen(["open", plot_path])  # open in Preview non-blocking


def print_results(songs):
    print(f"\n{'='*50}")
    print(f"Found {len(songs)} song(s):\n")
    for i, (s, e) in enumerate(songs):
        sm, ss = divmod(int(s), 60)
        em, es = divmod(int(e), 60)
        dur = int(e - s)
        print(f"  Song {i+1:>2}: {sm:02d}:{ss:02d} — {em:02d}:{es:02d}  ({dur}s / {dur//60}m{dur%60:02d}s)")
    print(f"\nTip: if splits look wrong, re-run with a different threshold:")
    print(f"     THRESHOLD = {THRESHOLD}  (lower = treat quieter sections as music)")


def extract_songs(video_path: str, songs: list, out_dir: str) -> list:
    """Cut detected songs from the original video using ffmpeg stream copy (no re-encode)."""
    ext = os.path.splitext(video_path)[1]
    out_paths = []

    print(f"\nExtracting {len(songs)} song(s) → {out_dir}")
    for i, (start, end) in enumerate(songs):
        song_num = i + 1
        # Resume: check for any existing file matching song_NN* (may already be renamed)
        existing = [
            f for f in os.listdir(out_dir)
            if re.match(rf"song_{song_num:02d}", f) and f.endswith(ext)
        ]
        if existing:
            existing_path = os.path.join(out_dir, existing[0])
            print(f"  Song {song_num}: already exists ({existing[0]}) — skipping", flush=True)
            out_paths.append((song_num, start, end, existing_path))
            continue

        out_path = os.path.join(out_dir, f"song_{song_num:02d}{ext}")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", out_path
        ]
        print(f"  Song {song_num}: {int(start//60):02d}:{int(start%60):02d} — {int(end//60):02d}:{int(end%60):02d}  →  {out_path}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            size_mb = os.path.getsize(out_path) / 1_000_000
            print(f"  Done ({size_mb:.0f} MB)", flush=True)
            out_paths.append((song_num, start, end, out_path))
    return out_paths



def find_chorus(transcript: str) -> str:
    """Find the most-repeated short phrase — almost always the chorus hook."""
    words = transcript.lower().split()
    best_phrase, best_score = "", 0
    for n in range(8, 3, -1):
        seen = {}
        for i in range(len(words) - n):
            phrase = " ".join(words[i:i+n])
            seen[phrase] = seen.get(phrase, 0) + 1
        for phrase, count in seen.items():
            score = count * len(phrase)
            if count > 1 and score > best_score:
                best_phrase, best_score = phrase, score
    return best_phrase


BANTER_PHRASES = ["my name is", "welcome", "i'll be here", "any requests", "thank you", "give it up", "how's everyone"]

def looks_like_banter(transcript: str) -> bool:
    t = transcript.lower()
    return any(p in t for p in BANTER_PHRASES) and len(transcript) < 300


def identify_with_claude(transcript: str, song_num: int):
    prompt = (
        "These are lyrics transcribed by Whisper from a live solo acoustic guitar cover. "
        "The transcription has errors — words may be wrong but the melody/phrases are close. "
        "Identify the song. Reply with ONLY this format: \"Song Title\" by Artist Name\n\n"
        f"Lyrics:\n{transcript}"
    )
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Claude error: {result.stderr[:200]}", flush=True)
        return None, None
    response = result.stdout.strip()
    m = re.match(r'^["\']?(.+?)["\']?\s+by\s+(.+)$', response, re.IGNORECASE)
    if m:
        return m.group(1).strip('"\''), m.group(2).strip()
    print(f"  Unexpected response: {response}", flush=True)
    return None, None


def identify_songs(songs_info: list) -> list:
    """Identify unrecognised songs via Whisper transcription + Claude. Returns updated songs_info."""
    path_map = {}
    needs_id = []
    for num, s, e, p in songs_info:
        base = os.path.splitext(os.path.basename(p))[0]
        if re.fullmatch(rf"song_{num:02d}", base):
            needs_id.append((num, s, e, p))
        elif re.fullmatch(rf"song_{num:02d}_unknown", base):
            print(f"  Song {num}: previously unidentifiable — skipping", flush=True)
            path_map[num] = p
        else:
            print(f"  Song {num}: already identified ({base}) — skipping", flush=True)
            path_map[num] = p

    if not needs_id:
        print("All songs already identified.", flush=True)
        return songs_info

    print(f"\nIdentifying {len(needs_id)} song(s) via Whisper + Claude...")
    print("(Whisper model cached at ~/.cache/whisper after first download)\n")
    model = whisper.load_model("medium")

    for song_num, start, end, video_path in needs_id:
        sample_path = f"/tmp/gig_whisper_{song_num}.wav"
        song_duration = end - start
        sample_start = min(40, song_duration * 0.3)

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(sample_start), "-t", "60",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            sample_path
        ], capture_output=True)

        size = os.path.getsize(sample_path) if os.path.exists(sample_path) else 0
        if size < 10000:
            print(f"  Song {song_num}: sample too small ({size} bytes) — skipping", flush=True)
            path_map[song_num] = video_path
            continue

        print(f"  Transcribing song {song_num} ({size//1000}KB sample)...", flush=True)
        result = model.transcribe(sample_path, language="en", fp16=False)
        transcript = result["text"].strip()
        if os.path.exists(sample_path):
            os.remove(sample_path)

        print(f"  Song {song_num} lyrics: \"{transcript}\"", flush=True)

        if looks_like_banter(transcript):
            print(f"  Detected banter — retrying at 60s offset...", flush=True)
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", "60", "-t", "60",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                sample_path
            ], capture_output=True)
            result = model.transcribe(sample_path, language="en", fp16=False)
            transcript = result["text"].strip()
            print(f"  Retry lyrics: \"{transcript}\"", flush=True)
            if os.path.exists(sample_path):
                os.remove(sample_path)

        if transcript:
            title, artist = identify_with_claude(transcript, song_num)
            if title:
                print(f"  Song {song_num}: \"{title}\" by {artist}", flush=True)
                safe = re.sub(r'[^\w\s-]', '', f"{artist} - {title}").strip()
                safe = re.sub(r'\s+', '_', safe)
                new_path = os.path.join(
                    os.path.dirname(video_path),
                    f"song_{song_num:02d}_{safe}{os.path.splitext(video_path)[1]}"
                )
                os.rename(video_path, new_path)
                print(f"  Renamed → {os.path.basename(new_path)}", flush=True)
                path_map[song_num] = new_path
                continue

        unknown_path = os.path.join(
            os.path.dirname(video_path),
            f"song_{song_num:02d}_unknown{os.path.splitext(video_path)[1]}"
        )
        os.rename(video_path, unknown_path)
        print(f"  Song {song_num}: could not identify — renamed to {os.path.basename(unknown_path)}", flush=True)
        path_map[song_num] = unknown_path

    return [(num, s, e, path_map.get(num, p)) for num, s, e, p in songs_info]


# ---------------------------------------------------------------------------
# Reel generation
# ---------------------------------------------------------------------------

def seconds_to_ass_time(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int(round((sec % 1) * 100))
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def find_clip_start(words_flat: list, song_duration: float, clip_length: int = 30) -> float:
    transcript = " ".join(w["word"] for w in words_flat)
    chorus = find_chorus(transcript)
    if chorus:
        chorus_tokens = chorus.lower().split()
        n = len(chorus_tokens)
        flat_lower = [w["word"].lower().strip(".,!?'\"- ") for w in words_flat]
        for i in range(len(flat_lower) - n):
            if flat_lower[i:i+n] == chorus_tokens:
                clip_start = max(0.0, words_flat[i]["start"] - 2)
                if clip_start + clip_length <= song_duration:
                    return clip_start
    return min(song_duration * 0.25, max(0.0, song_duration - clip_length))


def generate_ass(words_flat: list, clip_start: float, clip_length: int, output_path: str) -> None:
    clip_end = clip_start + clip_length
    in_window = [
        {"word": w["word"], "start": w["start"] - clip_start, "end": w["end"] - clip_start}
        for w in words_flat
        if w["start"] >= clip_start - 0.5 and w["end"] <= clip_end + 0.5
    ]

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial Black,72,&H00FFFFFF,&H00FFFF00,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,4,2,2,40,40,120,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogues = []
    for i in range(0, len(in_window), 5):
        chunk = in_window[i:i+5]
        if not chunk:
            continue
        line_start = max(0.0, chunk[0]["start"])
        line_end   = min(float(clip_length), chunk[-1]["end"])
        parts = []
        for w in chunk:
            cs = max(1, int(round((w["end"] - w["start"]) * 100)))
            parts.append(f"{{\\kf{cs}}}{w['word'].strip()}")
        dialogues.append(
            f"Dialogue: 0,{seconds_to_ass_time(line_start)},{seconds_to_ass_time(line_end)},"
            f"Default,,0,0,0,,{' '.join(parts)}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(dialogues))


def create_social_clips(songs_info: list, reels_dir: str, clip_length: int = 30, max_reels: int = None) -> None:
    print(f"\nCreating Instagram reels → {reels_dir}", flush=True)
    model = None  # loaded on first use, only if needed

    reels_made = 0
    for song_num, start, end, video_path in songs_info:
        if max_reels and reels_made >= max_reels:
            break
        if not os.path.exists(video_path):
            print(f"  Song {song_num}: file not found at {video_path}, skipping", flush=True)
            continue

        # Skip unidentified songs (plain song_NN.EXT) and known-unidentifiable (song_NN_unknown.EXT)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        if re.fullmatch(rf"song_{song_num:02d}(_unknown)?", basename):
            print(f"  Song {song_num}: not identified — skipping reel", flush=True)
            continue

        # Resume: skip if reel already exists for this song number
        existing_reel = next((
            f for f in os.listdir(reels_dir)
            if re.match(rf"song_{song_num:02d}", f) and f.endswith("_reel.mp4")
        ), None)
        if existing_reel:
            print(f"  Song {song_num}: reel already exists ({existing_reel}) — skipping", flush=True)
            continue

        song_duration = end - start
        base         = os.path.splitext(os.path.basename(video_path))[0]
        reel_path    = os.path.join(reels_dir, f"{base}_reel.mp4")
        tmp_portrait = f"/tmp/gig_portrait_{song_num}.mp4"
        tmp_ass      = f"/tmp/gig_captions_{song_num}.ass"
        tmp_wav      = f"/tmp/gig_song_{song_num}.wav"

        if model is None:
            print("Loading Whisper small model (cached after first use)...", flush=True)
            model = whisper.load_model("small")

        print(f"\n  [{song_num}] {os.path.basename(video_path)}", flush=True)

        # Transcribe a 90s window starting at 25% in — enough to catch the chorus
        # without processing the whole song
        sample_offset = min(song_duration * 0.25, song_duration - 90)
        sample_offset = max(0.0, sample_offset)
        print(f"    Transcribing 90s from {sample_offset:.0f}s for word timestamps...", flush=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(sample_offset), "-t", "90",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav
        ], capture_output=True)
        result = model.transcribe(tmp_wav, language="en", fp16=False, word_timestamps=True)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

        # Adjust word timestamps back to be relative to the song file start
        words_flat = [
            {"word": w["word"], "start": w["start"] + sample_offset, "end": w["end"] + sample_offset}
            for seg in result["segments"]
            for w in seg.get("words", [])
        ]
        print(f"    {len(words_flat)} words transcribed", flush=True)

        # Find the best 30-second window (chorus-first, then fallback)
        clip_start = find_clip_start(words_flat, song_duration, clip_length)
        print(f"    Clip: {clip_start:.0f}s – {clip_start+clip_length:.0f}s", flush=True)

        # Portrait conversion: blurred background + centered original
        print(f"    Converting to 9:16 portrait...", flush=True)
        r = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(clip_start), "-t", str(clip_length),
            "-i", video_path,
            "-vf", (
                "split[a][b];"
                "[a]scale=1080:1920,boxblur=20:5[bg];"
                "[b]scale=1080:-2[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            tmp_portrait
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Portrait error: {r.stderr[-400:]}", flush=True)
            continue

        # ASS subtitle file with karaoke word timing
        generate_ass(words_flat, clip_start, clip_length, tmp_ass)

        # Burn captions into portrait video
        print(f"    Burning captions...", flush=True)
        r = subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_portrait,
            "-vf", f"ass={tmp_ass}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            reel_path
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Caption error: {r.stderr[-400:]}", flush=True)
        else:
            size_mb = os.path.getsize(reel_path) / 1_000_000
            print(f"    Saved: {reel_path} ({size_mb:.0f} MB)", flush=True)
            reels_made += 1

        for tmp in [tmp_portrait, tmp_ass]:
            if os.path.exists(tmp):
                os.remove(tmp)


def get_video_duration(video_path: str) -> float:
    """Return duration in seconds via ffprobe."""
    r = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], capture_output=True, text=True)
    return float(r.stdout.strip())


def songs_info_from_dir(out_dir: str, ext: str) -> list:
    """Scan out_dir for identified song files (song_NN_<title>.EXT) and build songs_info."""
    pattern = re.compile(rf"^song_(\d+)_.+\{ext}$", re.IGNORECASE)
    matches = []
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m:
            matches.append((int(m.group(1)), os.path.join(out_dir, f)))
    matches.sort(key=lambda x: x[0])

    songs_info = []
    for song_num, path in matches:
        duration = get_video_duration(path)
        songs_info.append((song_num, 0.0, duration, path))
    return songs_info


def main():
    if len(sys.argv) < 2:
        print("Usage: python split_gig.py path/to/gig.mov [threshold] [--songs N] [--reels-only] [--reels N]")
        sys.exit(1)

    video_path  = sys.argv[1]
    threshold   = THRESHOLD
    max_songs   = None
    reels_only  = False
    max_reels   = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] in ("--songs", "-n") and i + 1 < len(args):
            max_songs = int(args[i + 1]); i += 2
        elif args[i] == "--reels-only":
            reels_only = True; i += 1
        elif args[i] == "--reels":
            if i + 1 < len(args) and args[i + 1].lstrip('-').isdigit():
                max_reels = int(args[i + 1]); i += 2
            else:
                print("Usage: --reels N  (N is a number)"); sys.exit(1)
        elif args[i].startswith("-"):
            print(f"Unknown option: {args[i]}"); sys.exit(1)
        else:
            threshold = float(args[i]); i += 1

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    out_dir   = os.path.dirname(os.path.abspath(video_path))
    reels_dir = os.path.join(out_dir, "reels")
    os.makedirs(reels_dir, exist_ok=True)
    ext       = os.path.splitext(video_path)[1]

    if reels_only:
        print(f"Reels-only mode — scanning {out_dir} for identified songs...", flush=True)
        songs_info = songs_info_from_dir(out_dir, ext)
        if not songs_info:
            print("No identified song files found. Run without --reels-only first.")
            sys.exit(1)
        print(f"Found {len(songs_info)} identified song(s):", flush=True)
        for num, s, e, path in songs_info:
            print(f"  {num}: {os.path.basename(path)} ({e:.0f}s)", flush=True)
        create_social_clips(songs_info, reels_dir, max_reels=max_reels)
        return

    audio_path     = os.path.join(out_dir, "gig_audio_full.mp3")
    songs_json     = os.path.join(out_dir, "gig_songs.json")
    plot_path      = os.path.join(out_dir, "gig_energy_plot.png")

    video_duration = get_video_duration(video_path)

    if os.path.exists(audio_path):
        audio_duration = get_video_duration(audio_path)
        if video_duration - audio_duration > 60:
            print(f"[1/4] Cached audio is {(video_duration - audio_duration)/60:.1f} min shorter than video — re-extracting", flush=True)
            os.remove(audio_path)
            if os.path.exists(songs_json):
                os.remove(songs_json)
            extract_audio(video_path, audio_path, PREVIEW_SECONDS)
        else:
            print(f"[1/4] Audio already extracted — skipping ({os.path.basename(audio_path)})", flush=True)
    else:
        extract_audio(video_path, audio_path, PREVIEW_SECONDS)

    cached_threshold = None
    if os.path.exists(songs_json):
        with open(songs_json) as f:
            data = json.load(f)
        cached_threshold = data["threshold"]

    if cached_threshold is not None and cached_threshold == threshold:
        songs = [tuple(s) for s in data["songs"]]
        print(f"[2-3/4] Song timestamps loaded from cache ({len(songs)} songs, threshold={cached_threshold}) — skipping detection", flush=True)
        if os.path.exists(plot_path):
            print(f"        Plot already exists — skipping", flush=True)
    else:
        if cached_threshold is not None and cached_threshold != threshold:
            print(f"[2-3/4] Threshold changed ({cached_threshold} → {threshold}) — re-running detection", flush=True)
        normalized, times, songs, threshold = detect_songs(audio_path, threshold)
        with open(songs_json, "w") as f:
            json.dump({"threshold": threshold, "songs": [list(s) for s in songs]}, f)
        plot_results(normalized, times, songs, threshold, out_dir)

    if max_songs:
        print(f"Limiting to first {max_songs} song(s)", flush=True)
        songs = songs[:max_songs]
    print_results(songs)
    songs_info = extract_songs(video_path, songs, out_dir)

    # Pick up any song files already on disk that aren't in this detection run
    # (e.g. extracted in a previous run with different --songs limit)
    known_nums = {num for num, *_ in songs_info}
    pattern = re.compile(rf"^song_(\d+).*\{re.escape(ext)}$", re.IGNORECASE)
    extras = []
    for f in sorted(os.listdir(out_dir)):
        m = pattern.match(f)
        if m:
            num = int(m.group(1))
            if num not in known_nums:
                path = os.path.join(out_dir, f)
                duration = get_video_duration(path)
                extras.append((num, 0.0, duration, path))
                print(f"  Found extra song on disk not in detection run: {f}", flush=True)
    if extras:
        songs_info = sorted(songs_info + extras, key=lambda x: x[0])

    songs_info = identify_songs(songs_info)
    create_social_clips(songs_info, reels_dir, max_reels=max_reels)


if __name__ == "__main__":
    main()
