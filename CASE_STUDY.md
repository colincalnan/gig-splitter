# Case Study: From iPhone Recording to Instagram Reels with Claude Code

**Project:** gig-splitter  
**Built by:** Colin Calnan, Fractional AI Consultant  
**Time to working prototype:** One session  
**Stack:** Python, ffmpeg, OpenAI Whisper, Claude, librosa

---

## The Problem

I played a two-hour solo acoustic gig at Predator Ridge, BC. I recorded the whole thing on my iPhone. Now what?

Manually splitting a two-hour video into individual songs in iMovie: most of a Saturday.  
Identifying each song: more time.  
Creating vertical Instagram Reels from each song with captions: probably a week of evenings.

Or: describe the problem to Claude Code and let it build the tool.

---

## What We Built

A single Python script that takes the raw iPhone video and produces:

1. **Song detection** — automatically finds where each song starts and ends using audio energy analysis, no manual timestamps needed
2. **Song identification** — names each song (artist and title) using Whisper transcription + Claude
3. **Song extraction** — cuts each song as a clean, lossless video file
4. **Instagram Reels** — generates portrait-format (9:16) clips from the best 30 seconds of each song, with karaoke-style word-synced captions burned in

One command. Walk away. Come back to a `reels/` folder ready for Instagram.

```bash
venv/bin/python split_gig.py ~/Movies/Gigs/IMG_2659.MOV
```

---

## The Journey

This is the part worth documenting. It was not a straight line.

### Song Boundary Detection

The first instinct was frequency analysis — a live band has strong bass energy between songs. But this was solo acoustic guitar. No bass signal to separate songs from silence.

The second instinct was wrong in a different direction. The assumption was: crowd noise between songs is louder than playing. The actual data showed the opposite. My playing and singing is consistently higher energy than the ambient crowd. Songs are the loud parts.

Once that was inverted, RMS energy analysis worked. A 15-second smoothing window prevents quiet strums mid-song from triggering a false split. Anything above a tuned threshold and longer than 90 seconds is a song.

Calibration: check known timestamps against the energy plot, tune the threshold, run again. Three iterations to get it right.

### Song Identification

Shazam failed on every track. Audio fingerprinting works by matching a recording against the original studio version. A solo acoustic cover on an iPhone in a bar sounds nothing like the original. No match.

We skipped Shazam entirely. Whisper transcribes the audio — imperfect but phonetically close. "Standing in the hall of shame" instead of "Hall of Fame." Feed the garbled transcript to Claude with the right prompt: "These are lyrics from a live acoustic cover, transcribed by Whisper. Errors expected. Identify the song." Claude gets it right every time.

This is better than Shazam for live acoustic music. Not a workaround. The better approach.

### Reel Generation

Three problems to solve:

**Which 30 seconds?** Find the chorus — the most-repeated n-gram (4 to 8 words) in the transcription is almost always the hook. Locate its first occurrence in word-level timestamps and start the clip there. Fallback: 25% into the song to skip the intro.

**Portrait video polish?** The recording is already vertical (portrait iPhone). A single ffmpeg filter chain blurs a copy of the frame to fill the 9:16 background and overlays the original centered on top. The blurred background look is standard on music content. No new libraries needed.

**Captions?** ASS subtitle format supports karaoke timing via `\kf{}` tags. Whisper's `word_timestamps=True` gives a start and end time per word. Group into lines of five words, build the ASS file, burn it in with ffmpeg. Words highlight as they are sung, frame-accurate.

---

## Iteration: Making It Robust

The first working version ran once and produced output. Then the real work started: making it safe to re-run, restart, and extend.

### Resume Logic

Long processes get interrupted. Every step now checks before doing work:

| Step | Skip condition |
|---|---|
| Audio extraction | `gig_audio_full.mp3` exists and matches video duration |
| Song detection | `gig_songs.json` exists with matching threshold |
| Song extraction | `song_NN*` file already exists for that number |
| Song identification | File already renamed (`song_NN_Title.EXT`) or marked `song_NN_unknown.EXT` |
| Reel generation | `song_NN*_reel.mp4` exists in `reels/` |

Kill the process and restart: picks up exactly where it left off.

### The Caching Bug

After the resume logic was in place, detection kept stopping at 15 songs — even though the gig had 28. The threshold was tuned, detection re-ran, and still found only 15.

The problem: `gig_audio_full.mp3` had been extracted during an early test run when the code had a preview-duration limit. The file was cached. The resume logic saw it existed and skipped re-extraction. Detection was correctly analyzing a complete audio file — just one that only covered the first 63 minutes of a 2-hour gig.

The fix: compare the cached audio duration against the source video duration on every run. If the audio is more than 60 seconds shorter than the video, delete the cache and re-extract. One line of guard logic prevents an invisible data loss that would have been nearly impossible to diagnose without the duration check.

This is the kind of bug that AI-assisted development surfaces faster — because the iteration loop is short enough to notice "15 songs again, that's suspicious" and keep pulling the thread.

### Threshold Invalidation

Song detection results are cached to `gig_songs.json` so the 20-minute audio analysis never runs twice. But if you pass a different threshold on the CLI, the cache is now invalidated and detection re-runs. Previously it silently ignored the new threshold.

### Unknown Song Marking

Songs that Whisper and Claude can't identify (MC announcements, banter, corrupted audio) get renamed from `song_01.MOV` to `song_01_unknown.MOV`. On re-run they are skipped immediately. Previously a failed identification left the file with the plain name, making it look like it had never been attempted — so Whisper retranscribed it every time.

---

## Problems Solved Along the Way

| Problem | Cause | Fix |
|---|---|---|
| Only 15 of 28 songs found | Audio cache truncated from early test run | Duration check on cache load; auto-invalidate if short |
| Song detection stops at wrong threshold | Cache ignores new CLI threshold | Compare cached vs. current threshold; invalidate on mismatch |
| Shazam panics on MP3 | Rust divide-by-zero on certain encodings | Removed Shazam entirely — Whisper + Claude is the better path |
| Failed songs retranscribed every run | `song_01.MOV` looks like "never tried" | Rename to `song_01_unknown.MOV` after failed identification |
| Empty Whisper transcriptions | Absolute timestamps used on already-relative song files | Fixed to relative offsets |
| Song 1 catches pre-show banter | Banter is high-energy too | `looks_like_banter()` detection, retry at 60s offset |
| Whisper model loads even when nothing to do | Loaded at top of function before skip checks | Lazy-load: model only created when a song actually needs transcribing |
| `--reels N` always skipped on resume | Song list sliced before checking existing reels | Scan all songs first; apply limit only to newly created reels |

---

## The Stack

| Tool | Role |
|---|---|
| librosa | Audio analysis (RMS energy, song boundary detection) |
| ffmpeg | Audio extraction, song cutting, portrait conversion, caption burn |
| openai-whisper | Speech-to-text with word-level timestamps |
| Claude (via CLI) | Song identification from imperfect lyrics |
| Python 3.11 | Orchestration |

No cloud APIs except the Claude CLI call. Everything runs locally. A two-hour gig takes roughly 20-30 minutes to process.

---

## The Output

For the Predator Ridge May 8 gig:
- 28 songs detected and extracted
- 26 identified (2 were MC announcements, not songs)
- 26 portrait Reels generated: `song_03_The_Beatles_-_All_You_Need_Is_Love_reel.mp4` etc.
- Each reel: 30 seconds, 1080x1920, H.264, karaoke captions

Files go directly to iPhone via AirDrop, upload straight to Instagram Reels.

---

## What This Demonstrates

The bugs were real: inverted energy assumption, Shazam failing on acoustic covers, timestamp offsets, a silent audio truncation that took three days to surface. None were solvable by prompting. They required debugging, instrumentation, testing against known data, and iteration.

The AI accelerated all of it. Not by writing perfect code on the first try — by shortening the gap between "something is wrong" and "here is the fix" enough to keep the debugging loop moving.

The result is a real tool, running on real data, used on real gigs.

---

## I Build This for Clients

If you have a workflow that feels like it should be automatable, it probably is.

The gig-splitter took one session to get working and a few more days to make robust. The kind of things that typically take that long:

- Document processing pipelines (invoices, contracts, reports)
- Data extraction from unstructured sources
- Content generation pipelines tied to real business triggers
- Internal tools that replace repetitive manual steps

If you can describe what you do, I can usually describe what a tool would look like. If the description makes sense, we build it.

**Colin Calnan** — Fractional AI Consultant  
[colin.calnan@gmail.com](mailto:colin.calnan@gmail.com)  
[GitHub: colincalnan/gig-splitter](https://github.com/colincalnan/gig-splitter)
