# Case Study: From iPhone Recording to Instagram Reels with Claude Code

**Project:** gig-splitter  
**Built by:** Colin Calnan, Fractional AI Consultant  
**Time to working prototype:** One session  
**Stack:** Python, ffmpeg, OpenAI Whisper, Shazam, Claude, librosa

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
2. **Song identification** — names each song (artist and title) using a combination of Shazam, Whisper transcription, and Claude
3. **Song extraction** — cuts each song as a clean, lossless video file
4. **Instagram Reels** — generates portrait-format (9:16) clips from the best 30 seconds of each song, with karaoke-style word-synced captions burned in

One command. Walk away. Come back to a `reels/` folder ready for Instagram.

```bash
venv/bin/python split_gig.py ~/Movies/Gigs/IMG_2659.MOV
```

---

## The Journey

This is the part worth documenting. It was not a straight line.

### Step 1: Finding the Song Boundaries

The first instinct was frequency analysis. A live band has strong bass energy between songs. But this was solo acoustic guitar. No bass signal to distinguish.

The second instinct was wrong in a different way. The assumption was: crowd noise between songs is louder than playing. The actual data showed the opposite. My playing and singing is consistently higher energy than the ambient crowd. Songs are the loud parts.

Once that was inverted, RMS energy analysis worked cleanly. A 15-second smoothing window prevents quiet strums mid-song from registering as song boundaries. Anything above a tuned threshold and longer than 90 seconds is a song.

The calibration loop: check known timestamps against the energy plot, tune the threshold, run again. Three iterations to get it right.

### Step 2: Identifying the Songs

Shazam failed on every track. Audio fingerprinting works by matching a recording against original studio versions. A solo acoustic cover recorded on an iPhone in a bar sounds nothing like the original. No match.

First pivot: scrape lyrics from DuckDuckGo and Genius based on partial transcription. Partly worked. Not reliable enough.

Second pivot: Whisper transcribes the audio (medium model, 60 seconds from the middle of each song). The transcription is imperfect. "Standing in the hall of shame" instead of "Hall of Fame." But it is phonetically close.

Feed the imperfect transcript to Claude with the right prompt: "These are lyrics from a live acoustic cover, transcribed by Whisper. Errors expected. Identify the song." Claude gets it right every time.

This is better than Shazam for live acoustic music. Not a workaround. The better approach.

### Step 3: Generating the Reels

The reel generator needs to solve three things:

**Which 30 seconds?** Find the chorus. The most-repeated n-gram (4 to 8 words) in the transcription is almost always the hook. Locate the first occurrence of that phrase in word-level timestamps and start the clip there. If no chorus found, start at 25% into the song to skip the intro.

**Portrait conversion?** A single ffmpeg filter chain: split the landscape frame into two streams, blur one to 1080x1920 for the background, scale the other to fit the width, overlay centered. The blurred background look is standard on music content. No new libraries.

**Captions?** ASS subtitle format supports karaoke timing via `\kf{}` tags. Whisper's `word_timestamps=True` gives a start and end time per word. Group into lines of five words, build the ASS file, burn it in with ffmpeg. The result is words that highlight as they are sung, synchronized frame-accurate.

### Problems Solved Along the Way

| Problem | Cause | Fix |
|---|---|---|
| `llvmlite` won't install | Python 3.13 incompatible with torch deps | Used Python 3.11 venv |
| numpy 2.x breaks torch | API changes in numpy 2 | Pinned `numpy<2` |
| Shazam panics on MP3 | Rust divide-by-zero on certain MP3s | Switched samples to WAV |
| Empty Whisper transcriptions | Using absolute timestamps from original video on already-extracted songs | Fixed to relative offsets |
| Song 1 always catches pre-show banter | Banter is high-energy too | Added `looks_like_banter()` detection, retry at 60s offset |
| `--reels 1` never generates a reel | `songs_info_from_dir` sliced the song list before checking which had reels | Moved the limit inside `create_social_clips`, scan all songs first |

### Resume Capability

Long processes get interrupted. Both stages have resume logic:

- `extract_songs()` checks for existing `song_NN*` files before re-cutting
- `create_social_clips()` checks for existing `song_NN*_reel.mp4` before re-processing
- `--reels-only` mode skips song detection entirely for a second run

---

## The Stack

| Tool | Role |
|---|---|
| librosa | Audio analysis (RMS energy, song boundary detection) |
| ffmpeg | Audio extraction, song cutting, portrait conversion, caption burn |
| shazamio | Audio fingerprint matching (fails gracefully on acoustic covers) |
| openai-whisper | Speech-to-text transcription with word-level timestamps |
| Claude (via CLI) | Song identification from imperfect lyrics |
| Python 3.11 | Orchestration |

No cloud APIs except the Claude CLI call. Everything runs locally. A two-hour gig takes roughly 20-30 minutes to process depending on how many songs need Whisper identification.

---

## The Output

For a two-hour gig recording:
- A labeled song file per detected song: `song_03_Bryan_Adams_-_Summer_of_69.MOV`
- A `reels/` folder containing portrait MP4s: `song_03_Bryan_Adams_-_Summer_of_69_reel.mp4`
- Each reel: 30 seconds, 1080x1920, H.264, AAC audio, karaoke captions

The reel files go directly to iPhone via AirDrop and upload straight to Instagram Reels.

---

## What This Demonstrates

This project is a useful illustration of what AI-assisted development actually looks like in practice.

The problems were real: incompatible dependencies, wrong assumptions about which end of the audio was louder, a primary identification strategy that didn't work, timestamp bugs that produced silent clips. None of these were solvable by prompting. They required debugging, instrumentation, testing against known data, and iteration.

The AI accelerated all of it. Not by writing perfect code on the first try, but by:
- Knowing which libraries exist and how they fit together
- Suggesting pivots when the first approach failed
- Writing boilerplate fast enough that each iteration took minutes instead of hours
- Holding the full context of a 600-line script across a long debugging session

The result is a real tool that runs on real data and produces real output. Not a demo.

---

## I Build This for Clients

If you have a workflow that feels like it should be automatable, it probably is.

The gig-splitter took one session. The kind of things that typically take that long to build:

- Document processing pipelines (invoices, contracts, reports)
- Data extraction and transformation from unstructured sources
- Content generation pipelines tied to real business triggers
- Internal tools that replace repetitive manual steps

If you can describe what you do, I can usually describe what a tool would look like. If the description makes sense, we build it.

**Colin Calnan** — Fractional AI Consultant  
[colin.calnan@gmail.com](mailto:colin.calnan@gmail.com)  
[GitHub: colincalnan/gig-splitter](https://github.com/colincalnan/gig-splitter)
