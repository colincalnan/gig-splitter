# Gig Splitter

One command turns a full-length gig recording into a file per song, named by what was actually played, plus a 30-second captioned reel for each one.

I built it for my own solo acoustic gigs. A phone clipped to the speaker stand records the whole set, and this does the part nobody does twice: finding the good bits in two hours of footage.

**The full story, with a reel it cut and what it still gets wrong:** https://colincalnan.github.io/portfolio/builds/gig-splitter/

## How it works

1. **Find the songs by loudness.** ffmpeg pulls the audio out and librosa measures its energy, smoothed over 15 seconds. Anything above a threshold (default `0.38`) for more than 90 seconds is a song.
2. **Cut each song** with ffmpeg, stream-copied, no re-encode.
3. **Name each song from garbled lyrics.** Whisper transcribes a sample and Claude names the song, knowing the words will be wrong but the phrases close. Shazam was tried first and failed on every live cover.
4. **Find the chorus.** The reel starts just before the most repeated 4 to 8 word phrase, which is almost always the hook.
5. **Cut the reel.** 30 seconds at 1080x1920 with word-timed captions burned in.

## Run it

Needs macOS (the energy plot opens in Preview), Python 3.11, [ffmpeg](https://ffmpeg.org/) on your path, and the [Claude Code](https://claude.com/claude-code) CLI (`claude`) on your path for naming songs.

```
./setup.sh
venv/bin/python split_gig.py path/to/gig.mov [threshold] [--songs N] [--reels-only] [--reels N]
```

Output lands next to the video. Every step skips work already on disk, so a rerun picks up where it stopped.

## Known limits

- One threshold doesn't fit every room. If two songs come out welded together, rerun with a different threshold.
- "Stand By Me" comes out as a song called "Me" by "Ben E King Stand": the title parser splits on the first "by" in Claude's answer.
- No model looks at the picture yet. The reel window comes from the lyrics, not the video.

The longer write-up of how it was built is in [CASE_STUDY.md](CASE_STUDY.md).

Built with Claude Code. Colin Calnan, Lake Country, BC.
