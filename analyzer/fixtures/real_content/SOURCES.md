# Real-content fixtures (never committed)

This directory holds real (non-synthetic) video used to validate the
face detector and VAD against genuine human faces and speech, which the
synthetic fixture generator (`syncsentry.synth.fixture_gen`) cannot
provide by design.

**Nothing in this directory except this file is committed to the repo.**
`analyzer/scripts/fetch_real_content.sh` reproducibly re-fetches it.

## Current source

- **Clip:** "Interview with biologist Dr Robyn Grant - Five things you
  never knew about whiskers, The Royal Society"
- **Author:** The Royal Society
- **Source:** [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Interview_with_biologist_Dr_Robyn_Grant_-_Five_things_you_never_knew_about_whiskers_%E2%80%93_The_Royal_Society.webm)
  (originally published on [YouTube](https://www.youtube.com/watch?v=q3h2CRhie3g))
- **License:** [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/deed.en),
  attribution required for any redistribution of the media itself. This
  project does not redistribute the media, only the fetch script.
- **Why this clip:** a real human face on screen for most of its
  runtime, real continuous speech, small file size, unambiguous license.
- **Local fixture used by tests:** `dialogue_clip.mkv`, a 50s trim of a
  continuous talking-head segment, re-encoded to H.264/PCM for
  consistent decoding. Regenerate with:

  ```bash
  bash analyzer/scripts/fetch_real_content.sh
  ```
