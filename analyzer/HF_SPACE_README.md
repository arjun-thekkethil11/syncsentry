---
title: SyncSentry API
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SyncSentry API

Backend for [SyncSentry](https://github.com/arjun-thekkethil11/syncsentry): detects and fixes
audio/video sync drift using lip motion and speech, no captions or OCR.

`GET /healthz` for status. `POST /v1/fix` to upload a video and get a corrected one back.
See the main repo's README for details.
