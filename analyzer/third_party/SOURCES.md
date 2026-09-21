# Vendored third-party code (never committed)

This whole directory is gitignored. Re-create it with:

```bash
bash analyzer/scripts/fetch_syncnet.sh
bash analyzer/scripts/fetch_light_asd.sh
bash analyzer/scripts/fetch_mtdvocalist.sh
```

## `syncnet_python/`

- **Source:** https://github.com/joonson/syncnet_python
- **License:** MIT.
- **Paper:** Chung, J.S. and Zisserman, A., "Out of time: automated lip
  sync in the wild", Workshop on Multi-view Lip-reading, ACCV 2016.
- **Weights:** `data/syncnet_v2.model` (SyncNet, trained on VoxCeleb2) and
  `detectors/s3fd/weights/sfd_face.pth` (S3FD face detector), fetched
  from the `lithiumice/syncnet` mirror on Hugging Face, since the
  original `robots.ox.ac.uk` host has an expired TLS certificate.

Used by `syncsentry/lipsync/syncnet_offset.py` as the learned lip-sync
estimator, invoked as an external tool (subprocess for face
detection/tracking/crop, direct import for SyncNet inference) rather
than ported into this project's own code.

## `light_asd/`

- **Source:** https://github.com/Junhua-Liao/Light-ASD
- **License:** MIT.
- **Paper:** Liao, J. et al., "A Light Weight Model for Active Speaker
  Detection", CVPR 2023.
- **Weights:** `weight/pretrain_AVA_CVPR.model`, trained on
  AVA-ActiveSpeaker, ships directly in the repo.

Used by `syncsentry/lipsync/active_speaker.py` as a multimodal
(audio + video) active-speaker classifier: an upgrade over a
mouth-motion-only gate, which can only tell if a face's mouth is moving,
not whether that face is the source of the audio right now. Only
`model/Model.py`'s `ASD_Model` and `loss.py`'s `lossAV` are imported
directly; this project's own wrapper handles device placement instead of
the upstream wrapper class, which hardcodes `.cuda()`. Reuses SyncNet's
own face-track crops instead of running a second face-detection pass,
since both use the same 224x224 at 25fps, 16kHz mono audio convention.

## `mtdvocalist/`

- **Source:** https://github.com/xjchenGit/MTDVocaLiST (a distilled,
  smaller version of https://github.com/vskadandale/vocalist).
- **License:** ships no license file of its own, and is built on
  VocaLiST (CC-BY-NC, non-commercial) and Wav2Lip (non-commercial) code.
  Treat this as non-commercial research use only.
- **Papers:** Chen, X. et al., "Multimodal Transformer Distillation for
  Audio-Visual Synchronization", ICASSP 2024; Kadandale, V.S. et al.,
  "VocaLiST: An Audio-Visual Synchronisation Model for Lips and Voices",
  Interspeech 2022.
- **Weights:** `pretrained/pure_MTDVocaLiST.pth`, fetched from a GitHub
  Releases asset.

Used by `syncsentry/lipsync/mtdvocalist_offset.py` as a second,
independent sync scorer over the same face-track crops SyncNet already
produced. Chosen because its published benchmark is much better than
SyncNet's at short context lengths (5 frames: 92.8% vs 75.8% on LRS2),
which matters for short or sparse clips. This model expects Wav2Lip's
tight, unpadded face-bbox crop convention rather than SyncNet's padded
one, so `mtdvocalist_offset.py` re-derives the tight region from
SyncNet's crop instead of running a second face-detection pass.
