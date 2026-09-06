# Vendored third-party code (never committed)

This whole directory is gitignored. Re-create it with:

```bash
bash analyzer/scripts/fetch_syncnet.sh
```

## `syncnet_python/`

- **Source:** https://github.com/joonson/syncnet_python
- **License:** MIT (`LICENSE.md` in that repo) -- code is freely
  usable/modifiable, unlike the real-content video clip elsewhere in this
  project.
- **Paper:** Chung, J.S. and Zisserman, A., "Out of time: automated lip
  sync in the wild", Workshop on Multi-view Lip-reading, ACCV 2016.
- **Weights:** `data/syncnet_v2.model` (SyncNet, trained on VoxCeleb2) and
  `detectors/s3fd/weights/sfd_face.pth` (S3FD face detector), fetched from
  the `lithiumice/syncnet` mirror on Hugging Face -- the original
  `robots.ox.ac.uk` host has a broken/self-signed TLS certificate as of
  this writing. Used here for research/portfolio purposes; see the
  upstream repo for any redistribution caveats on the weights themselves
  (the code is MIT, the *training data provenance* of the weights is not
  something this project controls).

Used by `syncsentry/lipsync/syncnet_offset.py` as the M3c learned
lip-sync estimator, invoked as an external tool (subprocess for the face
detection/tracking/crop stage, direct import for the SyncNet inference
stage) rather than ported into this project's own code -- see that
module's docstring and `docs/RESEARCH.md` section 1e.
