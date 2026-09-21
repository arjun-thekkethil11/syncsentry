#!/usr/bin/env bash
# Fetches the pretrained SyncNet reference implementation and weights,
# the learned lip-sync estimator used by --use-syncnet.
#
# Vendors joonson/syncnet_python (MIT-licensed code) into
# analyzer/third_party/syncnet_python/, and downloads its two pretrained
# model files (SyncNet itself and the S3FD face detector it depends on)
# from a Hugging Face mirror, since the original robotics.ox.ac.uk host
# has an expired TLS cert as of this writing. See
# analyzer/third_party/SOURCES.md for provenance and license notes.
#
# third_party/ is gitignored and never committed; re-run this script to
# fetch it again.
#
# Usage:
#   bash analyzer/scripts/fetch_syncnet.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../third_party/syncnet_python"

if [ ! -d "$DEST" ]; then
  echo "Cloning joonson/syncnet_python (MIT) ..."
  git clone --depth 1 https://github.com/joonson/syncnet_python.git "$DEST"
else
  echo "syncnet_python already cloned, skipping."
fi

mkdir -p "$DEST/data" "$DEST/detectors/s3fd/weights"

if [ ! -f "$DEST/data/syncnet_v2.model" ]; then
  echo "Downloading SyncNet weights (~52MB) from Hugging Face mirror ..."
  curl -fL -o "$DEST/data/syncnet_v2.model" \
    "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model"
else
  echo "SyncNet weights already present, skipping."
fi

if [ ! -f "$DEST/detectors/s3fd/weights/sfd_face.pth" ]; then
  echo "Downloading S3FD face-detector weights (~86MB) from Hugging Face mirror ..."
  curl -fL -o "$DEST/detectors/s3fd/weights/sfd_face.pth" \
    "https://huggingface.co/lithiumice/syncnet/resolve/main/sfd_face.pth"
else
  echo "S3FD weights already present, skipping."
fi

echo "Installing extra Python dependencies (python_speech_features, tqdm) ..."
pip install -q python_speech_features tqdm

# Patch: use CPU (not MPS) for S3FD face detection, and run detection
# across frames with a small thread pool instead of one at a time.
#
# CPU is consistently faster than MPS for this per-frame S3FD workload on
# Apple Silicon (roughly 1.4x, measured on real 1080p frames). CUDA is
# still preferred first where available.
#
# Threading (not multiprocessing, to avoid fork/spawn issues with
# cv2/torch on macOS) works because torch's native CPU tensor ops release
# the GIL during computation, so multiple Python threads, each holding
# its own S3FD instance, run concurrently on separate cores.
# `torch.set_num_threads(1)` per process avoids oversubscription, since
# each forward call would otherwise also spawn its own intra-op thread
# pool on top of the outer thread parallelism. 8 workers gives roughly a
# further 1.5x over a single worker at full frame-count scale on an
# 8-core machine, with identical per-frame face counts vs. the unthreaded
# baseline; only throughput and device selection change, not the
# underlying detection.
#
# Idempotent: checks before patching, and leaves the function untouched
# if already patched.
if ! grep -q "_s3fd_detect_one" "$DEST/run_pipeline.py"; then
  echo "Patching run_pipeline.py: CPU device + threaded face-detection loop ..."
  python3 - "$DEST/run_pipeline.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old = '''def inference_video(opt):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  DET = S3FD(device=device)

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = []

  with tqdm(enumerate(flist), total=len(flist), desc='Detecting faces') as pbar:
    for fidx, fname in pbar:

      image = cv2.imread(fname)

      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

      dets.append([])
      for bbox in bboxes:
        dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

      pbar.set_postfix(dets=len(dets[-1]))

  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)'''

new = '''def _s3fd_detect_one(args):
  fidx, fname, det, facedet_scale = args
  image = cv2.imread(fname)
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  bboxes = det.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])
  frame_dets = [{'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]} for bbox in bboxes]
  return fidx, frame_dets


def inference_video(opt):
  # CPU is faster than MPS for this workload; see fetch_syncnet.sh above.
  # Only use this frame-parallel thread pool when nothing else has
  # already claimed a thread budget via OMP_NUM_THREADS (set by
  # piecewise_offset.py before launching concurrent SyncNet subprocesses).
  # Otherwise fall back to a plain loop and let torch's own intra-op
  # parallelism respect that budget instead; layering an extra thread
  # pool on top of an already-small per-worker budget is a net loss.
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if os.environ.get('OMP_NUM_THREADS'):
    n_workers = 1
  else:
    n_workers = min(8, os.cpu_count() or 1)
    torch.set_num_threads(1)

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = [None] * len(flist)
  detectors = [S3FD(device=device) for _ in range(max(1, n_workers))]

  with tqdm(total=len(flist), desc='Detecting faces') as pbar:
    if n_workers <= 1:
      for fidx, fname in enumerate(flist):
        _, frame_dets = _s3fd_detect_one((fidx, fname, detectors[0], opt.facedet_scale))
        dets[fidx] = frame_dets
        pbar.update(1)
        pbar.set_postfix(dets=len(frame_dets))
    else:
      from concurrent.futures import ThreadPoolExecutor
      with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
          ex.submit(_s3fd_detect_one, (fidx, fname, detectors[fidx % n_workers], opt.facedet_scale))
          for fidx, fname in enumerate(flist)
        ]
        for future in futures:
          fidx, frame_dets = future.result()
          dets[fidx] = frame_dets
          pbar.update(1)

  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)'''

assert old in src, 'expected inference_video() body not found, upstream run_pipeline.py may have changed'
open(path, 'w').write(src.replace(old, new, 1))
PYEOF
else
  echo "run_pipeline.py already patched (CPU + threaded face detection), skipping."
fi

echo "Done. Try: syncsentry fix --video <file> --out-dir ./out --use-syncnet"
