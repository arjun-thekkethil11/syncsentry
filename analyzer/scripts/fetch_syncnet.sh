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

# Patch: resolve S3FD's weight file relative to its own module location
# instead of the process's current working directory.
#
# Needed for `syncnet_offset.py` to call `run_pipeline.run_face_track_pipeline`
# in-process rather than as a subprocess (see that patch below, and
# `syncsentry/lipsync/syncnet_offset.py::_run_face_track_crop`'s own
# comment for why in-process matters for memory): the old subprocess
# call used `cwd=` to make this relative path resolve correctly, but
# mutating the real process's cwd to do the same in-process would be a
# global change, unsafe from a worker thread during concurrent piecewise
# refinement. Resolving relative to `__file__` instead removes the cwd
# dependency entirely, for every caller, not just this one.
#
# Idempotent: checks before patching.
if ! grep -q "os.path.dirname(os.path.abspath(__file__))" "$DEST/detectors/s3fd/__init__.py"; then
  echo "Patching detectors/s3fd/__init__.py: absolute weight path ..."
  python3 - "$DEST/detectors/s3fd/__init__.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old_import = "import time, logging"
new_import = "import os, time, logging"
assert old_import in src, 'expected import line not found, upstream detectors/s3fd/__init__.py may have changed'
src = src.replace(old_import, new_import, 1)

old_path = "PATH_WEIGHT = './detectors/s3fd/weights/sfd_face.pth'"
new_path = ("PATH_WEIGHT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "
            "'weights', 'sfd_face.pth')")
assert old_path in src, 'expected PATH_WEIGHT line not found, upstream detectors/s3fd/__init__.py may have changed'
src = src.replace(old_path, new_path, 1)

open(path, 'w').write(src)
PYEOF
else
  echo "detectors/s3fd/__init__.py already patched (absolute weight path), skipping."
fi

# Patch: expose run_pipeline.py's face detection + tracking + crop
# pipeline as an importable function (`run_face_track_pipeline(opt)`)
# instead of only runnable as a subprocess.
#
# Upstream calls `parser.parse_args()` at module scope and runs
# everything immediately on import, so this used to be run only via
# `subprocess.run([sys.executable, "run_pipeline.py", ...])`
# (`syncsentry/lipsync/syncnet_offset.py::_run_face_track_crop`). That
# subprocess pays for its own separate import of torch/cv2/numpy, and
# its own resident copy of the S3FD model, on top of whatever the caller
# process has already loaded for `SyncNetInstance`. Measured directly,
# calling in-process instead (both stages sharing one loaded copy of
# everything) saves roughly 100MB of peak memory per request; on a host
# with a hard memory cap (Render's free tier, 512MB), that is real
# headroom, not a rounding error.
#
# This patch only moves code, it does not change what runs: arg parsing
# and the top-level execution both move under `_build_arg_parser()` /
# `if __name__ == "__main__":`, so `python3 run_pipeline.py --videofile
# ...` from the command line behaves exactly as before. Verified by
# diffing crop output on the same input, in-process vs. the old
# subprocess call: bit-for-bit identical.
#
# Idempotent: checks before patching.
if ! grep -q "^def run_face_track_pipeline" "$DEST/run_pipeline.py"; then
  echo "Patching run_pipeline.py: expose run_face_track_pipeline() for in-process use ..."
  python3 - "$DEST/run_pipeline.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old_parse = '''parser = argparse.ArgumentParser(description = "FaceTracker")
parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry')
parser.add_argument('--videofile',      type=str, default='',   help='Input video file')
parser.add_argument('--reference',      type=str, default='',   help='Video reference')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels')
parser.add_argument('--overwrite',      action='store_true',    help='Overwrite existing output directories')
opt = parser.parse_args()

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))'''

new_parse = '''def _build_arg_parser():
  parser = argparse.ArgumentParser(description = "FaceTracker")
  parser.add_argument('--data_dir',       type=str, default='data/work', help='Output direcotry')
  parser.add_argument('--videofile',      type=str, default='',   help='Input video file')
  parser.add_argument('--reference',      type=str, default='',   help='Video reference')
  parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
  parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box')
  parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration')
  parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
  parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped')
  parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels')
  parser.add_argument('--overwrite',      action='store_true',    help='Overwrite existing output directories')
  return parser


def _fill_derived_dirs(opt):
  setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
  setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
  setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
  setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
  setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))
  return opt'''

assert old_parse in src, 'expected arg-parsing block not found, upstream run_pipeline.py may have changed'
src = src.replace(old_parse, new_parse, 1)

old_exec = '''# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== DELETE EXISTING DIRECTORIES ==========

for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  path = os.path.join(d, opt.reference)
  if os.path.exists(path):
    if not opt.overwrite:
      sys.exit(f"Output directory already exists: {path}. Use --overwrite to overwrite.")
    logger.warning('Overwriting existing directory: %s', path)
    rmtree(path)

# ========== MAKE NEW DIRECTORIES ==========

for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
  os.makedirs(os.path.join(d, opt.reference), exist_ok=True)

# ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

logger.info('Converting video to 25fps: %s', opt.videofile)
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", opt.videofile, "-qscale:v", "2", "-async", "1", "-r", "25",
           os.path.join(opt.avi_dir, opt.reference, 'video.avi')]
subprocess.run(command, check=True)

logger.info('Extracting frames from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
           "-qscale:v", "2", "-threads", "1", "-f", "image2",
           os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')]
subprocess.run(command, check=True)

logger.info('Extracting audio from video')
command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
           "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
           os.path.join(opt.avi_dir, opt.reference, 'audio.wav')]
subprocess.run(command, check=True)

# ========== FACE DETECTION ==========

faces = inference_video(opt)

# ========== SCENE DETECTION ==========

scene = scene_detect(opt)

# ========== FACE TRACKING ==========

alltracks = []
vidtracks = []

for shot in scene:

  if shot[1].get_frames() - shot[0].get_frames() >= opt.min_track :
    alltracks.extend(track_shot(opt,faces[shot[0].get_frames():shot[1].get_frames()]))

# ========== FACE TRACK CROP ==========

for ii, track in enumerate(alltracks):
  vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

# ========== SAVE RESULTS ==========

savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

with open(savepath, 'wb') as fil:
  pickle.dump(vidtracks, fil)

rmtree(os.path.join(opt.tmp_dir,opt.reference))'''

new_exec = '''# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

def run_face_track_pipeline(opt):
  """Runs the full face detection + scene detection + tracking + crop
  pipeline for `opt` (an `argparse.Namespace`-like object with the same
  fields `_build_arg_parser()` produces, plus the derived `*_dir` fields
  from `_fill_derived_dirs()`).

  Factored out of module-level script code so this can be called
  in-process (see `syncsentry/lipsync/syncnet_offset.py::_run_face_track_crop`)
  instead of only via a subprocess: calling in-process means torch, cv2,
  and this module's other imports are loaded once and shared with the
  caller's own process, rather than paying a second full import (and
  second resident copy of every loaded model) for a separate interpreter.
  The `if __name__ == "__main__":` block below preserves the original
  CLI entry point exactly, unchanged.
  """
  # ========== DELETE EXISTING DIRECTORIES ==========

  for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
    path = os.path.join(d, opt.reference)
    if os.path.exists(path):
      if not opt.overwrite:
        raise SystemExit(f"Output directory already exists: {path}. Use --overwrite to overwrite.")
      logger.warning('Overwriting existing directory: %s', path)
      rmtree(path)

  # ========== MAKE NEW DIRECTORIES ==========

  for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
    os.makedirs(os.path.join(d, opt.reference), exist_ok=True)

  # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

  logger.info('Converting video to 25fps: %s', opt.videofile)
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i", opt.videofile, "-qscale:v", "2", "-async", "1", "-r", "25",
             os.path.join(opt.avi_dir, opt.reference, 'video.avi')]
  subprocess.run(command, check=True)

  logger.info('Extracting frames from video')
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
             "-qscale:v", "2", "-threads", "1", "-f", "image2",
             os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')]
  subprocess.run(command, check=True)

  logger.info('Extracting audio from video')
  command = ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(opt.avi_dir, opt.reference, 'video.avi'),
             "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
             os.path.join(opt.avi_dir, opt.reference, 'audio.wav')]
  subprocess.run(command, check=True)

  # ========== FACE DETECTION ==========

  faces = inference_video(opt)

  # ========== SCENE DETECTION ==========

  scene = scene_detect(opt)

  # ========== FACE TRACKING ==========

  alltracks = []
  vidtracks = []

  for shot in scene:

    if shot[1].get_frames() - shot[0].get_frames() >= opt.min_track :
      alltracks.extend(track_shot(opt,faces[shot[0].get_frames():shot[1].get_frames()]))

  # ========== FACE TRACK CROP ==========

  for ii, track in enumerate(alltracks):
    vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

  # ========== SAVE RESULTS ==========

  savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(vidtracks, fil)

  rmtree(os.path.join(opt.tmp_dir,opt.reference))

  return vidtracks


if __name__ == "__main__":
  _opt = _fill_derived_dirs(_build_arg_parser().parse_args())
  run_face_track_pipeline(_opt)'''

assert old_exec in src, 'expected EXECUTE DEMO block not found, upstream run_pipeline.py may have changed'
src = src.replace(old_exec, new_exec, 1)

open(path, 'w').write(src)
PYEOF
else
  echo "run_pipeline.py already patched (importable run_face_track_pipeline), skipping."
fi

# Patch: run SyncNetInstance's CNN forward passes under torch.no_grad(),
# and stop forcing a float64 intermediate copy of the (uint8) frame
# tensor before its final float32 cast.
#
# Neither changes SyncNet's output: no_grad() only turns off autograd's
# bookkeeping for a graph that a pure-inference call like this never
# uses (nothing here ever calls .backward()), and the float64 copy was
# immediately narrowed back to float32 by the very next call, i.e. it
# was already a no-op numerically, just an extra full-size allocation
# on the way there. Measured directly, together these roughly halve
# this stage's peak memory; most of the remaining reduction comes from
# fetch_syncnet.sh's caller passing a smaller opt.batch_size (see
# `syncsentry/lipsync/syncnet_offset.py`'s BATCH_SIZE), since batch size
# is this forward pass's dominant memory cost, upstream's own default
# (20) far more than needed on a memory-capped host.
#
# Idempotent: checks before patching.
if ! grep -q "with torch.no_grad():" "$DEST/SyncNetInstance.py"; then
  echo "Patching SyncNetInstance.py: no_grad() + drop float64 intermediate ..."
  python3 - "$DEST/SyncNetInstance.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old_loop = '''        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):

            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)'''
new_loop = '''        tS = time.time()
        with torch.no_grad():
            for i in range(0,lastframe,opt.batch_size):

                im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
                im_in = torch.cat(im_batch,0)
                im_out  = self.__S__.forward_lip(im_in.to(self.device))
                im_feat.append(im_out.data.cpu())

                cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
                cc_in = torch.cat(cc_batch,0)
                cc_out  = self.__S__.forward_aud(cc_in.to(self.device))
                cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)'''
assert old_loop in src, 'expected evaluate() batch loop not found, upstream SyncNetInstance.py may have changed'
src = src.replace(old_loop, new_loop, 1)

assert src.count('im.astype(float)') == 2, 'expected two im.astype(float) occurrences not found'
src = src.replace('torch.from_numpy(im.astype(float)).float()', 'torch.from_numpy(im).float()')
assert src.count('cc.astype(float)') == 1, 'expected one cc.astype(float) occurrence not found'
src = src.replace('torch.from_numpy(cc.astype(float)).float()', 'torch.from_numpy(cc).float()')

open(path, 'w').write(src)
PYEOF
else
  echo "SyncNetInstance.py already patched (no_grad + float32), skipping."
fi

echo "Done. Try: syncsentry fix --video <file> --out-dir ./out --use-syncnet"
