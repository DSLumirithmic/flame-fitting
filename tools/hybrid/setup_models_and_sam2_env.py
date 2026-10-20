'''
One-off setup for the hybrid-segmentation work (run with any Python 3; stdlib only).

1. Downloads the two segmentation models + their licence documents into
   outputs/hybrid_v1/provenance/models/ (never overwrites), records sha256.
     * MediaPipe selfie_multiclass_256x256 float32 v1 (.tflite) + Google model card PDF (Apache-2.0)
     * SAM 2.1 hiera-large checkpoint + SAM 2 LICENSE / LICENSE_cctorch at a pinned commit
2. Creates the SEPARATE uv env .venv-sam2 (py3.12, torch 2.7.1+cu128, sam2 at the pinned commit,
   SAM2_BUILD_CUDA=0, no build isolation). .venv / pyproject.toml / uv.lock are not touched:
   their sha256 is recorded before and after.
'''
import hashlib, json, os, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROV = os.path.join(REPO, 'outputs', 'hybrid_v1', 'provenance', 'models')
SAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
FILES = [
    ('selfie_multiclass_256x256_float32_v1.tflite',
     'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'),
    ('Model_Card_Multiclass_Segmentation.pdf',
     'https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf'),
    ('sam2.1_hiera_large.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'),
    ('SAM2_LICENSE_%s.txt' % SAM2_COMMIT[:7], 'https://raw.githubusercontent.com/facebookresearch/sam2/%s/LICENSE' % SAM2_COMMIT),
    ('SAM2_LICENSE_cctorch_%s.txt' % SAM2_COMMIT[:7], 'https://raw.githubusercontent.com/facebookresearch/sam2/%s/LICENSE_cctorch' % SAM2_COMMIT),
    ('SAM2_README_%s.md' % SAM2_COMMIT[:7], 'https://raw.githubusercontent.com/facebookresearch/sam2/%s/README.md' % SAM2_COMMIT),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def run(cmd, env=None):
    log('$', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    log(r.stdout[-4000:], r.stderr[-4000:], 'rc', r.returncode)
    if r.returncode:
        raise SystemExit('command failed')
    return r.stdout


def main():
    os.makedirs(PROV, exist_ok=True)
    guard = {f: sha256(os.path.join(REPO, f)) for f in ('pyproject.toml', 'uv.lock')}
    log('guard before', guard)
    man = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), sam2_commit=SAM2_COMMIT, files={})
    for name, url in FILES:
        p = os.path.join(PROV, name)
        if not os.path.exists(p):
            log('download', url)
            tmp = p + '.part'
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, p)
        man['files'][name] = dict(url=url, bytes=os.path.getsize(p), sha256=sha256(p))
        log(name, man['files'][name])
    # ---- .venv-sam2
    venv = os.path.join(REPO, '.venv-sam2')
    py = os.path.join(venv, 'Scripts', 'python.exe')
    if not os.path.exists(py):
        run(['uv', 'venv', '--python', '3.12', venv])
    run(['uv', 'pip', 'install', '--python', py, 'torch==2.7.1', 'torchvision==0.22.1',
         '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run(['uv', 'pip', 'install', '--python', py, 'setuptools>=61', 'wheel', 'numpy<2.3', 'pillow', 'opencv-python-headless==4.10.0.84'])
    env = dict(os.environ, SAM2_BUILD_CUDA='0')
    run(['uv', 'pip', 'install', '--python', py, '--no-build-isolation',
         'sam-2 @ git+https://github.com/facebookresearch/sam2@%s' % SAM2_COMMIT], env=env)
    out = run([py, '-c', 'import torch, sam2, sys; print(sys.version); print(torch.__version__, torch.cuda.is_available(), '
                         'torch.cuda.get_device_name(0)); print(sam2.__file__)'])
    man['venv_sam2'] = dict(python=py, check=out, freeze=run(['uv', 'pip', 'freeze', '--python', py]).splitlines(),
                            SAM2_BUILD_CUDA='0')
    after = {f: sha256(os.path.join(REPO, f)) for f in ('pyproject.toml', 'uv.lock')}
    man['repo_env_guard'] = dict(before=guard, after=after, unchanged=guard == after)
    log('guard after', after, 'unchanged', guard == after)
    mp_ = os.path.join(PROV, 'MANIFEST.json')
    if os.path.exists(mp_):
        mp_ = os.path.join(PROV, 'MANIFEST_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
    json.dump(man, open(mp_, 'w'), indent=1)
    log('SETUP DONE', mp_)


if __name__ == '__main__':
    main()
