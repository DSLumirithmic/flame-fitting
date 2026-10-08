'''
End-to-end GNM pipeline: takes a scan folder (model.obj + calibration.xml +
views/, see scan_landmarks/README.md for the exact layout) and produces a
fitted GNM mesh, in two steps:

  1. scan_landmarks/generate_scan_landmarks.py --gnm
     -- detects the full 68-point iBUG landmarks directly on the scan.
  2. fit_scan_gnm.py
     -- fits GNM's identity/expression/pose to the scan + those landmarks.

Everything is written under <input>/outputs/:

  <input>/outputs/
      landmarks_gnm/landmarks.npy, .obj, .json   (step 1)
      fit_scan_result_gnm.obj                     (step 2, scan's frame, with UVs)
      fit_scan_result_gnm.json                    (step 2, gnm_head_editor preset)
      scan_scaled.obj                             (step 2, for visual comparison)

Both steps run in this repo's own uv environment by default (it now declares
dlib-bin, face-alignment, open3d, beautifulsoup4/lxml and pillow alongside
torch/gnm-shape). --landmarks-env / --landmarks-python still let step 1 run in a
different interpreter if you need to. The dlib shape predictor defaults to the
copy shipped by the face-recognition-models package.

Usage:
  uv run python run_gnm_pipeline.py -i /path/to/scan_folder
  uv run python run_gnm_pipeline.py -i /path/to/scan_folder -o /path/to/out --device cuda
'''

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full GNM pipeline (landmarks -> fit) on a scan folder.")
    parser.add_argument('-i', '--input', required=True,
                         help="Scan folder containing model.obj, calibration.xml, views/.")
    parser.add_argument('-o', '--output', default=None,
                         help="Output folder (default: <input>/outputs).")
    parser.add_argument('--landmarks-env', default=None,
                         help="Optional: another project root whose uv .venv should run step 1 "
                              "instead of this repo's env (default: this env). Used to "
                              "find the python interpreter for step 1, and (unless "
                              "--facial-detector is given) the shape_predictor .dat file.")
    parser.add_argument('--landmarks-python', default=None,
                         help="Explicit python interpreter for step 1, overriding "
                              "--landmarks-env's <env>/.venv python.")
    parser.add_argument('--facial-detector', default=None,
                         help="Path to dlib's shape_predictor_68_face_landmarks.dat "
                              "(default: <landmarks-env>/shape_predictor_68_face_landmarks.dat).")
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'],
                         help="Device for the FAN neural landmark detector (step 1). Fitting "
                              "(step 2) always auto-selects cuda if available.")
    parser.add_argument('--scan_unit', default='m', choices=['m', 'cm', 'mm'],
                         help="Unit the scan is stored in (default: m).")
    return parser.parse_args()


def resolve_landmarks_python(args):
    if args.landmarks_python:
        return args.landmarks_python
    if not args.landmarks_env:
        return sys.executable  # this repo's env has the landmark deps
    venv = os.path.join(args.landmarks_env, '.venv')
    candidate = os.path.join(venv, 'Scripts', 'python.exe') if os.name == 'nt' \
        else os.path.join(venv, 'bin', 'python')
    if not os.path.isfile(candidate):
        sys.exit(
            f"[ERROR] No python interpreter found at {candidate}.\n"
            f"Step 1 (landmark detection) needs an environment with dlib/face-alignment/"
            f"open3d/beautifulsoup4/pillow installed -- this repo's own environment doesn't "
            f"have them (see scan_landmarks/README.md). Pass --landmarks-env pointing at a "
            f"project with a uv .venv containing them, or --landmarks-python pointing "
            f"directly at such an interpreter."
        )
    return candidate


def resolve_facial_detector(args):
    if args.facial_detector:
        return args.facial_detector
    if not args.landmarks_env:
        try:
            import importlib.util
            spec = importlib.util.find_spec('face_recognition_models')
            candidate = os.path.join(os.path.dirname(spec.origin), 'models',
                                     'shape_predictor_68_face_landmarks.dat')
            if os.path.isfile(candidate):
                return candidate
        except (ImportError, AttributeError, TypeError):
            pass
        sys.exit("[ERROR] No --facial-detector given and face_recognition_models isn't "
                 "installed -- pass --facial-detector /path/to/shape_predictor_68_face_landmarks.dat.")
    candidate = os.path.join(args.landmarks_env, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.isfile(candidate):
        sys.exit(
            f"[ERROR] No shape predictor found at {candidate}. Pass --facial-detector "
            f"pointing at dlib's shape_predictor_68_face_landmarks.dat."
        )
    return candidate


def run_step(cmd, label):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else os.path.join(input_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(input_dir, 'model.obj')
    calib_path = os.path.join(input_dir, 'calibration.xml')
    views_dir = os.path.join(input_dir, 'views')
    missing = [p for p in (model_path, calib_path, views_dir) if not os.path.exists(p)]
    if missing:
        sys.exit(
            "[ERROR] --input folder is missing expected file(s)/folder(s):\n"
            + "\n".join(f"    {p}" for p in missing)
            + f"\nExpected layout inside {input_dir}:\n"
              "    model.obj\n    calibration.xml\n    views/<photos>"
        )

    landmarks_python = resolve_landmarks_python(args)
    facial_detector = resolve_facial_detector(args)
    landmarks_dir = os.path.join(output_dir, 'landmarks_gnm')

    generate_landmarks_script = os.path.join(REPO_ROOT, 'scan_landmarks', 'generate_scan_landmarks.py')
    landmarks_cmd = [
        landmarks_python, generate_landmarks_script,
        '-i', input_dir, '--gnm', '-o', landmarks_dir,
        '--facial-detector', facial_detector,
    ]
    if args.device:
        landmarks_cmd += ['--device', args.device]
    run_step(landmarks_cmd, "Step 1/2: generating 68-point GNM landmarks")

    landmarks_npy = os.path.join(landmarks_dir, 'landmarks.npy')
    if not os.path.isfile(landmarks_npy):
        sys.exit(f"[ERROR] Step 1 did not produce {landmarks_npy} -- check its output above.")

    fit_script = os.path.join(REPO_ROOT, 'fit_scan_gnm.py')
    fit_cmd = [
        sys.executable, fit_script,
        '--scan', model_path, '--landmarks', landmarks_npy,
        '--output_dir', output_dir, '--scan_unit', args.scan_unit,
    ]
    run_step(fit_cmd, "Step 2/2: fitting GNM to the scan")

    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"landmarks:    {landmarks_dir}/landmarks.npy (.obj, .json)")
    print(f"fitted mesh:  {output_dir}/fit_scan_result_gnm.obj")
    print(f"preset json:  {output_dir}/fit_scan_result_gnm.json")


if __name__ == '__main__':
    main()
