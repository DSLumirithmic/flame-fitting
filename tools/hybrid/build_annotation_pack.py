'''
Annotation pack for the hybrid-segmentation evaluation (model-independent; no fitting).

For every view (all 13, held-out included) crops of the raw photo are cut around
  throat, chin_profile, jaw, ear_right, ear_left, lip_vermilion, mouth_opening
using ONLY the MediaPipe 478-point detections (no model, no masks, no overlays).
A person annotates them with tools/hybrid/annotation_tool (served locally); this script never
annotates anything. Held-out views are annotated too but are used for evaluation only.

Output (never overwrites): <out>/crops/<item>.png, <out>/manifest.json, <out>/README.md,
<out>/annotate.html, <out>/serve_annotations.py, <out>/START_ANNOTATION_TOOL.bat
'''
import argparse, hashlib, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from PIL import Image

from image_only import common as CM, mp_topology as T

REGIONS = ['throat', 'chin_profile', 'jaw', 'ear_right', 'ear_left', 'lip_vermilion', 'mouth_opening']

INSTR = {
    'throat': ('Trace the OUTLINE OF THE NECK SKIN: where neck skin meets the background (or hair / clothing). '
               'Side views: the front of the throat from just under the chin down to the clothing. Frontal / low views: '
               'the left and right sides of the neck (one stroke each). Stop where clothing starts. Trace the real skin '
               'edge, never a shadow line. Segments hidden by hair or clothing, or that you cannot place within ~2 px, '
               'go in an UNCERTAIN stroke (or are left out).'),
    'chin_profile': ('Trace the outline of the LOWER LIP -> CHIN -> UNDER-CHIN as seen in this view. In side views this '
                     'is the profile against the background. In frontal / low views trace the lower edge of the chin '
                     'where the chin surface turns away (over the neck). Do NOT trace the under-chin SHADOW: if you '
                     'can only see a shadow edge and not the surface edge, use an UNCERTAIN stroke or mark not visible.'),
    'jaw': ('Trace the JAW LINE on each side (one stroke per side), from below the ear towards the chin, where the '
            'jaw is visible as an outline (against the background or the neck). Do NOT trace the shadow under the jaw. '
            'Parts you cannot place within ~2 px go in an UNCERTAIN stroke. A side that is hidden (turned away) is '
            'simply not traced.'),
    'ear_right': ('SUBJECT\'S RIGHT ear (in a frontal view it appears on the LEFT of the image). Trace the outer rim of '
                  'the ear (helix + earlobe) where it is visible as an outline, against background, hair or skin. Skip '
                  'where the ear joins the head. Parts behind hair / headband: leave out or UNCERTAIN. If the ear is not '
                  'visible, tick "not visible".'),
    'ear_left': ('SUBJECT\'S LEFT ear (in a frontal view it appears on the RIGHT of the image). Trace the outer rim of '
                 'the ear (helix + earlobe) where it is visible as an outline, against background, hair or skin. Skip '
                 'where the ear joins the head. Parts behind hair / headband: leave out or UNCERTAIN. If the ear is not '
                 'visible, tick "not visible".'),
    'lip_vermilion': ('Trace the VERMILION BORDER: the outer edge of the red lip against the surrounding skin, all the '
                      'way round (upper lip and lower lip; one CLOSED stroke, or two open strokes corner to corner). '
                      'Do not trace the line between the lips here.'),
    'mouth_opening': ('If the mouth is OPEN, trace the inner opening (the dark gap between the lips) as one CLOSED '
                      'stroke. If it is CLOSED, tick "mouth closed" and trace the contact line between the lips from '
                      'corner to corner (one open stroke).'),
}


def sq(cx, cy, half, W, H):
    x0, y0 = int(np.floor(cx - half)), int(np.floor(cy - half))
    x1, y1 = int(np.ceil(cx + half)), int(np.ceil(cy + half))
    return [max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)]


def crops_for(P, W, H):
    s = float(np.linalg.norm(P[10] - P[152]))                    # forehead top -> chin
    out = {}
    lips = P[T.OUTER_UPPER_LIP + T.OUTER_LOWER_LIP]
    mc = lips.mean(0)
    mw = float(np.linalg.norm(P[61] - P[291]))
    half = max(0.8 * mw, 0.22 * s)
    out['lip_vermilion'] = sq(mc[0], mc[1], half, W, H)
    out['mouth_opening'] = out['lip_vermilion']
    c = 0.75 * P[152] + 0.25 * P[17]
    out['chin_profile'] = sq(c[0], c[1], 0.36 * s, W, H)
    J = P[T.JAW_CONTOUR]
    x0, x1 = J[:, 0].min() - 0.12 * s, J[:, 0].max() + 0.12 * s
    y0, y1 = min(P[61][1], P[291][1]) - 0.05 * s, J[:, 1].max() + 0.2 * s
    out['jaw'] = [max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)]
    x0, x1 = J[:, 0].min() - 0.2 * s, J[:, 0].max() + 0.2 * s
    y0, y1 = P[152][1] - 0.12 * s, P[152][1] + 0.95 * s
    out['throat'] = [max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)]
    mid = 0.5 * (P[6] + P[1])
    for side, a in (('ear_right', 234), ('ear_left', 454)):
        d = P[a] - mid
        d = d / max(np.linalg.norm(d), 1e-6)
        c = P[a] + 0.28 * s * d
        out[side] = sq(c[0], c[1], 0.46 * s, W, H)
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--detections', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--holdout', required=True, help='comma-separated held-out view labels (same as the fit config)')
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    os.makedirs(os.path.join(a.out, 'crops'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'annotations'), exist_ok=True)
    cams, det, _ = CM.load_capture(a.capture, a.detections)
    hold = a.holdout.split(',')
    items = []
    for k, c in enumerate(cams):
        img = np.array(Image.open(c.image_path).convert('RGB'))
        H, W = img.shape[:2]
        P = det['dense_478'][k][:, :2]
        boxes, s = crops_for(P, W, H)
        for r in REGIONS:
            x0, y0, x1, y1 = boxes[r]
            iid = '%s__%s' % (c.label, r)
            fn = os.path.join('crops', iid + '.png')
            Image.fromarray(img[y0:y1, x0:x1]).save(os.path.join(a.out, fn))
            items.append(dict(id=iid, view=c.label, region=r, crop=dict(x0=x0, y0=y0, w=x1 - x0, h=y1 - y0), file=fn.replace('\\', '/'),
                              heldout=c.label in hold, image_sha256=CM.sha256(c.image_path), face_scale_px=s))
    man = dict(schema='gnm_hybrid_annotation_pack_v1', created=time.strftime('%Y-%m-%d %H:%M:%S'),
               capture=a.capture, detections=a.detections, detections_sha256=CM.sha256(a.detections),
               coordinate_convention='annotation points are stored in FULL-IMAGE pixel coordinates (1080x1440), continuous: '
                                     'pixel (i, j) covers [i, i+1) x [j, j+1); crop pixel (u, v) -> (x0 + u, y0 + v)',
               crops_from='MediaPipe 478 detections only (no model, no masks, no overlays)',
               heldout_views=hold, heldout_use='evaluation only', regions=REGIONS, instructions=INSTR, items=items)
    json.dump(man, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation_tool')
    for f in ('annotate.html', 'serve_annotations.py', 'START_ANNOTATION_TOOL.bat', 'README.md'):
        shutil.copy(os.path.join(here, f), os.path.join(a.out, f))
    print('items', len(items))


if __name__ == '__main__':
    main()
