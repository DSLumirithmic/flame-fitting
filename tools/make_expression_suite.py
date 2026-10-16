'''
Builds tools/data/expression_suite_v1.npz: the fixed expression/pose suite used
for animation validation (tools/validate_animation.py) and by the fitter's
--anim_collision_weight.

Expressions are GNM's own semantic sampler prototypes (ExpressionSampler's
CVAE decoder at latent z = 0 for each class, i.e. the class mean), so they are
deterministic and in GNM's native expression space. Needs tensorflow, which the
fitting env does not have -- run once wherever TF is available; the npz is
committed so nothing else needs TF:

  python tools/make_expression_suite.py --gnm_shape_root ../GNM --out tools/data/expression_suite_v1.npz
'''
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gnm_shape_root', default=None, help='dir containing the gnm package')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data', 'expression_suite_v1.npz'))
    a = ap.parse_args()
    if a.gnm_shape_root:
        sys.path.insert(0, a.gnm_shape_root)
    from gnm.shape import semantic_sampler as ss
    smp = ss.ExpressionSampler()
    z = np.zeros((1, smp._latent_dim), np.float32)
    proto = {}
    for e in ss.Expression:
        oh = np.zeros((1, smp._num_classes), np.float32)
        oh[0, int(e)] = 1
        proto[e.name.lower()] = smp._decoder.predict([z, oh], verbose=0)[0].astype(np.float64)

    E = {'neutral': np.zeros(383)}
    for k, v in proto.items():
        E[k] = v
    # Both eyes closed: each wink's own eye block (0-99 / 100-199), no mouth.
    wl, wr = proto['wink_left'], proto['wink_right']
    blink = np.zeros(383)
    # pick, per eye block, the wink that moves it most
    for blk in (slice(0, 100), slice(100, 200)):
        blink[blk] = wl[blk] if np.abs(wl[blk]).sum() > np.abs(wr[blk]).sum() else wr[blk]
    E['blink'] = blink
    E['jaw_open'] = proto['stretch_face']
    E['tongue_out'] = proto['tongue_center']
    E['jaw_open_x1.5'] = 1.5 * proto['stretch_face']
    E['smile_wide_x1.5'] = 1.5 * proto['smile_wide']

    def rot(**kw):
        r = np.zeros((4, 3))
        for j, v in kw.items():
            r[['neck', 'head', 'left_eye', 'right_eye'].index(j)] = v
        return r
    gaze = lambda yaw, pitch: rot(left_eye=[pitch, yaw, 0], right_eye=[pitch, yaw, 0])
    poses = [  # (name, expression, rotations)
        ('neutral', 'neutral', rot()),
        ('blink', 'blink', rot()),
        ('wink_left', 'wink_left', rot()),
        ('gaze_left', 'neutral', gaze(0.40, 0.0)),
        ('gaze_right', 'neutral', gaze(-0.40, 0.0)),
        ('gaze_up', 'neutral', gaze(0.0, -0.30)),
        ('gaze_down', 'neutral', gaze(0.0, 0.30)),
        ('blink_gaze_down', 'blink', gaze(0.0, 0.30)),
        ('squint', 'squint', rot()),
        ('smile_wide', 'smile_wide', rot()),
        ('happy', 'happy', rot()),
        ('smile_wide_x1.5', 'smile_wide_x1.5', rot()),
        ('jaw_open', 'jaw_open', rot()),
        ('jaw_open_x1.5', 'jaw_open_x1.5', rot()),
        ('surprise', 'surprise', rot()),
        ('pucker', 'pucker', rot()),
        ('funneler', 'funneler', rot()),
        ('lips_roll_in', 'lips_roll_in', rot()),
        ('compress_face', 'compress_face', rot()),
        ('mouth_left', 'mouth_left', rot()),
        ('blow', 'blow', rot()),
        ('snarl', 'snarl', rot()),
        ('tongue_out', 'tongue_out', rot()),
        ('head_turn_jaw_open', 'jaw_open', rot(neck=[0.0, 0.35, 0.0], head=[0.2, 0.0, 0.0])),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, format='gnm_expression_suite_v1',
             pose_names=np.array([p[0] for p in poses]),
             pose_expr=np.array([p[1] for p in poses]),
             pose_rotations=np.stack([p[2] for p in poses]),
             **{'expr_' + k: v for k, v in E.items()})
    print('wrote', a.out, len(poses), 'poses,', len(E), 'expressions')


if __name__ == '__main__':
    main()
