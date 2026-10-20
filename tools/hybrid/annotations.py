''' Loading / densifying the human boundary annotations of tools/hybrid/annotation_tool (shared by gate + eval). '''
import json, os
import numpy as np

# annotation region -> hybrid observation family / compatible model regions (image_only.silhouette.REG names)
FAMILY = dict(throat='jaw_neck', chin_profile='jaw_neck', jaw='jaw_neck', ear_right='ear', ear_left='ear',
              lip_vermilion='vermilion', mouth_opening='mouth_line')
MODEL_COMPAT = dict(throat=['neck', 'neck_low', 'jaw_under'], chin_profile=['chin_jaw', 'jaw_under', 'lips'],
                    jaw=['chin_jaw', 'jaw_under', 'cheek'], ear_right=['ear'], ear_left=['ear'])


def load(pack, annotator, views=None):
    ''' latest saved annotation per item -> {item_id: dict}; views: optional set of view labels to keep '''
    man = json.load(open(os.path.join(pack, 'manifest.json')))
    d = os.path.join(pack, 'annotations', annotator)
    latest = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith('.json') and '__' in f:
            latest[f.rsplit('__', 1)[0]] = f
    items = {it['id']: it for it in man['items']}
    out = {}
    for iid, f in latest.items():
        it = items[iid]
        if views is not None and it['view'] not in views:
            continue
        a = json.load(open(os.path.join(d, f)))
        a['_file'] = os.path.join(d, f)
        a['_item'] = it
        out[iid] = a
    return out, man


def densify(points, closed=False, step=1.0):
    P = np.asarray(points, np.float64)
    if len(P) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if closed:
        P = np.concatenate([P, P[:1]])
    out, nrm = [], []
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        k = max(int(np.ceil(L / step)), 1)
        t = np.arange(k) / k
        out.append(a[None] + t[:, None] * (b - a)[None])
        d = (b - a) / max(L, 1e-9)
        nrm.append(np.repeat(np.array([[-d[1], d[0]]]), k, 0))
    if not closed:
        out.append(P[-1:]); nrm.append(nrm[-1][-1:])
    return np.concatenate(out), np.concatenate(nrm)


def points(ann, kind='certain'):
    ''' all strokes of one item of the given type, densified at 1 px -> (Q (m,2), N (m,2) unit normals, stroke id) '''
    Q, N, S = [], [], []
    if ann.get('not_visible'):
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)
    for si, st in enumerate(ann.get('strokes', [])):
        if st.get('type', 'certain') != kind or len(st.get('points', [])) < 2:
            continue
        q, n = densify(st['points'], st.get('closed', False))
        Q.append(q); N.append(n); S.append(np.full(len(q), si))
    if not Q:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)
    return np.concatenate(Q), np.concatenate(N), np.concatenate(S)


def in_crop(Q, crop):
    return (Q[:, 0] >= crop['x0']) & (Q[:, 0] < crop['x0'] + crop['w']) & (Q[:, 1] >= crop['y0']) & (Q[:, 1] < crop['y0'] + crop['h'])
