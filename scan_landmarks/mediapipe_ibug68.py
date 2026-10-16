'''
MediaPipe Face Mesh (478 points, refine_landmarks=True) -> iBUG-68 mapping.

MediaPipe's points are vertices of a fixed canonical face mesh regressed onto
the image; iBUG-68 points are annotation conventions. They are NOT the same
set, so every iBUG point here is defined by one of three rules, recorded per
point in MAPPING (and written into every detections file):

  vertex   a MediaPipe vertex that sits on the same anatomical point
           (eye canthi, mouth commissures, philtrum / lower-lip midline,
           subnasale, pronasale, nasion).
  arc      a fraction of arc length along a dense MediaPipe contour between
           two such anchors. iBUG places the in-between points of the lids,
           lips, nose bridge and jaw at roughly even spacing along the curve
           (Sagonas et al., 300-W markup), so e.g. iBUG 37/38 are taken at
           1/3 and 2/3 of the upper-lid arc from the outer to the inner
           canthus. This is a definition, not a fit to another detector.
  none     no defensible correspondence -> invalid in every view.

Caveats that cannot be mapped away (documented in the report):
  * jaw 0-16: MediaPipe's face oval is a fixed mesh contour. iBUG's jaw is the
    visible occluding contour, which in oblique views is a different curve on
    the far side. The fitter only uses the jaw as a view-dependent contour
    (point-to-line, sliding) and rejects samples that are not on a scan depth
    edge, so far-side oval points that are not on the silhouette are dropped
    there; they are still marked valid here.
  * brows 17-26: iBUG brow points follow the upper brow edge; MediaPipe's
    upper-brow vertices are used. Brows carry a low weight in the fit (texture-
    defined, 5-7 mm definition mismatch with GNM even for dlib).
  * MediaPipe reports a position for hidden points (it regresses the whole
    mesh); so do dlib and FAN. Occlusion is decided downstream from the scan.

Index conventions: iBUG 36-41 / 17-21 / 48 / 0 are on the SUBJECT's right
(image left in a frontal photo); MediaPipe 33/133/61 are the subject's right
eye outer/inner canthus and right mouth corner.
'''

import json

import numpy as np

# Dense MediaPipe contours (canonical face-mesh topology, refine_landmarks=True).
C = dict(
    r_upper_lid=[33, 246, 161, 160, 159, 158, 157, 173, 133],
    r_lower_lid=[33, 7, 163, 144, 145, 153, 154, 155, 133],
    l_upper_lid=[362, 398, 384, 385, 386, 387, 388, 466, 263],
    l_lower_lid=[362, 382, 381, 380, 374, 373, 390, 249, 263],
    outer_upper_lip=[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    outer_lower_lip=[291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    inner_upper_lip=[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    inner_lower_lip=[308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
    nose_bridge=[168, 6, 197, 195, 5, 4, 1],
    jaw=[234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
         361, 323, 454],
    r_brow_upper=[70, 63, 105, 66, 107],
    l_brow_upper=[336, 296, 334, 293, 300],
)


def _arc(contour, frac):
    return ('arc', contour, float(frac))


def _v(i):
    return ('vertex', int(i))


def _chain(contour_name, a, b):
    ''' Sub-contour of C[contour_name] from vertex a to vertex b (inclusive). '''
    c = C[contour_name]
    i, j = c.index(a), c.index(b)
    return c[i:j + 1] if i <= j else c[j:i + 1][::-1]


# iBUG-68 definition per point
MAPPING = {}
# jaw: 17 points evenly along the face oval (see caveat above)
for k in range(17):
    MAPPING[k] = _arc('jaw', k / 16.0)
# brows (upper edge), iBUG order: 17-21 subject-right outer->inner, 22-26 left inner->outer
for k, i in enumerate(C['r_brow_upper']):
    MAPPING[17 + k] = _v(i)
for k, i in enumerate(C['l_brow_upper']):
    MAPPING[22 + k] = _v(i)
# nose bridge 27-30: nasion (168) to pronasale (1), evenly
for k in range(4):
    MAPPING[27 + k] = _arc('nose_bridge', k / 3.0)
# nose base 31-35 (alar base / nostril / subnasale): set by visual validation, see NOSE_BASE
NOSE_BASE = [98, 97, 2, 326, 327]
for k, i in enumerate(NOSE_BASE):
    MAPPING[31 + k] = _v(i)
# eyes. right: 36 outer, 37/38 upper 1/3, 2/3 outer->inner, 39 inner, 40/41 lower 2/3, 1/3
MAPPING[36] = _v(33)
MAPPING[37] = _arc('r_upper_lid', 1 / 3.0)
MAPPING[38] = _arc('r_upper_lid', 2 / 3.0)
MAPPING[39] = _v(133)
MAPPING[40] = _arc('r_lower_lid', 2 / 3.0)
MAPPING[41] = _arc('r_lower_lid', 1 / 3.0)
# left: 42 inner, 43/44 upper 1/3, 2/3 inner->outer, 45 outer, 46/47 lower 2/3, 1/3 (from inner)
MAPPING[42] = _v(362)
MAPPING[43] = _arc('l_upper_lid', 1 / 3.0)
MAPPING[44] = _arc('l_upper_lid', 2 / 3.0)
MAPPING[45] = _v(263)
MAPPING[46] = _arc('l_lower_lid', 2 / 3.0)
MAPPING[47] = _arc('l_lower_lid', 1 / 3.0)
# outer lips 48-59. Anchors: commissures 61/291, cupid's-bow peaks 37/267,
# philtrum midline 0, lower midline 17; 49/53 halfway corner->peak; lower lip
# 55,56,58,59 at 1/6..5/6 of the lower arc (57 = midline).
MAPPING[48] = _v(61)
MAPPING[49] = ('arc_sub', 'outer_upper_lip', 61, 37, 0.5)
MAPPING[50] = _v(37)
MAPPING[51] = _v(0)
MAPPING[52] = _v(267)
MAPPING[53] = ('arc_sub', 'outer_upper_lip', 267, 291, 0.5)
MAPPING[54] = _v(291)
MAPPING[55] = _arc('outer_lower_lip', 1 / 6.0)
MAPPING[56] = _arc('outer_lower_lip', 2 / 6.0)
MAPPING[57] = _v(17)
MAPPING[58] = _arc('outer_lower_lip', 4 / 6.0)
MAPPING[59] = _arc('outer_lower_lip', 5 / 6.0)
# inner lips 60-67: inner commissures 78/308, midlines 13/14, quarter points
MAPPING[60] = _v(78)
MAPPING[61] = _arc('inner_upper_lip', 0.25)
MAPPING[62] = _v(13)
MAPPING[63] = _arc('inner_upper_lip', 0.75)
MAPPING[64] = _v(308)
MAPPING[65] = _arc('inner_lower_lip', 0.25)
MAPPING[66] = _v(14)
MAPPING[67] = _arc('inner_lower_lip', 0.75)

UNAVAILABLE = sorted(k for k, m in MAPPING.items() if m[0] == 'none')


def _arc_point(P, idx, frac):
    Q = P[idx]
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    t = frac * L[-1]
    j = int(np.clip(np.searchsorted(L, t, side='right') - 1, 0, len(seg) - 1))
    a = (t - L[j]) / max(seg[j], 1e-12)
    return Q[j] + a * (Q[j + 1] - Q[j])


def map_to_ibug68(P):
    ''' P (478,2) or (478,3) MediaPipe points in pixels -> (68,D), valid (68,).
    Arc lengths are measured in the image (2D) so spacing follows what the
    annotator would see. '''
    P = np.asarray(P, np.float64)
    out = np.zeros((68, P.shape[1]))
    valid = np.ones(68, bool)
    P2 = P[:, :2]
    for k in range(68):
        m = MAPPING[k]
        if m[0] == 'vertex':
            out[k] = P[m[1]]
        elif m[0] in ('arc', 'arc_sub'):
            idx = C[m[1]] if m[0] == 'arc' else _chain(m[1], m[2], m[3])
            frac = m[2] if m[0] == 'arc' else m[4]
            q2 = _arc_point(P2, idx, frac)
            # carry z (if any) with the same interpolation parameter
            Q = P[idx]
            seg = np.linalg.norm(np.diff(P2[idx], axis=0), axis=1)
            L = np.concatenate([[0.0], np.cumsum(seg)])
            t = frac * L[-1]
            j = int(np.clip(np.searchsorted(L, t, side='right') - 1, 0, len(seg) - 1))
            a = (t - L[j]) / max(seg[j], 1e-12)
            out[k] = Q[j] + a * (Q[j + 1] - Q[j])
            out[k, :2] = q2
        else:
            valid[k] = False
    return out, valid


def mapping_json():
    return json.dumps(dict(
        mediapipe='FaceMesh refine_landmarks=True (478 = 468 mesh + 10 iris)',
        contours=C, points={str(k): list(v) for k, v in MAPPING.items()},
        unavailable=UNAVAILABLE,
        rules=dict(vertex='same anatomical point', arc='fraction of 2D arc length along contour',
                   arc_sub='fraction of 2D arc length along contour between two anchor vertices',
                   none='no defensible correspondence -> invalid')))
