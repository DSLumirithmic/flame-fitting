'''
MediaPipe Face Mesh V2 (478 points) index groups used by the image-only fit.

Index conventions: MediaPipe 33/133/61 are the SUBJECT's right eye outer /
inner canthus and right mouth corner (image left in a frontal photo).
468-472 = subject-right iris (centre, then 4 contour points), 473-477 = left.
'''
import numpy as np

N_MESH = 468
N_ALL = 478

# lid margins (the visible palpebral fissure), outer canthus -> inner canthus
R_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
R_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
L_UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362]
L_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
EYE_CORNERS = dict(r_outer=33, r_inner=133, l_outer=263, l_inner=362)

# lips, right corner -> left corner
OUTER_UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
OUTER_LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
MOUTH_CORNERS = dict(r_outer=61, l_outer=291, r_inner=78, l_inner=308)

# full MediaPipe face-oval (mesh boundary loop), starting at the forehead top
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# The lower part (cheek at ear level -> chin -> cheek) is used as a VIEW-DEPENDENT
# occluding contour. The upper oval (temples / forehead / hairline) is the
# boundary of MediaPipe's mesh, not an image feature: no contour there.
JAW_CONTOUR = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
               288, 361, 323, 454]
UPPER_OVAL_EXCLUDED = [10, 338, 297, 332, 284, 251, 389, 356, 127, 162, 21, 54, 103, 67, 109]

R_IRIS = [468, 469, 470, 471, 472]
L_IRIS = [473, 474, 475, 476, 477]

# Anatomical anchors used ONLY to register MediaPipe's canonical face to the
# GNM template when building the correspondence (never in the objective):
# MediaPipe vertex -> iBUG-68 index of GNM's head_sparse_68 embedding.
REG_ANCHORS = {33: 36, 133: 39, 362: 42, 263: 45, 61: 48, 291: 54, 0: 51, 17: 57,
               1: 30, 168: 27, 2: 33, 98: 31, 327: 35}

BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276]


def lid_curves():
    return dict(r_upper=R_UPPER_LID, r_lower=R_LOWER_LID, l_upper=L_UPPER_LID, l_lower=L_LOWER_LID)


def lip_curves():
    return dict(outer_upper=OUTER_UPPER_LIP, outer_lower=OUTER_LOWER_LIP,
                inner_upper=INNER_UPPER_LIP, inner_lower=INNER_LOWER_LIP)


def eye_aperture_ratio(P):
    ''' P (478,2) -> (right, left) lid opening / eye width (median of the 7
    interior upper-lower pairs). Used to flag blinks. '''
    out = []
    for up, lo in ((R_UPPER_LID, R_LOWER_LID), (L_UPPER_LID, L_LOWER_LID)):
        w = np.linalg.norm(P[up[0]] - P[up[-1]])
        g = np.linalg.norm(P[up[1:-1]] - P[lo[1:-1]], axis=1)
        out.append(float(np.median(g) / max(w, 1e-6)))
    return out


def mouth_gap_ratio(P):
    ''' inner-lip opening at the 3 central pairs / mouth width. '''
    w = np.linalg.norm(P[61] - P[291])
    g = np.linalg.norm(P[INNER_UPPER_LIP[4:7]] - P[INNER_LOWER_LIP[4:7]], axis=1)
    return float(np.median(g) / max(w, 1e-6))
