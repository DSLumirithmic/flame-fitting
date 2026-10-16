'''
Region definitions on GNM's template, shared by the V2 fitter, the scan
confidence weights and the evaluation (all template-only, so vertex indices
are valid for every fit of every subject).

Editable / transition / protected (Stage 3)
-------------------------------------------
  editable    the face (GNM's `hockey_mask`), the ears and the under-chin
              band -- surface the capture actually constrains.
  protected   skin far (geodesically) from the editable set: crown, back of
              the head, back of the neck. Photogrammetry there sees hair,
              headbands and collars, not the skull, so any change the fit makes
              there is unsupported.
  transition  in between; the protection weight m(v) ramps smoothly from 0
              (editable side) to 1 (protected side) with a C1 smoothstep of the
              geodesic distance, so nothing that uses it can create a seam.

Non-skin components (eyes, teeth, tongue) are never protected: they are
edited only through the identity basis and are checked by the collision /
rigidity terms instead.

Why masking the scan loss is not enough: GNM's identity basis is global (PCA
over whole heads), so a coefficient chosen to fit the cheek also moves the
crown. `identity_influence()` measures that. The protection therefore acts on
the identity-induced DISPLACEMENT itself:
  soft  a penalty on m(v) * |B_id id|^2 over the protected/transition skin,
  hard  the identity displacement is attenuated in the forward pass,
        d_id(v) = (1 - m(v)) * (B_id id)(v), so protected vertices (m = 1) stay
        exactly at the template whatever id is. This is a fixed linear
        re-weighting of the identity basis -- still a bind-pose identity offset,
        so expression blendshapes, pose correctives and skinning apply on top
        of it exactly as for native GNM identities (see fitting/identity_io.py).
'''

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

DEFAULT_REGION_CFG = dict(
    editable_groups=('hockey_mask', 'ears'),
    chin_band=dict(y_min=0.135, y_max=0.200, z_min=0.030),  # under-chin / submandibular
    transition_start_m=0.010,   # geodesic distance from the editable set where m starts > 0
    transition_end_m=0.045,     # ... and where it reaches 1 (hard-protected beyond)
)

# Reporting regions (scan error by region, confidence priors). Each is a list
# of GNM vertex groups; a vertex takes the FIRST region that claims it.
# 'eyelids' (skin within EYELID_DIST_M of an eyeball) and 'lips' (GNM's
# upper_lip/lower_lip vermilion groups) are carved out first, so the low scan
# priors hit the lid margins and the red of the lips, not the whole orbit or
# the whole perioral area.
EYELID_DIST_M = 0.004
REPORT_REGIONS = (
    ('eyelids', None),
    ('lips', ('upper_lip', 'lower_lip')),
    ('periorbital', ('left_orbital_region', 'right_orbital_region')),
    ('perioral', ('upper_lip_region', 'lower_lip_region')),
    ('nose', ('nose_region',)),
    ('chin', ('chin_region',)),
    ('cheeks', ('left_cheek_region', 'right_cheek_region', 'left_zygomatic_region',
                'right_zygomatic_region', 'left_infraorbital_region',
                'right_infraorbital_region', 'left_parotid_region', 'right_parotid_region')),
    ('forehead_brows', ('forehead_region', 'left_brow_region', 'middle_brow_region',
                        'right_brow_region', 'left_temple_region', 'right_temple_region')),
    ('ears', ('ears',)),
    ('face_other', ('hockey_mask',)),
)


def _groups(npz):
    names = list(npz['vertex_group_names'])
    vg = npz['vertex_groups']
    return {n: vg[i] > 0.5 for i, n in enumerate(names)}


def edge_graph(V, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    n = len(V)
    return sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def protection_regions(npz, cfg=None):
    ''' Returns dict(editable, transition, protected (bool masks), weight m (V,)
    in [0,1], geodesic distance from the editable set, skin mask). '''
    cfg = dict(DEFAULT_REGION_CFG, **(cfg or {}))
    g = _groups(npz)
    V = npz['template_vertex_positions'].astype(np.float64)
    F = npz['triangles'].astype(np.int64)
    skin = g['skin']
    editable = np.zeros(len(V), bool)
    for k in cfg['editable_groups']:
        editable |= g[k]
    cb = cfg['chin_band']
    editable |= skin & (V[:, 1] > cb['y_min']) & (V[:, 1] < cb['y_max']) & (V[:, 2] > cb['z_min'])
    editable &= skin
    # Geodesic distance over the skin only (other components are separate
    # meshes anyway, but the mouth sock connects through the lips).
    A = edge_graph(V, F)
    d = csgraph.dijkstra(A, directed=False, indices=np.where(editable)[0], min_only=True)
    d[~np.isfinite(d)] = 1e3
    t0, t1 = cfg['transition_start_m'], cfg['transition_end_m']
    m = smoothstep((d - t0) / (t1 - t0))
    m[~skin] = 0.0
    m[editable] = 0.0
    protected = skin & (m >= 1.0 - 1e-12)
    transition = skin & ~protected & (m > 0.0)
    return dict(editable=editable, transition=transition, protected=protected, weight=m,
                geodesic=d, skin=skin, cfg=cfg)


def report_region_labels(npz):
    ''' (V,) int region id per vertex (-1 = none) and the names list, for
    skin vertices; non-skin parts get their own labels. '''
    g = _groups(npz)
    V = npz['template_vertex_positions']
    names = [r for r, _ in REPORT_REGIONS] + ['crown_back', 'neck', 'eyeballs', 'teeth', 'tongue']
    lab = np.full(len(V), -1, np.int64)
    for i, (r, groups) in enumerate(REPORT_REGIONS):
        m = np.zeros(len(V), bool)
        if r == 'eyelids':
            from scipy.spatial import cKDTree
            eye = g['left_eye'] | g['right_eye']
            d, _ = cKDTree(V[eye]).query(V)
            m = (d < EYELID_DIST_M) & (g['left_orbital_region'] | g['right_orbital_region'])
        else:
            for k in groups:
                m |= g[k]
        lab[(lab < 0) & m & g['skin']] = i
    rest = (lab < 0) & g['skin']
    neck = rest & (V[:, 1] < 0.20)
    lab[neck] = names.index('neck')
    lab[rest & ~neck] = names.index('crown_back')
    lab[g['left_eye'] | g['right_eye']] = names.index('eyeballs')
    lab[g['upper_teeth_and_gums'] | g['lower_teeth_and_gums']] = names.index('teeth')
    lab[g['tongue']] = names.index('tongue')
    return lab, names


def identity_influence(npz, regions):
    ''' Per identity component: RMS displacement (m per unit coefficient) on
    the protected, transition and editable skin. Shows why masking the data
    term is insufficient: every component that moves the face also moves the
    protected crown/back. '''
    B = npz['vertex_identity_basis'].astype(np.float64)        # (I, V, 3)
    mag = np.linalg.norm(B, axis=2)                            # (I, V)
    rms = lambda mask: np.sqrt((mag[:, mask] ** 2).mean(axis=1))
    return dict(protected=rms(regions['protected']), transition=rms(regions['transition']),
                editable=rms(regions['editable']))
