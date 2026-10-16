'''
Local annotation tool for eye-lid margins, eye corners and the mouth closure
line on the calibrated capture photos (standard library only).

  python tools/annotate/serve.py --scan_folder <gnm-fitting_input> --out <annotations dir> \
      [--nav <detections or observations .npz>] [--holdout frame_000332,frame_000496,frame_001152] [--port 8770]

Open http://127.0.0.1:8770/ . Binds to 127.0.0.1 only; nothing leaves the
machine. Images are read (never written) from <scan_folder>/views.

Annotations are an INDEPENDENT REFERENCE: nothing in the fitting code reads
them. One JSON per view in --out (every save also keeps a timestamped copy
under --out/history/). Coordinates are in the ORIGINAL image as stored:
continuous pixel coordinates with the origin at the top-left CORNER of the
top-left pixel (pixel (i, j) covers [i, i+1) x [j, j+1); its centre is
(i + 0.5, j + 0.5)). For this capture the calibration has zero distortion and
the images are stored at the calibrated size and orientation, so this is also
the pinhole space of fitting/cameras.project(..., distort=False) and of the
ray-cast renders (fitting/render.py casts at pixel centres i + 0.5). The
server checks the size against calibration.xml and refuses otherwise.

--nav only places the zoom presets (eye / mouth boxes); no detector point is
ever shown or stored.
'''
import argparse
import datetime
import hashlib
import http.server
import json
import os
import re
import sys
import urllib.parse
import webbrowser
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
SECTIONS = ['eye_r.upper_lid', 'eye_r.lower_lid', 'eye_r.outer_corner', 'eye_r.inner_corner',
            'eye_l.upper_lid', 'eye_l.lower_lid', 'eye_l.outer_corner', 'eye_l.inner_corner',
            'mouth.lip_line']
PIXEL_CONVENTION = ('continuous original-image coordinates, origin at the top-left CORNER of the top-left '
                    'pixel; pixel (i,j) covers [i,i+1)x[j,j+1), centre (i+0.5,j+0.5); image as stored '
                    '(no rotation, no undistortion: calibration distortion is zero); identical to '
                    'fitting/cameras.project(distort=False) and the ray-cast renders')


def image_size(path):
    from struct import unpack
    with open(path, 'rb') as f:
        head = f.read(32)
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        return unpack('>II', head[16:24])
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def calib_sizes(xml_path):
    out = {}
    if not os.path.isfile(xml_path):
        return out
    root = ET.parse(xml_path).getroot()
    chunk = root.find('chunk')
    sens = {}
    for s in chunk.iter('sensor'):
        r = s.find('resolution')
        if r is None:
            c = s.find('calibration')
            r = c.find('resolution') if c is not None else None
        if r is not None:
            sens[s.get('id')] = (int(r.get('width')), int(r.get('height')))
        c = s.find('calibration')
        if c is not None:
            dist = {k: float(c.find(k).text) for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2', 'b1', 'b2')
                    if c.find(k) is not None}
            sens[s.get('id') + '_dist'] = dist
    for c in chunk.iter('camera'):
        sid = c.get('sensor_id')
        out[c.get('label')] = dict(wh=sens.get(sid), dist=sens.get(sid + '_dist', {}))
    return out


def nav_regions(nav_path):
    ''' {view: {eye_r: [x0,y0,x1,y1], eye_l: ..., mouth: ...}} from a detections /
    observations npz (zoom presets only). '''
    if not nav_path or not os.path.isfile(nav_path):
        return {}
    import numpy as np
    z = np.load(nav_path, allow_pickle=False)
    labels = [str(x) for x in z['labels']]
    out = {}
    if 'dense_478' in z.files or 'dense_2d' in z.files:
        D = z['dense_478'][..., :2] if 'dense_478' in z.files else z['dense_2d']
        idx = dict(eye_r=[33, 133, 159, 145, 246, 7, 173, 155], eye_l=[362, 263, 386, 374, 398, 382, 466, 249],
                   mouth=[61, 291, 0, 17, 78, 308, 13, 14])
    else:
        D = z['landmarks_2d']
        idx = dict(eye_r=list(range(36, 42)), eye_l=list(range(42, 48)), mouth=list(range(48, 68)))
    for i, l in enumerate(labels):
        r = {}
        for k, ii in idx.items():
            P = D[i][ii]
            if not np.isfinite(P).all():
                continue
            c = P.mean(0)
            half = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 20.0) * (0.95 if k != 'mouth' else 0.75) + 12
            r[k] = [float(c[0] - half), float(c[1] - half * 0.7), float(c[0] + half), float(c[1] + half * 0.7)]
        out[l] = r
    return out


def make_handler(a):
    views_dir = os.path.join(a.scan_folder, 'views')
    out_dir = os.path.abspath(a.out)
    os.makedirs(os.path.join(out_dir, 'history'), exist_ok=True)
    calib = calib_sizes(os.path.join(a.scan_folder, 'calibration.xml'))
    hold = [h.strip() for h in a.holdout.split(',') if h.strip()]
    nav = nav_regions(a.nav)
    files = sorted(f for f in os.listdir(views_dir) if f.lower().endswith(EXTS))
    views = []
    for f in files:
        name = os.path.splitext(f)[0]
        p = os.path.join(views_dir, f)
        wh = image_size(p)
        c = calib.get(name, {})
        ok = c.get('wh') is None or tuple(wh) == tuple(c['wh'])
        nz = {k: v for k, v in (c.get('dist') or {}).items() if abs(v) > 0}
        views.append(dict(name=name, file=f, wh=list(wh), holdout=name in hold, calib_wh=c.get('wh'),
                          size_matches_calibration=ok, nonzero_distortion=nz, regions=nav.get(name, {})))
    bad = [v['name'] for v in views if not v['size_matches_calibration'] or v['nonzero_distortion']]
    if bad:
        sys.exit('refusing: image size != calibration or non-zero distortion for %s -- annotation pixel '
                 'space would not match the projection space' % bad)
    views.sort(key=lambda v: (not v['holdout'], v['name']))
    sha = {}

    def image_sha(v):
        if v['name'] not in sha:
            sha[v['name']] = hashlib.sha256(open(os.path.join(views_dir, v['file']), 'rb').read()).hexdigest()
        return sha[v['name']]

    class H(http.server.SimpleHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            p = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if p in ('/', '/index.html'):
                body = open(os.path.join(HERE, 'index.html'), 'rb').read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(body)
            elif p == '/api/views':
                for v in views:
                    fp = os.path.join(out_dir, v['name'] + '.json')
                    v['annotation'] = json.load(open(fp)) if os.path.isfile(fp) else None
                self._json(dict(views=views, sections=SECTIONS, pixel_convention=PIXEL_CONVENTION,
                                out_dir=out_dir))
            elif p.startswith('/img/'):
                name = p[len('/img/'):]
                v = next((v for v in views if v['name'] == name), None)
                if v is None:
                    self.send_error(404)
                    return
                body = open(os.path.join(views_dir, v['file']), 'rb').read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png' if v['file'].lower().endswith('.png') else 'image/jpeg')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/api/save':
                self.send_error(404)
                return
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n))
            name = req.get('view', '')
            v = next((v for v in views if v['name'] == name), None)
            if v is None or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                self._json(dict(error='unknown view'), 400)
                return
            secs = {}
            for k in SECTIONS:
                s = (req.get('sections') or {}).get(k) or {}
                status = s.get('status', 'todo')
                if status not in ('valid', 'unclear', 'occluded', 'todo'):
                    status = 'todo'
                pts = [[float(x), float(y)] for x, y in (s.get('points') or [])]
                if status == 'valid':
                    need = 1 if k.endswith('corner') else 2
                    if len(pts) < need:
                        status = 'todo'
                secs[k] = dict(status=status, points=pts if status == 'valid' else [], note=str(s.get('note', ''))[:500],
                               kind='point' if k.endswith('corner') else 'polyline',
                               draft_points=pts if status != 'valid' else [])
            doc = dict(format='gnm_fit_annotations_v1', view=name, image=os.path.join(views_dir, v['file']),
                       image_sha256=image_sha(v), image_wh=v['wh'], holdout=v['holdout'],
                       pixel_convention=PIXEL_CONVENTION,
                       eye_naming='eye_r = SUBJECT\'s right eye (image left in a frontal photo); eye_l = subject\'s left',
                       definitions=dict(upper_lid='visible edge where the upper lid margin / lash line meets the eyeball',
                                        lower_lid='visible edge where the lower lid margin meets the eyeball (not the iris edge)',
                                        outer_corner='lateral canthus: where upper and lower margins meet',
                                        inner_corner='medial canthus: where upper and lower margins meet (medial end of the opening)',
                                        lip_line='visible line where the upper and lower lips meet, commissure to commissure'),
                       valid_sections=[k for k in SECTIONS if secs[k]['status'] == 'valid'],
                       flagged_sections={k: secs[k]['status'] for k in SECTIONS if secs[k]['status'] in ('unclear', 'occluded')},
                       annotator=str(req.get('annotator', ''))[:80],
                       saved=datetime.datetime.now().isoformat(timespec='seconds'), tool='tools/annotate v1',
                       use='independent reference only; not read by any fitting code',
                       sections=secs)
            path = os.path.join(out_dir, name + '.json')
            with open(path, 'w') as f:
                json.dump(doc, f, indent=1)
            hist = os.path.join(out_dir, 'history', '%s_%s.json' % (name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            with open(hist, 'w') as f:
                json.dump(doc, f)
            self._json(dict(saved=path, valid=doc['valid_sections'], flagged=doc['flagged_sections']))

    return H, views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan_folder', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nav', default=None)
    ap.add_argument('--holdout', default='frame_000332,frame_000496,frame_001152')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--no_browser', action='store_true')
    a = ap.parse_args()
    H, views = make_handler(a)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), H)
    url = 'http://127.0.0.1:%d/' % a.port
    print('annotation tool: %s  (%d views, saving to %s)  Ctrl+C to stop' % (url, len(views), os.path.abspath(a.out)),
          flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
