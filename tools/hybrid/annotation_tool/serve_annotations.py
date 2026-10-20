'''
Local server for the annotation pack (Python standard library only).

  python serve_annotations.py [--port 8790]      (run from inside the pack folder, or pass --pack)
  then open http://127.0.0.1:8790/annotate.html

GET  /...                         static files of the pack
GET  /api/latest?annotator=NAME   latest saved annotation per item for that annotator
POST /api/save                    body = one item's annotation JSON; written to
                                  annotations/<annotator>/<item>__<YYYYmmdd_HHMMSS_ffffff>.json
Every save is a NEW file (nothing is ever overwritten); the newest file per item is the current one.
Binds to 127.0.0.1 only.
'''
import argparse, datetime, http.server, json, os, re, urllib.parse

SAFE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--pack', default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    pack = os.path.abspath(a.pack)
    man = json.load(open(os.path.join(pack, 'manifest.json')))
    ids = {it['id'] for it in man['items']}
    adir = os.path.join(pack, 'annotations')
    os.makedirs(adir, exist_ok=True)

    def latest(annot):
        d = os.path.join(adir, annot)
        out = {}
        if not os.path.isdir(d):
            return out
        for f in sorted(os.listdir(d)):
            if f.endswith('.json') and '__' in f:
                iid = f.rsplit('__', 1)[0]
                out[iid] = f                       # sorted -> last one wins (timestamped names)
        return {k: json.load(open(os.path.join(d, v))) for k, v in out.items()}

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=pack, **kw)

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == '/api/latest':
                q = urllib.parse.parse_qs(u.query)
                an = (q.get('annotator') or [''])[0]
                if not SAFE.match(an):
                    return self._json(400, dict(error='bad annotator name (letters, digits, _ . - only)'))
                return self._json(200, latest(an))
            return super().do_GET()

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != '/api/save':
                return self._json(404, dict(error='not found'))
            n = int(self.headers.get('Content-Length', 0))
            try:
                d = json.loads(self.rfile.read(n))
            except Exception as e:
                return self._json(400, dict(error='bad json: %s' % e))
            an, iid = d.get('annotator', ''), d.get('item', '')
            if not SAFE.match(an) or iid not in ids:
                return self._json(400, dict(error='bad annotator or item'))
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            d['saved_at'] = ts
            os.makedirs(os.path.join(adir, an), exist_ok=True)
            p = os.path.join(adir, an, '%s__%s.json' % (iid, ts))
            assert not os.path.exists(p)
            with open(p, 'x') as f:
                json.dump(d, f, indent=1)
            return self._json(200, dict(ok=True, file=os.path.relpath(p, pack)))

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), H)
    print('annotation tool: http://127.0.0.1:%d/annotate.html   (pack %s; Ctrl+C to stop)' % (a.port, pack), flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
