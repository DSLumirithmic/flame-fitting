'''
Local, offline server for the GNM comparison viewer (standard library only).

  python tools/viewer/serve.py --data <run>/viewer_data [--port 8765] [--no_browser]

Serves this folder at /, the data folder (made by build_viewer_data.py) at
/data/, and accepts POST /api/screenshot so "Save screenshot" writes PNGs to
<data>/screenshots/. Binds to 127.0.0.1 only; nothing leaves the machine.
A browser is needed only because WebGL module scripts cannot be loaded from
file:// URLs.
'''
import argparse
import base64
import http.server
import json
import os
import re
import sys
import time
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
MIME = {'.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
        '.bin': 'application/octet-stream', '.html': 'text/html', '.css': 'text/css',
        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.txt': 'text/plain'}


def make_handler(data_dir):
    data_dir = os.path.abspath(data_dir)

    class H(http.server.SimpleHTTPRequestHandler):
        # Windows can map .js to text/plain through the registry, which breaks
        # ES modules; force the types we serve.
        extensions_map = dict(http.server.SimpleHTTPRequestHandler.extensions_map, **MIME)

        def translate_path(self, path):
            p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
            if p.startswith('/data/'):
                root, rel = data_dir, p[len('/data/'):]
            else:
                root, rel = HERE, p.lstrip('/')
            full = os.path.abspath(os.path.join(root, *[x for x in rel.split('/') if x not in ('', '.', '..')]))
            return full if full.startswith(root) else root

        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache')
            super().end_headers()

        def log_message(self, fmt, *args):
            if '/api/' in (args[0] if args else ''):
                sys.stderr.write('%s\n' % (fmt % args))

        def do_POST(self):
            if self.path != '/api/screenshot':
                self.send_error(404)
                return
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n))
            name = re.sub(r'[^A-Za-z0-9_.-]+', '_', req.get('name', 'screenshot'))[:120]
            png = base64.b64decode(req['png'].split(',', 1)[-1])
            d = os.path.join(data_dir, 'screenshots')
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, '%s_%s.png' % (time.strftime('%Y%m%d_%H%M%S'), name))
            with open(path, 'wb') as f:
                f.write(png)
            body = json.dumps({'saved': path}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder written by build_viewer_data.py')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no_browser', action='store_true')
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.data, 'manifest.json')):
        sys.exit('%s has no manifest.json -- run tools/viewer/build_viewer_data.py first' % a.data)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(a.data))
    url = 'http://127.0.0.1:%d/' % a.port
    print('GNM comparison viewer: %s  (data: %s)  Ctrl+C to stop' % (url, os.path.abspath(a.data)), flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
