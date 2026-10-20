'''
Textured fits vs the photo silhouettes (no fitting). For each run the texture baked from its TRAINING views
(tools/bake_texture_image_only.py -> <run>/texture) is rendered (unlit, open3d ray casting) into all 13 views in
the run's state: training views = fitted, held-out views = interpolated prediction (texture never saw them).
Per view (head / neck crop):
  photo, render of each run composited over the dimmed photo (silhouette mismatch shows as dim photo / render
  spilling into the background), and an outline overlay: person matte (white), segmenter skin (green), model
  silhouette of each run (run colours).
Writes <out>/views/*.png, <out>/panels/*.jpg, <out>/contact_sheet.jpg and <out>/index.html (local flip viewer).
'''
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import cv2
from PIL import Image, ImageDraw

import fitting.py310_compat  # noqa: F401
import fit_eval
from fitting import identity_io as IO, render as RD
from tools import bake_texture as BT
from tools import bake_texture_image_only as BI
from image_only import common as CM

COLS = [(255, 60, 60), (0, 230, 255), (255, 200, 0), (200, 90, 255)]


def outline(mask, it=1):
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8), iterations=it)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--names', required=True)
    ap.add_argument('--capture', required=True); ap.add_argument('--skinseg', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit('refusing to overwrite %s' % a.out)
    for d in ('views', 'panels'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    runs, names = a.runs.split(','), a.names.split(',')
    npz_path = fit_eval.gnm_npz_path()
    npz = np.load(npz_path)
    g = IO.GNMNumpy(npz_path)
    F = g.triangles.astype(np.int64)
    mats = BT.triangle_materials(npz, F)
    meta = json.load(open(os.path.join(runs[0], 'fit', 'config_and_inputs.json')))
    hold = set(meta['config']['holdout'])
    cams, det, _ = CM.load_capture(a.capture, meta['inputs']['detections'])
    labels = [c.label for c in cams]
    R = {}
    for name, run in zip(names, runs):
        Z, states = BI.load_states(run, labels, [CM.frame_index(l) for l in labels])
        ident = np.asarray(Z['identity'], np.float64)
        tex = RD.load_textures(os.path.join(run, 'texture'), BT.MATERIALS)
        R[name] = (ident, states, tex)
    index = []
    sheet = []
    for k, c in enumerate(cams):
        im = np.asarray(Image.open(c.image_path))
        rgb, A = im[..., :3], im[..., 3] > 127
        H, W = A.shape
        P = det['dense_478'][k][:, :2]
        s = float(np.linalg.norm(P[10] - P[152]))
        x0, x1 = int(max(P[:, 0].min() - 0.45 * s, 0)), int(min(P[:, 0].max() + 0.45 * s, W))
        y0, y1 = int(max(P[:, 1].min() - 0.35 * s, 0)), int(min(P[152][1] + 0.9 * s, H))
        sk = sum(np.asarray(Image.open(os.path.join(a.skinseg, c.label, 'p_%s.png' % cl)), np.float32) / 255 for cl in ('body_skin', 'face_skin')) >= 0.5
        kind = 'held-out (predicted state)' if c.label in hold else 'training (fitted state)'
        crop = lambda x: x[y0:y1, x0:x1]
        files = dict(photo='views/%s__photo.png' % c.label)
        Image.fromarray(crop(rgb)).save(os.path.join(a.out, files['photo']))
        ov = (crop(rgb).astype(np.float32) * 0.5)
        ov[crop(outline(A))] = (255, 255, 255)
        ov[crop(outline(sk))] = (0, 255, 0)
        tiles = [crop(rgb)]
        for i, name in enumerate(names):
            ident, states, tex = R[name]
            st = states[c.label]
            bind = g.T + np.einsum('i,ivk->vk', ident, g.Bid) + np.einsum('i,ivk->vk', st.expr, g.Be)
            V = st.to_world(st.pose(g, bind, ident))
            ren, ids = RD.render_textured(V, F, g.triangle_uvs, mats, tex, c, return_ids=True, bg=0)
            M = ids >= 0
            comp = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
            comp[M] = ren[M]
            files[name] = 'views/%s__render_%s.png' % (c.label, name)
            Image.fromarray(crop(comp)).save(os.path.join(a.out, files[name]))
            ov[crop(outline(M))] = COLS[i % len(COLS)]
            tiles.append(crop(comp))
        files['outlines'] = 'views/%s__outlines.png' % c.label
        ovi = Image.fromarray(ov.clip(0, 255).astype(np.uint8))
        ovi.save(os.path.join(a.out, files['outlines']))
        tiles.append(np.asarray(ovi))
        h = tiles[0].shape[0]
        pan = np.concatenate(tiles, 1)
        pim = Image.fromarray(pan)
        d = ImageDraw.Draw(pim)
        wt = tiles[0].shape[1]
        for j, t in enumerate(['photo'] + ['%s textured' % n for n in names] + ['outlines']):
            d.rectangle([j * wt, 0, j * wt + 260, 18], fill=(0, 0, 0)); d.text((j * wt + 4, 3), '%s  %s' % (c.label, t), fill=(255, 255, 0))
        d.text((4, h - 16), kind + ' | outlines: white = person matte, green = segmenter skin, ' +
               ', '.join('%s = %s' % (n, ['red', 'cyan', 'yellow', 'violet'][i]) for i, n in enumerate(names)), fill=(255, 255, 0))
        pim.save(os.path.join(a.out, 'panels', '%s.jpg' % c.label), quality=90)
        sheet.append(pim)
        index.append(dict(view=c.label, kind=kind, files=files))
        print(c.label, flush=True)
    T = 1400
    th = [p.resize((T, int(p.height * T / p.width))) for p in sheet]
    S = Image.new('RGB', (T, sum(t.height for t in th)))
    y = 0
    for t in th:
        S.paste(t, (0, y)); y += t.height
    S.save(os.path.join(a.out, 'contact_sheet.jpg'), quality=85)
    legend = dict(names=names, colours=['red', 'cyan', 'yellow', 'violet'][:len(names)])
    html = HTML.replace('__DATA__', json.dumps(dict(views=index, legend=legend)))
    open(os.path.join(a.out, 'index.html'), 'w').write(html)
    print('done')


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Textured fits vs photo</title>
<style>body{margin:0;background:#1c1d20;color:#ddd;font:13px system-ui,Segoe UI,sans-serif;display:flex;height:100vh}
#side{width:230px;overflow:auto;background:#26282c;border-right:1px solid #000}#side div{padding:4px 10px;cursor:pointer}
#side div.cur{background:#0d47a1}#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:6px 10px;background:#26282c;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:#3b4046;color:#ddd;border:1px solid #555;border-radius:3px;padding:3px 9px;cursor:pointer}button.on{background:#1565c0}
#wrap{flex:1;overflow:auto;position:relative}#stack{position:relative;display:inline-block;transform-origin:0 0}
#stack img{display:block}#top{position:absolute;left:0;top:0}#help{color:#999;font-size:12px}</style></head>
<body><div id="side"></div><div id="main"><div id="bar"><b id="title"></b><span id="btns"></span>
overlay opacity <input type="range" id="op" min="0" max="100" value="100">
zoom <input type="range" id="zm" min="50" max="400" value="100">
<span id="help">keys: 1..9 layers, space = flip photo / last layer, up/down = view</span></div>
<div id="wrap"><div id="stack"><img id="base"><img id="top"></div></div></div>
<script>
const D=__DATA__;let vi=0,layer='photo',last='photo';const $=i=>document.getElementById(i);
const layers=()=>['photo',...D.legend.names,'outlines'];
function side(){$('side').innerHTML='';D.views.forEach((v,i)=>{const d=document.createElement('div');d.textContent=v.view.replace('frame_','')+(v.kind.startsWith('held')?' (held-out)':'');
 if(i===vi)d.className='cur';d.onclick=()=>{vi=i;show()};$('side').appendChild(d)})}
function show(){const v=D.views[vi];$('title').textContent=v.view+'  '+v.kind;$('base').src=v.files.photo;$('top').src=v.files[layer];
 $('top').style.opacity=layer==='photo'?0:$('op').value/100;$('btns').innerHTML='';layers().forEach((l,i)=>{const b=document.createElement('button');
 b.textContent=(i+1)+' '+l;if(l===layer)b.className='on';b.onclick=()=>{set(l)};$('btns').appendChild(b)});side()}
function set(l){if(l!=='photo')last=l;layer=l;show()}
$('op').oninput=show;$('zm').oninput=()=>{$('stack').style.transform='scale('+$('zm').value/100+')'};
window.onkeydown=e=>{const L=layers();if(e.key>='1'&&e.key<=String(L.length)){set(L[+e.key-1])}
 else if(e.key===' '){e.preventDefault();set(layer==='photo'?(last||L[1]):'photo')}
 else if(e.key==='ArrowDown'){vi=Math.min(D.views.length-1,vi+1);show()}else if(e.key==='ArrowUp'){vi=Math.max(0,vi-1);show()}};
last=D.legend.names[D.legend.names.length-1];show();
</script></body></html>"""

if __name__ == '__main__':
    main()
