'''
Adapter: turn a 2DGS reconstruction run into the scan-folder layout that
run_gnm_pipeline.py / generate_scan_landmarks.py expect:

    <out>/model.obj        fused mesh, scaled to metres (--scale)
    <out>/calibration.xml  Metashape-style camera export built from cameras.json
    <out>/views/<label>.png  the undistorted images 2DGS trained on

cameras.json is the 3DGS/2DGS export: 'rotation' is the camera->world rotation,
'position' the camera centre, OpenCV axes (COLMAP), PINHOLE intrinsics. Those
images are already undistorted, so the Metashape sensor gets k1..p2 = 0 and
cx/cy = principal-point offset from the image centre (Metashape convention).

--scale converts scene units to metres. SfM scenes have arbitrary scale; the
fit re-estimates scale itself (landmark Umeyama), but fit_eval's thresholds
and the reported mm are in the scan's own units, so the scan has to be
roughly metric going in. A good --scale makes the pipeline's own Umeyama
scale come out ~1.0.
'''
import argparse, json, os, shutil
import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--scale', type=float, required=True, help='metres per scene unit')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'views'), exist_ok=True)

    m = trimesh.load(a.ply, process=False, force='mesh')
    v = np.asarray(m.vertices, np.float64) * a.scale
    with open(os.path.join(a.out, 'model.obj'), 'w') as fh:
        np.savetxt(fh, v, fmt='v %.7f %.7f %.7f')
        np.savetxt(fh, np.asarray(m.faces, np.int64) + 1, fmt='f %d %d %d')
    print('model.obj: %d verts, %d faces, scale %.6f m/unit' % (len(v), len(m.faces), a.scale))

    cams = json.load(open(a.cameras))
    sensors, cam_xml = {}, []
    for c in cams:
        key = (c['width'], c['height'], round(c['fx'], 6), round(c['fy'], 6))
        if key not in sensors:
            sensors[key] = len(sensors)
        assert abs(c['fx'] - c['fy']) < 1e-3, 'Metashape model here assumes square pixels'
        T = np.eye(4)
        T[:3, :3] = np.asarray(c['rotation'], np.float64)
        T[:3, 3] = np.asarray(c['position'], np.float64) * a.scale
        cam_xml.append('      <camera id="%d" sensor_id="%d" label="%s">\n        <transform>%s</transform>\n      </camera>'
                       % (c['id'], sensors[key], c['img_name'], ' '.join('%.12g' % x for x in T.ravel())))
        src = None
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(a.images, c['img_name'] + ext)
            if os.path.exists(p):
                src = p; break
        if src is None:
            print('[WARN] no image for', c['img_name']); continue
        shutil.copy2(src, os.path.join(a.out, 'views', os.path.basename(src)))
    sens_xml = []
    for (w, h, fx, fy), sid in sensors.items():
        cx = 0.0  # 2DGS PINHOLE export: principal point at the image centre
        sens_xml.append(
            '      <sensor id="%d" label="2dgs" type="frame">\n        <resolution width="%d" height="%d"/>\n'
            '        <calibration type="frame" class="adjusted">\n          <resolution width="%d" height="%d"/>\n'
            '          <f>%.10g</f>\n          <cx>%g</cx>\n          <cy>%g</cy>\n'
            '          <k1>0</k1>\n          <k2>0</k2>\n          <k3>0</k3>\n          <p1>0</p1>\n          <p2>0</p2>\n'
            '        </calibration>\n      </sensor>' % (sid, w, h, w, h, fx, cx, cx))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<document version="1.4.0">\n  <chunk label="2dgs">\n'
           '    <sensors>\n%s\n    </sensors>\n    <cameras>\n%s\n    </cameras>\n  </chunk>\n</document>\n'
           % ('\n'.join(sens_xml), '\n'.join(cam_xml)))
    open(os.path.join(a.out, 'calibration.xml'), 'w').write(xml)
    print('calibration.xml: %d cameras, %d sensor(s)' % (len(cam_xml), len(sensors)))


if __name__ == '__main__':
    main()
