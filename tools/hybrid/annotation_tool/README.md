# Boundary annotation pack (hybrid segmentation evaluation)

**Who:** a person (Dhruv or a colleague). Nothing in this pack has been annotated by software or by Claude.

**What:** 13 views x 7 regions = 91 crops of the raw photos (no model, no masks, no overlays). Crops were placed
from the MediaPipe detections only. Some crops will show nothing useful (e.g. a hidden ear): tick **not visible**.

**How long:** roughly 30-45 min for all 91 items.

## Start
1. Double-click `START_ANNOTATION_TOOL.bat` (needs any Python 3 on PATH, standard library only; or run
   `python serve_annotations.py --port 8790`).
2. The browser opens `http://127.0.0.1:8790/annotate.html`. Type your name (letters/digits/_ . -) and press **load**.
3. Work through the list; **PgDn** saves and moves on.

## Rules
- Trace the **real surface edge** you see in the photo, as precisely as you can (zoom in; points every ~5-10 px).
- **Never trace shadows** (especially the shadow under the chin / jaw). If you can only see a shadow, use an
  **uncertain** stroke (key **U**) or leave it out.
- Segments hidden by hair, the headband or clothing, or that you cannot place within ~2 px: **uncertain** stroke or leave out.
- Each region's own instructions are shown above the image.
- Brightness / contrast sliders only change the display.

## Where things go
Every save writes a **new** file `annotations/<your name>/<item>__<timestamp>.json` (never overwritten); the newest
file per item is the current one. Points are stored in full-image pixel coordinates (see `manifest.json`).

Held-out views (listed in `manifest.json`) are annotated like the others; their annotations are used for evaluation
only, never for fitting, gating or weight selection.
