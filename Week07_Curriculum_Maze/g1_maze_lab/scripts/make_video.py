"""Assemble recorded frame folders into one mp4 with Thai title cards between clips.

usage: make_video.py <out.mp4> <spec.json>
spec: [{"title": "...", "sub": "...", "dir": "frames folder" (optional), "hold": 1.5}, ...]
Title cards are 2.5 s; a clip's last frame is held `hold` seconds. 25 fps, 1280x720, H.264.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FPS, W, H = 25, 1280, 720
THAI = "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"
THAI_B = "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"
LAT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LAT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path, size):
    return ImageFont.truetype(path if os.path.exists(path) else THAI, size)


def draw_mixed(d, xy, text, size, fill, bold=False):
    """Draw text mixing a Thai font (Thai runs) and a Latin font (everything else) on one baseline."""
    th, la = _font(THAI_B if bold else THAI, size), _font(LAT_B if bold else LAT, int(size * 0.92))
    x, y = xy
    runs, cur, is_th = [], "", None
    for ch in text:
        t = "฀" <= ch <= "๿"
        if is_th is None or t == is_th or ch == " ":
            cur += ch
            is_th = t if is_th is None or ch != " " else is_th
        else:
            runs.append((cur, is_th))
            cur, is_th = ch, t
    runs.append((cur, is_th))
    for s, t in runs:
        f = th if t else la
        d.text((x, y), s, font=f, fill=fill, anchor="ls")
        x += d.textlength(s, font=f)


def title_card(path, title, sub):
    im = Image.new("RGB", (W, H), "#0E1621")
    d = ImageDraw.Draw(im)
    d.rectangle((80, 300, 88, 420), fill="#7AC943")
    draw_mixed(d, (120, 360), title, 58, "#F4F3EE", bold=True)
    draw_mixed(d, (120, 420), sub, 32, "#B8C4D0")
    im.save(path, quality=92)


def main():
    out, spec = sys.argv[1], json.load(open(sys.argv[2]))
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)))
    n = 0

    def put(src, count):
        nonlocal n
        for _ in range(count):
            os.link(src, f"{tmp}/{n:06d}.jpg")
            n += 1

    for k, item in enumerate(spec):
        card = f"{tmp}/card_{k}.jpg"
        title_card(card, item["title"], item.get("sub", ""))
        put(card, int(item.get("card", 2.5) * FPS))
        if item.get("dir"):
            frames = sorted(f for f in os.listdir(item["dir"]) if f.endswith(".jpg"))
            for f in frames:
                put(os.path.join(item["dir"], f), 1)
            put(os.path.join(item["dir"], frames[-1]), int(item.get("hold", 1.5) * FPS))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{tmp}/%06d.jpg", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp)
    print(f"{out}: {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
