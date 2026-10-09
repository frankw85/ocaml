"""Overlay year / era / population captions on rendered frames and encode MP4.

    python3 compose.py --frames /path/to/frames --out aldermouth.mp4
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import timeline as tl

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
if not os.path.exists(BOLD):
    BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def fmt_year(y):
    y = int(round(y))
    return f"AD {y}" if y < 1000 else str(y)


def fmt_pop(p):
    if p >= 1e6:
        return f"Population  c. {p / 1e6:.1f} million"
    step = 1000 if p >= 10_000 else 100
    return f"Population  c. {int(round(p / step) * step):,}"


def shadow_text(img, xy, text, fnt, fill=(255, 255, 255), alpha=255, blur=3):
    """White text with a soft dark drop shadow."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((xy[0] + 2, xy[1] + 2), text, font=fnt, fill=(0, 0, 0, int(alpha * 0.7)))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    d = ImageDraw.Draw(layer)
    d.text(xy, text, font=fnt, fill=(*fill, alpha))
    img.alpha_composite(layer)


def bottom_gradient(size):
    w, h = size
    g = Image.new("L", (1, h))
    for y in range(h):
        t = max(0.0, (y - h * 0.55) / (h * 0.45))
        g.putpixel((0, y), int(150 * t ** 1.5))
    g = g.resize((w, h))
    black = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    black.putalpha(g)
    return black


def compose(img, frame):
    w, h = img.size
    s = h / 540
    img.alpha_composite(bottom_gradient(img.size))
    shadow_text(img, (int(16 * s), int(12 * s)), tl.CITY, font(BOLD, int(12 * s)), alpha=200)

    # title card, fading out into the timeline
    if frame < tl.ANIM_START + 12:
        fade = 1.0 if frame < tl.ANIM_START - 12 else max(0.0, (tl.ANIM_START + 12 - frame) / 24)
        a = int(255 * fade)
        f_sub = font(REG, int(13 * s))
        f_big = font(BOLD, int(84 * s))
        f_rng = font(REG, int(15 * s))
        d = ImageDraw.Draw(img)
        for text, fnt, y in ((tl.SUBTITLE, f_sub, h * 0.30), (tl.CITY, f_big, h * 0.35),
                             (f"AD {int(tl.YEARS[0])} - AD {int(tl.YEARS[-1])}", f_rng, h * 0.53)):
            tw = d.textlength(text, font=fnt)
            shadow_text(img, (int((w - tw) / 2), int(y)), text, fnt, alpha=a, blur=4)
        if frame < tl.ANIM_START - 12:
            return img

    year = tl.year_at(max(frame, tl.ANIM_START))
    x = int(18 * s)
    shadow_text(img, (x, int(h - 74 * s)), fmt_year(year), font(BOLD, int(46 * s)))
    shadow_text(img, (x, int(h - 104 * s)), tl.era_at(year), font(BOLD, int(13 * s)))
    shadow_text(img, (x, int(h - 88 * s)), fmt_pop(tl.population_at(year)), font(REG, int(11 * s)))

    ev = tl.event_at(year)
    if ev:
        title, desc = ev
        shadow_text(img, (x, int(h - 146 * s)), title, font(BOLD, int(15 * s)), fill=(255, 214, 150))
        shadow_text(img, (x, int(h - 126 * s)), desc, font(REG, int(11 * s)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "f_*.png")))
    with tempfile.TemporaryDirectory() as tmp:
        for i, fp in enumerate(files):
            frame = int(re.search(r"(\d+)\.png$", fp).group(1))
            img = compose(Image.open(fp).convert("RGBA"), frame)
            img.convert("RGB").save(os.path.join(tmp, f"c_{i:05d}.png"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(tl.FPS),
                        "-i", os.path.join(tmp, "c_%05d.png"), "-c:v", "libx264",
                        "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", args.out], check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
