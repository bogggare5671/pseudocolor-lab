#!/usr/bin/env python3
"""
curve_paint.py — окраска изображения кусочно-линейными кривыми R, G, B.

Кривые задаются ТОЛЬКО ключевыми точками (изломами). Всё, что между ними,
достраивается линейно: поставил отсчёты 0 и 4 — значения 1, 2, 3 появятся сами.

Режимы (что подставляется в аргумент кривой u из [0,1]):
  lut      — u = значение самого пикселя      -> цветокоррекция кривыми
  tint     — u = номер кадра                  -> кадр целиком красится одним цветом
  gradient — u = координата по ширине кадра   -> градиент вдоль строки развёртки

Цветовое пространство (--space) задаёт, К КАКИМ ТРЁМ КАНАЛАМ применяются кривые:
  srgb   R, G, B как есть (гамма-кодированные)
  linear R, G, B в линейном свете
  ycbcr  Y, Cb, Cr  (BT.601)
  lab    L, a, b    (CIE Lab, D65)

Запуск одной кнопкой (сам сделает тестовую картинку, если своей нет):
  python3 curve_paint.py
Свои данные:
  python3 curve_paint.py --image photo.jpg --curves curves_var1.json --mode tint --space ycbcr
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- кривые


def load_curves(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = float(data.get("n", 0))
    curves = {}
    for ch in ("R", "G", "B", "E"):
        if ch not in data:
            continue
        pts = np.asarray(data[ch], dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]
        span = (n - 1) if n > 1 else xs.max()
        curves[ch] = (xs / span, ys / 255.0)
    for ch in ("R", "G", "B"):
        if ch not in curves:
            raise SystemExit(f"в файле кривых нет канала {ch}")
    return curves, int(n)


def curve(curves, ch, u):
    """Кусочно-линейная интерполяция по ключевым точкам. u и результат в [0,1]."""
    xs, ys = curves[ch]
    return np.interp(np.clip(u, 0.0, 1.0), xs, ys)


# ------------------------------------------------- цветовые пространства

M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
M_RGB = np.linalg.inv(M_XYZ)
WP = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def to_space(rgb, space):
    """rgb: (..., 3) в [0,1] (sRGB). Возврат: три канала, нормированные в [0,1]."""
    if space == "srgb":
        return rgb.copy()
    if space == "linear":
        return srgb_to_linear(rgb)
    if space == "ycbcr":
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) / 1.772 + 0.5
        cr = (r - y) / 1.402 + 0.5
        return np.stack([y, cb, cr], axis=-1)
    if space == "lab":
        xyz = srgb_to_linear(rgb) @ M_XYZ.T / WP
        eps, kap = 216 / 24389, 24389 / 27
        f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16) / 116)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        bb = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L / 100.0, (a + 128) / 255.0, (bb + 128) / 255.0], axis=-1)
    raise SystemExit(f"неизвестное пространство: {space}")


def from_space(ch, space):
    """Обратно в sRGB [0,1]."""
    if space == "srgb":
        return np.clip(ch, 0.0, 1.0)
    if space == "linear":
        return linear_to_srgb(ch)
    if space == "ycbcr":
        y, cb, cr = ch[..., 0], ch[..., 1] - 0.5, ch[..., 2] - 0.5
        r = y + 1.402 * cr
        b = y + 1.772 * cb
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    if space == "lab":
        L = ch[..., 0] * 100.0
        a = ch[..., 1] * 255.0 - 128
        bb = ch[..., 2] * 255.0 - 128
        fy = (L + 16) / 116
        fx, fz = fy + a / 500, fy - bb / 200
        eps, kap = 216 / 24389, 24389 / 27
        f = np.stack([fx, fy, fz], axis=-1)
        xyz = np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kap)
        rgb = (xyz * WP) @ M_RGB.T
        return linear_to_srgb(rgb)
    raise SystemExit(f"неизвестное пространство: {space}")


# ----------------------------------------------------------- применение


def frame_lut(rgb, curves, space, t):
    """u = значение самого канала; t плавно вводит кривую (0 -> оригинал)."""
    ch = to_space(rgb, space)
    out = np.stack([curve(curves, c, ch[..., i]) for i, c in enumerate("RGB")], axis=-1)
    return from_space(ch * (1 - t) + out * t, space)


def frame_tint(rgb, curves, space, t):
    """u = номер кадра: один цвет на весь кадр, умножением по яркости."""
    color = np.array([curve(curves, c, t) for c in "RGB"])
    return from_space(to_space(rgb, space) * color, space)


def frame_gradient(rgb, curves, space, t, shift=0.0):
    """u = координата по ширине кадра, со сдвигом t (строка развёртки едет)."""
    w = rgb.shape[1]
    u = (np.linspace(0.0, 1.0, w)[None, :] + shift) % 1.0
    ch = to_space(rgb, space)
    color = np.stack([curve(curves, c, u) for c in "RGB"], axis=-1)
    return from_space(ch * color, space)


# ---------------------------------------------------------- вспомогательное


def test_image(w=480, h=320):
    """Тестовая картинка: серый клин + цветные поля + мелкая деталь."""
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.repeat(np.repeat(x, h, axis=0), 3, axis=2) * (0.35 + 0.65 * y)
    bars = np.zeros((h // 4, w, 3))
    for i, c in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]):
        bars[:, i * w // 6:(i + 1) * w // 6] = c
    img[: h // 4] = bars
    img[h // 2 - 2: h // 2 + 2, ::8] = 1.0
    return np.clip(img, 0, 1)


def save(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def contact_sheet(frames, path, cols=4):
    h, w, _ = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3))
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    save(sheet, path)


def plot_curves(curves, n, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    u = np.linspace(0, 1, 512)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, col in zip("RGB", ("#d62728", "#2ca02c", "#1f77b4")):
        xs, ys = curves[c]
        ax[0].plot(u * (n - 1), curve(curves, c, u) * 255, color=col, label=c)
        ax[0].plot(xs * (n - 1), ys * 255, "o", color=col, ms=4)
    ax[0].set_xlabel("отсчёт"), ax[0].set_ylabel("значение"), ax[0].legend()
    ax[0].set_ylim(-8, 264), ax[0].grid(alpha=.3), ax[0].set_title("кривые по ключевым точкам")

    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])  # B, R, G
    tri = V[[0, 1, 2, 0]]
    ax[1].plot(tri[:, 0], tri[:, 1], "k-", lw=1)
    rgb = np.array([[curve(curves, c, t) for c in "RGB"] for t in u])
    s = rgb.sum(axis=1, keepdims=True)
    ok = s[:, 0] > 1e-6
    bc = np.zeros_like(rgb)
    bc[ok] = rgb[ok] / s[ok]
    pt = bc[ok] @ V[[1, 2, 0]]  # r->R, g->G, b->B
    ax[1].plot(pt[:, 0], pt[:, 1], "-", color="#1d9e75", lw=2)
    ax[1].plot(*(np.ones(3) / 3 @ V[[1, 2, 0]]), "ko", ms=4)
    for (x, y), lab in zip(V, ("B", "R", "G")):
        ax[1].annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))
    ax[1].set_aspect("equal"), ax[1].axis("off"), ax[1].set_title("траектория цветности")
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
    return path


# ------------------------------------------------------------------ main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="исходное изображение (без него — тестовая картинка)")
    p.add_argument("--curves", default="curves_var1.json")
    p.add_argument("--mode", default="lut", choices=["lut", "tint", "gradient"])
    p.add_argument("--space", default="srgb", choices=["srgb", "linear", "ycbcr", "lab"])
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--out", default="out")
    p.add_argument("--all-spaces", action="store_true", help="прогнать все четыре пространства")
    args = p.parse_args()

    curves, n = load_curves(args.curves)
    if args.image:
        rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=float) / 255.0
    else:
        rgb = test_image()

    spaces = ["srgb", "linear", "ycbcr", "lab"] if args.all_spaces else [args.space]
    for space in spaces:
        root = os.path.join(args.out, f"{args.mode}_{space}")
        os.makedirs(os.path.join(root, "frames"), exist_ok=True)
        frames = []
        for k in range(args.frames):
            t = k / max(args.frames - 1, 1)
            if args.mode == "lut":
                f = frame_lut(rgb, curves, space, t)
            elif args.mode == "tint":
                f = frame_tint(rgb, curves, space, t)
            else:
                f = frame_gradient(rgb, curves, space, t, shift=t)
            frames.append(f)
            save(f, os.path.join(root, "frames", f"f{k:03d}.png"))
        contact_sheet(frames, os.path.join(root, "contact_sheet.png"))
        try:
            import imageio.v2 as imageio
            imageio.mimsave(os.path.join(root, "anim.mp4"),
                            [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames], fps=6)
        except Exception:
            pass
        print(f"{root}: {args.frames} кадров + contact_sheet.png")

    fig = plot_curves(curves, n, os.path.join(args.out, "curves.png"))
    if fig:
        print(f"{fig}: кривые и траектория цветности")


if __name__ == "__main__":
    main()
