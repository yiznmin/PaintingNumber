"""
package.py — 最終選擇打包工具

決定好要上架的組合後，修改下方 SELECTIONS 清單執行：
  py -3.10 package.py

輸出結構：
  final/<name>/<canvas>_<mode>_<difficulty>/
    original.jpg        原圖
    cropped.jpg         裁切後圖（符合畫布比例）
    template.svg        數字油畫模板
    filled.png          填色完成預覽
    palette.json        色號對照表
    color_swatches.png  顏色色卡總覽圖
    info.json           完整資訊（尺寸/難易度/顏色/定價/佔比）
"""

import cv2
import json
import math
import os
import shutil
import numpy as np

# ── 設定：要打包的組合 ────────────────────────────────────────────
SELECTIONS = [
    {
        "name":       "My",
        "canvas_cm":  (40, 60),
        "mode":       "sam_refine",
        "difficulty": "入門",
    },
    # 繼續加其他組合：
    # {"name": "My", "canvas_cm": (40, 60), "mode": "sam_refine", "difficulty": "初級"},
]
# ────────────────────────────────────────────────────────────────

IMAGES_DIR     = r"D:\website\PaintLearn\paint-by-number\images"
IMAGES_SAM_DIR = r"D:\website\PaintLearn\paint-by-number\images_sam"
OUTPUT_BASE    = r"D:\website\PaintLearn\paint-by-number\output"
FINAL_BASE     = r"D:\website\PaintLearn\paint-by-number\final"


def find_original(name):
    for d in [IMAGES_SAM_DIR, IMAGES_DIR]:
        p = os.path.join(d, f"{name}.jpg")
        if os.path.exists(p):
            return p
    return None


def crop_to_ratio(img_bgr, canvas_w_cm, canvas_h_cm, tolerance=0.01):
    """中央裁切圖片使比例符合畫布，容差 1%。"""
    ih, iw = img_bgr.shape[:2]
    target = canvas_w_cm / canvas_h_cm
    actual = iw / ih
    if abs(actual - target) < tolerance:
        return img_bgr, False
    if actual > target:
        new_w = int(ih * target)
        x0 = (iw - new_w) // 2
        return img_bgr[:, x0:x0 + new_w], True
    else:
        new_h = int(iw / target)
        y0 = (ih - new_h) // 2
        return img_bgr[y0:y0 + new_h, :], True


def make_color_swatches(palette, total_px, out_path):
    """
    產生顏色色卡總覽圖。
    每列：色塊 | 號碼 | HEX | RGB | 像素佔比
    """
    colors_sorted = sorted(palette, key=lambda x: -x.get("percent", 0))
    row_h, swatch_w, text_w = 40, 60, 320
    img_w = swatch_w + text_w
    img_h = row_h * len(colors_sorted) + 10
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 245

    for i, c in enumerate(colors_sorted):
        y = i * row_h + 5
        r, g, b = c["rgb"]
        # 色塊（BGR）
        canvas[y:y + row_h - 4, 5:swatch_w - 2] = (b, g, r)
        cv2.rectangle(canvas, (5, y), (swatch_w - 2, y + row_h - 4), (0, 0, 0), 1)
        # 文字
        pct  = c.get("percent", 0)
        text = f"#{c['id']:>2}  {c['hex']}  RGB{tuple(c['rgb'])}  {pct:.1f}%"
        cv2.putText(canvas, text, (swatch_w + 5, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)

    buf = cv2.imencode(".png", canvas)[1]
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())


def package_selection(sel):
    name       = sel["name"]
    canvas_cm  = tuple(sel["canvas_cm"])
    mode       = sel["mode"]
    difficulty = sel["difficulty"]
    canvas_str = f"{canvas_cm[0]}x{canvas_cm[1]}"

    # 來源資料夾
    src_dir = os.path.join(OUTPUT_BASE, name, mode, canvas_str, difficulty)
    if not os.path.exists(src_dir):
        print(f"❌ 找不到來源：{src_dir}")
        return

    # 目的資料夾
    folder_name = f"{canvas_str}_{mode}_{difficulty}"
    dst_dir = os.path.join(FINAL_BASE, name, folder_name)
    os.makedirs(dst_dir, exist_ok=True)

    # ── 1. 複製模板與結果 ────────────────────────────────────
    for fname in ["template.svg", "filled.png", "palette.json"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))

    # ── 2. 原圖 ─────────────────────────────────────────────
    orig_path = find_original(name)
    if orig_path:
        shutil.copy2(orig_path, os.path.join(dst_dir, "original.jpg"))
        orig_bgr = cv2.imdecode(np.fromfile(orig_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        orig_h, orig_w = orig_bgr.shape[:2]

        # ── 3. 裁切後原圖 ────────────────────────────────────
        cropped, was_cropped = crop_to_ratio(orig_bgr, canvas_cm[0], canvas_cm[1])
        crop_h, crop_w = cropped.shape[:2]
        buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        with open(os.path.join(dst_dir, "cropped.jpg"), "wb") as f:
            f.write(buf.tobytes())
    else:
        orig_w, orig_h, crop_w, crop_h, was_cropped = 0, 0, 0, 0, False

    # ── 4. 讀 summary.json ───────────────────────────────────
    summary_path = os.path.join(src_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    palette  = summary.get("colors", [])
    total_px = sum(c.get("pixels", 0) for c in palette)

    # ── 5. 色卡圖 ────────────────────────────────────────────
    if palette:
        make_color_swatches(palette, total_px, os.path.join(dst_dir, "color_swatches.png"))

    # ── 6. info.json ─────────────────────────────────────────
    info = {
        "image":          name,
        "canvas_cm":      {"width": canvas_cm[0], "height": canvas_cm[1]},
        "mode":           mode,
        "difficulty":     difficulty,
        "original_size":  {"width": orig_w, "height": orig_h},
        "cropped_size":   {"width": crop_w, "height": crop_h},
        "was_cropped":    was_cropped,
        "min_ratio":      summary.get("params", {}).get("min_ratio"),
        "num_colors":     len(palette),
        "colors": [
            {
                "id":      c["id"],
                "hex":     c["hex"],
                "rgb":     c["rgb"],
                "pixels":  c.get("pixels", 0),
                "percent": c.get("percent", 0),
            }
            for c in sorted(palette, key=lambda x: -x.get("percent", 0))
        ],
        "pricing_info":   summary.get("pricing_info", {}),
        "generated_at":   summary.get("generated_at", ""),
    }
    with open(os.path.join(dst_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✅ 打包完成：{dst_dir}")
    print(f"   尺寸：{orig_w}×{orig_h} → 裁切 {crop_w}×{crop_h}  畫布：{canvas_cm[0]}×{canvas_cm[1]} cm")
    print(f"   模式：{mode}  難易度：{difficulty}  顏色數：{len(palette)}")


def main():
    for sel in SELECTIONS:
        print(f"\n{'='*50}")
        print(f"  打包：{sel['name']} / {sel['canvas_cm']} / {sel['mode']} / {sel['difficulty']}")
        print(f"{'='*50}")
        package_selection(sel)
    print(f"\n全部完成，結果在：{FINAL_BASE}")


if __name__ == "__main__":
    main()
