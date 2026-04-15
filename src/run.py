"""
run.py — 數字油畫主入口

只需修改下方設定區，執行：
  py -3.10 run.py

三種模式：
  standard     — 直接轉換，不需遮罩
  sam_refine   — 選取區額外加色（基礎 N + 額外 M），需要遮罩
  sam_weighted — 選取區佔色數比重較高，非選取區更簡化，需要遮罩
"""

import cv2
import math
import os
import json
import numpy as np
from datetime import datetime
from pbn_gen import PbnGen

# ── 設定區（只需改這裡）────────────────────────────────
NAME         = "Mom"
STYLE_TAGS   = ["人物","自拍","卡通"]
MODE         = "sam_refine"   # "standard" / "sam_refine" / "sam_weighted"
DETAIL       = "細緻"          # "粗糙" / "標準" / "細緻" / "高級"
COMPARE_DETAILS =   False      # True = 跑所有細緻度，False = 只跑 DETAIL
COMPARE_DIFFICULTY = "中級"   # 比較模式下跑哪個難度，None = 跑所有難度

# sam_refine 參數
EXTRA_COLORS  = 10   # 選取區額外增加幾色

# sam_weighted 參數
WEIGHT_RATIO  = 0.65  # 選取區佔總色數比例（0.5~0.8）

# 畫布尺寸規格
# None = 根據圖片比例自動推薦 3 個尺寸
# 手動指定範例：[(30, 45), (40, 60), (50, 75)]
CANVAS_SIZES_CM   = None
MIN_BRUSH_DIAM_CM = 1  # 最小可塗色塊直徑（公分），建議 1.0~2.0
# ────────────────────────────────────────────────────────

IMAGES_DIR     = r"D:\website\PaintLearn\paint-by-number\images"
IMAGES_SAM_DIR = r"D:\website\PaintLearn\paint-by-number\images_sam"
OUTPUT_BASE    = r"D:\website\PaintLearn\paint-by-number\output"
STATS_PATH     = os.path.join(OUTPUT_BASE, "color_stats.json")


# 所有支援的畫布規格（寬, 高），單位公分
_STANDARD_SIZES = [
    # 正方形
    (20, 20), (30, 30), (40, 40), (50, 50), (60, 60),
    # 直幅
    (30, 40), (30, 50), (30, 60),
    (40, 50), (40, 60),
    (50, 60),
    # 橫幅
    (40, 30), (50, 30), (60, 30),
    (50, 40), (60, 40),
    (60, 50),
]


def suggest_canvas_sizes(img_w_px, img_h_px, n=3):
    """
    根據圖片長寬比，從標準畫布規格中挑出比例最接近的 n 個，
    由小到大排列。
    """
    img_ratio = img_w_px / img_h_px
    scored = []
    for w, h in _STANDARD_SIZES:
        ratio_diff = abs((w / h) - img_ratio)
        area = w * h
        scored.append((ratio_diff, area, (w, h)))
    scored.sort(key=lambda x: (round(x[0], 3), x[1]))
    # 取比例最接近的，且面積由小到大選 n 個不重複比例
    seen_ratios, result = set(), []
    for _, area, size in scored:
        r = round(size[0] / size[1], 3)
        if r not in seen_ratios:
            seen_ratios.add(r)
            result.append(size)
        if len(result) == n:
            break
    # 按面積從小到大排序
    result.sort(key=lambda s: s[0] * s[1])
    return result


def crop_to_canvas_ratio(img_bgr, mask, canvas_w_cm, canvas_h_cm):
    """
    將圖片（和遮罩）中央裁切成符合畫布比例。
    回傳 (cropped_bgr, cropped_mask)，cropped_mask 為 None 若輸入 mask 為 None。
    """
    ih, iw = img_bgr.shape[:2]
    target_ratio = canvas_w_cm / canvas_h_cm
    img_ratio    = iw / ih

    if abs(img_ratio - target_ratio) < 0.01:   # 比例差距 < 1%，不裁切
        return img_bgr, mask

    if img_ratio > target_ratio:
        # 圖片太寬 → 裁左右，保留全高
        new_w = int(ih * target_ratio)
        x0 = (iw - new_w) // 2
        cropped = img_bgr[:, x0:x0 + new_w]
        cropped_mask = mask[:, x0:x0 + new_w] if mask is not None else None
    else:
        # 圖片太高 → 裁上下，保留全寬
        new_h = int(iw / target_ratio)
        y0 = (ih - new_h) // 2
        cropped = img_bgr[y0:y0 + new_h, :]
        cropped_mask = mask[y0:y0 + new_h, :] if mask is not None else None

    cih, ciw = cropped.shape[:2]
    if ciw != iw or cih != ih:
        print(f"  裁切：{iw}×{ih} → {ciw}×{cih}（符合 {canvas_w_cm}:{canvas_h_cm} 比例）")
    return cropped, cropped_mask


def calc_min_ratio(canvas_w_cm, img_w_px, img_h_px,
                   min_brush_diam_cm=MIN_BRUSH_DIAM_CM):
    """根據畫布尺寸與最小筆觸直徑，計算 merge_tiny_colors 的 min_ratio。"""
    pixel_size_cm = canvas_w_cm / img_w_px          # 每 px 對應幾公分
    radius_px     = (min_brush_diam_cm / 2) / pixel_size_cm
    min_pixels    = math.pi * radius_px ** 2         # 圓形面積近似
    return min_pixels / (img_w_px * img_h_px)


def calc_min_radius_px(canvas_w_cm, img_w_px,
                       min_brush_diam_cm=MIN_BRUSH_DIAM_CM):
    """根據畫布實體尺寸，計算最小可下筆半徑（像素）。
    幾何標準：最大內切圓直徑 >= min_brush_diam_cm（預設 0.2cm）才不被合併。
    """
    pixel_size_cm = canvas_w_cm / img_w_px   # 每 px 多少公分
    return (min_brush_diam_cm / 2) / pixel_size_cm


def pricing_suggestion(sam_mask, mode, extra_colors, canvas_cm):
    """
    根據 SAM 遮罩選取資訊給出定價等級建議。
    sam_mask:     遮罩陣列（None = 無選取）
    mode:         standard / sam_refine / sam_weighted
    extra_colors: sam_refine 額外色數
    canvas_cm:    (寬, 高) 公分
    """
    w, h = canvas_cm
    lines = []

    if sam_mask is None or mode == "standard":
        lines.append("無選取細化區域（standard 模式）")
        lines.append("建議：基礎定價")
    else:
        mask_ratio = round(float((sam_mask > 127).sum()) / sam_mask.size, 3)
        lines.append(f"細化區域佔圖片 {mask_ratio*100:.1f}%，模式：{mode}")

        if mode == "sam_refine":
            if mask_ratio >= 0.25:
                lines.append(f"選取範圍大 × 細化 +{extra_colors} 色 → 整體還原度高")
                lines.append("定價參考：中～高階商品")
            else:
                lines.append(f"選取範圍小（局部細節）× 細化 +{extra_colors} 色")
                lines.append("定價參考：標準商品，局部精細")
        elif mode == "sam_weighted":
            lines.append("選取區高比重分色，非選取區簡化")
            if mask_ratio >= 0.3:
                lines.append("定價參考：中階商品")
            else:
                lines.append("定價參考：標準商品")

    lines.append(f"畫布：{w}×{h} cm")
    return lines


# 細緻度選項：根據年齡層 / 使用者喜好選擇，所有難易度共用同一套
# blur 越小、min_ratio 越低 → 格子越多越細緻（適合成人、小尺寸）
# blur 越大、min_ratio 越高 → 格子越少越粗（適合幼兒、大尺寸）
DETAIL_PRESETS = {
    "粗糙": {
        "blur_ksize": 31,
        "blur_sigma_color": 51,
        "blur_sigma_space": 51,
        "prune_iterations": 10,
        "bg_extra_blur": 21,
        "min_ratio_multiplier": 2.0,
    },
    "標準": {
        "blur_ksize": 21,
        "blur_sigma_color": 21,
        "blur_sigma_space": 14,
        "prune_iterations": 6,
        "bg_extra_blur": 21,
        "min_ratio_multiplier": 1.0,
    },
    "細緻": {
        "blur_ksize": 13,
        "blur_sigma_color": 13,
        "blur_sigma_space": 9,
        "prune_iterations": 3,
        "bg_extra_blur": 9,
        "min_ratio_multiplier": 0.6,
    },
    "高級": {
        "blur_ksize": 7,
        "blur_sigma_color": 7,
        "blur_sigma_space": 5,
        "prune_iterations": 1,
        "bg_extra_blur": 0,
        "min_ratio_multiplier": 0.3,
    },
}

# 難易度：只調顏色數量與相關門檻，細緻度由 DETAIL_PRESETS[DETAIL] 統一控制
DIFFICULTY_LEVELS = [
    {"name": "入門", "num_colors": 18, "pruning_threshold": 8e-4,    "refine_extra_colors": 8},
    {"name": "初級", "num_colors": 24, "pruning_threshold": 2e-4,    "refine_extra_colors": 12},
    {"name": "中級", "num_colors": 35, "pruning_threshold": 6.25e-5, "refine_extra_colors": 18},
    {"name": "進階", "num_colors": 50, "pruning_threshold": 1.5e-5,  "refine_extra_colors": 25},
]

# 合併選定的細緻度到每個難度
_detail = DETAIL_PRESETS.get(DETAIL, DETAIL_PRESETS["標準"])
for _lv in DIFFICULTY_LEVELS:
    _lv.update({k: v for k, v in _detail.items() if k not in _lv})


def load_sam_mask(name):
    mask_path = os.path.join(IMAGES_SAM_DIR, f"{name}_mask.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"[OK] 載入 SAM 遮罩：{mask_path}")
        return mask
    return None


def run_single_level(input_image_path, level_dir, level, mode, sam_mask,
                     canvas_cm=(40, 60), pricing_info=None):
    level_name = level["name"]
    print(f"\n{'='*40}")
    print(f"  [{level_name}] 模式：{mode}  畫布：{canvas_cm[0]}×{canvas_cm[1]} cm")
    print(f"{'='*40}")

    # 載入圖片並裁切成符合畫布比例
    img_bgr = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_bgr, sam_mask_cropped = crop_to_canvas_ratio(img_bgr, sam_mask, canvas_cm[0], canvas_cm[1])

    if mode == "sam_weighted" and sam_mask_cropped is not None:
        # weighted 模式：不走 set_final_pbn，改由 apply_weighted_region 直接量化
        pbn = PbnGen(
            img_bgr,
            num_colors=level["num_colors"],
            pruningThreshold=level["pruning_threshold"],
            fixed_palette=None,
        )
        pbn.apply_weighted_region(
            mask=sam_mask_cropped,
            total_colors=level["num_colors"],
            weight_ratio=WEIGHT_RATIO,
            bg_extra_blur=level["bg_extra_blur"],
        )
        # pruning 和邊框
        pbn.pruneClustersSimple(iterations=level["prune_iterations"])
        img = pbn.getImage()
        img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), 10)
        pbn.setImage(img)
    else:
        # standard / sam_refine：走正常流程
        pbn = PbnGen(
            img_bgr,
            num_colors=level["num_colors"],
            pruningThreshold=level["pruning_threshold"],
            fixed_palette=None,
        )
        pbn.set_final_pbn(
            blur_ksize=level["blur_ksize"],
            blur_sigma_color=level["blur_sigma_color"],
            blur_sigma_space=level["blur_sigma_space"],
            prune_iterations=level["prune_iterations"],
        )
        if mode == "sam_refine" and sam_mask_cropped is not None:
            extra_colors = level.get("refine_extra_colors", EXTRA_COLORS)
            print(f"  → 選取區細化 +{extra_colors} 色")
            pbn.refine_region(sam_mask_cropped, extra_colors=extra_colors)

    # 小色塊合併：幾何門檻（最大內切圓半徑）× 難易度倍數
    img_h, img_w = pbn.getImage().shape[:2]
    multiplier     = level.get("min_ratio_multiplier", 1.0)
    min_radius_px  = calc_min_radius_px(canvas_cm[0], img_w) * multiplier
    print(f"  畫布 {canvas_cm[0]}×{canvas_cm[1]} cm × {multiplier}× → min_radius={min_radius_px:.1f}px")
    merge_mask = sam_mask_cropped if (mode == "sam_refine" and sam_mask_cropped is not None) else None
    pbn.merge_tiny_colors(min_radius_px=min_radius_px, exclude_mask=merge_mask)

    svg_path  = os.path.join(level_dir, "template.svg")
    json_path = os.path.join(level_dir, "palette.json")

    palette_data = pbn.output_to_svg(svg_path, json_path,
                                     min_radius_px=min_radius_px,
                                     canvas_w_cm=canvas_cm[0],
                                     canvas_h_cm=canvas_cm[1])
    pbn.output_filled_from_template(os.path.join(level_dir, "filled_template.png"))

    # ── 顏色佔比分析（直接從記憶體 snapped_rgb 計算，不需要存 filled.png）─────
    filled_rgb = getattr(pbn, '_snapped_rgb', pbn.getImage().copy())
    if filled_rgb.shape[2] == 3:
        filled_rgb = cv2.cvtColor(filled_rgb, cv2.COLOR_BGR2RGB) \
                     if filled_rgb.dtype == np.uint8 else filled_rgb
    img_h, img_w = filled_rgb.shape[:2]
    total_px = img_h * img_w
    pixels_flat = filled_rgb.reshape(-1, 3)

    color_pixel_map = {}
    for item in palette_data:
        rgb = tuple(item["rgb"])
        match = np.all(pixels_flat == list(rgb), axis=1)
        color_pixel_map[item["template_id"]] = int(match.sum())
    # ──────────────────────────────────────────────────────────────────

    used_colors = sorted(palette_data, key=lambda x: x["template_id"])
    summary = {
        "image": NAME,
        "style_tags": STYLE_TAGS,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "image_size": {"width": img_w, "height": img_h},
        "difficulty": level_name,
        "params": {
            "num_colors_limit": level["num_colors"],
            "pruning_threshold": level["pruning_threshold"],
            "blur_ksize": level["blur_ksize"],
            "blur_sigma_color": level["blur_sigma_color"],
            "blur_sigma_space": level["blur_sigma_space"],
            "prune_iterations": level["prune_iterations"],
        },
        "sam_used": sam_mask is not None,
        "sam_mode": mode if sam_mask is not None else None,
        "sam_extra_colors": level.get("refine_extra_colors", EXTRA_COLORS) if mode == "sam_refine" else 0,
        "sam_weight_ratio": WEIGHT_RATIO if mode == "sam_weighted" else None,
        "num_colors_used": len(used_colors),
        "colors": [
            {
                "id": item["template_id"],
                "rgb": item["rgb"],
                "hex": "#{:02X}{:02X}{:02X}".format(*item["rgb"]),
                "pixels": color_pixel_map.get(item["template_id"], 0),
                "percent": round(
                    color_pixel_map.get(item["template_id"], 0) / total_px * 100, 2
                ),
            }
            for item in used_colors
        ],
        "canvas_cm": {"width": canvas_cm[0], "height": canvas_cm[1]},
        "min_radius_px": round(min_radius_px, 3),
        "pricing_info": pricing_info or {},
        "approved": False,
        "notes": "",
        "workflow_hints": {
            "color_issues": [],
            "param_adjustments": [],
            "recommended_for_type": STYLE_TAGS
        }
    }
    summary_path = os.path.join(level_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 印出顏色報表
    colors_by_pct = sorted(summary["colors"], key=lambda x: -x["percent"])
    print(f"\n  尺寸：{img_w} x {img_h} px　顏色數：{len(used_colors)}")
    print(f"  {'號':>3}  {'HEX':^8}  {'像素數':>8}  {'佔比':>6}")
    print(f"  {'-'*34}")
    for c in colors_by_pct:
        print(f"  {c['id']:>3}  {c['hex']}  {c['pixels']:>8,}  {c['percent']:>5.1f}%")

    print(f"\n實際使用顏色數：{summary['num_colors_used']}")
    return summary


def update_color_stats(summaries):
    stats = {}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)

    for summary in summaries:
        for color in summary["colors"]:
            hex_code = color["hex"]
            if hex_code not in stats:
                stats[hex_code] = {"rgb": color["rgb"], "hex": hex_code,
                                   "total_count": 0, "images": [], "style_tags": []}
            entry = stats[hex_code]
            entry["total_count"] += 1
            if summary["image"] not in entry["images"]:
                entry["images"].append(summary["image"])
            for tag in summary["style_tags"]:
                if tag not in entry["style_tags"]:
                    entry["style_tags"].append(tag)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[chart] 顏色統計已更新（共 {len(stats)} 個顏色）")


def main():
    # 圖片路徑：images_sam/ 優先，否則 images/
    sam_img = os.path.join(IMAGES_SAM_DIR, f"{NAME}.jpg")
    input_image_path = sam_img if os.path.exists(sam_img) \
                       else os.path.join(IMAGES_DIR, f"{NAME}.jpg")

    if not os.path.exists(input_image_path):
        print(f"[ERR] 找不到圖片：{NAME}.jpg")
        return

    # 載入遮罩
    sam_mask = load_sam_mask(NAME) if MODE in ("sam_refine", "sam_weighted") else None
    if MODE != "standard" and sam_mask is None:
        print(f"[!] 模式 {MODE} 需要遮罩，但找不到 {NAME}_mask.png，改用 standard 模式")

    # 決定要跑的畫布尺寸
    img = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]

    if CANVAS_SIZES_CM:
        canvas_list = CANVAS_SIZES_CM
        print(f"\n畫布規格（手動指定）：{canvas_list}")
    else:
        canvas_list = suggest_canvas_sizes(img_w, img_h)
        print(f"\n圖片尺寸：{img_w}×{img_h} px（比例 {img_w/img_h:.2f}）")
        print(f"自動推薦畫布規格：{canvas_list}")

    all_summaries = []
    for canvas_cm in canvas_list:
        canvas_str = f"{canvas_cm[0]}x{canvas_cm[1]}"
        print(f"\n{'#'*45}")
        print(f"  畫布規格：{canvas_cm[0]}×{canvas_cm[1]} cm")
        print(f"{'#'*45}")

        suggestions = pricing_suggestion(sam_mask, MODE, EXTRA_COLORS, canvas_cm)  # 用全域預設值做說明
        print("定價建議：")
        for s in suggestions:
            print(f"  {s}")

        pricing_info = {"canvas_cm": list(canvas_cm), "pricing_suggestion": suggestions}

        mode_dir = os.path.join(OUTPUT_BASE, NAME, MODE, canvas_str)
        os.makedirs(mode_dir, exist_ok=True)

        if COMPARE_DETAILS:
            # 比較模式：跑所有細緻度
            # COMPARE_DIFFICULTY = 指定難度名稱 → 只跑該難度；None → 跑所有難度
            if COMPARE_DIFFICULTY is not None:
                compare_levels = [lv for lv in DIFFICULTY_LEVELS if lv["name"] == COMPARE_DIFFICULTY]
            else:
                compare_levels = DIFFICULTY_LEVELS
            for base_level in compare_levels:
                for detail_name, detail_params in DETAIL_PRESETS.items():
                    compare_level = {**base_level, **detail_params}
                    level_dir = os.path.join(mode_dir, f"{base_level['name']}_{detail_name}")
                    os.makedirs(level_dir, exist_ok=True)
                    print(f"\n[比較] {base_level['name']} × {detail_name}")
                    summary = run_single_level(
                        input_image_path, level_dir, compare_level, MODE, sam_mask,
                        canvas_cm=canvas_cm, pricing_info=pricing_info
                    )
                    all_summaries.append(summary)
        else:
            for level in DIFFICULTY_LEVELS:
                level_dir = os.path.join(mode_dir, level["name"])
                os.makedirs(level_dir, exist_ok=True)
                summary = run_single_level(
                    input_image_path, level_dir, level, MODE, sam_mask,
                    canvas_cm=canvas_cm, pricing_info=pricing_info
                )
                all_summaries.append(summary)

    update_color_stats(all_summaries)
    print(f"\n[OK] 完成，結果在：{os.path.join(OUTPUT_BASE, NAME, MODE)}")


if __name__ == "__main__":
    main()
