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
import os
import json
from datetime import datetime
from pbn_gen import PbnGen

# ── 設定區（只需改這裡）────────────────────────────────
NAME         = "egg"
STYLE_TAGS   = ["食物", "暖色調"]
MODE         = "sam_weighted"   # "standard" / "sam_refine" / "sam_weighted"

# sam_refine 參數
EXTRA_COLORS  = 10   # 選取區額外增加幾色

# sam_weighted 參數
WEIGHT_RATIO  = 0.65  # 選取區佔總色數比例（0.5~0.8）
# ────────────────────────────────────────────────────────

IMAGES_DIR     = r"D:\website\PaintLearn\paint-by-number\images"
IMAGES_SAM_DIR = r"D:\website\PaintLearn\paint-by-number\images_sam"
OUTPUT_BASE    = r"D:\website\PaintLearn\paint-by-number\output"
STATS_PATH     = os.path.join(OUTPUT_BASE, "color_stats.json")

DIFFICULTY_LEVELS = [
    {
        "name": "入門",
        "num_colors": 15,
        "pruning_threshold": 8e-4,
        "blur_ksize": 31,
        "blur_sigma_color": 51,
        "blur_sigma_space": 51,
        "prune_iterations": 10,
        "bg_extra_blur": 21,   # weighted 模式：非選取區額外模糊
    },
    {
        "name": "初級",
        "num_colors": 20,
        "pruning_threshold": 2e-4,
        "blur_ksize": 25,
        "blur_sigma_color": 35,
        "blur_sigma_space": 35,
        "prune_iterations": 8,
        "bg_extra_blur": 15,
    },
    {
        "name": "中級",
        "num_colors": 35,
        "pruning_threshold": 6.25e-5,
        "blur_ksize": 21,
        "blur_sigma_color": 21,
        "blur_sigma_space": 14,
        "prune_iterations": 6,
        "bg_extra_blur": 9,
    },
    {
        "name": "進階",
        "num_colors": 50,
        "pruning_threshold": 1.5e-5,
        "blur_ksize": 13,
        "blur_sigma_color": 13,
        "blur_sigma_space": 9,
        "prune_iterations": 3,
        "bg_extra_blur": 0,   # 進階：非選取區不額外加模糊
    },
]


def load_sam_mask(name):
    mask_path = os.path.join(IMAGES_SAM_DIR, f"{name}_mask.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"✅ 載入 SAM 遮罩：{mask_path}")
        return mask
    return None


def run_single_level(input_image_path, level_dir, level, mode, sam_mask):
    level_name = level["name"]
    print(f"\n{'='*40}")
    print(f"  [{level_name}] 模式：{mode}")
    print(f"{'='*40}")

    if mode == "sam_weighted" and sam_mask is not None:
        # weighted 模式：不走 set_final_pbn，改由 apply_weighted_region 直接量化
        pbn = PbnGen(
            input_image_path,
            num_colors=level["num_colors"],
            pruningThreshold=level["pruning_threshold"],
            fixed_palette=None,
        )
        pbn.apply_weighted_region(
            mask=sam_mask,
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
            input_image_path,
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
        if mode == "sam_refine" and sam_mask is not None:
            print(f"  → 選取區細化 +{EXTRA_COLORS} 色")
            pbn.refine_region(sam_mask, extra_colors=EXTRA_COLORS)

    svg_path    = os.path.join(level_dir, "template.svg")
    filled_path = os.path.join(level_dir, "filled.png")
    json_path   = os.path.join(level_dir, "palette.json")

    palette_data = pbn.output_to_svg(svg_path, json_path)
    pbn.output_filled_image(filled_path)

    used_colors = sorted(palette_data, key=lambda x: x["template_id"])
    summary = {
        "image": NAME,
        "style_tags": STYLE_TAGS,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
        "sam_extra_colors": EXTRA_COLORS if mode == "sam_refine" else 0,
        "sam_weight_ratio": WEIGHT_RATIO if mode == "sam_weighted" else None,
        "num_colors_used": len(used_colors),
        "colors": [
            {
                "id": item["template_id"],
                "rgb": item["rgb"],
                "hex": "#{:02X}{:02X}{:02X}".format(*item["rgb"]),
            }
            for item in used_colors
        ],
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

    print(f"實際使用顏色數：{summary['num_colors_used']}")
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
    print(f"\n📊 顏色統計已更新（共 {len(stats)} 個顏色）")


def main():
    # 圖片路徑：images_sam/ 優先，否則 images/
    sam_img = os.path.join(IMAGES_SAM_DIR, f"{NAME}.jpg")
    input_image_path = sam_img if os.path.exists(sam_img) \
                       else os.path.join(IMAGES_DIR, f"{NAME}.jpg")

    if not os.path.exists(input_image_path):
        print(f"❌ 找不到圖片：{NAME}.jpg")
        return

    # 載入遮罩
    sam_mask = load_sam_mask(NAME) if MODE in ("sam_refine", "sam_weighted") else None
    if MODE != "standard" and sam_mask is None:
        print(f"⚠️ 模式 {MODE} 需要遮罩，但找不到 {NAME}_mask.png，改用 standard 模式")

    mode_dir = os.path.join(OUTPUT_BASE, NAME, MODE)
    os.makedirs(mode_dir, exist_ok=True)

    summaries = []
    for level in DIFFICULTY_LEVELS:
        level_dir = os.path.join(mode_dir, level["name"])
        os.makedirs(level_dir, exist_ok=True)
        summary = run_single_level(input_image_path, level_dir, level, MODE, sam_mask)
        summaries.append(summary)

    update_color_stats(summaries)
    print(f"\n✅ 完成，結果在：{mode_dir}")


if __name__ == "__main__":
    main()
