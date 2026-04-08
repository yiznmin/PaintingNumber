import numpy as np
import cv2
import os
import json
from datetime import datetime
from pbn_gen import PbnGen

# 4 個難易度設定（細緻度由模糊強度、合併門檻、迭代次數控制）
# num_colors 只是上限，實際顏色數由圖片內容決定
DIFFICULTY_LEVELS = [
    {
        "name": "入門",
        "num_colors": 15,
        "pruning_threshold": 8e-4,
        "blur_ksize": 31,
        "blur_sigma_color": 51,
        "blur_sigma_space": 51,
        "prune_iterations": 10,
    },
    {
        "name": "初級",
        "num_colors": 20,
        "pruning_threshold": 2e-4,
        "blur_ksize": 25,
        "blur_sigma_color": 35,
        "blur_sigma_space": 35,
        "prune_iterations": 8,
    },
    {
        "name": "中級",
        "num_colors": 35,
        "pruning_threshold": 6.25e-5,
        "blur_ksize": 21,
        "blur_sigma_color": 21,
        "blur_sigma_space": 14,
        "prune_iterations": 6,
    },
    {
        "name": "進階",
        "num_colors": 50,
        "pruning_threshold": 1.5e-5,
        "blur_ksize": 13,
        "blur_sigma_color": 13,
        "blur_sigma_space": 9,
        "prune_iterations": 3,
    },
]

# 跨圖片顏色統計檔案路徑
STATS_PATH = r"D:\website\PaintLearn\paint-by-number\output\color_stats.json"


def run_single_level(input_image_path, level_dir, level, image_name, style_tags):
    level_name = level["name"]

    print(f"\n{'='*40}")
    print(f"  [{level_name}] 上限 {level['num_colors']} 色")
    print(f"{'='*40}")

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

    svg_path    = os.path.join(level_dir, "template.svg")
    filled_path = os.path.join(level_dir, "filled.png")
    json_path   = os.path.join(level_dir, "palette.json")

    palette_data = pbn.output_to_svg(svg_path, json_path)
    pbn.output_filled_image(filled_path)

    used_colors = sorted(palette_data, key=lambda x: x["template_id"])

    # summary.json：記錄這次的所有參數、風格標籤、顏色結果
    summary = {
        "image": image_name,
        "style_tags": style_tags,
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
        "num_colors_used": len(used_colors),
        "colors": [
            {
                "id": item["template_id"],
                "rgb": item["rgb"],
                "hex": "#{:02X}{:02X}{:02X}".format(*item["rgb"]),
            }
            for item in used_colors
        ],
        "approved": False,   # 人工確認滿意後改為 True
        "notes": ""          # 備註欄位
    }
    summary_path = os.path.join(level_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"實際使用顏色數：{summary['num_colors_used']}")
    print(f"輸出資料夾：{level_dir}")

    return summary


def update_color_stats(summaries):
    """
    累積統計：每個 HEX 顏色被使用了幾次、出現在哪些圖片與風格。
    """
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {}

    for summary in summaries:
        image = summary["image"]
        tags  = summary["style_tags"]
        for color in summary["colors"]:
            hex_code = color["hex"]
            if hex_code not in stats:
                stats[hex_code] = {
                    "rgb": color["rgb"],
                    "hex": hex_code,
                    "total_count": 0,
                    "images": [],
                    "style_tags": []
                }
            entry = stats[hex_code]
            entry["total_count"] += 1
            if image not in entry["images"]:
                entry["images"].append(image)
            for tag in tags:
                if tag not in entry["style_tags"]:
                    entry["style_tags"].append(tag)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n📊 顏色統計已更新：{STATS_PATH}（共 {len(stats)} 個顏色）")


def test_pbn_effect():
    # 只需改這兩行
    name       = "egg"
    style_tags = ["食物", "暖色調"]   # 自由填寫風格標籤，用於累積分析

    images_dir = r"D:\website\PaintLearn\paint-by-number\images"
    input_image_path = os.path.join(images_dir, f"{name}.jpg")

    if not os.path.exists(input_image_path):
        print(f"❌ 找不到測試圖片: {input_image_path}")
        return

    base_output_dir = os.path.join(
        r"D:\website\PaintLearn\paint-by-number\output", name
    )
    os.makedirs(base_output_dir, exist_ok=True)

    summaries = []
    for level in DIFFICULTY_LEVELS:
        level_dir = os.path.join(base_output_dir, level["name"])
        os.makedirs(level_dir, exist_ok=True)
        summary = run_single_level(input_image_path, level_dir, level, name, style_tags)
        summaries.append(summary)

    # 更新跨圖片顏色統計
    update_color_stats(summaries)

    print(f"\n✅ 全部完成，結果在：{base_output_dir}")


if __name__ == "__main__":
    test_pbn_effect()
