"""
export_pdf.py — 將指定的 template.svg 轉換成 PDF（文字轉路徑）

用法：
  # 轉換單一 SVG
  py -3.10 export_pdf.py output/Mom/sam_refine/40x60/入門/template.svg

  # 轉換整個難度資料夾下所有 template.svg
  py -3.10 export_pdf.py output/Mom/sam_refine/40x60/入門/

  # 轉換某個模式下所有尺寸、所有難度
  py -3.10 export_pdf.py output/Mom/sam_refine/
"""

import sys
import os
import subprocess

INKSCAPE = r"D:\Inkscape\bin\inkscape.exe"
OUTPUT_BASE = r"D:\website\PaintLearn\paint-by-number\output"


def convert_svg_to_pdf(svg_path: str):
    pdf_path = svg_path.replace(".svg", ".pdf")
    result = subprocess.run([
        INKSCAPE,
        svg_path,
        "--export-text-to-path",
        f"--export-filename={pdf_path}",
    ], capture_output=True, encoding="utf-8", errors="ignore")
    if result.returncode == 0:
        print(f"[OK] {os.path.relpath(pdf_path, OUTPUT_BASE)}")
    else:
        print(f"[ERR] {svg_path}\n{result.stderr}")


def convert_path(target: str):
    target = os.path.abspath(target)

    if target.endswith(".svg") and os.path.isfile(target):
        # 單一 SVG 檔案
        convert_svg_to_pdf(target)

    elif os.path.isdir(target):
        # 資料夾：遞迴找所有 template.svg
        found = []
        for root, _, files in os.walk(target):
            for f in files:
                if f == "template.svg":
                    found.append(os.path.join(root, f))

        if not found:
            print(f"[!] 找不到任何 template.svg：{target}")
            return

        print(f"找到 {len(found)} 個 template.svg，開始轉換...\n")
        for svg_path in sorted(found):
            convert_svg_to_pdf(svg_path)

    else:
        print(f"[ERR] 路徑不存在：{target}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：py export_pdf.py <svg檔案或資料夾路徑>")
        sys.exit(1)

    convert_path(sys.argv[1])
    print("\n完成")
