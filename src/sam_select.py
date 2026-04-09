"""
sam_select.py — 互動式區域選取工具

使用方式：
  py -3.10 sam_select.py <圖片名稱>
  例：py -3.10 sam_select.py Mom

兩種模式同時作用，結果合併成一張遮罩：

【SAM 模式】按 S 啟用 — 智慧點選
  左鍵  = 前景點（綠）
  右鍵  = 背景點（紅）

【多邊形模式】按 P 啟用 — 手動框選
  左鍵  = 新增頂點
  右鍵  = 閉合目前多邊形，可繼續選下一個

【共用按鍵】
  S / P   = 切換模式
  U       = 撤銷最後一步
  C       = 清除全部重來
  Enter   = 儲存
  ESC     = 離開不儲存

若已有儲存記錄，啟動時自動載入繼續編輯。
"""

import sys
import os
import cv2
import json
import numpy as np

SAM_MODEL_PATH = r"D:\website\PaintLearn\paint-by-number\models\sam_vit_b.pth"
IMAGES_SAM_DIR = r"D:\website\PaintLearn\paint-by-number\images_sam"
IMAGES_DIR     = r"D:\website\PaintLearn\paint-by-number\images"


def load_sam(model_path):
    from segment_anything import sam_model_registry, SamPredictor
    print("載入 SAM 模型中...")
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to("cpu")
    predictor = SamPredictor(sam)
    print("SAM 載入完成")
    return predictor


def predict_mask_sam(predictor, points, labels):
    if not points:
        return None
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points, dtype=np.float32),
        point_labels=np.array(labels, dtype=np.int32),
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def polys_to_mask(completed_polys, current_poly, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in completed_polys:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32).reshape(-1, 1, 2)], 255)
    if len(current_poly) >= 3:
        cv2.fillPoly(mask, [np.array(current_poly, dtype=np.int32).reshape(-1, 1, 2)], 255)
    return mask


def merge_masks(sam_mask, poly_mask, image_shape):
    """SAM 遮罩 + 多邊形遮罩 → 聯集"""
    result = np.zeros(image_shape[:2], dtype=np.uint8)
    if sam_mask is not None:
        result = np.logical_or(result, sam_mask).astype(np.uint8) * 255
    if poly_mask is not None and np.any(poly_mask):
        result = np.logical_or(result, poly_mask > 0).astype(np.uint8) * 255
    return result if np.any(result) else None


def draw_overlay(image, mode, sam_points, sam_labels,
                 completed_polys, current_poly, final_mask):
    display = image.copy()

    # 合併遮罩半透明覆蓋
    if final_mask is not None:
        overlay = display.copy()
        overlay[final_mask > 0] = (
            overlay[final_mask > 0] * 0.5 + np.array([0, 200, 100]) * 0.5
        ).astype(np.uint8)
        display = overlay

    # SAM 點
    for (x, y), label in zip(sam_points, sam_labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display, (x, y), 6, color, -1)
        cv2.circle(display, (x, y), 6, (255, 255, 255), 1)

    # 已完成多邊形（橘色）
    for poly in completed_polys:
        if len(poly) >= 2:
            for i in range(len(poly) - 1):
                cv2.line(display, poly[i], poly[i + 1], (0, 165, 255), 2)
            cv2.line(display, poly[-1], poly[0], (0, 165, 255), 2)
        for pt in poly:
            cv2.circle(display, pt, 4, (0, 165, 255), -1)

    # 目前進行中的多邊形（青色）
    if len(current_poly) >= 2:
        for i in range(len(current_poly) - 1):
            cv2.line(display, current_poly[i], current_poly[i + 1], (0, 255, 255), 2)
    for pt in current_poly:
        cv2.circle(display, pt, 5, (0, 255, 255), -1)
        cv2.circle(display, pt, 5, (255, 255, 255), 1)

    h = display.shape[0]
    mode_text = f"模式：{'SAM 智慧點選 (S)' if mode == 'sam' else '多邊形框選 (P)'}  |  已完成 {len(completed_polys)} 個多邊形 / {len(sam_points)} 個SAM點"
    hint_text = "S=SAM模式  P=多邊形模式  U=撤銷  C=清除  Enter=儲存  ESC=離開"
    cv2.putText(display, mode_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)
    cv2.putText(display, hint_text, (10, h - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    return display


def load_saved_state(points_path):
    """載入已儲存的點選記錄"""
    if not os.path.exists(points_path):
        return [], [], [], []
    with open(points_path, "r") as f:
        record = json.load(f)
    sam_points      = [tuple(p) for p in record.get("sam_points", [])]
    sam_labels      = record.get("sam_labels", [])
    completed_polys = [[tuple(pt) for pt in poly] for poly in record.get("completed_polys", [])]
    # 相容舊格式（單一 polygon）
    old_poly = record.get("polygon", [])
    if old_poly and len(old_poly) >= 3:
        completed_polys.append([tuple(pt) for pt in old_poly])
    return sam_points, sam_labels, completed_polys, []


def main():
    if len(sys.argv) < 2:
        print("用法：py -3.10 sam_select.py <圖片名稱>")
        sys.exit(1)

    name = sys.argv[1]
    img_path = os.path.join(IMAGES_SAM_DIR, f"{name}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGES_DIR, f"{name}.jpg")
    if not os.path.exists(img_path):
        print(f"❌ 找不到圖片：{name}.jpg")
        sys.exit(1)

    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = load_sam(SAM_MODEL_PATH)
    predictor.set_image(image_rgb)

    # 嘗試載入已儲存記錄
    points_path = os.path.join(IMAGES_SAM_DIR, f"{name}_points.json")
    sam_points, sam_labels, completed_polys, _ = load_saved_state(points_path)
    if sam_points or completed_polys:
        print(f"✅ 載入已儲存記錄：{len(sam_points)} 個SAM點、{len(completed_polys)} 個多邊形")

    current_poly = []
    mode         = "sam"

    def get_final_mask():
        sam_mask  = predict_mask_sam(predictor, sam_points, sam_labels)
        poly_mask = polys_to_mask(completed_polys, current_poly, image_bgr.shape)
        return merge_masks(sam_mask, poly_mask, image_bgr.shape)

    window = f"區域選取 — {name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def refresh():
        cv2.imshow(window, draw_overlay(
            image_bgr, mode, sam_points, sam_labels,
            completed_polys, current_poly, get_final_mask()
        ))

    def on_mouse(event, x, y, flags, param):
        if mode == "sam":
            if event == cv2.EVENT_LBUTTONDOWN:
                sam_points.append((x, y)); sam_labels.append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                sam_points.append((x, y)); sam_labels.append(0)
            else:
                return
        elif mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                current_poly.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current_poly) >= 3:
                    completed_polys.append(list(current_poly))
                    current_poly.clear()
                    print(f"區域 {len(completed_polys)} 完成，可繼續選下一個")
            else:
                return
        refresh()

    cv2.setMouseCallback(window, on_mouse)
    refresh()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # Enter
            if len(current_poly) >= 3:
                completed_polys.append(list(current_poly))
                current_poly.clear()
            final_mask = get_final_mask()
            if final_mask is None:
                print("⚠️ 尚未選取任何區域")
                continue
            mask_path = os.path.join(IMAGES_SAM_DIR, f"{name}_mask.png")
            cv2.imwrite(mask_path, final_mask)
            record = {
                "sam_points": sam_points,
                "sam_labels": sam_labels,
                "completed_polys": completed_polys,
            }
            with open(points_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"✅ 儲存完成：{mask_path}（SAM {len(sam_points)} 點 + {len(completed_polys)} 個多邊形）")
            break

        elif key == ord('s') or key == ord('S'):
            mode = "sam"
            print("切換到：SAM 智慧點選")
            refresh()

        elif key == ord('p') or key == ord('P'):
            mode = "polygon"
            print("切換到：多邊形框選")
            refresh()

        elif key == ord('u') or key == ord('U'):
            if mode == "sam" and sam_points:
                sam_points.pop(); sam_labels.pop()
            elif mode == "polygon":
                if current_poly:
                    current_poly.pop()
                elif completed_polys:
                    current_poly.extend(completed_polys.pop())
            refresh()

        elif key == ord('c') or key == ord('C'):
            sam_points.clear(); sam_labels.clear()
            current_poly.clear(); completed_polys.clear()
            refresh()

        elif key == 27:
            print("已離開，未儲存")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
