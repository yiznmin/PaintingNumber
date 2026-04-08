"""
sam_select.py — 互動式 SAM 區域點選工具

使用方式：
  py -3.10 sam_select.py <圖片名稱>
  例：py -3.10 sam_select.py face

操作說明：
  左鍵點擊  = 新增前景點（要保留的區域，綠點）
  右鍵點擊  = 新增背景點（要排除的區域，紅點）
  U         = 撤銷最後一個點
  C         = 清除所有點重新選
  Enter     = 確認遮罩並儲存
  ESC       = 離開不儲存

輸出：
  images_sam/<name>_mask.png    — 遮罩（白=選取區域，黑=其餘）
  images_sam/<name>_points.json — 點選記錄
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


def predict_mask(predictor, points, labels):
    """points: [(x,y), ...], labels: [1=前景, 0=背景, ...]"""
    pts = np.array(points, dtype=np.float32)
    lbs = np.array(labels, dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=True,
    )
    # 選分數最高的遮罩
    best = np.argmax(scores)
    return masks[best]


def draw_overlay(image, points, labels, mask=None):
    display = image.copy()
    if mask is not None:
        overlay = display.copy()
        overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)
        display = overlay
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display, (x, y), 6, color, -1)
        cv2.circle(display, (x, y), 6, (255, 255, 255), 1)
    return display


def main():
    if len(sys.argv) < 2:
        print("用法：py -3.10 sam_select.py <圖片名稱>")
        print("例：py -3.10 sam_select.py face")
        sys.exit(1)

    name = sys.argv[1]
    # 嘗試 images_sam/ 和 images/ 兩個資料夾
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

    points = []
    labels = []
    current_mask = None

    window = f"SAM 點選 — {name}  (左鍵=前景 右鍵=背景 Enter=確認 U=撤銷 C=清除 ESC=離開)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal current_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)
        else:
            return
        if points:
            current_mask = predict_mask(predictor, points, labels)
        refresh()

    def refresh():
        frame = draw_overlay(image_bgr, points, labels, current_mask)
        h, w = frame.shape[:2]
        info = f"點數: {len(points)}  (綠=前景 紅=背景)"
        cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window, frame)

    cv2.setMouseCallback(window, on_mouse)
    refresh()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter — 確認儲存
            if current_mask is None:
                print("⚠️ 尚未選取任何區域")
                continue
            # 儲存遮罩
            mask_path   = os.path.join(IMAGES_SAM_DIR, f"{name}_mask.png")
            points_path = os.path.join(IMAGES_SAM_DIR, f"{name}_points.json")
            mask_img = (current_mask.astype(np.uint8) * 255)
            cv2.imwrite(mask_path, mask_img)
            with open(points_path, "w") as f:
                json.dump({"points": points, "labels": labels}, f, indent=2)
            print(f"✅ 遮罩儲存：{mask_path}")
            print(f"✅ 點選記錄：{points_path}")
            break
        elif key == ord('u') or key == ord('U'):  # 撤銷
            if points:
                points.pop()
                labels.pop()
                current_mask = predict_mask(predictor, points, labels) if points else None
                refresh()
        elif key == ord('c') or key == ord('C'):  # 清除
            points.clear()
            labels.clear()
            current_mask = None
            refresh()
        elif key == 27:  # ESC
            print("已離開，未儲存")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
