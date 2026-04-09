# PaintingNumber 開發日誌

## 2026-04-09 — 色彩品質、畫布尺寸規劃與打包系統

---

### 色彩細化品質改善（`pbn_gen.py`）

#### 稀少顏色保護（`refine_region`）
- K-Means 完成後，對指派誤差前 15% 的高誤差像素再跑 mini K-Means（最多 5 色）
- 防止嘴唇紅、特殊顏色被大面積膚色吞掉
- **亮度保護**：LAB L* > 78 的像素（眼白、高光）強制加入 outlier，確保眼白有自己的顏色群

#### 微小色塊合併（`merge_tiny_colors`）
- `sam_refine` 模式：遮罩內外分開統計像素總數計算門檻，避免細化色被稀釋
- `standard` / `sam_weighted`：全圖統一計算

---

### 畫布尺寸系統（`run.py`）

#### 支援 22 種標準畫布規格
```
正方形：30×30 40×40 50×50 60×60 70×70 80×80 100×100
直幅：30×40 30×50 40×50 40×60 50×60 50×70
      60×80 60×90 60×120 70×90 70×100 80×100 80×120 90×120 100×150
橫幅：對應直幅翻轉版
```

#### 自動推薦畫布規格（`suggest_canvas_sizes`）
- 根據圖片長寬比，從標準規格中找出比例最接近的 3 個
- `CANVAS_SIZES_CM = None` → 自動推薦；手動指定 → `[(40,60)]`

#### 裁切對齊（`crop_to_canvas_ratio`）
- 中央裁切圖片與 SAM 遮罩，確保比例完全符合畫布
- 容差 1%（`< 0.01`），超出才裁切
- 每個尺寸分別裁切，PbnGen 收到的是已對齊的圖片

#### min_ratio 雙因子計算
```
min_ratio = calc_min_ratio(canvas_w_cm, img_w, img_h) × min_ratio_multiplier
```
- `calc_min_ratio`：根據畫布寬度換算最小筆觸（1.5cm）對應的像素比例
- `min_ratio_multiplier`：各難易度獨立設定

---

### 難易度參數擴充（`DIFFICULTY_LEVELS`）

| 難易度 | refine_extra_colors | min_ratio_multiplier |
|--------|--------------------|--------------------|
| 入門 | 8 色 | ×2.0（合併最多） |
| 初級 | 12 色 | ×1.5 |
| 中級 | 18 色 | ×1.0（基準） |
| 進階 | 25 色 | ×0.6（保留最多） |

- `refine_extra_colors`：取代全域 `EXTRA_COLORS`，各難易度獨立控制細化色數
- `min_ratio_multiplier`：難易度 × 尺寸共同決定最小可塗色塊門檻

---

### 多尺寸批次執行

- `run.py` 一次跑完所有推薦尺寸 × 4 難易度
- 輸出結構：`output/<name>/<mode>/<canvas>/入門|初級|中級|進階/`
- `summary.json` 新增：`image_size`、`canvas_cm`、`min_ratio`、`colors[].pixels`、`colors[].percent`

---

### 最終打包工具（`package.py`）

執行 `py -3.10 package.py`，根據 `SELECTIONS` 清單打包選定的組合：

```
final/<name>/<canvas>_<mode>_<difficulty>/
  original.jpg        原圖
  cropped.jpg         裁切後（符合畫布比例）
  template.svg        數字油畫模板
  filled.png          填色完成預覽
  palette.json        色號對照表
  color_swatches.png  顏色色卡總覽圖（色塊 + 號碼 + HEX + 佔比）
  info.json           完整資訊（尺寸/難易度/顏色/定價/佔比）
```

---

### 顏色分析報表

每個難易度跑完後，terminal 自動印出：
```
  號  HEX       像素數    佔比
  2   #E0D7CD  116,188   13.6%
  ...
```

---

### 商業流程認知整理

- **先決定畫布尺寸，再產模板**：尺寸決定 min_ratio，影響哪些色塊被合併
- **不同尺寸要分開跑**：SVG 座標系對應裁切後的圖片，不能共用
- **定價影響因素**：畫布尺寸 × 難易度（色數）× 遮罩選取範圍 × 模式

---

### Git 提交記錄

| Commit | 內容 |
|--------|------|
| `8709930` | 稀少顏色保護、亮度保護、多尺寸批次、裁切對齊、package.py |
| `5fe8d5f` | 難易度獨立 refine_extra_colors 與 min_ratio_multiplier |

---

## 2026-04-08 — 系統建構與功能完善

---

### 環境建置

- 修正 `pip install requirements.txt` → 正確用法 `pip install -r requirements.txt`
- 解決 Python 3.14 與舊套件不相容問題（`contourpy`, `numpy`），改用 Python 3.10
- 安裝 SAM 依賴：`segment-anything`, `torch`, `torchvision`
- 下載 SAM vit_b 模型（375MB）至 `models/sam_vit_b.pth`

---

### 核心演算法改進（`pbn_gen.py`）

#### 顏色匹配改用 LAB 色彩空間
- 原本用 RGB 歐氏距離 snap 固定色盤，人眼感知不準確
- 改用 LAB 色彩空間計算距離，數學距離 ≈ 人眼感知色差
- 特別改善暗部、膚色、低飽和度區域的顏色對應準確度

#### 細緻度參數化（`set_final_pbn`）
- 新增參數：`blur_ksize`, `blur_sigma_color`, `blur_sigma_space`, `prune_iterations`
- 細緻度由三個維度控制：
  - **模糊強度**：越大 → 邊界消失 → 色塊越大越簡單
  - **pruningThreshold**：越高 → 更多小色塊被合併
  - **prune_iterations**：迭代次數越多 → 越簡化

#### 填色效果圖輸出（`output_filled_image`）
- 直接輸出量化後影像，不重繪輪廓（避免黑點、虛線殘影）
- 修正 Windows 中文路徑導致 `cv2.imwrite` 靜默失敗問題，改用 `imencode` + 直接寫檔

#### SVG 模板改進（`output_to_svg`）
- 改回白底 + 1px 黑邊線（適合印刷）
- 分析印刷 vs 螢幕解析度差異：sub-pixel gap 在印刷 300 DPI 下不可見

#### 修正 `pruneClustersSimple` dtype 問題
- 發現 pruning 內部將影像轉為 `int32` 後未轉回
- 修正：`self.setImage(image.astype(np.uint8))`

#### 新增 SAM 區域細化方法
- `refine_region(mask, extra_colors)`：選取區額外加色，直接覆蓋原量化結果
- `apply_weighted_region(mask, total_colors, weight_ratio, bg_extra_blur)`：在固定色數預算內，選取區佔比重更高，非選取區更簡化

---

### 四個難易度等級

| 難易度 | 色數上限 | blur ksize | pruning 門檻 | 迭代次數 |
|--------|---------|------------|-------------|---------|
| 入門   | 15      | 31         | 8e-4        | 10      |
| 初級   | 20      | 25         | 2e-4        | 8       |
| 中級   | 35      | 21         | 6.25e-5     | 6       |
| 進階   | 50      | 13         | 1.5e-5      | 3       |

> 顏色數是上限，實際數量由圖片內容決定。

---

### 三種商業模式（`run.py`）

#### `standard` — 直接轉換
- 不需要遮罩
- 適合所有圖片類型

#### `sam_refine` — 選取區額外加色
- 需要 SAM 遮罩
- 基礎 N 色 + 選取區額外 M 色
- 適合需要局部細化但整體色數不需增加太多的圖片

#### `sam_weighted` — 色數比重重新分配
- 需要 SAM 遮罩
- 選取區佔總色數 65%（可調），非選取區佔 35% 且可額外加模糊
- 適合人像眼睛、臉部細節等需要高還原度的局部區域

---

### 輸出資料夾結構

```
output/
  <name>/
    standard/
      入門/ 初級/ 中級/ 進階/
        template.svg    ← 數字油畫模板
        filled.png      ← 上色完成預覽
        palette.json    ← 色號對照表
        summary.json    ← 參數、顏色、記錄欄位
    sam_refine/
      ...
    sam_weighted/
      ...
  color_stats.json      ← 跨圖片顏色使用頻率統計
```

---

### SAM 互動點選工具（`sam_select.py`）

```bash
py -3.10 sam_select.py <圖片名稱>
```

| 操作 | 功能 |
|------|------|
| 左鍵 | 前景點（保留區域，綠點）|
| 右鍵 | 背景點（排除區域，紅點）|
| U    | 撤銷最後一個點 |
| C    | 清除全部重來 |
| Enter | 確認儲存遮罩 |
| ESC  | 離開不儲存 |

輸出：`images_sam/<name>_mask.png` + `images_sam/<name>_points.json`

---

### 顏色統計與記錄系統

- `summary.json` 記錄每次產出的參數、使用顏色、SAM 設定
- `color_stats.json` 累積所有圖片的顏色使用頻率，供顏料採購決策
- `summary.json` 內含 `approved`, `notes`, `workflow_hints` 欄位供人工標注

---

### 已知圖片類型問題紀錄

#### 食物類（egg）
- **問題**：牛皮紙袋 snap 到 SKIN TONE 變粉色；水泥背景 snap 到深綠色
- **原因**：色盤缺少中深灰和暖棕褐色系
- **建議**：補充 `RGB(80~120, 80~120, 80~120)` 和 `RGB(170~200, 140~160, 100~120)` 色系

#### 花卉類（flower）
- **問題**：浮雕紋理圖在進階等級出現大量黑點
- **原因**：pruning 不足，大量雜訊小色塊殘留
- **建議**：中級等級效果最佳；改用 `output_filled_image` 直接輸出量化圖解決

#### 人像類（Mom）
- **建議流程**：先 `standard` 分析 → `sam_select.py` 點選眼睛/嘴唇 → `sam_weighted` 輸出

---

### Git 提交記錄

| Commit | 內容 |
|--------|------|
| `d139e8c` | 初始提交：pbn_gen, run_test, assign_color, .gitignore |
| `1af6985` | SAM 整合、weighted 模式、3 種商業模式、bug 修正 |

---

*建立於 2026-04-08*
