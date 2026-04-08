# paint-by-number

Converts images to paint by number SVGs that can be colored in with javascript.

Fully functioning web app is deployed [here](https://paint-by-number-21987.web.app/) where you can upload photos, see them converted to paint by numbers, and fill them in.

_Completed for Computational Photography (CS445) at UIUC_

![demo.gif](demo.gif)

## How to Run PBNgen

The easiest way to run the code that generates a paint by number from an image is by uploading your image to our [web app](https://paint-by-number-21987.web.app/). Very big (high resolution) images might take a long time or time out due to limitations on cloud resources. Therefore, if you want to run the paint by number generator locally, follow the steps below:

- go into the project directory `cd paint-by-number`
- install the python dependencies `pip install -r requirements.txt`
- run `python main.py <image_path>` - this will output the B/W SVG and the JSON palette to the same directory as the input image
  - for example: `python main.py images/red_panda.jpg`
  - the image path is relative to the directory you are running your code
  - images should be in jpg or png format

## 進階使用（PaintingNumber 擴充功能）

### 安裝依賴
```bash
py -3.10 -m pip install -r requirements.txt
py -3.10 -m pip install segment-anything torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 下載 SAM 模型
SAM 模型為 Meta 公開發布，需手動下載後放至 `models/` 資料夾：
```bash
mkdir models
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o models/sam_vit_b.pth
```

### 使用方式

**Step 1（可選）：用 SAM 點選要細化的區域**
```bash
cd src
py -3.10 sam_select.py <圖片名稱>
# 左鍵=選取區域  右鍵=排除  Enter=儲存  ESC=離開
```

**Step 2：執行轉換**

修改 `src/run.py` 頂端設定：
```python
NAME  = "your_image"         # 圖片名稱（不含副檔名）
MODE  = "standard"           # standard / sam_refine / sam_weighted
```

```bash
py -3.10 run.py
```

### 三種模式

| 模式 | 需要遮罩 | 說明 |
|------|---------|------|
| `standard` | 否 | 直接轉換，4 個難易度 |
| `sam_refine` | 是 | 選取區額外增加顏色細節 |
| `sam_weighted` | 是 | 選取區佔色數較高比重，非選取區更簡化 |

### 輸出結構
```
output/<name>/<mode>/
  入門/ 初級/ 中級/ 進階/
    template.svg   ← 數字油畫模板
    filled.png     ← 填色完成預覽
    palette.json   ← 色號對照表
    summary.json   ← 參數與顏色記錄
```

---

## Project Structure

- `src`
  - the Python source code for generating a paint by number from an image
  - this contains a `PbnGen` class that can be invoked as `PbnGen("images/input_image.jpg")` with some optional parameters
  - to get the final pbn you must run `self.set_final_pbn()` which will set the internal image of the class to be the paint by number image
  - then you must run `self.output_to_svg()` to get the final SVG image and JSON color palette
- `frontend`
  - the React app for filling in SVG paint by number images
- `functions`
  - the PBN generator deployed to google cloud functions
