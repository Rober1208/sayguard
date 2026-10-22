[English](LICENSES.md) | 繁體中文

# 第三方資料、模型與程式碼

競賽規則要求參賽者對作品用到的每張圖片、每段影片、每個模型和資料集都要有使用權。以下項目都依各自公開的授權條款使用，這個 repo 也沒有再散布任何受限制的內容。

SayGuard 本身：Copyright (C) 2026 逼寶隊，以 GNU AGPL-3.0 授權釋出（[LICENSE](../LICENSE)）。

## 資料

| 項目 | 來源 | 授權 | 用途 |
|---|---|---|---|
| Hard Hats v2（19,745 張圖片，類別 `hardhat` / `no-hardhat`） | Roboflow Universe "hard-hats-fhbh5"；Hugging Face 鏡像為 `keremberke/hard-hat-detection` | CC BY 4.0（需標示出處） | 安全帽偵測模型的訓練、驗證與測試（`scripts/prepare_dataset.py` 把類別改名為 `helmet` / `head`）；裁切圖用於警報複核；Hailo INT8 編譯用的校正圖片（不取自測試集） |
| 工地模式的示範畫面 | Mixkit（見 `docs/VIDEO_SOURCES.zh-TW.md`） | Mixkit Stock Video Free License | 模擬監視器串流；影片**不放**在 repo 裡，需從列出的頁面手動下載（`scripts/fetch_videos.py list` / `check`） |
| NVIDIA PhysicalAI-SmartSpaces，MTMC_Tracking_2026（合成倉庫：時間同步、已校正的多鏡頭影片，附校正參數、平面圖和 2D/3D 標準答案） | [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | CC BY 4.0（需標示出處） | 倉庫模式的示範畫面、評估，以及倉庫偵測模型的訓練資料；用到的每個場景及用途列在下方。不放在 repo 裡；`scripts/fetch_smartspaces.py` 一次下載一個場景 |

所有儀表板畫面和存檔快照裡的人臉，都在現場打上馬賽克（`sayguard/privacy.py`；目前在 PC 原型上執行，Stage II 改在 UGen300 上）：工地模式遮住偵測到的頭部，倉庫模式遮住每個人員框的上緣區域（人員框下半被畫面或貨架切掉、或人蹲著時，遮得更高）。偵測模型漏掉的人臉不會打上馬賽克，漏掉的比例尚未量測。

### Hard Hats v2

出處標示："Hard Hats v2, Roboflow Universe (https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5), CC BY 4.0"，透過 Hugging Face 鏡像 https://huggingface.co/datasets/keremberke/hard-hat-detection 取得。修改內容：類別改名為 `helmet` / `head`，標註從 COCO 格式轉成 YOLO 格式（`scripts/prepare_dataset.py`）。

### NVIDIA PhysicalAI-SmartSpaces：用到的場景

出處標示："NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0"。場景名稱沿用資料集路徑；`train/`、`val/` 是 NVIDIA 的資料夾名稱，不代表 SayGuard 怎麼使用該場景。SayGuard 給每個場景的用途見第二欄（評估方式見 [SITE_MODE.zh-TW.md](SITE_MODE.zh-TW.md)，數字見 [FACTS.md](FACTS.md)）。

| 場景 | 在 SayGuard 的用途 | 使用內容 |
|---|---|---|
| `train/Warehouse_010` | 即時示範（`configs/warehouse.yaml`），也是**測試**場景。不算盲測：追蹤器是看著這個場景開發的，管制區也直接沿用示範設定 | 11 支攝影機的影片、校正參數、平面圖；標準答案只用來評分 |
| `train/Warehouse_013`、`train/Warehouse_015`、`train/Warehouse_018`、`train/Warehouse_019`、`val/Warehouse_022` | **測試**：偵測模型沒用這些場景訓練過，量測用的參數也不是在這裡決定的；平面圖都和訓練場景不同。W018 和開發警報邏輯用的 W017 是同一棟建築、同一組鏡頭 | 攝影機影片、校正參數、平面圖；標準答案只用來評分 |
| `train/Warehouse_012` | **驗證** | 攝影機影片、校正參數、平面圖；標準答案只用來評分 |
| `val/Warehouse_020` | **驗證**：倉庫模式的參數（工人信心門檻、車輛信心門檻、堆高機距離補償、「看得到」的 40 px 高度門檻）只用這個場景決定。和訓練場景 W000-W007 是同一棟建築；有 16 支攝影機，超出 12 支的規格 | 攝影機影片、校正參數、平面圖；標準答案只用來評分 |
| `train/Warehouse_009`、`train/Warehouse_017` | **僅供開發**：W009 和訓練場景 W008 是同一棟建築、同一組鏡頭，W017 和測試場景 W018 也是 | 攝影機影片、校正參數、平面圖；標準答案只用來評分 |
| `train/Warehouse_000`、`_002`、`_004`、`_006`、`_008`、`_011`、`_014`、`_016` | **倉庫偵測模型訓練**：5 棟建築、8 個場景，共 8,000 張圖片（W000、002、004、006 是同一棟建築；W016 的攝影機 0-8 用於訓練，其餘攝影機當作訓練時的驗證集）；從未用於評估 | 從攝影機影片取樣的影格，標籤取自標準答案中看得到的 2D 框（`scripts/prepare_site_dataset.py`） |
| `train/Warehouse_001`、`_003`、`_005`、`_007`、`val/Warehouse_021` | 未用於訓練或評估 | 只有校正參數、平面圖和標準答案，沒有影片 |

修改內容（依 CC BY 4.0 要求註明）：

- 示範與評估串流：每 3 張來源影格保留 1 張（30 → 10 fps），縮放成 1280 px 寬，重新編碼為無音軌的 H.264（`scripts/make_site_streams.py`）；儀表板上另外疊加 SayGuard 的標示和人臉馬賽克。
- 偵測模型訓練：取樣的影格存成 1280 px 寬的 JPEG，YOLO 標籤由標準答案轉換而來（`scripts/prepare_site_dataset.py`）。
- 標準答案不會進入即時流程，只有 `scripts/eval_site.py` 用它來評分。

## 模型

| 模型 | 授權 |
|---|---|
| YOLOv8s（COCO 預訓練）與我們微調的權重（安全帽偵測模型，2 類；倉庫偵測模型，6 類）、YOLO-World v2-S 權重 | AGPL-3.0（Ultralytics） |
| CLIP ViT-B/32：文字編碼器（YOLO-World 的類別嵌入）與影像編碼器（警報複核）；Hailo Model Zoo 的 Hailo-10H 版 `clip_vit_b_32_image_encoder` | MIT（OpenAI） |
| Whisper-Small | MIT（OpenAI） |
| Qwen2.5-1.5B-Instruct | Apache-2.0（Alibaba Qwen） |
| Qwen2-VL-2B-Instruct（選用的 VLM 複核） | Apache-2.0（Alibaba Qwen） |

我們微調的權重不在這個 repo 裡（放在 `@work/models/`）。倉庫偵測模型是用 SmartSpaces 的影格訓練的，分享這份權重時，請一併附上前述的 CC BY 4.0 出處標示。

## 程式相依套件

Ultralytics（AGPL-3.0）、FastAPI/Starlette（MIT）、Uvicorn（BSD-3）、OpenCV（Apache-2.0）、PyTorch（BSD-3）、Transformers（Apache-2.0）、Ollama（MIT）、opencc-python-reimplemented（Apache-2.0）、huggingface_hub（Apache-2.0，`scripts/fetch_smartspaces.py`）、motmetrics（MIT，`scripts/eval_site.py`）、pycocotools（BSD-2-Clause，`hailo/eval_int8.py`）、NumPy（BSD-3）、PyYAML（MIT）、Pydantic（MIT）、HTTPX（BSD-3）、psutil（BSD-3）、Accelerate（Apache-2.0）、Pillow（MIT-CMU）、python-multipart（Apache-2.0）、ftfy（Apache-2.0）、regex（Apache-2.0）、Ultralytics 的 CLIP 分支（MIT）、pytest（MIT）、onnxruntime（MIT，`hailo/eval_same_nms.py`）、edge-tts（LGPL-3.0，只有 `scripts/eval_spoken_rules.py` 使用）、Hailo Model Zoo（MIT）；Hailo Dataflow Compiler 與 HailoRT（Hailo 授權，需從 Hailo Developer Zone 下載，本專案不再散布）。FFmpeg（依建置方式為 LGPL 或 GPL）由串流與資料集腳本當作獨立程式呼叫，同樣不再散布。

偵測模型的程式用到 Ultralytics（AGPL-3.0），因此本專案也以 AGPL-3.0 釋出（見 `LICENSE`），讓整體授權保持相容。另外，競賽的智慧財產條款（得獎者授權 ASUS/Intel/Hailo 使用參賽作品）不影響第三方授權。
