[English](LICENSES.md) | 繁體中文

# 第三方資料、模型與程式碼

競賽規則要求參賽者對使用的每一張影像、每一段影片、每一個模型與資料集都擁有權利。以下所有項目都依其公開的
授權條款使用；本 repository 沒有再散布任何受限制的內容。

## 資料

| 項目 | 來源 | 授權 | 用途 |
|---|---|---|---|
| Hard Hats v2（19,745 張影像，類別 `hardhat` / `no-hardhat`） | Roboflow Universe "hard-hats-fhbh5"，Hugging Face 上的鏡像為 `keremberke/hard-hat-detection` | CC BY 4.0（需標示出處） | 安全帽偵測器的訓練／驗證／測試（類別由 `scripts/prepare_dataset.py` 改名為 `helmet` / `head`）；警報的是非判斷複核用的裁切影像；Hailo INT8 建置用的校正影像（絕不使用測試集） |
| 工地模式（安全帽）的示範攝影機畫面 | Mixkit（見 `docs/VIDEO_SOURCES.zh-TW.md`） | Mixkit Stock Video Free License | 模擬的 CCTV 串流；影片**沒有**放進 repository：要從列出的頁面手動下載（`scripts/fetch_videos.py list` / `check`） |
| NVIDIA PhysicalAI-SmartSpaces，MTMC_Tracking_2026（合成倉庫：時間同步且已校正的影片、相機校正、平面圖、2D/3D 標註真值） | [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | CC BY 4.0（需標示出處） | 倉庫模式：示範畫面、評估，以及倉庫偵測器的訓練資料；用到的每個場景與其角色列於下方。沒有放進 repository；`scripts/fetch_smartspaces.py` 一次下載一個場景 |

人臉在現場本地打馬賽克（目前在 PC 原型上；第二階段改在 UGen300 上），範圍涵蓋所有儀表板畫面與儲存的快照
（`sayguard/privacy.py`）：工地模式（安全帽）依每個偵測到的頭部，倉庫模式依每個偵測到的人物框上緣。偵測器漏掉的臉
不會被打碼；這種情況的頻率沒有量測。

### NVIDIA PhysicalAI-SmartSpaces：用到的場景

出處標示："NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0"。場景以資料集路徑命名：
`train/` 與 `val/` 是 NVIDIA 的資料夾名稱，不是 SayGuard 的角色。SayGuard 為每個場景指定的角色在第二欄
（評估規範見 [SITE_MODE.zh-TW.md](SITE_MODE.zh-TW.md)，數字見 [FACTS.md](FACTS.md)）。

| 場景 | 在 SayGuard 的角色 | 使用內容 |
|---|---|---|
| `train/Warehouse_010` | 即時示範（`configs/warehouse.yaml`）兼**測試**場景。不是盲測：追蹤器是看著它開發的，區域沿用展示設定 | 11 支攝影機的影片、相機校正、平面圖；標註真值只用於評分 |
| `train/Warehouse_013`、`train/Warehouse_015`、`train/Warehouse_018`、`train/Warehouse_019`、`val/Warehouse_022` | **測試**：偵測器從未用它們訓練，量測選定的設定也不是在它們上面選的；平面圖與訓練場景不同。W018 是開發場景 W017 的雙胞胎，而 W017 被用來發展警報邏輯 | 攝影機影片、相機校正、平面圖；標註真值只用於評分 |
| `train/Warehouse_012` | **驗證** | 攝影機影片、相機校正、平面圖；標註真值只用於評分 |
| `val/Warehouse_020` | **驗證**：倉庫模式中經量測選定的設定（人員信心度過濾、車輛信心度、堆高機距離補償、40 px 的「看得到」高度）只在這個場景上選定。與訓練場景 W000-W007 是同一棟建築，16 支攝影機（超出 12 支攝影機的規格） | 攝影機影片、相機校正、平面圖；標註真值只用於評分 |
| `train/Warehouse_009`、`train/Warehouse_017` | **僅供開發**：W009 是訓練場景 W008 的雙胞胎（同一棟建築、同一組攝影機），W017 是測試場景 W018 的雙胞胎 | 攝影機影片、相機校正、平面圖；標註真值只用於評分 |
| `train/Warehouse_000`、`_002`、`_004`、`_006`、`_008`、`_011`、`_014`、`_016` | **倉庫偵測器訓練**：來自 5 棟建築、8 個場景的 8,000 張影像（W000、002、004、006 是同一棟建築；W016：攝影機 0-8 用於訓練，其餘攝影機作為訓練時的驗證集）；從未用於評估 | 從攝影機影片取樣的影格，標籤取自標註真值中的可見 2D 框（`scripts/prepare_site_dataset.py`） |
| `train/Warehouse_001`、`_003`、`_005`、`_007`、`val/Warehouse_021` | 未用於訓練或評估 | 只有相機校正、平面圖與標註真值；沒有影片 |

所做的修改（CC BY 4.0 要求標示）：

- 示範與評估串流：每 3 張來源影格保留 1 張（30 → 10 fps），縮放為 1280 px 寬，重新編碼為不含音訊的 H.264
  （`scripts/make_site_streams.py`）；儀表板上另外疊加 SayGuard 的標示並做人臉馬賽克。
- 偵測器訓練：取樣的影格存成 1280 px 寬的 JPEG，YOLO 標籤由標註真值轉換而來（`scripts/prepare_site_dataset.py`）。
- 標註真值從不進入即時流程；`scripts/eval_site.py` 只用它來評分。

## 模型

| 模型 | 授權 |
|---|---|
| YOLOv8s（COCO 預訓練）與我們微調的權重（安全帽偵測器，2 類；倉庫偵測器，6 類）、YOLO-World v2-S 權重 | AGPL-3.0（Ultralytics） |
| CLIP ViT-B/32：文字編碼器（YOLO-World 類別嵌入）與影像編碼器（警報的是非判斷複核）；Hailo Model Zoo 的 Hailo-10H 版 `clip_vit_b_32_image_encoder` | MIT（OpenAI） |
| Whisper-Small | MIT（OpenAI） |
| Qwen2.5-1.5B-Instruct | Apache-2.0（Alibaba Qwen） |
| Qwen2-VL-2B-Instruct（選用的 VLM 複核） | Apache-2.0（Alibaba Qwen） |

我們微調的權重不在本 repository 中（存放在 `@work/models/`）。倉庫偵測器的權重是用 SmartSpaces 影格訓練的，
因此分享這份權重時應附上上述 CC BY 4.0 出處標示。

## 程式相依套件

Ultralytics（AGPL-3.0）、FastAPI/Starlette（MIT）、Uvicorn（BSD-3）、
OpenCV（Apache-2.0）、PyTorch（BSD-3）、Transformers（Apache-2.0）、Ollama（MIT）、
opencc-python-reimplemented（Apache-2.0）、huggingface_hub（Apache-2.0，`scripts/fetch_smartspaces.py`）、
motmetrics（MIT，`scripts/eval_site.py`）、pycocotools（BSD-2-Clause，`hailo/eval_int8.py`）、
Hailo Model Zoo（MIT）；Hailo Dataflow Compiler 與 HailoRT（Hailo 授權，從 Hailo Developer Zone 下載，
本 repository 不再散布）。FFmpeg（依建置方式為 LGPL 或 GPL）由串流與資料集腳本以獨立程式呼叫，不再散布。

由於偵測器的程式路徑使用 Ultralytics（AGPL-3.0），本 repository 也以 AGPL-3.0 釋出（見 `LICENSE`），
讓整體組合符合授權。另外請注意，競賽的智慧財產條款（得獎者授權 ASUS/Intel/Hailo 使用參賽作品）
不會改變第三方授權。
