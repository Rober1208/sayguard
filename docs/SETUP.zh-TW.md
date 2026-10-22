[English](SETUP.md) | 繁體中文

# 執行原型、重現數據與專案結構

這份文件寫給想自己跑原型、或重跑量測的開發者。產品介紹請看 [README](../README.zh-TW.md)。

## 執行原型（Windows 11、NVIDIA GPU）

需要準備：

- Python 3.12（64 位元）、git，以及加進 `PATH` 的 `ffmpeg` / `ffprobe`（例如 `winget install Gyan.FFmpeg`）
- 記憶體約 8 GB 以上的 NVIDIA 顯示卡（原型的數據是在 RTX 5060 Ti 16 GB 上量的）
- [Ollama](https://ollama.com)，負責提供本地 LLM

資料集、模型權重、影片和警報資料庫這類大檔案，都放在 repo 外的工作目錄，由 `SAYGUARD_WORK`
指定，預設是 `D:/ugen-work`。設定檔裡以 `@work/` 開頭的路徑都指向這裡。要放別的位置，先設好再執行：

```powershell
$env:SAYGUARD_WORK = "E:/sayguard-work"         # 只對這個終端機有效；要一直生效用 setx SAYGUARD_WORK E:\sayguard-work
```

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
ollama serve                                   # 另開一個終端機執行（Stage II 改由 UGen300 上的 hailo-ollama 提供）
ollama pull qwen2.5:1.5b
```

從原作者那裡下載模型，放進工作目錄：

```powershell
$W = $env:SAYGUARD_WORK
hf download openai/whisper-small --local-dir "$W/models/hf/whisper-small"
python -c "import clip, os; clip.load('ViT-B/32', device='cpu', download_root=os.path.join(r'$W', 'models', 'clip'))"   # 產生 models/clip/ViT-B-32.pt
python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt')"; Move-Item yolov8s-worldv2.pt "$W/models/"
hf download Qwen/Qwen2-VL-2B-Instruct --local-dir "$W/models/hf/Qwen2-VL-2B-Instruct"   # 選用：改用 VLM 複核時才需要
```

`models/clip/ViT-B-32.pt` 由 YOLO-World 和警報複核共用。自己訓練的權重（安全帽偵測、複核判斷層、倉庫偵測）不隨 repo 提供，照下面的步驟就能訓練出來。

單元測試不需要上面這些（不用顯示卡、不用模型、也不用工作目錄）：

```powershell
python -m pytest -q tests
```

### 工地模式（安全帽）

```powershell
# Hard Hats v2（Roboflow Universe，CC BY 4.0），Hugging Face 上有鏡像
hf download keremberke/hard-hat-detection --repo-type dataset --local-dir "$W/datasets/raw/keremberke"
foreach ($s in "train", "valid", "test") { Expand-Archive "$W/datasets/raw/keremberke/data/$s.zip" "$W/datasets/raw/keremberke/ex/$s" }
python scripts/prepare_dataset.py              # 把 Hard Hats v2 轉成 YOLO 格式
python scripts/train_detector.py --epochs 40   # 產生 @work/models/helmet_yolov8s.pt（另外匯出 .onnx 給 Hailo DFC）
python scripts/train_probe.py                  # 訓練複核用的是非分類頭，產生 @work/models/helmet_probe.npz
python scripts/fetch_videos.py list            # 列出 10 個 Mixkit 頁面，影片要自己逐一下載（見 VIDEO_SOURCES.zh-TW.md）
python scripts/fetch_videos.py check           # 確認影片都到齊，並寫出攝影機清單
python scripts/make_streams.py                 # 轉成 10 路循環播放的示範串流
python -m sayguard serve                       # 讀 configs/site.yaml，開啟 http://127.0.0.1:8000（?lang=en 或 ?lang=zh）
```

`start_demo.cmd` 會一次把 Ollama 和工地模式示範都開起來。它讀 `SAYGUARD_WORK`（預設 `D:\ugen-work`），找不到時改用 `PATH` 上的 `python` 和 `ollama`。

### 倉庫模式

```powershell
# 1. 示範場景：校正資料、平面圖、標準答案（只有評估用得到，加 --no-gt 可以跳過）、11 支影片
python scripts/fetch_smartspaces.py train/Warehouse_010
python scripts/make_site_streams.py train/Warehouse_010    # 逐格對齊的 10 fps 串流，以及 streams.json

# 2. 倉庫偵測模型（權重不隨 repo 提供），只用訓練場景：
#    prepare_site_dataset.py 會拒絕所有評估與開發用場景（009、010、012、013、015、017–020、022）。
python scripts/fetch_smartspaces.py train/Warehouse_000    # 002、004、006、008、011、014、016 也照樣下載
python scripts/prepare_site_dataset.py --name site_v3 --train train/Warehouse_000 train/Warehouse_002 train/Warehouse_004 train/Warehouse_006 train/Warehouse_008 train/Warehouse_011 train/Warehouse_014 "train/Warehouse_016:~9,10" --val "train/Warehouse_016:9,10"
python scripts/train_detector.py --data "@work/datasets/site_v3/data.yaml" --name site_v8s_v3 --out site_yolov8s.pt --imgsz 960 --batch 16 --epochs 20

# 3. 執行
python -m sayguard serve --config configs/warehouse.yaml   # http://127.0.0.1:8000
```

串流準備好之後，`start_demo_warehouse.cmd` 一樣能一鍵開啟倉庫模式。

訓練資料共 8,000 張圖片，來自 SmartSpaces 的 8 個訓練場景、5 棟合成建築（Warehouse_000/002/004/006
是同一棟）。實測用的權重（`@work/models/site_yolov8s.pt`）分兩階段訓練：第一階段從 COCO YOLOv8s
出發，在 Warehouse_000/004/008/014 上訓練 30 個 epoch，用 016 驗證；第二階段就是上面最後一條指令，
加上 `--model` 指向第一階段的 checkpoint 繼續訓練。照上面的指令原樣執行，則會從 COCO YOLOv8s 開始訓練。

儀表板的攝影機清單，以及「用說的加規則」底下的範例句子，都從設定檔讀取（`configs/warehouse.yaml`
裡的 `cameras:` 和 `examples:`）。

## 重現數據

```powershell
python scripts/eval_detector.py              # 安全帽偵測模型，測試集 -> detector_metrics.json
python scripts/benchmark.py                  # 處理量表格 -> benchmark.md
python scripts/eval_rules.py                 # 工地模式的規則解析（打字輸入的句子）-> rule_parsing.json（需要 Ollama）
python scripts/eval_rules.py --site          # 倉庫模式的規則解析（打字輸入的句子）-> rule_parsing_site.json
python scripts/eval_spoken_rules.py          # 同一批保留測試句改用講的（合成語音、加工地噪音）-> rule_parsing_spoken.json（需要 Ollama、edge-tts）
python scripts/eval_report.py                # 日報改善措施，20 天模擬資料 -> report_actions.json（需要 Ollama）
python scripts/train_probe.py                # 是非分類頭 -> probe_train.json
python scripts/eval_vlm.py --per-class 100   # 用生成式 VLM 判斷同一批裁切圖 -> vlm_check.json
python scripts/eval_yesno.py                 # 在 193 張沒參與訓練的裁切圖上比較各種複核方法 -> yesno_check.json（較早的比較表）

python scripts/eval_site.py --all-zones                                              # Warehouse_010 -> site_eval.json
python scripts/eval_site.py --config configs/eval/warehouse_018.yaml --all-zones     # -> site_eval_warehouse_018.json
```

幾點說明：

- **結果放哪裡**：同時寫到 `docs/results/` 和 `@work/results/`。
- **其他倉庫場景**：每個 `configs/eval/warehouse_0XX.yaml` 的開頭都寫了該場景要跑的三條指令：下載、
  產生串流、評估。Warehouse_020 和 _022 在資料集的 `val/`，其他都在 `train/`。
- **`eval_site.py` 的選項**：
  - `--dets-cache dets.pkl`：第一次執行時把偵測結果存下來，之後直接重播，不用再跑 GPU。
  - `--tag NAME`：實驗結果只寫到 `@work/results/`，不會動到公開的結果檔。
- **用講的規則測試**：`eval_spoken_rules.py` 另外需要 `pip install edge-tts`（不在 `requirements.txt`
  裡，語音是向微軟的線上服務取得），以及 Ollama 的 `qwen2.5:1.5b` 和上面下載的 Whisper-Small。有 GPU
  會比較快；公開的結果是在 Colab L4 上跑的。加 `--limit N` 則每組只跑 N 句，方便先快速試跑，
  但結果一樣會覆寫公開的 `rule_parsing_spoken.json`。
- **保留原始紀錄的三項結果**：以下三項直接保存當時的結果檔，不由上面的腳本重新產生：
  - `rule_parsing_site_v1_first_run.json`：倉庫第 1 組測試句的第一次結果。之後解析器針對這組句子
    做了改進，現在第 1 組是 100%（LLM + 檢查器）。
  - 倉庫模式的凍結版結果，放在 `docs/results/frozen/`，是兩項警報邏輯修正之前跑的。
  - 日報改善措施的 AI 評分。評分不在腳本裡：拿去評分的那次結果是 `report_actions_judged.json`，
    提示詞、分批內容和評分結果在 `report_actions_judging/`。重跑 `eval_report.py` 會產生新的
    `report_actions.json`，裡面沒有評分。
- **Hailo 編譯與模擬器驗證**放在 `hailo/`，見下一節。

## 重建 Hailo 檔案

在 WSL Ubuntu 22.04 上用 CPU 執行：Dataflow Compiler 需要的 TensorFlow 版本不支援 RTX 50 系列 GPU。
倉庫偵測模型則改在 Colab 的 T4 GPU 上編譯，這版 TensorFlow 支援 T4。編譯器的 wheel（5.4.0）要從
Hailo Developer Zone 下載。結果會寫進 `docs/results/`：`hailo_int8.json`、`pc_baseline.json`、
`clip_probe_int8.json`、`site_same_nms.json`、`site_r544_same_nms.json` 和 `helmet_onnx_same_nms.json`；
兩個倉庫 HEF 的編譯資訊記在 `site_hef.json`。編譯流程做了什麼、量到什麼，見 [HAILO_PORT.zh-TW.md](HAILO_PORT.zh-TW.md)。

```bash
bash hailo/setup_dfc.sh                       # 在 WSL venv 裝好 Dataflow Compiler 和 Hailo Model Zoo

# 安全帽偵測模型 -> HEF（從驗證集取 512 張校正圖片，加上 activation clipping）
python hailo/build_helmet.py parse
python hailo/build_helmet.py optimize --calib 512 --tag _c512clip \
  --extra "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"
python hailo/build_helmet.py compile --tag _c512clip --performance --install   # -> @work/models/helmet_yolov8s.hef

# 倉庫偵測模型 -> HEF（512 張校正影格只取自訓練場景）。960x960 編譯約兩小時，
# 完成後把 site_perf.hef 複製成 @work/models/site_yolov8s.hef
python hailo/build_site.py parse
python hailo/build_site.py optimize --calib 512
python hailo/build_site.py compile --performance
# 攝影機的 16:9 比例：先用 Ultralytics 以 imgsz=(544, 960) 匯出同一組權重，再
python hailo/build_site.py parse --onnx site_yolov8s_r544.onnx --tag _r544
python hailo/build_site.py optimize --tag _r544 --calib 512
python hailo/build_site.py compile --tag _r544 --performance

# 用同一套評分比較浮點與 INT8（相同 letterbox、HEF 的 NMS 設定、pycocotools）
python hailo/eval_same_nms.py site --har ~/hailo/build/site_optimized.har
python hailo/eval_same_nms.py site --tag _r544 --har ~/hailo/build/site_r544_optimized.har
python hailo/eval_same_nms.py helmet --onnx-only

# 在 2,001 張測試圖片上比較浮點與 INT8；PyTorch 基準（SayGuard venv）
python hailo/eval_int8.py --har ~/hailo/build/helmet_c512clip_optimized.har
python hailo/eval_pc_baseline.py

# CLIP 編碼器：先產生裁切圖和 PyTorch 特徵（SayGuard venv），再到 WSL 量化並模擬
python hailo/clip_probe_int8.py dump
hailomz parse clip_vit_b_32_image_encoder --hw-arch hailo10h
python hailo/clip_probe_int8.py emulate --har clip_vit_b_32_image_encoder.har
```

## 專案結構

```
sayguard/            主流程、規則引擎與解析器、偵測模型（Ultralytics / HailoRT）、追蹤、
                     人臉馬賽克、警報紀錄、警報複核（CLIP 分類頭 / VLM）、語音辨識、日報、伺服器
  site.py            倉庫模式：攝影機校正、平面圖、管制區
  sitetrack.py       倉庫模式：跨鏡頭合併同一位工人，以輪子著地點定位車輛
  site_pipeline.py   倉庫模式：即時迴圈（和 scripts/eval_site.py 共用）
  site_view.py       倉庫模式：各鏡頭畫面與即時平面圖
web/                 儀表板（原生 JS，可離線使用）
scripts/             資料集準備、訓練、評估、挑選示範影片、SmartSpaces 下載與串流、效能測試
hailo/               Hailo Dataflow Compiler 流程：ONNX -> INT8 -> HEF，以及模擬器準確率驗證
configs/site.yaml    工地模式：十支攝影機、管制區、模型、預設規則
configs/warehouse.yaml  倉庫模式：Warehouse_010 的 11 支校正攝影機、地面管制區、範例規則
configs/eval/        每個評估場景一份設定，Warehouse_010 除外（管制區是在看到任何結果之前，依各場景的鏡頭涵蓋範圍圖畫好的）
docs/                FACTS.md（所有公開數字）、SITE_MODE.md（倉庫模式完整說明）、results/、
                     Hailo 移植計畫、授權、影片來源、示範腳本、簡報大綱（多數附 .zh-TW.md 中文版）
tests/               單元測試（規則、解析器、倉庫規則、跨鏡頭追蹤、追蹤器、事件、日報、
                     複核、語音解碼、同步影像來源、人臉遮罩、效能指標、長時間執行、API 防護），
                     以及倉庫模式的規則測試句
start_demo.cmd       工地模式示範的 Windows 啟動檔
start_demo_warehouse.cmd  倉庫模式示範的 Windows 啟動檔
```
