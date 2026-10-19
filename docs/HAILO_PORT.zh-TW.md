[English](HAILO_PORT.md) | 繁體中文

# 第二階段：把 SayGuard 移到 ASUS UGen300（Hailo-10H）上

原型裡的每一項工作，都對應到 Hailo Model Zoo 或 Hailo GenAI Model Zoo 中已有的 Hailo-10H 模型，所以第二階段是「換平台」，
不是重新設計。主機負責銜接的工作：解碼攝影機串流、追蹤、平面圖融合（倉庫模式）、規則引擎、網頁介面。

**目前進度。** 第一階段在 PC GPU 上執行，還沒有任何東西在 UGen300 上跑過。只有安全帽偵測器編譯成 HEF
（尚未在硬體上執行）；CLIP 影像編碼器做了量化，但沒有編譯。兩者都在 Hailo 的模擬器中做過位元精確（bit-exact）
的檢查（見下方「第一階段已用 Hailo 工具鏈完成」）。
**倉庫模式的偵測器還沒有 HEF：** `configs/warehouse.yaml` 已經指向 `@work/models/site_yolov8s.hef`，
但這個檔案尚未建置（見下方「倉庫模式偵測器：尚無 HEF」）。目前程式碼裡只有偵測器有 HailoRT 後端
（`sayguard/detectors/hailo_backend.py`）；YOLO-World、Whisper 與 CLIP 編碼器仍在 PC 上透過
PyTorch/transformers 執行。

| 功能 | PC 原型（第一階段） | UGen300（第二階段） | Hailo 參考數值（Hailo-10H） |
|---|---|---|---|
| 安全帽偵測，所有攝影機 | YOLOv8s 微調（2 類），Ultralytics/CUDA | 同一份權重 → ONNX（opset 11）→ 用 Hailo Dataflow Compiler 編成 INT8 HEF（`hailo/build_helmet.py`），NMS 由主機上的 HailoRT 執行；`sayguard/detectors/hailo_backend.py`。**HEF 已建置，尚未在硬體上執行** | yolov8s：166 FPS（batch 1）、252 FPS（batch 8） |
| 倉庫偵測，所有攝影機（倉庫模式） | YOLOv8s 微調（6 類，960 px），Ultralytics/CUDA | 同樣流程，換成它自己的 ONNX、輸入尺寸、6 類 NMS 設定與校正影格 → `site_yolov8s.hef`；同一個後端（類別順序取自 `detector.labels`）。**HEF 尚未建置** | yolov8s 的數值是 640 px 輸入；960 px 沒有任何量測 |
| 口述規則中提到的物件 | YOLO-World v2-S（Ultralytics） | `yolo_world_v2s` HEF + 主機端文字嵌入 | 46.6 FPS |
| 語音 → 文字 | Whisper-Small（transformers） | Whisper-Small（Hailo GenAI Model Zoo） | 10.6 tokens/s |
| 口述規則 → 規則 JSON（國語或英語、區域名稱）；每日報告的規劃（從列出的問題與審核過的改善做法中挑選） | Qwen2.5-1.5B-Instruct，經 Ollama API，每次呼叫都帶 JSON schema 的 `format` | Qwen2.5-1.5B-Instruct，經 **hailo-ollama**（API 相同：改 `llm.base_url` 即可） | 7.35 tokens/s，首個 token 0.37 s |
| 警報的是非判斷複核 | CLIP ViT-B/32 影像編碼器 + 2×512 線性分類頭 | `clip_vit_b_32_image_encoder` HEF（已量化並在模擬器中檢查，尚未編譯）；分類頭只是主機端一次矩陣乘法 | 81.5 FPS（batch 1） |
| 選用的 VLM 複核 | Qwen2-VL-2B-Instruct（transformers） | Qwen2-VL-2B-Instruct，經 HailoRT GenAI VLM API（hailo-ollama 只提供文字 LLM） | Hailo GenAI Model Zoo（見 [results/benchmark.md](results/benchmark.md)） |

FPS/TPS 數值來源：Hailo Model Zoo（公開模型，HAILO10H，以 DFC v5.4.0 編譯，經 PCIe Gen3 x4 量測）與
Hailo GenAI Model Zoo。UGen300 的 USB 版本可能比 PCIe 略低；第二階段會在實機上量測。

## 運算預算

**工地模式（安全帽），10 支攝影機。** 十支攝影機、每支目標每秒 10 張影格，全部送進同一個批次化的 YOLOv8s
（在 PC 上實測約 105 張/秒，每批 10 張 31-40 ms）。以名目的每秒 100 張計算，估計**約佔 batch 8 容量（252 FPS）
的 40 %**（未在裝置上量測），剩下的空間留給有物件規則的攝影機跑 YOLO-World，以及警報的是非判斷複核（每次警報跑
一次 CLIP 編碼器，batch 1 為 81.5 FPS）。語言模型只在需要時才執行（一條口述規則、一份報告），從不逐張影格執行。
依參考速度推估的 UGen300 數字（[results/benchmark.md](results/benchmark.md)，未量測）：口述規則 → 規則 JSON
約 11.3 s，Whisper 轉寫一條 5 s 的規則約 2.4 s。本 repository 所有的 LLM 準確度都是在 PC GPU 上用 Ollama
`qwen2.5:1.5b` 量的；Hailo GenAI 版本（量化方式不同）沒有量測。

儀表板會即時顯示這個估計值（「UGen300 預估負載」），依據是實測的每支攝影機影格率，以及設定檔中的
yolov8s 參考數值（`hailo_reference`）。

**倉庫模式。** 部署規格是每個區域一台 UGen300、最多 12 支攝影機；這是來自測試場景的涵蓋規格，不是運算量的
數字。在 PC 上，11 支攝影機以 10 fps 經過 6 類倉庫偵測器（960 px），每批 11 張約 63-95 ms（RTX 5060 Ti）。一顆
Hailo-10H 能否撐住 12 支攝影機、960 px 輸入，**沒有量測過**（排在第二階段第 1 週）：Model Zoo 的數值是 640 px 輸入，960 px 每張影格的運算量更大，而且目前還
沒有 HEF 可量。倉庫模式的「UGen300 預估負載」用的也是同一組 640 px 的 yolov8s 參考數值
（`configs/warehouse.yaml`），所以會低估負載。

## 第一階段已用 Hailo 工具鏈完成（`hailo/`）

還沒有實機，但工地模式（安全帽）的兩個視覺網路都已走過真正的 Hailo 流程：偵測器編譯成 HEF，CLIP 編碼器做了
量化，兩者都在 Hailo 的位元精確模擬器中檢查過；完整結果見
[results/README.zh-TW.md](results/README.zh-TW.md#hailo-10h已編譯並在-hailo-模擬器中逐位元驗證hailo)。

1. 環境（WSL Ubuntu 22.04，只用 CPU：DFC 指定版本的 TensorFlow 不支援 RTX 50 系列 GPU）：
   `bash hailo/setup_dfc.sh`，搭配從 Hailo Developer Zone 下載的 Dataflow Compiler 5.4.0 wheel；
   腳本也會 clone Hailo Model Zoo。
2. 偵測器，實際採用的版本（512 張校正影像加上 activation clipping）：
   ```bash
   python hailo/build_helmet.py parse
   python hailo/build_helmet.py optimize --calib 512 --tag _c512clip \
     --extra "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"
   python hailo/build_helmet.py compile --tag _c512clip --performance --install
   ```
   ONNX 在 YOLOv8 的六個 head 卷積處截斷，套用 Model Zoo 的 `yolov8s` model script 與 2 類 NMS 設定；
   校正資料是驗證集中固定的隨機樣本（seed 0），絕不使用測試集。`--performance` 就是
   `hailomz compile --performance` 會加上的設定；`--install` 會把 HEF 複製到
   `@work/models/helmet_yolov8s.hef`。
3. 偵測器準確度：`python hailo/eval_int8.py --har ~/hailo/build/helmet_c512clip_optimized.har`
   （在 2,001 張測試影像上比較浮點與位元精確的 INT8 模擬；寫入 `hailo_int8.json`），以及
   `python hailo/eval_pc_baseline.py`（SayGuard venv；PyTorch，相同的評估器與 NMS 設定）。
4. CLIP 編碼器：先 `python hailo/clip_probe_int8.py dump`（SayGuard venv），再到 WSL 執行
   `hailomz parse clip_vit_b_32_image_encoder --hw-arch hailo10h` 與
   `python hailo/clip_probe_int8.py emulate --har clip_vit_b_32_image_encoder.har`
   （寫入 `clip_probe_int8.json`）。

模擬器量到的結果（`docs/results/hailo_int8.json`、`docs/results/clip_probe_int8.json`）：

| 檢查項目 | 結果 |
|---|---|
| 安全帽偵測器，2,001 張測試影像：Hailo 浮點圖 → INT8 | mAP50 0.8927 → 0.889（下降 0.0037）；mAP50-95 0.5662 → 0.56（下降 0.0062） |
| CLIP 影像特徵與 PyTorch 的相似度（cosine） | Hailo 浮點圖 0.9999；INT8 0.9634 |
| 在 INT8 特徵上做是非判斷複核，分類頭用 PC 特徵訓練 | 準確率 94.3 %，真實違規保留 92.9 %（在 PC 特徵上：95.3 %、94.9 %） |
| 在 INT8 特徵上做是非判斷複核，分類頭改用 INT8 特徵重新訓練 | 準確率 95.3 %；誤報濾除 96.8 %，真實違規保留 93.9 % |

這裡的偵測器 mAP50（分數 ≥ 0.2 的 NMS、pycocotools）與 Ultralytics 驗證的 0.939（分數 ≥ 0.001）是不同的評估：
0.889 只能和 0.893 比較。

## 倉庫模式偵測器：尚無 HEF

`configs/warehouse.yaml` 已經有 `detector.hef: "@work/models/site_yolov8s.hef"` 與 `detector.labels`
（六個類別的輸出順序；`hailo_backend.py` 會使用它們，且遇到非安全帽的類別組合時不做 helmet/head 合併）。
HEF 本身還沒有建置。`scripts/train_detector.py` 已經匯出 `@work/models/site_yolov8s.onnx`（opset 11），
但 `hailo/build_helmet.py` 是為安全帽模型寫的：ONNX 路徑、640×640 輸入與 2 類 NMS 設定都寫死在腳本裡，
只有校正資料夾是參數（`--calib-dir`）。要建出 `site_yolov8s.hef`，需要一個改版腳本，換成倉庫 ONNX、
960×960 輸入、6 類 NMS 設定，以及取自訓練場景（絕不用評估場景）的校正影格；之後還要做同樣的浮點對 INT8
檢查，才能引用任何晶片上的倉庫模式數字。

## 在 UGen300 上的第二階段

1. 在主機（Windows 11 或 Linux）上安裝 HailoRT 5.x 與 UGen300 的 USB/PCIe 驅動程式。
2. 工地模式（安全帽）：在 `configs/site.yaml` 設定 `detector.backend: hailo`（`detector.hef` 已指向
   `@work/models/helmet_yolov8s.hef`）。倉庫模式：等 `site_yolov8s.hef` 建好後，在
   `configs/warehouse.yaml` 做同樣設定。
3. 驗證 `sayguard/detectors/hailo_backend.py`（依 HailoRT 5.x 撰寫，尚未在硬體上執行）：NMS 輸出格式、
   letterbox 縮放、batch 8 的吞吐量。
4. 在晶片上做是非判斷複核：用 HailoRT 執行 CLIP 影像編碼器 HEF，並用晶片自己的 INT8 特徵重新訓練 2×512 分類頭
   （在模擬器中這樣的準確率為 95.3 %，真實違規保留 93.9 %）。
5. 用 Qwen2.5-1.5B 啟動 hailo-ollama，並把 `llm.base_url` 指向它。要確認它支援 JSON schema 的
   `format` 欄位：規則解析器（`sayguard/rules/parser.py`）與報告的行動選單（`sayguard/report.py`）
   每次請求都會帶上。Whisper-Small 取自 Hailo GenAI Model Zoo。
6. 量測：每支攝影機的 fps、端到端警報延遲、裝置負載、功耗；Hailo GenAI 版 Qwen2.5-1.5B 在規則解析與報告
   測試上的準確度與延遲；倉庫模式（第 1 週）另外要量一台裝置能否容納 12 支攝影機、960 px 輸入。
