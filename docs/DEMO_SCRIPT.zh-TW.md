[English](DEMO_SCRIPT.md) | 繁體中文

# 3 分鐘 Demo 影片：腳本與分鏡（Stage I）

目標長度 2:50，英文旁白（競賽規則 §6.1），上限 3:00。以 1920×1080 錄製儀表板（OBS 擷取顯示器，
瀏覽器全螢幕），旁白事後再配，才能對照每一段實際錄到的畫面。
需要錄兩段：**工地模式（安全帽）**（`configs/site.yaml`）和**倉庫模式**（`configs/warehouse.yaml`），兩種模式不能同時執行。

**本文件是給主講人看的中文對照；錄旁白時請照英文版 [DEMO_SCRIPT.md](DEMO_SCRIPT.md) 念。**
畫面上的按鈕名稱以英文介面為準（括號內為中文介面的對應文字）。

畫面上的每個量測數字都來自 [FACTS.md](FACTS.md)（由 [results/](results/README.md) 的原始結果產生）。UGen300 的
吞吐量、負載與延遲是依 Hailo Model Zoo 公開資料推算的估計值，畫面上要標明是估計值。影片裡的原型跑在 PC GPU 上：
說明現在做的事要講「在本地」或「在現場」，計畫則講「Stage II 在 UGen300 上」。

## 這支影片要讓評審記住的三件事

1. **一台設備看住所有攝影機。** 在工地模式（安全帽），它檢查十支攝影機有沒有人沒戴安全帽；在倉庫模式，它把
   十一支攝影機合成一張平面圖。現在這台設備是 PC GPU，Stage II 是一台 UGen300。
2. **本地 LLM 做兩件偵測器做不到的事。** 它把說出或打出的一句話變成規則（中文或英文都可以），也負責規劃每日報告。
3. **LLM 提案、程式檢查、安全主管確認規則。** 報告部分，LLM 只能從列出的問題和審核過的措施裡選。影片裡的每個數字
   都有量測，或標明是估計值。

## 分鏡表

| 時間 | 段落 | 畫面 | 旁白（中文對照） |
|---|---|---|---|
| 0:00–0:13 | 問題 | 片頭卡 2 秒：「SayGuard · one UGen300 per site area · every camera watched · rules you can say」，接著切到工地模式（安全帽）儀表板與十個循環播放 demo 串流的畫面。 | 「現場到處都是攝影機，卻很少有人即時在看。SayGuard 是錄影主機旁的一台設備：現在是 PC GPU，Stage II 是一台 ASUS UGen300。它看住每一支攝影機；安全主管只要開口說，就能新增規則。」 |
| 0:13–0:37 | 工地模式（安全帽） | 攝影機牆。指向上方 KPI 列（「Frames analysed / s」每秒分析影格、「Est. UGen300 load」UGen300 預估負載）。在有人沒戴安全帽的畫面（我們的素材是 House Frame），先出現紅框「NO HARD HAT」，再出現警報卡，標記從「Checking…」（複核中…）變成「Confirmed by double-check」（複核確認）。放大一張打了馬賽克的臉。字幕 1。 | 「在工地模式，十支攝影機、每支目標每秒十張影格，共用一個批次處理的 YOLOv8s 偵測器：實測每秒約 105 張影格。以名目上的每秒 100 張計算，估計只用到一台 UGen300 的 40 %。沒戴安全帽要持續兩秒才算數，接著 CLIP 是非判斷複核檢查裁切畫面，確認這則警報，或把它標成誤報。臉部在本地就打上馬賽克。」 |
| 0:37–1:12 | 語音規則（LLM 的第一個用途） | (1) 按住麥克風按鈕說「Alert me if more than two people are at the site gate after 10 pm.」。畫面出現轉錄文字，接著出現「Proposed rule」卡片，標記「LLM + checker」，「When」顯示 22:00–06:00。點開「Checker notes」（檢查器備註），可以看到 LLM 給的時段和檢查器採用的時段。按「Cancel」。(2) 輸入「A區有人靠近挖土機三公尺內要立刻警告」並按「Check」。卡片顯示「Person within 3 m of an excavator」、Zone A、嚴重度 High。按「Activate rule」（啟用規則）。(3) Excavation Pit 畫面出現橘色「excavator」框、以公尺標示的紅色距離線，以及警報。字幕 2。 | 「這是 LLM 的第一個特殊用途：關於已知物件、區域、距離、人數或時段的新規則，就是一句話。Whisper 轉成文字，Qwen2.5-1.5B 寫成結構化 JSON，再由確定性的程式核對它在實際說出的話裡找到的數字、時間和物件。這次模型把『晚上 10 點後』的結束時間設在午夜；檢查器套用明訂的慣例 22:00 到 06:00，並把修正顯示出來。中文也一樣：經主管確認的挖土機規則，在這台 PC 上幾秒內就生效。開放詞彙偵測器只靠『挖土機』這個詞就找到它。」 |
| 1:12–1:44 | 倉庫模式 | 切到倉庫模式那一段。平面圖畫面標題是「Floor plan · each worker counted once」，顯示人員、區域和車輛。KPI「Workers on site」（現場人數）顯示不重複人數，下方是各攝影機偵測總數。出現警報「Worker within 3 m of a forklift」：這名人員在平面圖上、以及在每一支看得到他的攝影機畫面裡都變紅，警報卡列出相關攝影機（「seen by …」）和以公尺計的距離。接著按住麥克風說「Alert me if anyone enters the dock.」，卡片上「Where」顯示 Dock。按「Activate rule」。字幕 3。 | 「倉庫模式是多鏡頭版本，這裡用的是合成倉庫。十一支校正過的攝影機合成一張平面圖，所以每個人只算一次，而不是每支攝影機各算一次；如果把各攝影機的人數直接相加，人數大約會變成兩倍。區域只要在平面圖上畫一次，和堆高機的距離以公尺量測，每個人每次進入只觸發一則警報，而不是每支攝影機各一則，並在每一支看得到他的攝影機畫面上標示。規則一樣用說的，區域名稱照大家平常的叫法。」 |
| 1:44–2:10 | 每日報告（LLM 的第二個用途） | 回到已累積至少 15 分鐘紀錄的工地模式（安全帽）。按「Daily safety report」（每日安全報告）。視窗顯示「Writing the report with the on-device LLM…」。接著報告出現摘要、各攝影機表格，以及三項「Recommended actions」，格式為 **攝影機** · 問題：措施。放大頁尾：「numbers and tables computed by SayGuard, actions drafted by on-device LLM (qwen2.5:1.5b)」。字幕 4。 | 「LLM 的第二個用途是每日報告，但每個數字、每張表都由程式計算。模型只負責規劃：從實際發生的問題裡挑出重要的三個，再替每個問題選一項審核過的改善措施，而且是用編號選擇。所以它不會把問題算到錯的攝影機頭上，也不會編造數字。兩位 AI 盲評的結果：讓 LLM 自由撰寫的措施有 20 % 指錯問題，這種做法則是零。」 |
| 2:10–2:31 | 數字 | 全螢幕結果卡（字幕 5）。 | 「每個數字都有量測，或標明是估計值，腳本都在 repo 裡。在工地模式打字輸入的保留測試句上，檢查器把小模型完全正確的比例從 40 % 拉到 95 %。在六棟偵測器從沒訓練過的測試建築上，經過兩項警報邏輯修正後，倉庫模式涵蓋了全部 53 個門口與碼頭事件，其中 52 個有自己觸發的警報。」 |
| 2:31–2:50 | 請求 | 模型對照表（用途 · 模型 · Hailo-10H 參考值），接著是結尾卡「SayGuard · github.com/Rober1208/sayguard」。字幕 6。 | 「每項工作都對應到一個 Hailo 公開的 Hailo-10H 模型，我們的安全帽偵測器也已經為它編譯好：在 Hailo 模擬器裡，INT8 的 mAP50 是 0.889，浮點版本是 0.893。Stage II 我們會把所有模型搬到每個場區一台、功耗 2.5 瓦的 UGen300 上，並在上面實際量測。我們希望得到這個機會，在真正的現場證明。」 |

如果 0:37 那段錄到的卡片沒有檢查器備註（代表 LLM 這次時段答對了），把「午夜」那句旁白換成：
「When the model gets a detail wrong, the checker corrects it and shows the correction.」（模型答錯細節時，檢查器會修正並顯示出來。）
午夜那個例子是在實機伺服器上記錄到的（FACTS.md 的「Rule-parsing examples」）；LLM 以 temperature 0 執行，
同一句話通常會得到相同的草稿。

## 畫面字幕（照抄英文）

1. `10 cameras at a 10 fps target · ≈ 105 frames/s measured through one batched YOLOv8s · ≈ 40 % of one UGen300 at the nominal 100 frames/s (estimate from Hailo's YOLOv8s reference, 252 FPS at batch 8)`
2. `Whisper-Small → Qwen2.5-1.5B (JSON schema) → deterministic checker → officer confirms · rule parse 0.35–0.40 s mean (max 0.64 s) on the PC · UGen300 estimate ≈ 11.3 s per rule (not measured)`
3. `Warehouse_010 (NVIDIA PhysicalAI-SmartSpaces, synthetic; our demo scene) · 11 calibrated cameras · people present on average 24.97 · fused on one floor plan 26.56 · summed over cameras 51.64`
4. `20 synthetic days · 2 blind AI judges per action · wrong problem: LLM writing freely 20.0 % → LLM choosing a vetted fix 0.0 %`
5. 結果卡：
   ```
   Rule parsing, typed held-out sentences (rules exactly right; speech recognition not measured)
     Hard-hat mode, 20 sentences:     LLM alone 40 %     →  LLM + checker 95 %     (rule extractor alone 90 %)
     Warehouse mode, 2 × 25:          LLM alone 16–24 %  →  LLM + checker 84–88 %  (rule extractor alone 68–80 %)
     LLM measured with Ollama qwen2.5:1.5b on the PC; the Hailo build is not measured yet
   Daily report actions, 2 blind AI judges: wrong problem 20.0 % → 0.0 %
   Yes/no check (CLIP, probe_train.json):   95.7 % of hard-hat crops passed (proxy for false flags) · 94.9 % of violations kept
   Warehouse mode, 6 test buildings the detector never trained on
     final run, after two alert-logic fixes developed on non-test scenes once the frozen run had been seen
     covered = an alert of its own, or inside the cooldown of an alert for the same person
                        covered             own alert (strict)   false alerts
     door / dock        53 of 53 (100 %)    52 of 53 (98 %)      0 of 52
     forklift 3 m       56 of 61 (92 %)     25 of 61 (41 %)*     1 of 26
     lanes / aisles     98 of 118 (83 %)    71 of 118 (60 %)     3 of 77
     work areas         323 of 338 (96 %)   237 of 338 (70 %)    26 of 362
     other zones        43 of 48 (90 %)     34 of 48 (71 %)      6 of 44
     frozen run, before the fixes: covered 94 / 89 / 64 / 83 / 71 % · false alerts 0 of 26 · 1 of 24 · 1 of 40 · 16 of 244 · 4 of 30
     * the 60 s cooldown deliberately does not re-alert a worker stepping in and out of 3 m of a parked forklift
     W010 is the demo scene (zones from the demo config); W017 (development) is W018's twin
   Hard-hat detector:  mAP50 0.9393 (Ultralytics evaluator) · Hailo's emulator, a different evaluation: INT8 0.889 vs float 0.8927 (compare only these two)
   ```
   （中文說明：規則解析用的是打字輸入的保留測試句，不含語音辨識；「Hard-hat mode」是工地模式（安全帽），「Warehouse
   mode」是倉庫模式；LLM 是在 PC 上用 Ollama 量測，Hailo 版本尚未量測。「涵蓋」= 有自己觸發的警報，或落在同一個人
   一則警報的冷卻時間內；「own alert」是嚴格召回率。堆高機規則的 60 秒冷卻時間刻意不對在停放的堆高機 3 公尺內進進
   出出的人重複警報，所以嚴格召回率是 41 %。表中主要數字是兩項警報邏輯修正後的最終版本，這兩項修正是在看過凍結版本
   之後、在非測試場景上開發的；修正前的凍結版本涵蓋率較低，誤報也較少。W010 是 demo 場景（區域來自 demo 設定檔），
   W017（開發用）是 W018 的雙胞胎場景。0.9393 和 0.889 / 0.8927 是不同的評估，只能拿 0.889 和 0.8927 比較。）
6. `Hailo-10H references: YOLOv8s 252 FPS (batch 8) · Qwen2.5-1.5B-Instruct 7.35 tokens/s · Whisper-Small 10.6 tokens/s · CLIP ViT-B/32 81.5 FPS · one UGen300 per area of ≤ 12 cameras (coverage spec; warehouse-mode load measured in Stage II)`

## 錄影前準備

### 第 1 段：工地模式（安全帽）

- 先啟動 Ollama（`ollama serve`，並已 pull `qwen2.5:1.5b`），再執行 `python -m sayguard serve`，開啟
  `http://127.0.0.1:8000/?lang=en`。在 Windows 上，`start_demo.cmd` 會一次做完這三件事；模型載入期間
  網頁會自動重新連線。
- 在「Active rules」（啟用中規則）刪掉測試規則，只留「Hard hat required on site」。讓它跑大約 15 分鐘，
  報告才會有好幾支攝影機的紀錄。至少要發生三組（攝影機、問題）組合，LLM 才會規劃建議措施；
  不足三組時，措施改由範本產生，報告頁尾會寫「template report」。
- 按下錄影前先暖機：
  - 先說一條用不到的規則，再按取消。Whisper 在第一次使用時才載入。
  - 等第一筆安全帽警報出現，這會載入 CLIP 是非判斷複核。
  - 啟用一條任何物件的規則再刪掉。YOLO-World 在第一次有規則提到物件時載入，之後會保持載入。
- KPI「On-device LLM」（裝置端 LLM）必須顯示 Online。若不是，規則卡上的標記會是「Rule extractor」而不是
  「LLM + checker」，LLM 的段落就錄不了。

### 第 2 段：倉庫模式

- 只需做一次：`python scripts/fetch_smartspaces.py train/Warehouse_010`，以及
  `python scripts/make_site_streams.py train/Warehouse_010`。倉庫偵測器權重要放在
  `@work/models/site_yolov8s.pt`（訓練方式見 [SITE_MODE.zh-TW.md](SITE_MODE.zh-TW.md)）。
- 先關掉第 1 段：兩種模式預設都用 8000 埠，也都要用同一張 GPU。接著啟動 Ollama，
  執行 `python -m sayguard serve --config configs/warehouse.yaml`。在 Windows 上，
  `start_demo_warehouse.cmd` 會一次做完這兩件事並開啟網頁。
- 兩條預設規則（「Keep 3 m from forklifts」、「No walking in aisle 7 (forklift parked)」）會產生警報。
  影片會循環播放，每次循環時平面圖狀態會重新開始，所以平面圖鏡頭要在同一輪內錄完。
- 介面是英文時請說英文。儀表板會把介面語言傳給語音模型，所以要用中文說規則，得先切到「中文」；
  用打字輸入中文則兩種介面都可以。

### 兩段共通

- 關掉其他吃 GPU 的程式（例如訓練工作），讓每支攝影機維持每秒 10 張影格。
- 誠實檢查：倉庫素材要說明是合成影片，UGen300 負載要說明是估計值。儀表板上的「on-device LLM」標示指的是設計目標；
  旁白要說「在本地」，因為 demo 是在 PC 上跑的。倉庫模式從不顯示人員編號，因為編號幾十秒內就會變（Warehouse_010
  的 5 分鐘片段中每人平均 10.7 個編號），不要把它說成穩定的身分。挖土機是靠開放詞彙偵測找到的，工地模式（安全帽）
  的距離線則是由頭部大小估算；兩者的準確率都沒有量測，不要把它們當成量測結果引用。
