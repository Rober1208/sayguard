[English](DEMO_SCRIPT.md) | 繁體中文

# 3 分鐘示範影片：腳本與分鏡（Stage I）

目標長度 2:50，上限 3:00，旁白用英文（競賽規則 §6.1）。以 1920×1080 錄製儀表板（OBS 擷取整個螢幕，瀏覽器全螢幕），旁白事後再配，才能對上每段實際錄到的畫面。需要錄兩段：**工地模式**（`configs/site.yaml`）和**倉庫模式**（`configs/warehouse.yaml`），兩種模式無法同時執行。

**這份是給主講人看的中文對照；錄旁白時請照英文版 [DEMO_SCRIPT.md](DEMO_SCRIPT.md) 念。** 畫面上的按鈕名稱以英文介面為準（括號內是中文介面的對應文字）。

畫面上的每個量測數字都來自 [FACTS.md](FACTS.md)（由 [results/](results/README.zh-TW.md) 的原始結果產生）。UGen300 的處理量、負載和延遲都是依 Hailo Model Zoo 公開數據推算的估計值，畫面上要標明是估計值。影片裡的原型跑在 PC 的 GPU 上：講目前做到的事，說「在本地」或「在現場」；講計畫，則說「Stage II 在 UGen300 上」。

## 這支影片要讓評審記住的三件事

1. **一台盒子顧好所有攝影機。** 工地模式檢查十支攝影機裡有沒有人沒戴安全帽；倉庫模式把十一支攝影機整合成一張平面圖。目前這台盒子是 PC 的 GPU，Stage II 會換成一台 UGen300。
2. **本地 LLM 做兩件偵測模型做不到的事。** 把說出或打字輸入的一句話變成規則（國語或英文都可以），以及規劃工安日報。
3. **LLM 起草、程式把關、工安人員確認規則。** 日報則是 LLM 只能從列出的問題和事先審過的改善措施裡挑選。影片中的每個數字不是實測，就是標明為估計值。

## 分鏡表

| 時間 | 段落 | 畫面 | 旁白（中文對照） |
|---|---|---|---|
| 0:00–0:13 | 問題 | 片頭卡 2 秒：「SayGuard · one UGen300 per site area · every camera watched · rules you can say」，接著切到工地模式儀表板，十格畫面循環播放示範串流。 | 「現場到處都是攝影機，卻很少有人即時在看。SayGuard 是放在錄影主機旁的一台盒子：現在是 PC 的 GPU，Stage II 換成一台 ASUS UGen300。它顧好每一支攝影機，工安人員只要開口說，就能新增規則。」 |
| 0:13–0:37 | 工地模式 | 攝影機牆。指向上方的 KPI 列：「Frames analysed / s」（每秒分析影格）和「Est. UGen300 load」（UGen300 預估負載）。在有人沒戴安全帽的畫面（素材中的 House Frame）上，先出現紅框「NO HARD HAT」，接著跳出警報卡，標記從「Checking…」（複核中…）變成「Confirmed by double-check」（複核確認）。放大一張打了馬賽克的臉。字幕 1。 | 「工地模式下，十支攝影機、每支以每秒十張畫面為目標，共用一個批次處理的 YOLOv8s 偵測模型，每秒 100 張全部分析完，估計約佔一台 UGen300 的 40 %。沒戴安全帽要持續兩秒才算數；接著由 CLIP 是非判斷複核檢查裁切畫面，確認這則警報，或標成誤報。臉部在本地就打上馬賽克。」 |
| 0:37–1:12 | 用說的加規則（LLM 的第一項工作） | （1）按住麥克風按鈕說「Alert me if more than two people are at the site gate after 10 pm.」。畫面先出現轉寫文字，接著出現「Proposed rule」（建議規則）卡片，標記為「LLM + checker」（LLM + 檢查器），「When」（時段）顯示 22:00–06:00。點開「Checker notes」（檢查器備註），可以看到 LLM 給的時段和檢查器採用的時段。按「Cancel」（取消）。（2）輸入「A區有人靠近挖土機三公尺內要立刻警告」，按「Check」（解析）。卡片顯示「Person within 3 m of an excavator」、Zone A、嚴重度 High。按「Activate rule」（啟用規則）。（3）Excavation Pit 畫面出現橘色「excavator」框、標出公尺數的紅色距離線，以及警報。字幕 2。 | 「這是 LLM 的第一項工作：已知物件、區域、距離、人數或時段的新規則，一句話就能加。Whisper 先轉成文字，Qwen2.5-1.5B 起草結構化的 JSON，再由固定邏輯的程式核對句子裡實際講到的數字、時間和物件。這次模型把『晚上 10 點後』的結束時間設在午夜；檢查器套用明訂的慣例，改成 22:00 到 06:00，並把修正顯示出來。國語也一樣能用：確認後的挖土機規則，在這台 PC 上幾秒內就生效。開放詞彙偵測模型光憑『挖土機』這個詞，就找到了挖土機。」 |
| 1:12–1:44 | 倉庫模式 | 切到倉庫模式的錄影。平面圖畫面的標題是「Floor plan · each worker counted once」（平面圖・每人只算一次），顯示工人、管制區和車輛。KPI「Workers on site」（現場人數）顯示不重複的人數，下方是各攝影機的偵測總數。出現警報「Worker within 3 m of a forklift」：這名工人在平面圖上和每一支拍到他的攝影機畫面裡都變成紅色，警報卡列出相關攝影機（「seen by …」）和以公尺計的距離。接著按住麥克風說「Alert me if anyone enters the dock.」，卡片上的「Where」（位置）顯示 Dock。按「Activate rule」。字幕 3。 | 「倉庫模式是多鏡頭版本，這裡用的是合成倉庫。十一支校正過的攝影機整合成一張平面圖，所以每個人只算一次，而不是每支攝影機各算一次；如果直接把各攝影機的人數相加，大約會多算一倍。管制區只要在平面圖上畫一次，和堆高機的距離用公尺量；每個人每次進入只發一則警報，而不是每支攝影機各一則，並在每一支拍到他的攝影機畫面上標出來。規則一樣用說的，區域名稱照大家平常的叫法。」 |
| 1:44–2:10 | 工安日報（LLM 的第二項工作） | 回到已累積至少 15 分鐘紀錄的工地模式。按「Daily safety report」（每日安全報告）。視窗顯示「Writing the report with the on-device LLM…」。接著日報出現摘要、各攝影機的表格，以及三項「Recommended actions」，格式為 **攝影機** · 問題：措施。放大頁尾：「numbers and tables computed by SayGuard, actions drafted by on-device LLM (qwen2.5:1.5b)」。字幕 4。 | 「LLM 的第二項工作是工安日報，但每個數字、每張表都由程式計算。模型只負責規劃：從實際發生的問題裡挑出最重要的三個，再替每個問題選一項事先審過的改善措施，而且是用編號選。所以它不會把問題歸到錯的攝影機，也不會編造數字。兩位 AI 評審盲評的結果：讓 LLM 自由撰寫時，有 20 % 的措施配錯問題；採用這種做法則是零。」 |
| 2:10–2:31 | 數字 | 全螢幕結果卡（字幕 5）。 | 「每個數字不是實測，就是標明為估計值，腳本都在 repo 裡。在工地模式打字輸入的保留測試句上，檢查器讓小模型的規則完全正確率從 40 % 提升到 95 %。在六棟偵測模型沒訓練過的測試建築上，經過兩項警報邏輯修正後，倉庫模式涵蓋了全部 53 個門口與碼頭事件，其中 52 個有自己的警報。」 |
| 2:31–2:50 | 請求 | 模型對照表（用途 · 模型 · Hailo-10H 參考值），接著是結尾卡「SayGuard · github.com/Rober1208/sayguard」。字幕 6。 | 「每項工作都對應到一個 Hailo 公開的 Hailo-10H 模型，我們的安全帽偵測模型也已經編譯成 Hailo-10H 格式：在 Hailo 模擬器裡，INT8 的 mAP50 是 0.889，浮點版是 0.893。Stage II 我們會把所有模型搬到 UGen300 上（每個區域一台、功耗 2.5 瓦），並在上面實測。我們希望有機會在真正的現場證明它。」 |

如果 0:37 那段錄到的卡片沒有檢查器備註（代表 LLM 這次時段答對了），就把「午夜」那句旁白換成：「When the model gets a detail wrong, the checker corrects it and shows the correction.」（模型答錯細節時，檢查器會改正並顯示出來。）午夜的例子是在實際運作的伺服器上錄到的（FACTS.md 的「Rule-parsing examples」）；LLM 以 temperature 0 執行，同一句話通常會得到相同的草稿。

## 畫面字幕（照抄英文）

1. `10 cameras at 10 fps · all 100 frames/s analysed through one batched YOLOv8s · ≈ 40 % of one UGen300 (estimate from Hailo's YOLOv8s reference, 252 FPS at batch 8)`
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
   （中文說明：規則解析用的是打字輸入的保留測試句，不含語音辨識；「Hard-hat mode」是工地模式，「Warehouse mode」是倉庫模式；LLM 是在 PC 上以 Ollama 量測，Hailo 版本尚未量測。「covered」（涵蓋）＝事件有自己的警報，或落在同一人前一則警報的冷卻時間內；「own alert」是嚴格召回率。堆高機規則設有 60 秒冷卻時間，刻意不對在停放堆高機 3 公尺邊緣進進出出的人重複發警報，所以嚴格召回率只有 41 %。表中的主要數字是兩項警報邏輯修正後的最終版，這兩項修正是看過凍結版結果之後、在非測試場景上開發的；修正前的凍結版涵蓋率較低，誤報也較少。W010 是示範場景（管制區來自示範設定檔），W017（開發用）和 W018 是同一棟建築、同一組鏡頭。0.9393 和 0.889 / 0.8927 是不同的評估方式，只能拿 0.889 和 0.8927 比較。）
6. `Hailo-10H references: YOLOv8s 252 FPS (batch 8) · Qwen2.5-1.5B-Instruct 7.35 tokens/s · Whisper-Small 10.6 tokens/s · CLIP ViT-B/32 81.5 FPS · one UGen300 per area of ≤ 12 cameras (coverage spec; warehouse-mode load measured in Stage II)`

## 錄影前準備

### 第 1 段：工地模式

- 先啟動 Ollama（`ollama serve`，需先 pull `qwen2.5:1.5b`），再執行 `python -m sayguard serve`，打開 `http://127.0.0.1:8000/?lang=en`。在 Windows 上，`start_demo.cmd` 會一次做完這三件事；模型載入期間，網頁會自動重新連線。
- 在「Active rules」（啟用中規則）刪掉測試規則，只留「Hard hat required on site」。讓系統跑約 15 分鐘，日報才會有多支攝影機的紀錄。至少要發生三組（攝影機、問題）組合，LLM 才會規劃改善措施；不到三組時，措施改由範本產生，日報頁尾會寫「template report」。
- 按下錄影前先暖機：
  - 先說一條用不到的規則再取消，因為 Whisper 第一次使用時才載入。
  - 等第一則安全帽警報出現，這會載入 CLIP 複核模型。
  - 啟用一條提到任何物件的規則再刪掉。YOLO-World 在第一次有規則提到物件時載入，之後會一直保持載入。
- KPI「On-device LLM」（裝置端 LLM）必須顯示 Online。否則規則卡上的標記會是「Rule extractor」（規則擷取器）而不是「LLM + checker」，LLM 的段落就無法錄製。

### 第 2 段：倉庫模式

- 只需做一次：`python scripts/fetch_smartspaces.py train/Warehouse_010` 和 `python scripts/make_site_streams.py train/Warehouse_010`。倉庫偵測模型的權重要放在 `@work/models/site_yolov8s.pt`（訓練方式見 [SETUP.zh-TW.md](SETUP.zh-TW.md)）。
- 先關掉第 1 段的伺服器：兩種模式預設都使用 8000 埠，也共用同一張 GPU。接著啟動 Ollama，執行 `python -m sayguard serve --config configs/warehouse.yaml`。在 Windows 上，`start_demo_warehouse.cmd` 會一次做完這兩件事並打開網頁。
- 兩條預設規則（「Keep 3 m from forklifts」、「No walking in aisle 7 (forklift parked)」）會產生警報。影片會循環播放，每輪重新開始時平面圖的狀態也會重置，所以平面圖的鏡頭要在同一輪內錄完。
- 介面是英文時請說英文。儀表板會把介面語言傳給語音模型，所以要用國語說規則，得先切到「中文」；打字輸入國語則兩種介面都可以。

### 兩段共通

- 關掉其他佔用大量 GPU 的程式（例如訓練工作），讓每支攝影機維持每秒 10 張畫面。
- 如實說明：倉庫素材要說明是合成影片，UGen300 負載要講明是估計值。儀表板上的「on-device LLM」字樣指的是設計目標；示範是在 PC 上跑的，所以旁白要說「在本地」。倉庫模式從不顯示人員編號，因為編號幾十秒內就會換（Warehouse_010 的 5 分鐘片段中每人平均 10.7 個編號），不要把編號說成固定的身分。挖土機是靠開放詞彙偵測找到的，工地模式的距離線則是用頭部大小估算的；兩者的準確率都未量測，不要當成量測結果引用。
