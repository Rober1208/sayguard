[English](VIDEO_SOURCES.md) | 繁體中文

# 示範影片來源

SayGuard 的兩種模式各用一種畫面：工地模式用十段素材影片，倉庫模式用時間同步、已校正的多鏡頭合成倉庫場景。兩種畫面都不放在這個 repo 裡；下列腳本負責下載或檢查這些畫面，再轉成示範串流。

## 工地模式：十段 Mixkit 影片

十路示範串流剪自 [Mixkit](https://mixkit.co/) 的免費素材影片，依 [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree) 使用。這份授權不要求標示出處，列在這裡是為了方便追查來源。每路串流從表中的起點取 60 秒，縮放成 960 px 寬、10 fps，並移除聲音（不到 60 秒的影片會循環播放）。

| 攝影機 | 名稱 | Mixkit 影片 | 存檔名稱 | 起點（秒） | 授權 |
|---|---|---|---|---|---|
| cam01 | Site Gate | [Man on the phone near a construction site](https://mixkit.co/free-stock-video/man-on-the-phone-near-a-construction-site-1442/) | `mixkit-1442.mp4` | 0 | Mixkit Stock Video Free License |
| cam02 | Site Office | [Young people working together in the office](https://mixkit.co/free-stock-video/young-people-working-together-in-the-office-42641/) | `mixkit-42641.mp4` | 0 | Mixkit Stock Video Free License |
| cam03 | Excavation Pit | [Excavation crane lifts a rock into a truck](https://mixkit.co/free-stock-video/excavation-crane-lifts-a-rock-into-a-truck-4069/) | `mixkit-4069.mp4` | 0 | Mixkit Stock Video Free License |
| cam04 | Scaffold | [Two construction workers](https://mixkit.co/free-stock-video/two-construction-workers-1436/) | `mixkit-1436.mp4` | 0 | Mixkit Stock Video Free License |
| cam05 | House Frame | [Construction workers at a house under construction](https://mixkit.co/free-stock-video/construction-workers-at-a-house-under-construction-1459/) | `mixkit-1459.mp4` | 0 | Mixkit Stock Video Free License |
| cam06 | Welding Bay | [Welder working at a construction](https://mixkit.co/free-stock-video/welder-working-at-a-construction-4125/) | `mixkit-4125.mp4` | 0 | Mixkit Stock Video Free License |
| cam07 | Warehouse Aisle | [Young worker doing inventory](https://mixkit.co/free-stock-video/young-worker-doing-inventory-4705/) | `mixkit-4705.mp4` | 0 | Mixkit Stock Video Free License |
| cam08 | Loading Dock | [Courier worker preparing boxes on a loading truck](https://mixkit.co/free-stock-video/courier-worker-preparing-boxes-on-a-loading-truck-31346/) | `mixkit-31346.mp4` | 0 | Mixkit Stock Video Free License |
| cam09 | Maintenance Shop | [Mechanic pouring oil into an engine](https://mixkit.co/free-stock-video/mechanic-pouring-oil-into-an-engine-41937/) | `mixkit-41937.mp4` | 0 | Mixkit Stock Video Free License |
| cam10 | Site Walk | [Architect talking to businessmen](https://mixkit.co/free-stock-video/architect-talking-to-businessmen-4648/) | `mixkit-4648.mp4` | 0 | Mixkit Stock Video Free License |

### 取得影片

授權只允許使用這些影片，不允許再散布，所以 repo 裡沒有附檔。Mixkit 的條款也禁止用腳本或機器人大量下載，因此本專案不提供自動下載。請手動下載這十段影片：

1. `python scripts/fetch_videos.py list` 會列出上表的頁面、每段影片該存成的檔名，以及 raw 資料夾的位置（`<work dir>/videos/raw`；工作目錄預設是 `D:/ugen-work`，可用 `SAYGUARD_WORK` 改到別處）。
2. 用瀏覽器打開每個頁面，按 Download（示範時用的是 720p 版本；其他解析度也行，串流都會縮放到 960 px），以「存檔名稱」欄的檔名存進 raw 資料夾，例如 `mixkit-1442.mp4`。
3. `python scripts/fetch_videos.py check` 會列出缺少或損壞的檔案。如果瀏覽器存成 Mixkit 原本的檔名（或預期檔名的檔案其實是一個網頁），可以把影片留在 raw 資料夾，或用 `--inbox` 指向下載資料夾，再加上 `--rename`：檔名含影片 id 的檔案會改成預期的檔名（來自 `--inbox` 的檔案則是複製過去）。十段都到齊後，`check` 會寫出 `raw/manifest.json`；如果 `videos/selection.json` 還不存在，也會依上表的攝影機分配產生這個檔案（加 `--force` 則覆蓋既有檔案）。
4. `python scripts/make_streams.py` 把影片轉成 `videos/streams/cam01.mp4` … `cam10.mp4`。

想換成別的畫面，可以把任何採用 Stock Video Free License（不是 Restricted License）的 Mixkit 影片存成 `mixkit-<id>.mp4` 放進 raw 資料夾，再執行 `python scripts/score_clips.py`：它會用示範用的偵測模型掃過資料夾裡的每段影片，替 `configs/site.yaml` 裡的攝影機挑出最合適的十段，並改寫 `selection.json`。

## 倉庫模式：NVIDIA PhysicalAI-SmartSpaces 場景

倉庫模式使用 **NVIDIA PhysicalAI-SmartSpaces，MTMC_Tracking_2026**：一組合成倉庫場景，以時間同步、已校正、視野互相重疊的 1080p 30 fps 攝影機拍攝，附平面圖和標準答案，授權為 **CC BY 4.0**（[資料集說明頁](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)）。出處標示："NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0"；修改內容：每 3 張影格保留 1 張（10 fps）、縮放到 1280 px、加上 SayGuard 的標示和人臉馬賽克。即時示範（`configs/warehouse.yaml`）播放的是 `train/Warehouse_010`，下表其他場景只用於評估。所有用到的 SmartSpaces 場景（包括偵測模型的訓練場景）及各自的用途，都列在 [LICENSES.zh-TW.md](LICENSES.zh-TW.md)。

### 轉成串流的場景

| 場景 | 用途 | 攝影機數 | 串流影格數（10 fps） | 設定檔 |
|---|---|---|---|---|
| `train/Warehouse_010` | 即時示範，兼作測試 | 11 | 3000 | `configs/warehouse.yaml` |
| `train/Warehouse_013` | 測試 | 10 | 3000 | `configs/eval/warehouse_013.yaml` |
| `train/Warehouse_015` | 測試 | 10 | 3000 | `configs/eval/warehouse_015.yaml` |
| `train/Warehouse_018` | 測試 | 3 | 3000 | `configs/eval/warehouse_018.yaml` |
| `train/Warehouse_019` | 測試 | 3 | 3000 | `configs/eval/warehouse_019.yaml` |
| `val/Warehouse_022` | 測試 | 4 | 3090 | `configs/eval/warehouse_022.yaml` |
| `train/Warehouse_012` | 驗證 | 8 | 3000 | `configs/eval/warehouse_012.yaml` |
| `val/Warehouse_020` | 驗證：倉庫模式的參數在這裡決定；16 支攝影機，超出 12 支的規格 | 16 | 3000 | `configs/eval/warehouse_020.yaml` |
| `train/Warehouse_009` | 僅供開發（和訓練場景 W008 是同一棟建築、同一組鏡頭） | 12 | 3000 | `configs/eval/warehouse_009.yaml` |
| `train/Warehouse_017` | 僅供開發（和測試場景 W018 是同一棟建築、同一組鏡頭） | 3 | 3000 | `configs/eval/warehouse_017.yaml` |

偵測模型從沒用測試場景訓練，參數也是在驗證場景上決定的，測試場景的建築和訓練場景不同。其中兩個不算盲測：W010 是開發追蹤時一直在看的示範場景，W018 和開發用的 W017 是同一個空間。偵測模型的訓練場景不轉成串流：`scripts/prepare_site_dataset.py` 直接從下載的影片取樣影格。

### 製作串流

兩支腳本都用資料集路徑指定場景，輸出寫到工作目錄（預設 `D:/ugen-work`，可用 `SAYGUARD_WORK` 改到別處）。下載需要 `huggingface_hub`；轉檔需要 PATH 上有 `ffmpeg` 和 `ffprobe`。

```bash
python scripts/fetch_smartspaces.py train/Warehouse_010     # 校正參數、平面圖、標準答案、所有影片
python scripts/make_site_streams.py train/Warehouse_010     # 影格鎖定的 10 fps 串流
```

1. `fetch_smartspaces.py` 會把 `calibration.json`、`map.png`、`ground_truth.json` 和 `videos/Camera_XXXX.mp4` 放到 `@work/datasets/smartspaces/<split>/<scene>/`。深度圖不下載；中斷後可以續傳，失敗會自動重試。`--no-videos` 只抓校正參數、標準答案和平面圖，`--no-gt` 不抓標準答案，`--cameras 0 1 2` 只抓指定的攝影機。
2. `make_site_streams.py` 會寫出 `@work/videos/site/<scene>/Camera_XXXX.mp4`（1280×720、10 fps、H.264、無音軌）和 `streams.json`。選項：`--cameras`、`--step`（預設 3：30 → 10 fps）、`--width`（預設 1280）、`--jobs`（同時執行的 ffmpeg 行程數，預設 4）、`--threads`（每個行程的執行緒數）、`--force`（已通過驗證的輸出也重新編碼）。

`make_site_streams.py` 怎麼產生串流，以及為什麼這樣做：

- **影格鎖定**：倉庫模式把所有攝影機整合到同一張平面圖上，標準答案又是依來源影格編號建立索引，所以每支攝影機輸出的第 k 張都必須是來源的第 3k 張。腳本依編號挑影格（對影格編號做 `select`，事後再改寫時間戳），而不用 `fps=`：`fps=` 依時間戳挑選，可能差一張。結尾湊不滿一組的影格直接捨棄，讓每支攝影機都停在同一張。
- **檢查**：還在複製中（檔案大小仍在變動）或不完整（封包數比容器索引記載的少）的來源會直接跳過。每次編碼都要確認解碼器輸出了每一張來源影格，且沒有解碼錯誤（ffmpeg 6.1 以上才會回報這項資訊；版本較舊時，腳本會警告無法確認對齊）；輸出也必須剛好是 ⌊來源影格數 / step⌋ 張、影格率為來源 fps / step、寬度符合設定。輸出會先寫到 `.part.mp4`，所以中途中斷也不會留下看似已完成的串流。
- **固定影格率**：播放檔案來源時，速度依 OpenCV 回報的影格率而定，所以每路串流都以精確的固定影格率編碼；如果各攝影機的平均影格率不同，腳本會發出警告，因為其中一路會播得比較快，逐漸失去同步。
- **`streams.json`** 記錄場景、來源 fps、step、fps、畫面尺寸、影格數（以最短的攝影機為準）和攝影機清單。`scripts/eval_site.py` 從這裡讀取 step 和 fps，把每張串流影格對回對應的標準答案影格。
- 重新執行時，會先驗證既有的輸出，通過就跳過；如果某路串流的來源已經不在磁碟上，只有在上一次的 `streams.json` 以相同設定列出它時才會保留。

即時示範時，伺服器以影格鎖定的方式播放這些檔案（`sayguard/sources.py` 裡的 `SyncedFileGroup`）：同一個讀取器取出每個檔案的第 k 張影格，一起送出；播到最短的檔案結尾時，所有檔案一起回到開頭。

準備評估場景也是同樣兩個指令，換成該場景的路徑即可，例如 `python scripts/fetch_smartspaces.py val/Warehouse_022` 和 `python scripts/make_site_streams.py val/Warehouse_022`。每個 `configs/eval/warehouse_0XX.yaml` 的檔頭都附上了這兩個指令；評估方式見 [SITE_MODE.zh-TW.md](SITE_MODE.zh-TW.md)。
