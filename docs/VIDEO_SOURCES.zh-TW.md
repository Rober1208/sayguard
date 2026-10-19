[English](VIDEO_SOURCES.md) | 繁體中文

# 示範影片來源

SayGuard 用兩種畫面，每種模式一種：安全帽模式用十段素材影片，倉庫模式用時間同步、已校正的多攝影機倉庫場景。
兩者都不在本 repository 中；下面的腳本負責下載或檢查這些畫面，並轉成示範串流。

## 安全帽模式：十段 Mixkit 影片

十路示範攝影機串流剪自 [Mixkit](https://mixkit.co/) 的免費素材影片，依 [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree) 使用。這份授權不要求標示出處；列在這裡是為了讓來源可追溯。每路串流從表中的起點開始取 60 s，縮放為 960 px 寬、10 fps、移除音訊（短於 60 s 的影片會循環播放）。

| 攝影機 | 名稱 | Mixkit 影片 | 存檔名稱 | 起點 (s) | 授權 |
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

本 repository 不含這些畫面：授權涵蓋的是使用影片，不是再散布。Mixkit 的條款也禁止用腳本或機器人大量下載，所以這裡沒有任何程式會替你下載。請手動下載這十段影片：

1. `python scripts/fetch_videos.py list` 會列出上表的頁面、每段影片要存成的檔名，以及 raw 資料夾（`<work dir>/videos/raw`；工作目錄（work dir）預設為 `D:/ugen-work`，可用 `SAYGUARD_WORK` 另行指定）。
2. 用瀏覽器開啟每個頁面，按 Download（示範用的是 720p 檔；任何解析度都可以，串流會縮放到 960 px），並以「存檔名稱」欄的檔名存進 raw 資料夾，例如 `mixkit-1442.mp4`。
3. `python scripts/fetch_videos.py check` 會回報缺少或損壞的檔案。如果瀏覽器保留了 Mixkit 原本的檔名（或預期檔名的檔案其實是存下來的網頁），可以把下載的影片留在 raw 資料夾，或用 `--inbox` 指向下載資料夾，再加上 `--rename`：檔名含有影片 id 的檔案會被改成預期的檔名（來自 `--inbox` 的檔案則是複製過去）。十段都到齊後，`check` 會寫出 `raw/manifest.json`，並在 `videos/selection.json` 不存在時依上表的攝影機分配寫出它（`--force` 會覆蓋既有檔案）。
4. `python scripts/make_streams.py` 把它們轉成 `videos/streams/cam01.mp4` … `cam10.mp4`。

想換其他畫面時，把任何以 Stock Video Free License（不是 Restricted License）提供的 Mixkit 影片存進 raw 資料夾，檔名用 `mixkit-<id>.mp4`，再執行 `python scripts/score_clips.py`：它會用示範所用的偵測器跑過資料夾中的每段影片，為 `configs/site.yaml` 中的攝影機挑出最適合的十段，並據此重寫 `selection.json`。

## 倉庫模式：NVIDIA PhysicalAI-SmartSpaces 場景

倉庫模式使用 **NVIDIA PhysicalAI-SmartSpaces，MTMC_Tracking_2026**：由時間同步、已校正、視野互相重疊的
1080p 30 fps 攝影機拍攝的合成倉庫，附平面圖與標註真值，授權為 **CC BY 4.0**
（[資料集說明頁](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)）。出處標示：
"NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0"；所做修改：每 3 張影格保留 1 張（10 fps）、
縮放為 1280 px、加上 SayGuard 的標示與人臉馬賽克。即時示範（`configs/warehouse.yaml`）播放的是
`train/Warehouse_010` 場景；下表其他場景只用於評估。用到的每一個 SmartSpaces 場景（包括偵測器訓練場景）
及其角色，都列在 [LICENSES.zh-TW.md](LICENSES.zh-TW.md)。

### 轉成串流的場景

| 場景 | 角色 | 攝影機數 | 串流影格數（10 fps） | 設定檔 |
|---|---|---|---|---|
| `train/Warehouse_010` | 即時示範兼測試 | 11 | 3000 | `configs/warehouse.yaml` |
| `train/Warehouse_013` | 測試 | 10 | 3000 | `configs/eval/warehouse_013.yaml` |
| `train/Warehouse_015` | 測試 | 10 | 3000 | `configs/eval/warehouse_015.yaml` |
| `train/Warehouse_018` | 測試 | 3 | 3000 | `configs/eval/warehouse_018.yaml` |
| `train/Warehouse_019` | 測試 | 3 | 3000 | `configs/eval/warehouse_019.yaml` |
| `val/Warehouse_022` | 測試 | 4 | 3090 | `configs/eval/warehouse_022.yaml` |
| `train/Warehouse_012` | 驗證 | 8 | 3000 | `configs/eval/warehouse_012.yaml` |
| `val/Warehouse_020` | 驗證：倉庫模式的設定在此選定；16 支攝影機，超出 12 支攝影機的規格 | 16 | 3000 | `configs/eval/warehouse_020.yaml` |
| `train/Warehouse_009` | 僅供開發（與訓練場景 W008 是同一棟建築的成對場景） | 12 | 3000 | `configs/eval/warehouse_009.yaml` |
| `train/Warehouse_017` | 僅供開發（與測試場景 W018 是同一棟建築的成對場景） | 3 | 3000 | `configs/eval/warehouse_017.yaml` |

測試場景從未用於訓練或調參，建築也與訓練場景不同。偵測器訓練場景不轉成串流：
`scripts/prepare_site_dataset.py` 直接從下載的影片中取樣影格。

### 製作串流

兩支腳本都以資料集路徑指定場景，並寫入工作目錄（預設為 `D:/ugen-work`，可用 `SAYGUARD_WORK` 另行指定）。
下載需要 `huggingface_hub`；轉檔需要 PATH 上有 `ffmpeg` 與 `ffprobe`。

```bash
python scripts/fetch_smartspaces.py train/Warehouse_010     # 相機校正、平面圖、標註真值、所有影片
python scripts/make_site_streams.py train/Warehouse_010     # 影格鎖定的 10 fps 串流
```

1. `fetch_smartspaces.py` 會把 `calibration.json`、`map.png`、`ground_truth.json` 與
   `videos/Camera_XXXX.mp4` 放進 `@work/datasets/smartspaces/<split>/<scene>/`。深度圖會略過；下載可續傳並自動重試。
   `--no-videos` 只抓相機校正、標註真值與平面圖，`--no-gt` 略過標註真值，`--cameras 0 1 2` 只抓指定的攝影機。
2. `make_site_streams.py` 會寫出 `@work/videos/site/<scene>/Camera_XXXX.mp4`（1280×720、10 fps、H.264、
   無音訊）與 `streams.json`。選項：`--cameras`、`--step`（預設 3：30 → 10 fps）、`--width`（預設 1280）、
   `--jobs`（平行執行的 ffmpeg 行程數，預設 4）、`--threads`（每個行程的執行緒數）、`--force`（重新編碼已通過驗證的輸出）。

`make_site_streams.py` 怎麼做串流，以及為什麼這樣做：

- **影格鎖定。** 倉庫模式把所有攝影機融合到同一張平面圖上，而標註真值是以來源影格編號索引，所以每支攝影機的
  第 k 張輸出影格都必須是第 3k 張來源影格。因此影格是依編號挑選（對影格編號做 `select`，事後再重寫時間戳），
  而不是用 `fps=`，因為 `fps=` 依時間戳挑選，可能差一張。最後不足一組的影格會丟掉，讓每支攝影機都結束在同一張影格。
- **檢查。** 仍在複製中（檔案大小還在變）或被截斷（封包數少於容器索引所列）的來源會被略過。每次編碼都必須確認
  解碼器交出了每一張來源影格、且沒有解碼錯誤（ffmpeg 6.1 以上才會回報這項資訊；版本較舊時腳本會警告對齊未經檢查），
  輸出也必須剛好有 ⌊來源影格數 / step⌋ 張影格、影格率為來源 fps / step、寬度符合要求。輸出會先寫到 `.part.mp4`，
  所以中斷的執行不會留下看似完成的串流。
- **固定影格率。** 檔案來源依 OpenCV 回報的影格率控制播放速度，所以每路串流都以精確的固定影格率編碼；平均影格率
  不同的攝影機會被回報出來，因為其中一路會播得比較快，脫離影格鎖定。
- **`streams.json`** 記錄場景、來源 fps、step、fps、影格尺寸、影格數（以最短的攝影機為準）與攝影機清單。
  `scripts/eval_site.py` 從中讀取 step 與 fps，把每張串流影格對到它的標註真值影格。
- 重新執行時會驗證既有輸出並略過；來源已不在磁碟上的串流，只有在上一次的 `streams.json` 以相同設定列出它時才會保留。

在即時示範中，伺服器以影格鎖定方式播放這些檔案（`sayguard/sources.py` 中的 `SyncedFileGroup`）：由一個讀取器
取出每個檔案的第 k 張影格並一起發布，播到最短的檔案結尾時，所有檔案一起倒回開頭。

評估場景用同樣兩個指令、換成它自己的路徑來準備，例如
`python scripts/fetch_smartspaces.py val/Warehouse_022` 與
`python scripts/make_site_streams.py val/Warehouse_022`。每個 `configs/eval/warehouse_0XX.yaml` 的檔頭都重列了
它的指令；評估本身的說明見 [SITE_MODE.zh-TW.md](SITE_MODE.zh-TW.md)。
