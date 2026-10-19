/* SayGuard dashboard: plain JavaScript with no network dependencies, because it has to run
   offline on a factory LAN.

   Why some things are done the way they are:
   - Browsers open at most ~6 HTTP/1.1 connections per host and every MJPEG <img> holds one
     open for good, so ten camera streams on one host would starve the API.  Tiles are spread
     over the two loopback names (127.0.0.1 / localhost) so two connections stay free; when no
     second name reaches the server (LAN access) a tile grabs single frames from the same
     stream instead.  ?stream=mjpeg or ?stream=poll forces one mode.
   - /api/state is polled by a self-scheduling timer (requests never overlap); events and rules
     are refetched only when their version counter changes.
   - Server data is written with textContent; the report markdown is HTML-escaped before the
     small renderer adds its own tags. */
"use strict";

(() => {
  // ------------------------------------------------------------------ constants
  const MAX_PER_HOST = 6;
  const API_RESERVE = 2;
  const POLL_FPS = 6;
  const FLASH_MS = 6000;
  const HOLD_MS = 350;
  const MIN_REC_MS = 600;
  const MAX_REC_MS = 15000;
  const EVENTS_LIMIT = 30;
  const SEV = ["low", "medium", "high"];

  const OBJ_ZH = {
    "forklift": "堆高機", "hand cart": "推車", "ladder": "梯子", "fire extinguisher": "滅火器",
    "fire": "火焰", "smoke": "煙霧", "cardboard box": "箱子", "pallet": "棧板", "excavator": "挖土機",
    "crane": "吊車", "truck": "卡車", "motorcycle": "機車", "bicycle": "腳踏車", "car": "汽車",
    "safety vest": "反光背心", "cell phone": "手機", "backpack": "背包", "cigarette": "香菸",
    "knife": "刀子", "bucket": "水桶", "power drill": "電鑽", "dog": "狗",
    "robot": "機器人", "humanoid_robot": "人形機器人", "mobile_robot": "移動機器人", "transporter": "搬運車",
    "pallet_truck": "托板車",
  };

  // ------------------------------------------------------------------ i18n
  const I18N = {
    en: {
      tagline: "One UGen300 · {n} cameras · rules you can say",
      site: "Site",
      reconnecting: "Reconnecting…",
      lang_group: "Language",
      kpis_label: "Key figures",
      kpi_cams: "Cameras online",
      kpi_cams_sub: "live streams",
      kpi_fps: "Frames analysed / s",
      kpi_fps_sub: "batch {b} · {ms} ms per batch",
      kpi_fps_sub0: "batched across all cameras",
      kpi_load: "Est. UGen300 load",
      kpi_load_caption: "Estimate from Hailo Model Zoo {model} (Hailo-10H, batch 8)",
      kpi_load_na: "Not measured",
      kpi_load_na_caption: "No Hailo reference for the {px} px detector; measured in Stage II",
      kpi_load_b1: "Batch 1 capacity: {n} FPS",
      kpi_rules: "Active rules",
      kpi_rules_sub: "of {n} total",
      kpi_confirmed: "Alerts confirmed · 24 h",
      kpi_confirmed_sub: "of {n} raised",
      kpi_filtered: "False alarms filtered by double-check",
      kpi_filtered_sub: "24 h · {n} awaiting check",
      kpi_filtered_na: "Hard-hat alerts only: {n} alerts from other rules are not double-checked",
      kpi_llm: "On-device LLM",
      llm_online: "Online",
      llm_offline: "Offline",
      llm_fallback: "rule extractor fallback",
      kpi_site: "Workers on site",
      kpi_site_unit: "unique",
      kpi_site_sub: "{n} camera detections",
      site_workers: "Workers on site: {unique} unique ({sum} camera detections)",
      wall_title: "Live cameras",
      wall_meta: "{n} cameras · {fps} FPS target each · click to enlarge",
      offline: "Offline",
      connecting: "Connecting…",
      people: "People",
      no_helmet: "No hard hat",
      fps_unit: "fps",
      open_camera: "Open {name} enlarged",
      map_title: "Floor plan · each worker counted once",
      open_map: "Open the floor plan enlarged",
      map_workers: "Workers on site, each counted once",
      vehicles: "Vehicles",
      alert: "Alert",
      alerts_title: "Live alerts",
      alerts_meta: "Newest first · click a card for details",
      alerts_empty: "No alerts yet. New alerts show up here within a second.",
      vlm_confirmed: "Confirmed by double-check",
      vlm_false: "False alarm filtered",
      vlm_checking: "Checking…",
      vlm_unchecked: "Not checked",
      vlm_only_helmet: "Only hard-hat alerts are double-checked",
      seen_by: "seen by {list}",
      dist_m: "{d} m",
      dist_tip: "Distance measured on the floor plan",
      sev_high: "High",
      sev_medium: "Medium",
      sev_low: "Low",
      say_title: "Say a rule",
      say_pipeline: "Whisper → LLM → checker",
      mic_idle: "Hold to talk",
      mic_idle_sub: "or click to start and stop · Mandarin or English",
      mic_rec: "Listening…",
      mic_rec_sub_hold: "Release to finish",
      mic_rec_sub_toggle: "Click again to finish",
      mic_wait: "Working…",
      mic_wait_sub: "speech to text, then text to rule",
      mic_label: "Record a spoken rule: hold to talk, or click to start and stop",
      mic_unsupported: "This browser cannot record here (needs MediaRecorder on localhost or HTTPS). Type the rule instead.",
      mic_denied: "Microphone access is blocked. Allow it in the browser, or type the rule.",
      mic_error: "Could not start the microphone: {detail}",
      mic_short: "Too short. Hold the button while you speak.",
      input_ph: "Type or say a safety rule…",
      input_label: "Safety rule in plain language",
      check: "Check",
      examples: "Try an example",
      transcribing: "Transcribing speech…",
      heard: "Heard in {ms} on {device}",
      parsing: "Turning it into a rule…",
      asr_empty: "Nothing was heard. Please try again.",
      asr_busy: "The speech model cannot load right now (memory): {detail}",
      err_asr: "Speech recognition failed: {detail}",
      err_parse: "Could not parse the rule: {detail}",
      proposal_title: "Proposed rule",
      proposal_failed: "Rule not understood",
      parser_llm: "LLM + checker",
      parser_rules: "Rule extractor",
      parser_failed: "Not understood",
      parser_llm_tip: "A small LLM proposed the rule; a deterministic checker verified the numbers, times and objects stated in the sentence",
      parser_rules_tip: "LLM offline, so the deterministic rule extractor was used",
      parser_failed_tip: "Neither the LLM nor the extractor could tell what to watch for",
      f_watch: "Watch for",
      f_where: "Where",
      f_when: "When",
      f_hold: "Hold for",
      f_severity: "Severity",
      f_message: "Alert text",
      f_repeat: "Repeat",
      any_time: "Any time",
      all_cameras: "All cameras",
      except: "except {z}",
      overnight: "overnight",
      covers: "Covers {list}",
      covers_n: "Covers {n} of {total} cameras",
      covers_none: "No camera matches this scope",
      hold_s: "{s} s before alerting",
      hold_short: "≥ {s} s",
      cooldown_s: "at most every {s} s",
      notes: "Checker notes ({n})",
      problems: "Needs fixing",
      activate: "Activate rule",
      saving: "Activating…",
      cancel: "Cancel",
      dismiss: "Dismiss",
      activated: "Rule activated: {name}",
      rules_title: "Active rules",
      rules_meta: "{on} of {n} on",
      rules_empty: "No rules yet. Say one above.",
      enable_rule: "Enable rule {name}",
      delete_rule: "Delete rule {name}",
      delete: "Delete",
      confirm_delete: "Delete the rule \"{name}\"?",
      deleted: "Rule deleted",
      rule_on: "Rule enabled",
      rule_off: "Rule paused",
      report_btn: "Daily safety report",
      report_title: "Daily safety report",
      report_loading: "Writing the report with the on-device LLM…",
      report_meta: "Generated by {gen} in {s}",
      copy: "Copy",
      copied: "Copied to clipboard",
      copy_failed: "Copy failed",
      regenerate: "Regenerate",
      close: "Close",
      err_report: "Could not build the report: {detail}",
      sys_detector: "Detector",
      sys_ov: "Open vocabulary",
      sys_vlm: "Alert double-check",
      sys_asr: "Speech (ASR)",
      batch_info: "{ms} ms / batch · {cap} FPS capacity",
      ov_none: "idle (no object rules)",
      ov_error: "unavailable",
      ov_loading: "loading",   // object rules exist: YOLO-World + CLIP load in the background
      asr_idle: "idle · loads on first use",
      st_idle: "idle",
      st_loading: "loading",
      st_ready: "ready",
      st_disabled: "disabled",
      st_unavailable: "unavailable",
      st_error: "error",
      queue_n: "queue {n}",
      checked_n: "{n} checked",
      type_no_helmet: "Person without a hard hat",
      type_person_present: "Anyone present",
      type_person_count_over: "More than {n} people at once",
      type_object_present: "{obj} visible",
      type_object_near_person: "Person within {n} m of {art} {obj}",
      ev_time: "Time",
      ev_camera: "Camera",
      ev_zone: "Zone",
      ev_rule: "Rule",
      ev_severity: "Severity",
      ev_detail: "Detail",
      ev_spoken: "Spoken rule",
      ev_vlm: "Double-check",
      ev_track: "Track ID",
      ev_seen_by: "Seen by",
      ev_distance: "Distance",
      ev_world: "Floor position",
      snapshot_missing: "No snapshot",
      rules_here: "Rules on this camera",
      recent_here: "Recent alerts on this camera",
      map_unique: "Workers",
      map_sum: "Camera detections",
      map_zones: "Workers per zone",
      map_recent: "Recent alerts on site",
      none: "None",
      err_network: "Server unreachable",
      err_timeout: "The server took too long to answer",
      err_server: "Server error",
      err_generic: "Something went wrong: {detail}",
    },
    zh: {
      tagline: "一台 UGen300 · {n} 支攝影機 · 用說的就能加規則",
      site: "廠區",
      reconnecting: "重新連線中…",
      lang_group: "語言",
      kpis_label: "關鍵指標",
      kpi_cams: "攝影機在線",
      kpi_cams_sub: "即時串流",
      kpi_fps: "每秒分析影格",
      kpi_fps_sub: "批次 {b} · 每批 {ms} ms",
      kpi_fps_sub0: "所有攝影機批次推論",
      kpi_load: "UGen300 預估負載",
      kpi_load_caption: "依 Hailo Model Zoo {model} 數據估算（Hailo-10H，batch 8）",
      kpi_load_na: "尚未量測",
      kpi_load_na_caption: "{px} px 偵測模型沒有 Hailo 參考數據，Stage II 實測",
      kpi_load_b1: "Batch 1 容量：{n} FPS",
      kpi_rules: "啟用中規則",
      kpi_rules_sub: "共 {n} 條",
      kpi_confirmed: "已確認警報 · 24 小時",
      kpi_confirmed_sub: "共觸發 {n} 次",
      kpi_filtered: "複核濾除的誤報",
      kpi_filtered_sub: "24 小時 · {n} 筆待複核",
      kpi_filtered_na: "只複核安全帽警報：其他規則的 {n} 筆警報不經複核",
      kpi_llm: "裝置端 LLM",
      llm_online: "連線",
      llm_offline: "離線",
      llm_fallback: "改用規則擷取器",
      kpi_site: "現場人數",
      kpi_site_unit: "人",
      kpi_site_sub: "各鏡頭合計 {n}",
      site_workers: "現場人數：{unique} 人（各鏡頭合計 {sum}）",
      wall_title: "即時影像",
      wall_meta: "{n} 支攝影機 · 每支目標 {fps} FPS · 點擊放大",
      offline: "離線",
      connecting: "連線中…",
      people: "人數",
      no_helmet: "未戴安全帽",
      fps_unit: "fps",
      open_camera: "放大檢視 {name}",
      map_title: "平面圖・每人只算一次",
      open_map: "放大檢視平面圖",
      map_workers: "現場人數（每人只算一次）",
      vehicles: "車輛",
      alert: "警報",
      alerts_title: "即時警報",
      alerts_meta: "最新在前 · 點擊卡片查看細節",
      alerts_empty: "目前沒有警報，新警報會在一秒內出現在這裡。",
      vlm_confirmed: "複核確認",
      vlm_false: "已濾除誤報",
      vlm_checking: "複核中…",
      vlm_unchecked: "未檢查",
      vlm_only_helmet: "只複核安全帽警報",
      seen_by: "鏡頭：{list}",
      dist_m: "{d} 公尺",
      dist_tip: "在平面圖上量測的距離",
      sev_high: "高",
      sev_medium: "中",
      sev_low: "低",
      say_title: "用說的加規則",
      say_pipeline: "Whisper → LLM → 檢查器",
      mic_idle: "按住說話",
      mic_idle_sub: "或點一下開始／結束 · 國語或英語",
      mic_rec: "聆聽中…",
      mic_rec_sub_hold: "放開即完成",
      mic_rec_sub_toggle: "再點一下完成",
      mic_wait: "處理中…",
      mic_wait_sub: "語音轉文字，再轉成規則",
      mic_label: "錄製口述規則：按住說話，或點一下開始與結束",
      mic_unsupported: "這個瀏覽器無法在此錄音（需要 localhost 或 HTTPS 與 MediaRecorder），請直接輸入規則。",
      mic_denied: "麥克風權限被封鎖，請在瀏覽器允許，或直接輸入規則。",
      mic_error: "無法啟動麥克風：{detail}",
      mic_short: "太短了，請按住按鈕再說話。",
      input_ph: "輸入或說出一條安全規則…",
      input_label: "以口語描述的安全規則",
      check: "解析",
      examples: "試試範例",
      transcribing: "語音辨識中…",
      heard: "辨識耗時 {ms}（{device}）",
      parsing: "轉換成規則中…",
      asr_empty: "沒有聽到內容，請再試一次。",
      asr_busy: "語音模型目前無法載入（記憶體不足）：{detail}",
      err_asr: "語音辨識失敗：{detail}",
      err_parse: "規則解析失敗：{detail}",
      proposal_title: "建議規則",
      proposal_failed: "無法理解這條規則",
      parser_llm: "LLM + 檢查器",
      parser_rules: "規則擷取器",
      parser_failed: "解析失敗",
      parser_llm_tip: "小型 LLM 提出規則，再由確定性檢查器核對句中明確的數字、時間與物件",
      parser_rules_tip: "LLM 離線，改用確定性規則擷取器",
      parser_failed_tip: "LLM 與擷取器都無法判斷要偵測什麼",
      f_watch: "偵測",
      f_where: "位置",
      f_when: "時段",
      f_hold: "持續",
      f_severity: "嚴重度",
      f_message: "警報文字",
      f_repeat: "重複間隔",
      any_time: "全天",
      all_cameras: "全部攝影機",
      except: "（{z}除外）",
      overnight: "跨夜",
      covers: "涵蓋：{list}",
      covers_n: "涵蓋 {n}／{total} 支攝影機",
      covers_none: "沒有攝影機符合這個範圍",
      hold_s: "持續 {s} 秒才警報",
      hold_short: "≥ {s} 秒",
      cooldown_s: "最多每 {s} 秒一次",
      notes: "檢查器備註（{n}）",
      problems: "需要修正",
      activate: "啟用規則",
      saving: "啟用中…",
      cancel: "取消",
      dismiss: "關閉",
      activated: "已啟用規則：{name}",
      rules_title: "啟用中規則",
      rules_meta: "{on}／{n} 條啟用",
      rules_empty: "還沒有規則，在上方說一條吧。",
      enable_rule: "啟用規則「{name}」",
      delete_rule: "刪除規則「{name}」",
      delete: "刪除",
      confirm_delete: "要刪除規則「{name}」嗎？",
      deleted: "已刪除規則",
      rule_on: "規則已啟用",
      rule_off: "規則已暫停",
      report_btn: "每日安全報告",
      report_title: "每日安全報告",
      report_loading: "裝置端 LLM 撰寫報告中…",
      report_meta: "產生方式：{gen} · 耗時 {s}",
      copy: "複製",
      copied: "已複製到剪貼簿",
      copy_failed: "複製失敗",
      regenerate: "重新產生",
      close: "關閉",
      err_report: "報告產生失敗：{detail}",
      sys_detector: "偵測器",
      sys_ov: "開放詞彙",
      sys_vlm: "警報複核",
      sys_asr: "語音辨識",
      batch_info: "每批 {ms} ms · 容量 {cap} FPS",
      ov_none: "待命（沒有物件規則）",
      ov_error: "無法使用",
      ov_loading: "載入中",
      asr_idle: "待命 · 第一次使用時載入",
      st_idle: "待命",
      st_loading: "載入中",
      st_ready: "就緒",
      st_disabled: "已停用",
      st_unavailable: "無法使用",
      st_error: "錯誤",
      queue_n: "佇列 {n}",
      checked_n: "已檢查 {n}",
      type_no_helmet: "有人未戴安全帽",
      type_person_present: "有人出現",
      type_person_count_over: "同時超過 {n} 人",
      type_object_present: "出現{obj}",
      type_object_near_person: "有人在{obj} {n} 公尺內",
      ev_time: "時間",
      ev_camera: "攝影機",
      ev_zone: "區域",
      ev_rule: "規則",
      ev_severity: "嚴重度",
      ev_detail: "細節",
      ev_spoken: "口述原句",
      ev_vlm: "複核",
      ev_track: "追蹤 ID",
      ev_seen_by: "鏡頭",
      ev_distance: "距離",
      ev_world: "平面位置",
      snapshot_missing: "沒有快照",
      rules_here: "套用在此攝影機的規則",
      recent_here: "此攝影機近期警報",
      map_unique: "工人",
      map_sum: "各鏡頭合計",
      map_zones: "各區人數",
      map_recent: "現場近期警報",
      none: "無",
      err_network: "無法連線到伺服器",
      err_timeout: "伺服器回應逾時",
      err_server: "伺服器錯誤",
      err_generic: "發生錯誤：{detail}",
    },
  };

  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch { /* storage blocked */ } },
  };

  let lang = (new URLSearchParams(location.search).get("lang") || store.get("sayguard.lang")) === "zh" ? "zh" : "en";

  function t(key, vars) {
    let s = (I18N[lang] && I18N[lang][key]) ?? I18N.en[key] ?? key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
    return s;
  }

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const SVGNS = "http://www.w3.org/2000/svg";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const numOr = (v, fallback = null) => (isNum(v) ? v : fallback);
  const locale = () => (lang === "zh" ? "zh-TW" : "en-GB");
  const sep = () => (lang === "zh" ? "、" : ", ");
  const ucfirst = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "dataset") Object.assign(node.dataset, v);
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid instanceof Node ? kid : String(kid));
    }
    return node;
  }

  const ICONS = {
    person: "M12 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM5 20.5c0-3.9 3.1-6.5 7-6.5s7 2.6 7 6.5",
    nohat: "M4.5 15.5a7.5 7.5 0 0 1 15 0M2.5 15.5h19M10.5 8.2V6.5h3v1.7M3.5 3.5l17 17",
    check: "M5 12.5l4.5 4.5L19 7.5",
    x: "M6 6l12 12M18 6L6 18",
    minus: "M6 12h12",
    trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
    copy: "M9 9h11v11H9zM15 9V4H4v11h5",
    refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
    camoff: "M3 3l18 18M10.5 6H15l2 3h2a1 1 0 0 1 1 1v7.5M17 19H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1h2",
    shield: "M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6zM9 12l2 2 4-4",
    dist: "M3 12h18M6.5 8.5L3 12l3.5 3.5M17.5 8.5L21 12l-3.5 3.5",
    truck: "M2.5 6.5h11v9h-11zM13.5 9.5h4l3 3.5v2.5h-7M5 18a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0M15 18a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0",
  };

  function icon(name, cls = "ico") {
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", cls);
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(SVGNS, "path");
    path.setAttribute("d", ICONS[name] || "");
    svg.append(path);
    return svg;
  }

  function setText(id, text) {
    const node = $(id);
    if (node && node.textContent !== text) node.textContent = text;
  }

  const fmt1 = (v) => (isNum(v) ? v.toFixed(1) : "–");
  const fmtInt = (v) => (isNum(v) ? Math.round(v).toLocaleString(locale()) : "–");
  const fmtMs = (v) => (!isNum(v) ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
  const fmtNumber = (v) => (!isNum(v) ? "?" : Number.isInteger(v) ? String(v) : String(+v.toFixed(1)));

  function fmtTime(ts) {
    if (!isNum(ts)) return "–";
    return new Date(ts * 1000).toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }

  function fmtDateTime(ts) {
    if (!isNum(ts)) return "–";
    return new Date(ts * 1000).toLocaleString(locale(), {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  }

  const sevKey = (s) => (SEV.includes(s) ? s : "medium");
  const sevChip = (s) => h("span", { class: `sev sev-${sevKey(s)}`, text: t(`sev_${sevKey(s)}`) });

  // ------------------------------------------------------------------ state
  const S = {
    config: null,
    site: null,
    cams: [],
    camMap: new Map(),
    state: null,
    rules: [],
    rulesLoaded: false,
    events: [],
    rulesVersion: null,
    eventsVersion: null,
    eventsLoaded: false,
    seen: new Set(),
    fails: 0,
    offline: false,
    skewMs: 0,
    proposal: null,
    heardNote: "",
    busy: false,
    newRuleId: null,
    camModal: null,
  };

  const serverNow = () => Date.now() + S.skewMs;
  const zoneName = (z) => {
    const names = S.config && S.config.zones && S.config.zones[z];
    if (names) return names[lang] || names.en || z;
    const planZone = S.site && Array.isArray(S.site.zones) ? S.site.zones.find((x) => x && x.id === z) : null;
    return planZone ? planZone.name || z : z || "";
  };
  const camName = (id) => (S.camMap.get(id) || {}).name || id;
  const objName = (o) => (lang === "zh" ? OBJ_ZH[o] || o : o) || "?";

  // ------------------------------------------------------------------ site mode
  const MAP_ID = "map";   // reserved tile id: the server routes /stream/map ahead of /stream/{cam_id}
  const camTag = (id) => String(id || "").toUpperCase();
  const camLabel = (id) => (camName(id) !== id ? `${camTag(id)} · ${camName(id)}` : camTag(id));
  const fmtDist = (m) => t("dist_m", { d: m < 10 ? m.toFixed(1) : String(Math.round(m)) });
  const siteStatus = () => (S.site && S.state && S.state.site && typeof S.state.site === "object" ? S.state.site : null);

  // One alert per worker, however many cameras see them.  Outside site mode these fields are
  // ignored, so the single-camera dashboard renders exactly as it always has.
  function siteInfo(ev) {
    const x = ev && ev.extra;
    if (!S.site || !x || typeof x !== "object" || !isNum(x.global_id)) return null;
    return {
      cams: Array.isArray(x.cameras) ? x.cameras.filter((c) => typeof c === "string" && c) : [],
      dist: numOr(x.distance_m),
      world: Array.isArray(x.world) && isNum(x.world[0]) && isNum(x.world[1]) ? x.world : null,
    };
  }


  // ------------------------------------------------------------------ API
  class ApiError extends Error {
    constructor(status, detail) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  }

  function errDetail(data, status) {
    const d = data && data.detail;
    if (typeof d === "string" && d) return d;
    if (Array.isArray(d)) return d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ");
    if (status >= 500) return `${t("err_server")} (${status})`;
    return `HTTP ${status}`;
  }

  async function api(path, { method = "GET", json, form, timeout = 15000 } = {}) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeout);
    const init = { method, signal: ctl.signal, cache: "no-store", headers: {} };
    if (json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(json);
    } else if (form) {
      init.body = form;
    }
    let res;
    let text = "";
    try {
      res = await fetch(path, init);
      text = await res.text();
    } catch (e) {
      throw new ApiError(0, e && e.name === "AbortError" ? t("err_timeout") : t("err_network"));
    } finally {
      clearTimeout(timer);
    }
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = null; }
    if (!res.ok) throw new ApiError(res.status, errDetail(data, res.status));
    return data;
  }

  // Coalesces bursts: a call while a fetch is running schedules exactly one more run.
  function makeLoader(fn) {
    let running = false;
    let again = false;
    return async function run() {
      if (running) { again = true; return; }
      running = true;
      try {
        do { again = false; await fn(); } while (again);
      } finally {
        running = false;
      }
    };
  }

  // ------------------------------------------------------------------ polling
  let pollTimer = 0;
  let pollInFlight = false;

  function schedulePoll(ms) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollState, ms);
  }

  async function pollState() {
    if (pollInFlight) return;
    pollInFlight = true;
    let delay = 1000;
    try {
      const st = await api("/api/state", { timeout: 5000 });
      if (!st || typeof st !== "object") throw new ApiError(0, "bad state");
      // a restarted server counts rule and event versions from zero again: refetch both
      if (st.boot && S.boot && st.boot !== S.boot) { S.rulesLoaded = false; S.eventsLoaded = false; }
      if (st.boot) S.boot = st.boot;
      // another site configuration (hard-hat mode <-> warehouse mode): cameras, zones and examples on
      // this page are stale, so start over, but not while the officer is typing, recording or confirming
      if (st.config_id && S.configId && st.config_id !== S.configId) S.siteChanged = true;
      if (st.config_id && !S.configId) S.configId = st.config_id;
      if (S.siteChanged && !userBusy()) { location.reload(); return; }
      S.state = st;
      S.fails = 0;
      if (isNum(st.time)) S.skewMs = st.time * 1000 - Date.now();
      setOffline(false);
      renderState();
      if (!S.rulesLoaded || st.rules_version !== S.rulesVersion) loadRules();
      if (!S.eventsLoaded || st.events_version !== S.eventsVersion) loadEvents();
    } catch {
      S.fails += 1;
      if (S.fails >= 2) setOffline(true);
      delay = Math.min(5000, 1000 * S.fails);
    } finally {
      pollInFlight = false;
      schedulePoll(document.hidden ? Math.max(delay, 3000) : delay);
    }
  }

  // detector class names inside an alert detail, in the UI language
  const OBJ_KEYS = Object.keys(OBJ_ZH).sort((a, b) => b.length - a.length);   // "humanoid_robot" before "robot"
  function detailText(d) {
    let s = String(d);
    if (lang === "zh") for (const k of OBJ_KEYS) s = s.split(k).join(OBJ_ZH[k]);
    return s.replace(/_/g, " ");
  }

  function userBusy() {
    const draft = $("ruleText");
    return !!(S.busy || S.proposal || S.camModal || R.phase !== "idle" || (draft && draft.value.trim()));
  }

  function setOffline(off) {
    if (off === S.offline) return;
    S.offline = off;
    $("conn").hidden = !off;
    document.body.classList.toggle("is-stale", off);
    if (!off) restartStreams();
  }

  const loadRules = makeLoader(async () => {
    const version = S.state ? S.state.rules_version : null;
    try {
      const list = await api("/api/rules");
      S.rules = Array.isArray(list) ? list.filter((r) => r && r.id) : [];
      S.rulesLoaded = true;
      S.rulesVersion = version;
      renderRules();
      renderKpis();
      if (S.camModal) S.camModal.update();
    } catch { /* retried on the next poll: the version still differs */ }
  });

  const loadEvents = makeLoader(async () => {
    const version = S.state ? S.state.events_version : null;
    try {
      const list = await api(`/api/events?limit=${EVENTS_LIMIT}`);
      const events = Array.isArray(list) ? list.filter((e) => e && e.id) : [];
      const fresh = [];
      for (const ev of events) {
        if (S.seen.has(ev.id)) continue;
        S.seen.add(ev.id);
        if (S.eventsLoaded) fresh.push(ev);
      }
      if (S.seen.size > 1000) S.seen = new Set(events.map((e) => e.id));
      S.events = events;
      S.eventsVersion = version;
      S.eventsLoaded = true;
      renderAlerts(new Set(fresh.map((e) => e.id)));
      const now = serverNow();
      for (const ev of fresh.slice().reverse()) {
        if (isNum(ev.ts) && now - ev.ts * 1000 < 60000) for (const id of alertTiles(ev)) flashTile(id, ev);
      }
      if (S.camModal) S.camModal.update();
    } catch { /* retried on the next poll */ }
  });

  // ------------------------------------------------------------------ KPIs
  function renderState() {
    renderKpis();
    renderWallState();
    renderSys();
    updateVerdicts();
    if (S.camModal) S.camModal.update();
  }

  function renderKpis() {
    const st = S.state || {};
    const cams = st.cameras && typeof st.cameras === "object" ? st.cameras : null;
    const total = S.cams.length || (cams ? Object.keys(cams).length : 0);
    const online = cams ? Object.values(cams).filter((c) => c && c.online).length : null;
    setText("kCams", online == null ? "–" : String(online));
    setText("kCamsTotal", total ? `/${total}` : "");

    const stats = st.stats || {};
    setText("kFps", fmt1(stats.frames_per_s));
    setText("kFpsSub", isNum(stats.batch_size) && stats.batch_size > 0
      ? t("kpi_fps_sub", { b: fmt1(stats.batch_size), ms: fmt1(stats.det_batch_ms) })
      : t("kpi_fps_sub0"));

    const hl = st.hailo || {};
    const ref = (S.config && S.config.hailo_reference) || {};
    const cap8 = numOr(hl.capacity_batch8, numOr(ref.fps_batch8));
    const cap1 = numOr(hl.capacity_batch1, numOr(ref.fps_batch1));
    const demand = numOr(hl.demand_fps);
    // warehouse mode runs a 960 px detector: Hailo's 640 px figures do not apply, so no estimate
    const comparable = hl.comparable !== false;
    let pct = comparable ? numOr(hl.load_batch8_pct) : null;
    if (comparable && pct == null && demand != null && cap8) pct = (100 * demand) / cap8;
    setText("kLoad", !comparable ? t("kpi_load_na")
      : demand != null && cap8 ? `${Math.round(demand)} / ${Math.round(cap8)} FPS` : "–");
    setText("kLoadPct", pct != null ? `${Math.round(pct)}%` : "");
    const meter = $("kLoadMeter");
    const fill = Math.max(0, Math.min(100, pct || 0));
    $("kLoadFill").style.width = `${fill}%`;
    meter.classList.toggle("is-warn", pct != null && pct >= 70 && pct < 90);
    meter.classList.toggle("is-crit", pct != null && pct >= 90);
    meter.setAttribute("aria-valuenow", String(Math.round(fill)));
    const tick = $("kLoadTick");
    if (comparable && cap1 && cap8 && cap1 < cap8) {
      tick.style.display = "block";
      tick.style.left = `${(100 * cap1) / cap8}%`;
      tick.title = t("kpi_load_b1", { n: Math.round(cap1) });
    } else {
      tick.style.display = "none";
    }
    const model = String(hl.model || ref.model || "yolov8s").replace(/^yolo/i, "YOLO");
    setText("kLoadCaption", comparable ? t("kpi_load_caption", { model }) : t("kpi_load_na_caption", { px: hl.imgsz || "?" }));

    const enabled = S.rules.filter((r) => r.enabled !== false).length;
    setText("kRules", S.rulesLoaded ? String(enabled) : "–");
    setText("kRulesSub", S.rulesLoaded ? t("kpi_rules_sub", { n: S.rules.length }) : "");

    const a = st.alerts_24h || {};
    setText("kConfirmed", fmtInt(a.confirmed));
    setText("kConfirmedSub", isNum(a.total) ? t("kpi_confirmed_sub", { n: fmtInt(a.total) }) : "");
    setText("kFiltered", fmtInt(a.false_alarms));
    // unchecked = hard-hat alerts still waiting for the double-check; other rule types are never checked (not_applicable)
    setText("kFilteredSub", isNum(a.unchecked) ? t("kpi_filtered_sub", { n: fmtInt(a.unchecked) }) : "");
    const filteredSub = $("kFilteredSub");
    if (filteredSub) filteredSub.title = isNum(a.not_applicable) && a.not_applicable > 0 ? t("kpi_filtered_na", { n: fmtInt(a.not_applicable) }) : "";

    const llmModel = (S.config && S.config.models && S.config.models.llm) || "";
    const on = st.llm_online;
    const dot = $("kLlmDot");
    dot.classList.toggle("is-on", on === true);
    dot.classList.toggle("is-off", on === false);
    setText("kLlm", on === true ? t("llm_online") : on === false ? t("llm_offline") : "–");
    setText("kLlmSub", on === false ? `${llmModel} · ${t("llm_fallback")}` : llmModel);

    if (S.site) {
      // the gap between the two numbers is the point: one worker in three views counts once
      const site = siteStatus() || {};
      const unique = numOr(site.unique_people);
      const sum = numOr(site.camera_sum);
      setText("kSite", unique == null ? "–" : String(unique));
      setText("kSiteUnit", unique == null ? "" : t("kpi_site_unit"));
      setText("kSiteSub", sum == null ? "" : t("kpi_site_sub", { n: fmtInt(sum) }));
      $("kSiteBox").title = unique == null ? "" : t("site_workers", { unique, sum: sum == null ? "–" : fmtInt(sum) });
    }
  }

  // ------------------------------------------------------------------ camera wall
  const tiles = new Map();

  // The floor plan is a tile like any camera (/stream/map), so it shares the stream budget,
  // the poll fallback and the reconnect logic.
  function makeTile(id, name, badges, open, cls = "tile") {
    const img = h("img", { class: "tile-img", alt: "", decoding: "async", draggable: "false" });
    const waitText = h("span", { class: "wait-text" });
    const offText = h("span", { class: "off-text" });
    const alertTag = h("span", { class: "tile-alert" });
    const nameEl = h("span", { class: "tile-name", text: name });
    const zone = h("span", { class: "tile-zone" });
    const tile = h("figure", { class: cls, tabindex: "0", role: "button", dataset: { cam: id } },
      h("div", { class: "tile-media" },
        img,
        h("div", { class: "tile-wait" }, h("span", { class: "spin", "aria-hidden": "true" }), waitText),
        h("div", { class: "tile-off" }, icon("camoff"), offText),
        alertTag),
      h("figcaption", { class: "tile-cap" },
        nameEl,
        zone,
        h("span", { class: "tile-badges" }, badges)));
    const T = {
      id, tile, img, waitText, offText, alertTag, nameEl, zone,
      mode: "mjpeg", origin: "", errors: 0, everLoaded: false, retry: 0, flashTimer: 0,
      poll: { due: 0, fails: 0, busy: false }, objUrl: null,
    };
    tile.addEventListener("click", open);
    tile.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
    });
    img.addEventListener("load", () => markFrame(T));
    img.addEventListener("error", () => onStreamError(T));
    tiles.set(id, T);
    return T;
  }

  function buildWall() {
    const wall = $("wall");
    wall.textContent = "";
    tiles.clear();
    if (S.site) {
      const people = h("span", { class: "v" }, "–");
      const vehicles = h("span", { class: "v" }, "–");
      const peopleBadge = h("span", { class: "badge badge-people" }, icon("person"), people);
      const vehBadge = h("span", { class: "badge badge-veh" }, icon("truck"), vehicles);
      const T = makeTile(MAP_ID, t("map_title"), [peopleBadge, vehBadge], openMap, "tile tile-map");
      Object.assign(T, { isMap: true, people, vehicles, peopleBadge, vehBadge });
      wall.append(T.tile);
    }
    for (const cam of S.cams) {
      const people = h("span", { class: "v" }, "–");
      const nohat = h("span", { class: "v" }, "–");
      const fps = h("span", { class: "v" }, "–");
      const peopleBadge = h("span", { class: "badge badge-people" }, icon("person"), people);
      const nohatBadge = h("span", { class: "badge badge-nohat" }, icon("nohat"), nohat);
      const fpsBadge = h("span", { class: "badge badge-fps" }, fps);
      const T = makeTile(cam.id, cam.name || cam.id, [peopleBadge, nohatBadge, fpsBadge], () => openCamera(cam.id));
      Object.assign(T, { cam, people, nohat, fps, peopleBadge, nohatBadge, fpsBadge });
      wall.append(T.tile);
    }
    renderWallLabels();
    setupStreams();
  }

  function renderWallLabels() {
    const fpsTarget = S.config ? S.config.fps_target : null;
    setText("wallMeta", S.cams.length ? t("wall_meta", { n: S.cams.length, fps: isNum(fpsTarget) ? fmtNumber(fpsTarget) : "–" }) : "");
    for (const T of tiles.values()) {
      if (T.isMap) {
        T.nameEl.textContent = t("map_title");
        T.zone.textContent = S.site.name || "";
        T.tile.setAttribute("aria-label", t("open_map"));
        T.waitText.textContent = t("connecting");
        T.offText.textContent = t("offline");
        T.peopleBadge.title = t("map_workers");
        T.vehBadge.title = t("vehicles");
        continue;
      }
      T.zone.textContent = zoneName(T.cam.zone);
      T.tile.setAttribute("aria-label", t("open_camera", { name: T.cam.name || T.id }));
      T.waitText.textContent = t("connecting");
      T.offText.textContent = t("offline");
      T.peopleBadge.title = t("people");
      T.nohatBadge.title = t("no_helmet");
    }
  }

  function renderWallState() {
    const cams = (S.state && S.state.cameras) || {};
    for (const T of tiles.values()) {
      if (T.isMap) {
        const site = siteStatus() || {};
        if (T.img.naturalWidth > 0) markFrame(T);
        T.people.textContent = isNum(site.unique_people) ? String(site.unique_people) : "–";
        T.vehicles.textContent = isNum(site.vehicles) ? String(site.vehicles) : "–";
        continue;
      }
      const c = cams[T.id];
      T.tile.classList.toggle("is-offline", !!c && c.online === false);
      if (T.img.naturalWidth > 0) markFrame(T);
      const people = c ? numOr(c.people) : null;
      const nohat = c ? numOr(c.no_helmet) : null;
      const fps = c ? numOr(c.fps) : null;
      T.people.textContent = people == null ? "–" : String(people);
      T.nohat.textContent = nohat == null ? "–" : String(nohat);
      T.nohatBadge.classList.toggle("is-hot", (nohat || 0) > 0);
      T.fps.textContent = fps == null ? "–" : `${fps.toFixed(1)} ${t("fps_unit")}`;
    }
  }

  function markFrame(T) {
    if (!T.everLoaded || T.errors) {
      T.everLoaded = true;
      T.errors = 0;
    }
    T.tile.classList.add("has-frame");
  }

  // A site alert lights the floor plan and every camera that sees the worker, not only the one
  // that took the snapshot.
  function alertTiles(ev) {
    const site = siteInfo(ev);
    return site ? [MAP_ID, ...new Set([ev.cam_id, ...site.cams])] : [ev.cam_id];
  }

  function flashTile(camId, ev) {
    const T = tiles.get(camId);
    if (!T) return;
    T.alertTag.textContent = ev.message || ev.rule_name || t("alert");
    T.tile.classList.remove("flash");
    void T.tile.offsetWidth; // restart the pulse when a second alert lands during the first
    T.tile.classList.add("flash");
    clearTimeout(T.flashTimer);
    T.flashTimer = setTimeout(() => T.tile.classList.remove("flash"), FLASH_MS);
  }

  // ------------------------------------------------------------------ streams
  const streamMode = new URLSearchParams(location.search).get("stream");
  let grabActive = 0;
  let pumpTimer = 0;

  function aliasOrigin() {
    const host = location.hostname;
    const alt = host === "127.0.0.1" ? "localhost" : host === "localhost" ? "127.0.0.1" : null;
    return alt ? `${location.protocol}//${alt}${location.port ? `:${location.port}` : ""}` : null;
  }

  const mjpegOnPage = () => [...tiles.values()].filter((T) => T.mode === "mjpeg" && !T.origin).length;

  function setupStreams() {
    const alias = streamMode ? null : aliasOrigin();
    let onAlias = 0;
    let onPage = 0;
    for (const T of tiles.values()) {
      if (streamMode === "poll") {
        T.mode = "poll";
      } else if (streamMode === "mjpeg") {
        T.mode = "mjpeg";
        T.origin = "";
      } else if (alias && onAlias < MAX_PER_HOST) {
        T.mode = "mjpeg";
        T.origin = alias;
        onAlias += 1;
      } else if (alias && onPage < MAX_PER_HOST - API_RESERVE) {
        T.mode = "mjpeg";
        T.origin = "";
        onPage += 1;
      } else {
        T.mode = "poll";
      }
      if (T.mode === "mjpeg") startMjpeg(T, false);
    }
    if ([...tiles.values()].some((T) => T.mode === "poll")) ensurePump();
  }

  function startMjpeg(T, bust) {
    clearTimeout(T.retry);
    const url = `${T.origin}/stream/${encodeURIComponent(T.id)}${bust ? `?r=${Date.now()}` : ""}`;
    T.img.src = url;
  }

  function onStreamError(T) {
    if (T.mode !== "mjpeg") return;
    T.tile.classList.remove("has-frame");
    T.errors += 1;
    // the second loopback name never delivered a frame while the API answers: it does not reach the server
    if (T.origin && !T.everLoaded && T.errors >= 2 && !S.offline && S.state) {
      T.mode = "poll";
      T.origin = "";
      T.img.removeAttribute("src");
      ensurePump();
      return;
    }
    clearTimeout(T.retry);
    T.retry = setTimeout(() => startMjpeg(T, true), Math.min(15000, 1000 * 2 ** Math.min(T.errors, 4)));
  }

  function restartStreams() {
    for (const T of tiles.values()) {
      if (T.mode === "mjpeg") startMjpeg(T, true);
      else T.poll.due = 0;
    }
  }

  function ensurePump() {
    if (!pumpTimer) pumpTimer = setTimeout(pumpPolls, 0);
  }

  function pumpPolls() {
    pumpTimer = 0;
    const allowed = Math.max(1, Math.min(3, MAX_PER_HOST - API_RESERVE - mjpegOnPage()));
    const now = performance.now();
    const cams = (S.state && S.state.cameras) || {};
    const due = [];
    for (const T of tiles.values()) {
      if (T.mode !== "poll" || T.poll.busy || T.poll.due > now) continue;
      if (cams[T.id] && cams[T.id].online === false) { T.poll.due = now + 5000; continue; }
      due.push(T);
    }
    due.sort((a, b) => a.poll.due - b.poll.due);
    while (grabActive < allowed && due.length) grab(due.shift());
    pumpTimer = setTimeout(pumpPolls, 40);
  }

  async function grab(T) {
    const P = T.poll;
    P.busy = true;
    grabActive += 1;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 4000);
    try {
      const jpeg = await readOneFrame(`/stream/${encodeURIComponent(T.id)}`, ctl.signal);
      showFrame(T, jpeg);
      P.fails = 0;
      // hidden or occluded windows (e.g. one being screen-recorded) keep updating, just slower
      P.due = performance.now() + (document.hidden ? 1000 : 1000 / POLL_FPS);
    } catch {
      P.fails += 1;
      P.due = performance.now() + Math.min(10000, 400 * 2 ** Math.min(P.fails, 5));
    } finally {
      clearTimeout(timer);
      ctl.abort();
      P.busy = false;
      grabActive -= 1;
    }
  }

  async function readOneFrame(url, signal) {
    const res = await fetch(url, { signal, cache: "no-store" });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    let buf = new Uint8Array(0);
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) throw new Error("stream closed");
        if (value && value.length) buf = concatBytes(buf, value);
        const frame = jpegFromPart(buf);
        if (frame) return frame;
        if (buf.length > 8e6) throw new Error("no frame boundary");
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  function concatBytes(a, b) {
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function indexOfBytes(buf, seq, from = 0) {
    outer: for (let i = from; i <= buf.length - seq.length; i++) {
      for (let j = 0; j < seq.length; j++) if (buf[i + j] !== seq[j]) continue outer;
      return i;
    }
    return -1;
  }

  const HEADER_END = [13, 10, 13, 10];
  const JPEG_END = [0xff, 0xd9];
  const latin1 = new TextDecoder("latin1");

  // One multipart part: "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n<jpeg>\r\n"
  function jpegFromPart(buf) {
    const end = indexOfBytes(buf, HEADER_END);
    if (end < 0) return null;
    const start = end + 4;
    const m = /content-length:\s*(\d+)/i.exec(latin1.decode(buf.subarray(0, end)));
    if (m) {
      const len = Number(m[1]);
      return buf.length >= start + len ? buf.slice(start, start + len) : null;
    }
    const eoi = indexOfBytes(buf, JPEG_END, start);
    return eoi >= 0 ? buf.slice(start, eoi + 2) : null;
  }

  function showFrame(T, bytes) {
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
    const old = T.objUrl;
    T.objUrl = url;
    T.img.src = url;
    const release = () => { if (old) URL.revokeObjectURL(old); };
    if (T.img.decode) T.img.decode().then(release, release);
    else setTimeout(release, 1000);
  }

  // ------------------------------------------------------------------ alerts feed
  const cards = new Map();

  function verdict(ev) {
    if (ev.verified === true) return { cls: "vlm-ok", icon: "check", text: t("vlm_confirmed") };
    if (ev.verified === false) return { cls: "vlm-false", icon: "x", text: t("vlm_false") };
    const vs = S.state && S.state.verifier ? S.state.verifier.state : null;
    const pending = !ev.verify_note && ev.rule_type === "no_helmet" && vs && vs !== "disabled" && vs !== "unavailable"
      && isNum(ev.ts) && serverNow() - ev.ts * 1000 < 300000;
    if (pending) return { cls: "vlm-checking", spin: true, text: t("vlm_checking") };
    return { cls: "vlm-none", icon: "minus", text: t("vlm_unchecked") };
  }

  function verdictNote(ev) {
    if (ev.verify_note) return String(ev.verify_note);
    return ev.rule_type && ev.rule_type !== "no_helmet" ? t("vlm_only_helmet") : "";
  }

  function verdictBadge(ev) {
    const v = verdict(ev);
    const note = verdictNote(ev);
    return h("span", { class: `vlm ${v.cls}`, title: note || null },
      v.spin ? h("span", { class: "spin", "aria-hidden": "true" }) : icon(v.icon),
      h("span", { text: v.text }));
  }

  const alertSig = (ev) => [ev.verified, ev.verify_note || "", lang, verdict(ev).cls].join("|");
  const snapUrl = (id) => `/api/events/${encodeURIComponent(id)}/snapshot`;

  function thumb(ev) {
    const box = h("div", { class: "alert-thumb" });
    if (!ev.snapshot) {
      box.append(h("span", { class: "thumb-missing", text: t("snapshot_missing") }));
      return box;
    }
    const img = h("img", { src: snapUrl(ev.id), alt: "", loading: "lazy", decoding: "async" });
    img.addEventListener("error", () => {
      img.remove();
      box.append(h("span", { class: "thumb-missing", text: t("snapshot_missing") }));
    }, { once: true });
    box.append(img);
    return box;
  }

  function alertCard(ev) {
    const node = h("article", { class: "alert", tabindex: "0", role: "button" });
    node.append(thumb(ev), h("div", { class: "alert-body" }));
    node.addEventListener("click", () => openEvent(ev.id));
    node.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openEvent(ev.id); }
    });
    node.addEventListener("animationend", (e) => { if (e.animationName === "glow") node.classList.remove("is-new"); });
    return node;
  }

  function fillAlertCard(node, ev) {
    const sev = sevKey(ev.severity);
    const title = ev.message || ev.rule_name || "—";
    const where = [ev.cam_name || camName(ev.cam_id), zoneName(ev.zone)].filter(Boolean).join(" · ");
    node.className = `alert sev-${sev}${ev.verified === false ? " is-false" : ""}${node.classList.contains("is-new") ? " is-new" : ""}`;
    node.title = [ev.rule_name, ev.extra && ev.extra.detail].filter(Boolean).join(" — ");
    node.setAttribute("aria-label", `${t(`sev_${sev}`)}: ${title}, ${where}, ${fmtTime(ev.ts)}, ${verdict(ev).text}`);
    const site = siteInfo(ev);
    const trail = site && site.cams.length
      ? h("div", { class: "alert-trail", text: t("seen_by", { list: site.cams.map(camTag).join(" · ") }), title: site.cams.map(camLabel).join(", ") })
      : null;
    const dist = site && site.dist != null
      ? h("span", { class: "alert-dist", title: t("dist_tip") }, icon("dist"), fmtDist(site.dist))
      : null;
    const body = node.querySelector(".alert-body");
    body.replaceChildren(...[
      h("div", { class: "alert-top" }, sevChip(sev), h("span", { class: "alert-time", text: fmtTime(ev.ts) })),
      h("div", { class: "alert-title", text: title }),
      h("div", { class: "alert-meta", text: where }),
      trail,
      h("div", { class: dist ? "alert-foot has-dist" : "alert-foot" }, verdictBadge(ev), dist),
    ].filter(Boolean));
  }

  function renderAlerts(freshIds = new Set()) {
    const list = $("alerts");
    if (!S.eventsLoaded) return;
    if (!S.events.length) {
      cards.clear();
      list.replaceChildren(h("div", { class: "alerts-empty" }, icon("shield"), h("span", { text: t("alerts_empty") })));
      return;
    }
    const empty = list.querySelector(".alerts-empty");
    if (empty) empty.remove();
    const keep = new Set();
    const nodes = S.events.map((ev) => {
      keep.add(ev.id);
      let c = cards.get(ev.id);
      if (!c) {
        c = { node: alertCard(ev), sig: "" };
        cards.set(ev.id, c);
      }
      if (freshIds.has(ev.id)) c.node.classList.add("is-new");
      const sig = alertSig(ev);
      if (sig !== c.sig) {
        fillAlertCard(c.node, ev);
        c.sig = sig;
      }
      return c.node;
    });
    for (const id of [...cards.keys()]) if (!keep.has(id)) cards.delete(id);
    nodes.forEach((node, i) => {
      if (list.children[i] !== node) list.insertBefore(node, list.children[i] || null);
    });
    while (list.children.length > nodes.length) list.lastElementChild.remove();
  }

  // "Checking…" depends on the verifier state and on the alert's age, so re-check every poll.
  function updateVerdicts() {
    for (const ev of S.events) {
      const c = cards.get(ev.id);
      if (!c) continue;
      const sig = alertSig(ev);
      if (sig !== c.sig) {
        fillAlertCard(c.node, ev);
        c.sig = sig;
      }
    }
  }

  // ------------------------------------------------------------------ rule wording
  function whatText(r) {
    const n = fmtNumber(r.threshold);
    const obj = objName(r.object);
    let s;
    switch (r.type) {
      case "no_helmet": s = t("type_no_helmet"); break;
      case "person_present": s = t("type_person_present"); break;
      case "person_count_over": s = t("type_person_count_over", { n }); break;
      case "object_present": s = t("type_object_present", { obj }); break;
      case "object_near_person": s = t("type_object_near_person", { n, obj, art: /^[aeiou]/i.test(String(obj || "")) ? "an" : "a" }); break;
      default: s = String(r.type || "?");
    }
    return lang === "en" ? ucfirst(s) : s;
  }

  // Mirrors Rule.covers / covers_any on the server.  In site mode a camera sees several floor-plan
  // zones (cam.zones), so "in aisle 3" covers every camera that sees aisle 3.
  function ruleAppliesTo(r, cam) {
    const cams = Array.isArray(r.cameras) && r.cameras.length ? r.cameras : ["*"];
    if (!cams.includes("*") && !cams.includes(cam.id)) return false;
    const seen = Array.isArray(cam.zones) && cam.zones.length ? cam.zones : [cam.zone];
    if (Array.isArray(r.zones) && r.zones.length && !seen.some((z) => r.zones.includes(z))) return false;
    if (Array.isArray(r.zones_exclude) && seen.every((z) => r.zones_exclude.includes(z))) return false;
    return true;
  }

  function whereText(r) {
    const cams = Array.isArray(r.cameras) && r.cameras.length ? r.cameras : ["*"];
    const zones = Array.isArray(r.zones) ? r.zones : [];
    const excl = Array.isArray(r.zones_exclude) ? r.zones_exclude : [];
    const all = cams.includes("*");
    let s;
    if (all && !zones.length) s = t("all_cameras");
    else if (all) s = zones.map(zoneName).join(sep());
    else {
      s = cams.filter((c) => c !== "*").map(camName).join(sep());
      if (zones.length) s += ` (${zones.map(zoneName).join(sep())})`;
    }
    if (excl.length) s += (lang === "zh" ? "" : " ") + t("except", { z: excl.map(zoneName).join(sep()) });
    return s;
  }

  function coverText(r) {
    const matched = S.cams.filter((c) => ruleAppliesTo(r, c));
    if (!S.cams.length) return { text: "", bad: false };
    if (!matched.length) return { text: t("covers_none"), bad: true };
    if (matched.length <= 4) return { text: t("covers", { list: matched.map((c) => c.name || c.id).join(sep()) }), bad: false };
    return { text: t("covers_n", { n: matched.length, total: S.cams.length }), bad: false };
  }

  function whenText(r) {
    const tw = r.time_window;
    if (!tw || !tw.start || !tw.end) return t("any_time");
    const overnight = tw.end < tw.start;
    return `${tw.start}–${tw.end}${overnight ? ` (${t("overnight")})` : ""}`;
  }

  function ruleSummary(r) {
    const parts = [whatText(r), whereText(r), whenText(r)];
    if (isNum(r.min_duration_s)) parts.push(t("hold_short", { s: fmtNumber(r.min_duration_s) }));
    return parts.join(" · ");
  }

  // ------------------------------------------------------------------ rules list
  function renderRules() {
    const ul = $("rules");
    const enabled = S.rules.filter((r) => r.enabled !== false).length;
    setText("rulesMeta", S.rules.length ? t("rules_meta", { on: enabled, n: S.rules.length }) : "");
    if (!S.rules.length) {
      ul.replaceChildren(h("li", { class: "rules-empty", text: S.rulesLoaded ? t("rules_empty") : "" }));
      return;
    }
    const items = S.rules.slice().sort((a, b) => (numOr(b.created_at, 0)) - (numOr(a.created_at, 0)));
    ul.replaceChildren(...items.map((r) => ruleItem(r)));
  }

  function ruleItem(r) {
    const on = r.enabled !== false;
    const name = r.name || whatText(r);
    const sw = h("button", {
      type: "button", class: "switch", role: "switch", "aria-checked": String(on),
      "aria-label": t("enable_rule", { name }),
    });
    sw.addEventListener("click", () => toggleRule(r, sw));
    const del = h("button", { type: "button", class: "icon-btn danger", "aria-label": t("delete_rule", { name }), title: t("delete") }, icon("trash"));
    del.addEventListener("click", () => deleteRule(r));
    const item = h("li", { class: `rule sev-${sevKey(r.severity)}${on ? "" : " is-off"}${r.id === S.newRuleId ? " is-new" : ""}` },
      h("div", { class: "rule-main" },
        h("div", { class: "rule-name" }, h("span", { text: name, title: name }), sevChip(r.severity)),
        h("div", { class: "rule-sum", text: ruleSummary(r) }),
        r.source_text ? h("div", { class: "rule-src", text: `“${r.source_text}”`, title: r.source_text }) : null),
      h("div", { class: "rule-ctl" }, sw, del));
    item.addEventListener("animationend", (e) => {
      if (e.animationName !== "glow") return;
      item.classList.remove("is-new");
      if (S.newRuleId === r.id) S.newRuleId = null;
    });
    return item;
  }

  async function toggleRule(r, sw) {
    const want = r.enabled === false;
    sw.disabled = true;
    sw.setAttribute("aria-checked", String(want));
    try {
      await api(`/api/rules/${encodeURIComponent(r.id)}`, { method: "PATCH", json: { enabled: want } });
      r.enabled = want;
      toast(want ? t("rule_on") : t("rule_off"));
    } catch (e) {
      sw.setAttribute("aria-checked", String(!want));
      toast(t("err_generic", { detail: e.detail || e.message }), true);
    } finally {
      sw.disabled = false;
      renderKpis();
      loadRules();
    }
  }

  async function deleteRule(r) {
    if (!window.confirm(t("confirm_delete", { name: r.name || r.id }))) return;
    try {
      await api(`/api/rules/${encodeURIComponent(r.id)}`, { method: "DELETE" });
      S.rules = S.rules.filter((x) => x.id !== r.id);
      renderRules();
      renderKpis();
      toast(t("deleted"));
    } catch (e) {
      toast(t("err_generic", { detail: e.detail || e.message }), true);
    } finally {
      loadRules();
    }
  }

  // ------------------------------------------------------------------ say a rule
  function setSayStatus(msg, kind) {
    const box = $("sayStatus");
    box.classList.toggle("is-error", kind === "error");
    box.replaceChildren();
    if (!msg) return;
    if (kind === "busy") box.append(h("span", { class: "spin", "aria-hidden": "true" }));
    box.append(h("span", { text: msg }));
  }

  function setBusy(busy) {
    S.busy = busy;
    $("ruleText").disabled = busy;
    $("parseBtn").disabled = busy;
    document.querySelectorAll("#examples .chip").forEach((c) => { c.disabled = busy; });
    updateMicUi();
  }

  async function submitText(raw) {
    const text = (raw || "").trim();
    if (!text || S.busy) return;
    setBusy(true);
    S.proposal = null;
    renderProposal();
    setSayStatus(t("parsing"), "busy");
    try {
      const res = await api("/api/rules/parse", { method: "POST", json: { text }, timeout: 120000 });
      S.proposal = { text, res: res && typeof res === "object" ? res : {}, saving: false, error: "" };
      setSayStatus(S.heardNote);
    } catch (e) {
      setSayStatus(t("err_parse", { detail: e.detail || e.message }), "error");
    } finally {
      setBusy(false);
      renderProposal();
    }
  }

  function field(label, value) {
    return [h("dt", { text: label }), h("dd", null, value)];
  }

  function renderProposal() {
    const box = $("proposal");
    const P = S.proposal;
    $("examples").hidden = !!P;
    if (!P) {
      box.hidden = true;
      box.replaceChildren();
      return;
    }
    const res = P.res || {};
    const r = res.rule && typeof res.rule === "object" ? res.rule : null;
    const problems = Array.isArray(res.problems) ? res.problems : [];
    const notes = Array.isArray(res.notes) ? res.notes : [];
    const parserKey = res.parser === "llm+check" ? "parser_llm" : res.parser === "rules" ? "parser_rules" : "parser_failed";
    box.className = `proposal${r ? "" : " is-failed"}`;

    const kids = [h("div", { class: "prop-head" },
      h("span", { class: "prop-kicker", text: r ? t("proposal_title") : t("proposal_failed") }),
      h("span", { class: `prop-parser${res.parser === "llm+check" ? " is-llm" : ""}`, title: t(`${parserKey}_tip`), text: t(parserKey) }),
      isNum(res.ms) ? h("span", { class: "prop-ms", text: fmtMs(res.ms) }) : null)];

    if (r) {
      const cover = coverText(r);
      const where = h("div", null, whereText(r),
        cover.text ? h("div", { class: "muted", style: cover.bad ? "color: var(--critical-hi)" : null, text: cover.text }) : null);
      const fields = [
        ...field(t("f_watch"), whatText(r)),
        ...field(t("f_where"), where),
        ...field(t("f_when"), whenText(r)),
      ];
      if (isNum(r.min_duration_s)) fields.push(...field(t("f_hold"), t("hold_s", { s: fmtNumber(r.min_duration_s) })));
      fields.push(...field(t("f_severity"), sevChip(r.severity)));
      if (r.message) fields.push(...field(t("f_message"), `“${r.message}”`));
      if (isNum(r.cooldown_s)) fields.push(...field(t("f_repeat"), t("cooldown_s", { s: fmtNumber(r.cooldown_s) })));
      kids.push(h("div", { class: "prop-name", text: r.name || whatText(r) }), h("dl", { class: "prop-fields" }, fields));
    } else {
      kids.push(h("div", { class: "prop-name", text: `“${P.text}”` }));
    }

    if (problems.length) {
      kids.push(h("div", { class: "prop-problems", role: "alert" },
        h("strong", { text: t("problems") }),
        h("ul", null, problems.map((p) => h("li", { text: String(p) })))));
    }
    if (notes.length) {
      kids.push(h("details", { class: "prop-notes" },
        h("summary", { text: t("notes", { n: notes.length }) }),
        h("ul", null, notes.map((n) => h("li", { text: String(n) })))));
    }
    if (P.error) kids.push(h("div", { class: "prop-error", role: "alert", text: P.error }));

    // a rule whose scope matches no camera could never fire; the server refuses it too
    const canActivate = !!r && res.ok !== false && !problems.length && !coverText(r).bad;
    const actions = h("div", { class: "prop-actions" });
    if (r) {
      const go = h("button", { type: "button", class: "btn btn-go", disabled: !canActivate || P.saving },
        icon("check"), h("span", { text: P.saving ? t("saving") : t("activate") }));
      go.addEventListener("click", activateProposal);
      actions.append(go);
    }
    const cancel = h("button", { type: "button", class: "btn btn-ghost", text: r ? t("cancel") : t("dismiss"), disabled: P.saving });
    cancel.addEventListener("click", () => {
      S.proposal = null;
      renderProposal();
      $("ruleText").focus({ preventScroll: true });
    });
    actions.append(cancel);
    kids.push(actions);

    box.replaceChildren(...kids);
    box.hidden = false;
  }

  async function activateProposal() {
    const P = S.proposal;
    if (!P || !P.res || !P.res.rule || P.saving) return;
    P.saving = true;
    P.error = "";
    renderProposal();
    try {
      const saved = await api("/api/rules", { method: "POST", json: { rule: P.res.rule } });
      S.newRuleId = saved && saved.id ? saved.id : null;
      toast(t("activated", { name: (saved && saved.name) || P.res.rule.name || "" }));
      S.proposal = null;
      S.heardNote = "";
      $("ruleText").value = "";
      setSayStatus("");
      renderProposal();
      await loadRules();
    } catch (e) {
      if (S.proposal === P) {
        P.saving = false;
        P.error = e.detail || e.message;
        renderProposal();
      }
    }
  }

  // ------------------------------------------------------------------ microphone
  const R = {
    phase: "idle", mode: "hold", stream: null, rec: null, chunks: [], mime: "", t0: 0, dur: 0,
    ctx: null, raf: 0, timer: 0, level: 0, pressAt: 0, stopOnUp: false, stopWhenReady: false,
  };

  const micSupported = () => !!(window.isSecureContext !== false && navigator.mediaDevices
    && navigator.mediaDevices.getUserMedia && window.MediaRecorder);

  function pickMime() {
    const options = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return "";
    return options.find((m) => MediaRecorder.isTypeSupported(m)) || "";
  }

  function extFor(mime) {
    if (/ogg/.test(mime)) return "ogg";
    if (/mp4|aac|m4a/.test(mime)) return "m4a";
    if (/wav/.test(mime)) return "wav";
    return "webm";
  }

  function updateMicUi() {
    const btn = $("micBtn");
    const rec = R.phase === "starting" || R.phase === "recording";
    const working = S.busy || R.phase === "stopping";
    btn.classList.toggle("is-rec", rec);
    btn.classList.toggle("is-busy", working && !rec);
    btn.setAttribute("aria-pressed", String(rec));
    btn.disabled = working && !rec;
    setText("micState", rec ? t("mic_rec") : working ? t("mic_wait") : t("mic_idle"));
    setText("micSub", rec ? (R.mode === "toggle" ? t("mic_rec_sub_toggle") : t("mic_rec_sub_hold"))
      : working ? t("mic_wait_sub") : t("mic_idle_sub"));
    const timer = $("micTimer");
    timer.classList.toggle("is-rec", rec);
    if (!rec) timer.textContent = "0:00";
  }

  function tickTimer() {
    const ms = performance.now() - R.t0;
    const s = Math.floor(ms / 1000);
    $("micTimer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    if (ms >= MAX_REC_MS) stopRec();
  }

  function startLevel(stream) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      const ctx = new AC();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      R.ctx = ctx;
      const btn = $("micBtn");
      const loop = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const x = (buf[i] - 128) / 128;
          sum += x * x;
        }
        const level = Math.min(1, Math.sqrt(sum / buf.length) * 5);
        R.level = 0.6 * R.level + 0.4 * level;
        btn.style.setProperty("--level", R.level.toFixed(3));
        R.raf = requestAnimationFrame(loop);
      };
      loop();
    } catch { /* level meter is cosmetic */ }
  }

  async function startRec(mode) {
    if (R.phase !== "idle" || S.busy) return;
    if (!micSupported()) {
      setSayStatus(t("mic_unsupported"), "error");
      return;
    }
    R.phase = "starting";
    R.mode = mode;
    R.stopWhenReady = false;
    updateMicUi();
    setSayStatus("");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      R.phase = "idle";
      updateMicUi();
      const denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
      setSayStatus(denied ? t("mic_denied") : t("mic_error", { detail: (e && (e.message || e.name)) || e }), "error");
      return;
    }
    if (R.phase !== "starting") {
      stream.getTracks().forEach((tr) => tr.stop());
      return;
    }
    const mime = pickMime();
    let rec;
    try {
      rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (e) {
      stream.getTracks().forEach((tr) => tr.stop());
      R.phase = "idle";
      updateMicUi();
      setSayStatus(t("mic_error", { detail: (e && e.message) || e }), "error");
      return;
    }
    R.stream = stream;
    R.rec = rec;
    R.chunks = [];
    R.mime = rec.mimeType || mime || "audio/webm";
    rec.addEventListener("dataavailable", (e) => { if (e.data && e.data.size) R.chunks.push(e.data); });
    rec.addEventListener("stop", onRecStop, { once: true });
    rec.start(250);
    R.t0 = performance.now();
    R.phase = "recording";
    startLevel(stream);
    clearInterval(R.timer);
    R.timer = setInterval(tickTimer, 200);
    updateMicUi();
    if (R.stopWhenReady) stopRec();
  }

  function stopRec() {
    if (R.phase === "starting") { R.stopWhenReady = true; return; }
    if (R.phase !== "recording") return;
    R.phase = "stopping";
    R.dur = performance.now() - R.t0;
    updateMicUi();
    try {
      R.rec.stop();
    } catch {
      onRecStop();
    }
  }

  function releaseMic() {
    clearInterval(R.timer);
    cancelAnimationFrame(R.raf);
    if (R.stream) R.stream.getTracks().forEach((tr) => tr.stop());
    if (R.ctx) R.ctx.close().catch(() => {});
    R.stream = null;
    R.rec = null;
    R.ctx = null;
    R.level = 0;
    $("micBtn").style.setProperty("--level", "0");
  }

  async function onRecStop() {
    if (R.phase !== "stopping") return;
    const chunks = R.chunks;
    const mime = R.mime;
    const dur = R.dur;
    R.chunks = [];
    releaseMic();
    R.phase = "idle";
    updateMicUi();
    if (dur < MIN_REC_MS || !chunks.length) {
      setSayStatus(t("mic_short"), "error");
      return;
    }
    await transcribe(new Blob(chunks, { type: mime }), mime);
  }

  async function transcribe(blob, mime) {
    setBusy(true);
    S.proposal = null;
    renderProposal();
    setSayStatus(t("transcribing"), "busy");
    const form = new FormData();
    form.append("audio", blob, `speech.${extFor(mime)}`);
    form.append("lang", lang); // decode in the UI language instead of the server default
    let text = "";
    try {
      const res = await api("/api/asr", { method: "POST", form, timeout: 180000 });
      text = String((res && res.text) || "").trim();
      if (!text) {
        setSayStatus(t("asr_empty"), "error");
        return;
      }
      $("ruleText").value = text;
      S.heardNote = t("heard", { ms: fmtMs(res.ms), device: res.device || "?" });
    } catch (e) {
      setSayStatus(e.status === 503 ? t("asr_busy", { detail: e.detail }) : t("err_asr", { detail: e.detail || e.message }), "error");
      return;
    } finally {
      setBusy(false);
    }
    await submitText(text);
  }

  function bindMic() {
    const btn = $("micBtn");
    btn.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || btn.disabled) return;
      e.preventDefault();
      try { btn.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
      R.pressAt = performance.now();
      if (R.phase === "recording" || R.phase === "starting") {
        R.stopOnUp = true;
        return;
      }
      R.stopOnUp = false;
      startRec("hold");
    });
    btn.addEventListener("pointerup", () => {
      if (R.stopOnUp) {
        R.stopOnUp = false;
        stopRec();
        return;
      }
      const held = performance.now() - R.pressAt;
      if (R.phase !== "recording" && R.phase !== "starting") return;
      if (held >= HOLD_MS) stopRec();
      else {
        R.mode = "toggle"; // a short click keeps recording until the next click
        updateMicUi();
      }
    });
    btn.addEventListener("pointercancel", () => {
      if (R.mode === "hold") stopRec();
    });
    btn.addEventListener("keydown", (e) => {
      if ((e.key === " " || e.key === "Enter") && !e.repeat) {
        e.preventDefault();
        if (R.phase === "recording" || R.phase === "starting") stopRec();
        else startRec("toggle");
      }
    });
    btn.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // ------------------------------------------------------------------ modal
  const modal = $("modal");
  let modalToken = 0;
  let modalOnClose = null;
  let modalReturn = null;

  function openModal({ title, body, actions = [], narrow = false, onClose = null }) {
    closeModal();
    modalToken += 1;
    $("modalTitle").textContent = title || "";
    $("modalActions").replaceChildren(...actions);
    $("modalBody").replaceChildren(body);
    modal.querySelector(".modal-dialog").classList.toggle("is-narrow", narrow);
    modalReturn = document.activeElement;
    modalOnClose = onClose;
    modal.hidden = false;
    $("modalClose").focus({ preventScroll: true });
    return modalToken;
  }

  const modalIsOpen = (token) => !modal.hidden && token === modalToken;

  function closeModal() {
    if (modal.hidden) return;
    modal.hidden = true;
    const fn = modalOnClose;
    modalOnClose = null;
    if (fn) { try { fn(); } catch { /* ignore */ } }
    $("modalBody").replaceChildren();
    $("modalActions").replaceChildren();
    if (modalReturn && document.contains(modalReturn) && modalReturn.focus) modalReturn.focus({ preventScroll: true });
    modalReturn = null;
  }

  function trapFocus(e) {
    const focusables = [...modal.querySelectorAll("button, [href], input, summary, [tabindex]:not([tabindex='-1'])")]
      .filter((el) => !el.disabled && el.offsetParent !== null);
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  function detailList(rows) {
    return h("dl", { class: "detail" }, rows.filter(Boolean).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", null, v)]));
  }

  // ------------------------------------------------------------------ camera view
  function viewStat(label) {
    const b = h("b", null, "–");
    const box = h("div", { class: "view-stat" }, b, h("span", { text: label }));
    return { box, b };
  }

  // Copy frames from the tile instead of opening another stream (connection budget); returns stop().
  function mirrorTile(T, canvas) {
    const ctx = canvas.getContext("2d");
    let raf = 0;
    let last = 0;
    const draw = (ts) => {
      raf = requestAnimationFrame(draw);
      if (ts - last < 50) return;
      last = ts;
      const img = T.img;
      if (!img.naturalWidth) return;
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      try { ctx.drawImage(img, 0, 0); } catch { /* frame not decodable yet */ }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }

  function openCamera(id) {
    const T = tiles.get(id);
    const cam = S.camMap.get(id);
    if (!T || !cam) return;
    const canvas = h("canvas", { width: "640", height: "360", role: "img", "aria-label": cam.name || id });
    const sPeople = viewStat(t("people"));
    const sNohat = viewStat(t("no_helmet"));
    const sFps = viewStat(t("fps_unit").toUpperCase());
    const rulesUl = h("ul", { class: "view-list" });
    const alertsUl = h("ul", { class: "view-list" });
    const side = h("div", { class: "view-side" },
      h("div", { class: "view-stats" }, sPeople.box, sNohat.box, sFps.box),
      h("div", null, h("div", { class: "view-h", text: t("rules_here") }), rulesUl),
      h("div", null, h("div", { class: "view-h", text: t("recent_here") }), alertsUl));
    const body = h("div", { class: "view" }, h("div", { class: "view-media" }, canvas), side);
    const stop = mirrorTile(T, canvas);

    const update = () => {
      const c = (S.state && S.state.cameras && S.state.cameras[id]) || {};
      sPeople.b.textContent = isNum(c.people) ? String(c.people) : "–";
      sNohat.b.textContent = isNum(c.no_helmet) ? String(c.no_helmet) : "–";
      sNohat.box.classList.toggle("is-hot", (c.no_helmet || 0) > 0);
      sFps.b.textContent = isNum(c.fps) ? c.fps.toFixed(1) : "–";
      const rules = S.rules.filter((r) => r.enabled !== false && ruleAppliesTo(r, cam));
      rulesUl.replaceChildren(...(rules.length
        ? rules.map((r) => h("li", null, h("div", { text: r.name || whatText(r) }), h("div", { class: "muted", text: ruleSummary(r) })))
        : [h("li", { class: "empty", text: t("none") })]));
      const seenHere = (e) => {
        const site = siteInfo(e);
        return e.cam_id === id || (!!site && site.cams.includes(id));
      };
      alertsUl.replaceChildren(...recentItems(S.events.filter(seenHere).slice(0, 4)));
    };
    update();
    openModal({
      title: `${id.toUpperCase()} · ${cam.name || ""} — ${zoneName(cam.zone)}`,
      body,
      onClose: () => {
        stop();
        S.camModal = null;
      },
    });
    S.camModal = { id, update };
  }

  function recentItems(evs) {
    return evs.length
      ? evs.map((e) => h("li", null,
        h("div", { text: e.message || e.rule_name || "—" }),
        h("div", { class: "muted", text: `${fmtTime(e.ts)} · ${verdict(e).text}` })))
      : [h("li", { class: "empty", text: t("none") })];
  }

  // ------------------------------------------------------------------ floor plan view
  function openMap() {
    const T = tiles.get(MAP_ID);
    if (!T || !S.site) return;
    const canvas = h("canvas", { width: "640", height: "360", role: "img", "aria-label": t("map_title") });
    const sUnique = viewStat(t("map_unique"));
    const sSum = viewStat(t("map_sum"));
    const sVeh = viewStat(t("vehicles"));
    const zonesUl = h("ul", { class: "view-list" });
    const alertsUl = h("ul", { class: "view-list" });
    const side = h("div", { class: "view-side" },
      h("div", { class: "view-stats" }, sUnique.box, sSum.box, sVeh.box),
      h("div", null, h("div", { class: "view-h", text: t("map_zones") }), zonesUl),
      h("div", null, h("div", { class: "view-h", text: t("map_recent") }), alertsUl));
    const body = h("div", { class: "view" }, h("div", { class: "view-media view-plan" }, canvas), side);
    const stop = mirrorTile(T, canvas);

    const update = () => {
      const site = siteStatus() || {};
      const num = (v) => (isNum(v) ? String(v) : "–");
      sUnique.b.textContent = num(site.unique_people);
      sSum.b.textContent = num(site.camera_sum);
      sVeh.b.textContent = num(site.vehicles);
      const counts = site.zones && typeof site.zones === "object" ? site.zones : {};
      const ids = (Array.isArray(S.site.zones) ? S.site.zones : []).map((z) => z && z.id).filter(Boolean);
      for (const z of Object.keys(counts)) if (!ids.includes(z)) ids.push(z);
      zonesUl.replaceChildren(...(ids.length
        ? ids.map((z) => {
          const n = numOr(counts[z], 0);
          return h("li", { class: `zone-count${n > 0 ? " is-busy" : ""}` }, h("span", { text: zoneName(z) }), h("b", { text: String(n) }));
        })
        : [h("li", { class: "empty", text: t("none") })]));
      alertsUl.replaceChildren(...recentItems(S.events.slice(0, 4)));
    };
    update();
    openModal({
      title: S.site.name ? `${t("map_title")} — ${S.site.name}` : t("map_title"),
      body,
      onClose: () => {
        stop();
        S.camModal = null;
      },
    });
    S.camModal = { id: MAP_ID, update };
  }

  // ------------------------------------------------------------------ alert detail
  function openEvent(id) {
    const ev = S.events.find((e) => e.id === id);
    if (!ev) return;
    const media = h("div", { class: "view-media" });
    if (ev.snapshot) {
      const img = h("img", { src: snapUrl(ev.id), alt: ev.message || ev.rule_name || "" });
      img.addEventListener("error", () => img.replaceWith(h("div", { class: "alerts-empty", style: "position:absolute;inset:0;border:0", text: t("snapshot_missing") })), { once: true });
      media.append(img);
    } else {
      media.append(h("div", { class: "alerts-empty", style: "position:absolute;inset:0;border:0", text: t("snapshot_missing") }));
    }
    const extra = ev.extra && typeof ev.extra === "object" ? ev.extra : {};
    const note = verdictNote(ev);
    const site = siteInfo(ev);
    const rows = [
      [t("ev_time"), fmtDateTime(ev.ts)],
      [t("ev_camera"), `${ev.cam_name || camName(ev.cam_id)} (${String(ev.cam_id || "").toUpperCase()})`],
      site && site.cams.length ? [t("ev_seen_by"), h("div", null, site.cams.map((c) => h("div", { text: camLabel(c) })))] : null,
      [t("ev_zone"), zoneName(ev.zone) || "–"],
      [t("ev_rule"), ev.rule_name || "–"],
      [t("ev_severity"), sevChip(ev.severity)],
      site && site.dist != null ? [t("ev_distance"), fmtDist(site.dist)] : null,
      site && site.world ? [t("ev_world"), `x ${site.world[0].toFixed(1)} m · y ${site.world[1].toFixed(1)} m`] : null,
      extra.detail ? [t("ev_detail"), detailText(extra.detail)] : null,
      extra.source_text ? [t("ev_spoken"), `“${extra.source_text}”`] : null,
      [t("ev_vlm"), h("div", null, verdictBadge(ev), note ? h("div", { class: "muted", style: "margin-top:6px;color:var(--text-3);font-size:12.5px", text: note }) : null)],
      ev.track_id != null && !site ? [t("ev_track"), String(ev.track_id)] : null,   // site worker ids are not shown
    ];
    const body = h("div", { class: "view" }, media, h("div", { class: "view-side" }, detailList(rows)));
    openModal({ title: ev.message || ev.rule_name || t("alert"), body });
  }

  // ------------------------------------------------------------------ report
  function openReport() {
    const body = h("div");
    const copyBtn = h("button", { type: "button", class: "btn btn-ghost", disabled: true }, icon("copy"), h("span", { text: t("copy") }));
    const regenBtn = h("button", { type: "button", class: "btn btn-ghost", disabled: true }, icon("refresh"), h("span", { text: t("regenerate") }));
    let markdown = "";
    let token = 0;
    const run = async () => {
      copyBtn.disabled = true;
      regenBtn.disabled = true;
      body.replaceChildren(h("div", { class: "report-loading" }, h("span", { class: "spin", "aria-hidden": "true" }), h("span", { text: t("report_loading") })));
      try {
        const res = await api("/api/report", { method: "POST", json: { lang, hours: 24 }, timeout: 240000 });
        if (!modalIsOpen(token)) return;
        markdown = String((res && res.markdown) || "");
        const content = h("div", { class: "md" });
        content.innerHTML = renderMarkdown(markdown); // renderMarkdown escapes all input first
        body.replaceChildren(
          h("div", { class: "report-meta", text: t("report_meta", { gen: (res && res.generator) || "?", s: fmtMs(res && res.ms) }) }),
          content);
        copyBtn.disabled = !markdown;
      } catch (e) {
        if (!modalIsOpen(token)) return;
        body.replaceChildren(h("div", { class: "prop-error", role: "alert", text: t("err_report", { detail: e.detail || e.message }) }));
      } finally {
        regenBtn.disabled = false;
      }
    };
    copyBtn.addEventListener("click", async () => {
      toast((await copyText(markdown)) ? t("copied") : t("copy_failed"), false);
    });
    regenBtn.addEventListener("click", run);
    token = openModal({ title: t("report_title"), body, actions: [copyBtn, regenBtn], narrow: true });
    run();
  }

  async function copyText(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch { /* fall through */ }
    const ta = h("textarea", { style: "position:fixed;left:-9999px;top:0", "aria-hidden": "true" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
    return ok;
  }

  // ------------------------------------------------------------------ markdown
  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function mdInline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
  }

  const splitRow = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  const isTableSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
  const RE_UL = /^\s*[-*+]\s+(.*)$/;
  const RE_OL = /^\s*\d+[.)]\s+(.*)$/;
  const RE_H = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RE_HR = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
  const RE_QUOTE = /^\s*&gt;\s?(.*)$/;

  function isBlockStart(line, next) {
    return RE_H.test(line) || RE_HR.test(line) || RE_UL.test(line) || RE_OL.test(line) || RE_QUOTE.test(line)
      || (line.includes("|") && next != null && isTableSep(next));
  }

  // Headings, bold/italic/code, bullet and numbered lists, tables, quotes, rules, paragraphs.
  function renderMarkdown(md) {
    const lines = escapeHtml(md || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m;
      if ((m = RE_H.exec(line))) {
        const level = m[1].length;
        out.push(`<h${level}>${mdInline(m[2])}</h${level}>`);
        i++;
        continue;
      }
      if (RE_HR.test(line)) { out.push("<hr>"); i++; continue; }
      if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = splitRow(line);
        const align = splitRow(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? "center" : /-+:$/.test(c) ? "right" : ""));
        const cell = (tag, c, k) => `<${tag}${align[k] ? ` style="text-align:${align[k]}"` : ""}>${mdInline(c)}</${tag}>`;
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          const cells = splitRow(lines[i]);
          rows.push(`<tr>${head.map((_, k) => cell("td", cells[k] || "", k)).join("")}</tr>`);
          i++;
        }
        out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c, k) => cell("th", c, k)).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`);
        continue;
      }
      if (RE_UL.test(line) || RE_OL.test(line)) {
        const ordered = !RE_UL.test(line);
        const re = ordered ? RE_OL : RE_UL;
        const items = [];
        while (i < lines.length && (m = re.exec(lines[i]))) {
          let text = m[1];
          i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !RE_UL.test(lines[i]) && !RE_OL.test(lines[i])) {
            text += ` ${lines[i].trim()}`;
            i++;
          }
          items.push(`<li>${mdInline(text)}</li>`);
        }
        const tag = ordered ? "ol" : "ul";
        out.push(`<${tag}>${items.join("")}</${tag}>`);
        continue;
      }
      if (RE_QUOTE.test(line)) {
        const parts = [];
        while (i < lines.length && (m = RE_QUOTE.exec(lines[i]))) { parts.push(m[1]); i++; }
        out.push(`<blockquote>${mdInline(parts.join(" "))}</blockquote>`);
        continue;
      }
      const para = [line.trim()];
      i++;
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i], lines[i + 1])) {
        para.push(lines[i].trim());
        i++;
      }
      out.push(`<p>${mdInline(para.join(" "))}</p>`);
    }
    return out.join("\n");
  }

  // ------------------------------------------------------------------ system strip
  const sysItems = {};

  function buildSys() {
    const dl = $("sys");
    dl.replaceChildren();
    for (const key of ["detector", "ov", "vlm", "asr"]) {
      const dt = h("dt");
      const dot = h("span", { class: "state-dot", "aria-hidden": "true" });
      const main = h("span");
      const sub = h("span", { class: "muted" });
      const dd = h("dd", null, dot, main, sub);
      dl.append(h("div", { class: "sys-item" }, dt, dd));
      sysItems[key] = { dt, dd, dot, main, sub };
    }
  }

  function setSys(key, label, state, main, sub, title) {
    const it = sysItems[key];
    if (!it) return;
    if (it.dt.textContent !== label) it.dt.textContent = label;
    it.dot.className = `state-dot${state ? ` ${state}` : ""}`;
    if (it.main.textContent !== main) it.main.textContent = main;
    const subText = sub ? ` · ${sub}` : "";
    if (it.sub.textContent !== subText) it.sub.textContent = subText;
    if (title) it.dd.title = title;
    else it.dd.removeAttribute("title");
  }

  const stateName = (s) => (s ? t(`st_${s}`) : "–");
  const stateDot = (s) => (s === "ready" ? "ok" : s === "loading" ? "busy" : s === "unavailable" || s === "error" ? "bad" : "");

  function renderSys() {
    const st = S.state || {};
    const models = (S.config && S.config.models) || {};
    const stats = st.stats || {};

    const running = isNum(stats.frames_per_s) && stats.frames_per_s > 0;
    const det = st.detector || models.detector || "–";
    const batch = isNum(stats.det_batch_ms) && stats.det_batch_ms > 0
      ? t("batch_info", { ms: fmt1(stats.det_batch_ms), cap: fmtInt(stats.det_capacity_fps) }) : "";
    setSys("detector", t("sys_detector"), running ? "ok" : "", det, batch, det);

    const ov = st.open_vocab || {};
    const classes = Array.isArray(ov.classes) ? ov.classes : [];
    if (ov.error) setSys("ov", t("sys_ov"), "bad", t("ov_error"), classes.join(", "), String(ov.error));
    else if (!classes.length) setSys("ov", t("sys_ov"), "", t("ov_none"), "", models.open_vocab || "");
    else if (ov.loaded) setSys("ov", t("sys_ov"), "ok", classes.join(", "), isNum(ov.ms) && ov.ms > 0 ? `${Math.round(ov.ms)} ms` : "", models.open_vocab || "");
    else setSys("ov", t("sys_ov"), "busy", classes.join(", "), t("ov_loading"), models.open_vocab || "");

    const v = st.verifier || {};
    const vSub = [];
    if (isNum(v.avg_ms) && v.avg_ms > 0) vSub.push(fmtMs(v.avg_ms));
    if (v.device) vSub.push(String(v.device));
    if (isNum(v.checked) && v.checked > 0) vSub.push(t("checked_n", { n: v.checked }));
    if (isNum(v.queue) && v.queue > 0) vSub.push(t("queue_n", { n: v.queue }));
    if (v.backend) vSub.unshift(String(v.backend));
    setSys("vlm", t("sys_vlm"), stateDot(v.state), stateName(v.state), vSub.join(" · "), v.reason || models.verifier || "");

    const a = st.asr || {};
    const aSub = [];
    if (a.device) aSub.push(String(a.device));
    if (isNum(a.avg_ms) && a.avg_ms > 0) aSub.push(fmtMs(a.avg_ms));
    const aMain = a.state === "idle" || !a.state ? t("asr_idle") : stateName(a.state);
    setSys("asr", t("sys_asr"), stateDot(a.state), aMain, aSub.join(" · "), a.reason || models.asr || "");
  }

  // ------------------------------------------------------------------ toast, clock
  let toastTimer = 0;

  function toast(msg, isError = false) {
    const node = $("toast");
    node.textContent = msg;
    node.classList.toggle("is-error", !!isError);
    node.classList.add("is-on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("is-on"), isError ? 5000 : 2600);
  }

  function renderClock() {
    const d = new Date();
    setText("clockTime", d.toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }));
    setText("clockDate", d.toLocaleDateString(locale(), { weekday: "short", month: "short", day: "numeric" }));
  }

  function tickClock() {
    renderClock();
    setTimeout(tickClock, 1000 - (Date.now() % 1000) + 10);
  }

  // ------------------------------------------------------------------ language
  function applyStatic() {
    document.documentElement.lang = lang === "zh" ? "zh-Hant-TW" : "en";
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = el.dataset.i18n === "tagline" ? t("tagline", { n: S.cams.length || 10 }) : t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    document.querySelectorAll("[data-i18n-aria]").forEach((el) => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); });
    document.querySelectorAll(".seg-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  }

  function setLang(next) {
    lang = next === "zh" ? "zh" : "en";
    store.set("sayguard.lang", lang);
    applyStatic();
    updateMicUi();
    renderWallLabels();
    renderKpis();
    renderWallState();
    renderSys();
    renderRules();
    renderAlerts();
    renderProposal();
    if (S.heardNote && !S.busy) setSayStatus("");
    S.heardNote = "";
    renderClock();
  }

  // ------------------------------------------------------------------ init
  async function loadConfig() {
    for (let attempt = 0; ; attempt++) {
      try {
        const c = await api("/api/config", { timeout: 6000 });
        if (c && typeof c === "object") {
          S.config = c;
          if (c.config_id) S.configId = c.config_id;
          S.site = c.site && typeof c.site === "object" ? c.site : null;
          S.cams = Array.isArray(c.cameras) ? c.cameras.filter((x) => x && x.id) : [];
          S.camMap = new Map(S.cams.map((x) => [x.id, x]));
          setText("siteName", c.site_name || (S.site && S.site.name) || "—");
          document.querySelectorAll('[data-i18n="tagline"]').forEach((el) => {
            el.textContent = t("tagline", { n: S.cams.length || 10 });
          });
          // the configuration's example rules (site mode speaks of floor zones and forklifts, not hard hats)
          if (Array.isArray(c.examples) && c.examples.length) {
            $("examples").querySelector(".chips").replaceChildren(
              ...c.examples.map((x) => h("button", { type: "button", class: "chip", text: String(x) })));
          }
          $("kSiteBox").hidden = !S.site;
          $("kpis").classList.toggle("has-site", !!S.site);
          setOffline(false);
          return;
        }
      } catch {
        if (attempt >= 1) setOffline(true);
      }
      await sleep(Math.min(5000, 1000 * (attempt + 1)));
    }
  }

  function bindUi() {
    document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
    $("ruleForm").addEventListener("submit", (e) => {
      e.preventDefault();
      submitText($("ruleText").value);
    });
    $("ruleText").addEventListener("input", () => { S.heardNote = ""; });
    $("examples").addEventListener("click", (e) => {        // delegated: the chips can come from the config
      const chip = e.target instanceof Element ? e.target.closest(".chip") : null;
      if (!chip || chip.disabled) return;
      $("ruleText").value = chip.textContent.trim();
      S.heardNote = "";
      submitText(chip.textContent);
    });
    $("reportBtn").addEventListener("click", openReport);
    modal.addEventListener("click", (e) => {
      if (e.target instanceof Element && e.target.closest("[data-close]")) closeModal();
    });
    document.addEventListener("keydown", (e) => {
      if (modal.hidden) return;
      if (e.key === "Escape") { e.preventDefault(); closeModal(); }
      else if (e.key === "Tab") trapFocus(e);
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && !pollInFlight) schedulePoll(0);
    });
    window.addEventListener("pagehide", () => {
      releaseMic();
      for (const T of tiles.values()) if (T.objUrl) URL.revokeObjectURL(T.objUrl);
    });
    bindMic();
  }

  async function init() {
    applyStatic();
    bindUi();
    buildSys();
    updateMicUi();
    tickClock();
    renderAlerts();
    renderRules();
    await loadConfig();
    buildWall();
    renderKpis();
    renderSys();
    pollState();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
