"""HTTP API + dashboard.  All endpoints are local (127.0.0.1) and JSON unless noted.

GET  /                          dashboard (web/index.html)
GET  /stream/{cam_id}           MJPEG tile stream (multipart/x-mixed-replace)
GET  /stream/map                site mode: live floor plan (workers by global id, zones, vehicles), same
                                MJPEG format; 404 without a site
GET  /api/config                site, cameras, zones, model names; "site" {name, zones, cameras} or null
GET  /api/state                 live stats, per-camera status, Hailo load estimate, verifier/ASR status;
                                site mode adds "site" {unique_people, camera_sum, zones, vehicles}
GET  /api/events?limit=&since=  recent alerts (newest first), 1 <= limit <= 500
GET  /api/events/{id}/snapshot  anonymised JPEG
GET  /api/rules                 active rules
POST /api/rules/parse  {text}   natural language -> rule draft (not activated)
POST /api/rules        {rule}   activate a (confirmed) rule; 422 if incomplete or no camera is in scope
PATCH /api/rules/{id}  {enabled}
DELETE /api/rules/{id}
POST /api/asr  multipart 'audio' [+ 'lang' zh|en|auto]  speech -> text
POST /api/report {lang, hours}  daily safety report (markdown)
GET  /api/health                memory headroom
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import resources
from .config import ROOT, resolve
from .pipeline import Pipeline
from .rules import Rule
from .rules import parser as rule_parser

WEB = ROOT / "web"


class ParseIn(BaseModel):
    text: str


class RuleIn(BaseModel):
    rule: dict


class EnableIn(BaseModel):
    enabled: bool


class ReportIn(BaseModel):
    lang: str = "zh"
    hours: float = 24.0


def site_vocabulary(cfg: dict) -> dict[str, str]:
    """Words a safety officer will say, from the configured zones/cameras plus fixed PPE terms,
    used as Whisper's decoder prompt (configurable with asr.vocabulary)."""
    custom = cfg.get("asr", {}).get("vocabulary")
    if custom:
        return custom
    zones = cfg.get("zones", {}).values()
    zh = "、".join([z.get("zh", "") for z in zones if z.get("zh")] +
                  ["安全帽", "堆高機", "挖土機", "梯子", "公尺", "秒", "超過", "警告"])
    en = ", ".join([z.get("en", "") for z in zones if z.get("en")] +
                   [c["name"] for c in cfg.get("cameras", [])] + ["hard hat", "forklift", "excavator", "metres"])
    return {"zh": zh + "。", "en": en + "."}


def mjpeg(frame: Callable[[], tuple[int, bytes | None]]) -> StreamingResponse:
    """Stream frame() -> (seq, jpeg) as MJPEG, a part only when seq moves on.  Polled instead of
    waiting on pipe.cond: a threading.Condition cannot be awaited without a thread per viewer."""
    async def gen():
        seq = -1
        while True:
            s, data = frame()
            if data is not None and s != seq:
                seq = s
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode()
                       + b"\r\n\r\n" + data + b"\r\n")
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


def create_app(pipe: Pipeline) -> FastAPI:
    app = FastAPI(title="SayGuard", docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
    cfg = pipe.cfg
    # site mode: "zones" = the floor-plan zones a camera sees (set by SitePipeline), without which the
    # parser's scope check and the dashboard's coverage preview reject "in aisle 3"
    cams = [{"id": c["id"], "name": c["name"], "zone": c.get("zone", ""),
             **({"zones": list(c["zones"])} if c.get("zones") else {})} for c in cfg["cameras"]]
    zones = cfg.get("zones", {})
    asr_holder: dict = {}

    def site_config() -> dict | None:
        """What the dashboard needs to label the floor plan; None outside site mode.  Zones are the
        loaded Site's: the top-level `zones:` entries that carry a polygon or plan_polygon."""
        site = getattr(pipe, "site", None)
        if site is None:
            return None
        return {"name": (cfg.get("site") or {}).get("name") or cfg.get("site_name", ""),
                "zones": [{"id": z.id, "name": z.name} for z in site.zones],
                "cameras": [c["id"] for c in cams]}

    def asr():
        if "asr" not in asr_holder:
            from .asr import SpeechToText
            a = cfg.get("asr", {})
            asr_holder["asr"] = SpeechToText(str(resolve(a.get("model", ""))), a.get("language", "zh"),
                                             vocabulary=site_vocabulary(cfg))
        return asr_holder["asr"]

    @app.on_event("startup")
    async def _start() -> None:
        pipe.start()

    @app.on_event("shutdown")
    async def _stop() -> None:
        pipe.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))

    # registered before /stream/{cam_id}, which would otherwise take "map" for a camera id
    @app.get("/stream/map")
    async def stream_map():
        if getattr(pipe, "site", None) is None:
            raise HTTPException(404, "not in site mode")
        return mjpeg(lambda: (pipe.map_seq, pipe.map_jpeg))

    @app.get("/stream/{cam_id}")
    async def stream(cam_id: str):
        st = pipe.states.get(cam_id)
        if st is None:
            raise HTTPException(404, "unknown camera")
        return mjpeg(lambda: (st.jpeg_seq, st.jpeg))

    config_id = hashlib.sha1(json.dumps(
        {"site_name": cfg.get("site_name", ""), "site": site_config(), "cameras": cams, "zones": zones,
         "examples": cfg.get("examples") or []}, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    @app.get("/api/config")
    async def config():
        return {"site_name": cfg.get("site_name", ""), "site": site_config(), "cameras": cams, "zones": zones,
                "models": {"detector": pipe.detector.name, "llm": cfg["llm"]["model"],
                           "asr": cfg.get("asr", {}).get("model"), "verifier": pipe.verifier.label,
                           "open_vocab": cfg.get("open_vocab", {}).get("weights")},
                "hailo_reference": cfg.get("hailo_reference", {}), "fps_target": pipe.fps_target,
                "privacy": {"blur_faces": pipe.blur},
                "examples": [str(x) for x in (cfg.get("examples") or [])][:6], "config_id": config_id}

    boot_id = uuid.uuid4().hex[:12]

    @app.get("/api/state")
    async def state():
        s = await run_in_threadpool(pipe.status)
        s["asr"] = asr_holder["asr"].status() if "asr" in asr_holder else {"state": "idle"}
        s["llm_online"] = await run_in_threadpool(pipe.llm.available)
        s["time"] = time.time()
        s["boot"] = boot_id          # a new id: the server restarted (rule and event versions start over)
        s["config_id"] = config_id    # a new id: another site configuration (cameras, zones, examples)
        return s

    @app.get("/api/events")
    async def events(limit: int = Query(50, ge=1, le=500), since: float | None = None):
        # bounded both ways: SQLite reads LIMIT -1 as "no limit", i.e. the whole table under the store lock
        return await run_in_threadpool(pipe.store.recent, limit, since)

    @app.get("/api/events/{event_id}/snapshot")
    async def snapshot(event_id: str):
        ev = pipe.store.get(event_id)
        if not ev or not ev.get("snapshot") or not Path(ev["snapshot"]).exists():
            raise HTTPException(404, "no snapshot")
        return FileResponse(ev["snapshot"], media_type="image/jpeg")

    @app.get("/api/rules")
    async def rules():
        return [r.model_dump() for r in pipe.engine.list()]

    # rule types this pipeline cannot evaluate (site mode has no hard-hat detector): refused, so
    # a rule is never listed as active while it can never fire
    unsupported = set(getattr(pipe, "unsupported_rule_types", ()))

    def unsupported_msg(rtype: str) -> str:
        return f"rule type {rtype} is not available in this mode (site mode tracks whole people, not hard hats)"

    @app.post("/api/rules/parse")
    async def parse(body: ParseIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty text")
        llm = pipe.llm if await run_in_threadpool(pipe.llm.available) else None
        t0 = time.perf_counter()
        res = await run_in_threadpool(rule_parser.parse, text, cams, zones, llm)
        if res.rule is not None and res.rule.type in unsupported:
            res.problems.append(unsupported_msg(res.rule.type))
        return {"ok": res.rule is not None and not res.problems, "rule": res.rule.model_dump() if res.rule else None,
                "parser": res.parser, "notes": res.notes, "problems": res.problems,
                "ms": int((time.perf_counter() - t0) * 1000)}

    @app.post("/api/rules")
    async def add_rule(body: RuleIn):
        try:
            rule = Rule(**body.rule)
        except Exception as e:
            raise HTTPException(422, f"invalid rule: {e}")
        problems = rule.check_complete()
        if rule.type in unsupported:
            problems.append(unsupported_msg(rule.type))
        if problems:
            raise HTTPException(422, "; ".join(problems))
        if not pipe.engine.scope_cameras(rule):   # e.g. cam01 + zone_b while cam01 is at the entrance
            raise HTTPException(422, "no camera matches this scope")
        return pipe.engine.add(rule).model_dump()

    @app.patch("/api/rules/{rule_id}")
    async def toggle_rule(rule_id: str, body: EnableIn):
        r = pipe.engine.set_enabled(rule_id, body.enabled)
        if r is None:
            raise HTTPException(404, "unknown rule")
        return r.model_dump()

    @app.delete("/api/rules/{rule_id}")
    async def delete_rule(rule_id: str):
        if not pipe.engine.remove(rule_id):
            raise HTTPException(404, "unknown rule")
        return {"ok": True}

    @app.post("/api/asr")
    async def speech(audio: UploadFile = File(...), lang: str | None = Form(None)):
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        try:
            return await run_in_threadpool(asr().transcribe, data, audio.filename or "audio.webm", lang or None)
        except resources.ResourceError as e:
            raise HTTPException(503, str(e))
        except ValueError as e:  # undecodable or too-short recording, unsupported lang
            raise HTTPException(400, str(e))

    @app.post("/api/report")
    async def report(body: ReportIn):
        from .report import build_report
        llm = pipe.llm if await run_in_threadpool(pipe.llm.available) else None
        return await run_in_threadpool(build_report, pipe.store, llm, cams, body.lang, body.hours)

    @app.get("/api/health")
    async def health():
        return resources.snapshot()

    return app
