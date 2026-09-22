from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from app.config import settings
from app.services import inference
from app.storage import load_records, normalize_image_path, queue_payload, resolve_image_path

app = FastAPI(title="Hyundai Service ANPR")


def _latest_fallback_image():
    if settings.live_frame_file.exists():
        return settings.live_frame_file
    if not settings.image_dir.exists():
        return None
    images = sorted(settings.image_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return images[0] if images else None


def _live_stream():
    while True:
        frame = _latest_fallback_image()
        if frame and frame.exists():
            try:
                data = frame.read_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n" + data + b"\r\n"
                )
            except OSError:
                pass
        time.sleep(0.18)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/api/queue")
def get_queue_api():
    return JSONResponse(content=queue_payload())


@app.get("/queue")
def get_queue():
    return JSONResponse(content=queue_payload())


@app.get("/api/inference/status")
def get_inference_status():
    return JSONResponse(content=inference.status())


@app.post("/api/inference/start")
def start_inference():
    status = inference.start()
    return JSONResponse(content=status, status_code=200 if status["configured"] else 400)


@app.post("/api/inference/stop")
def stop_inference():
    return JSONResponse(content=inference.stop())


@app.get("/live-frame")
def get_live_frame():
    frame = _latest_fallback_image()
    if not frame:
        raise HTTPException(status_code=404, detail="Live frame is not available yet")
    try:
        data = frame.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Live frame is being refreshed") from exc
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/live-stream")
def get_live_stream():
    return StreamingResponse(
        _live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/images/{item_id}")
def get_image_by_id(item_id: int, kind: str = Query("auto", pattern="^(auto|vehicle|plate)$")):
    target = next((x for x in load_records() if x.get("id") == item_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Item not found")

    full = resolve_image_path(target, kind)
    if not full:
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path=str(full), media_type="image/jpeg", filename=full.name)


@app.get("/image-file")
def get_image_by_name(name: str = Query(..., min_length=1)):
    full = normalize_image_path(name)
    if not full.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path=str(full), media_type="image/jpeg", filename=full.name)


@app.get("/images/by-name")
def get_legacy_image_by_name(name: str = Query(..., min_length=1)):
    return get_image_by_name(name)


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hyundai Service ANPR</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#f3f6fa; --panel:#fff; --ink:#101820; --muted:#667085;
      --line:#d7dde6; --blue:#002c5f; --blue2:#005eb8;
      --queue:#a15c07; --garage:#087443; --danger:#b42318;
      --shadow:0 18px 46px rgba(16,24,40,.10);
    }
    *{box-sizing:border-box} body{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink)}
    header{background:#071a33;border-bottom:1px solid #0f2f55;color:#fff;padding:18px 28px}
    .topbar{width:min(1500px,100%);margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:18px}
    .brand{display:flex;align-items:center;gap:14px;min-width:260px}
    .emblem{width:78px;height:42px;border:3px solid #fff;border-radius:50%;display:grid;place-items:center;font-size:26px;font-weight:900;font-style:italic;transform:skew(-8deg)}
    .brand-title{font-size:12px;color:#9fd3ff;font-weight:900;letter-spacing:1.8px;text-transform:uppercase}
    h1{margin:3px 0 0;font-size:23px;line-height:1.2;color:#fff}
    .meta{display:flex;align-items:center;gap:12px;color:#d8e3ef;font-size:13px;flex-wrap:wrap;justify-content:flex-end}
    .control-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .live-dot{width:9px;height:9px;border-radius:999px;background:#16a34a;box-shadow:0 0 0 5px rgba(22,163,74,.14);display:inline-block;margin-right:7px}
    main{width:min(1500px,100%);margin:0 auto;padding:24px 28px 36px}
    .overview{display:grid;grid-template-columns:minmax(520px,1.45fr) minmax(360px,.9fr);gap:18px;align-items:stretch;margin-bottom:18px}
    .panel,.stat,.table-wrap{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}
    .camera{overflow:hidden;min-height:470px;display:flex;flex-direction:column}
    .panel-head{min-height:58px;padding:14px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px}
    .panel-title{margin:0;font-size:16px;line-height:1.2;font-weight:900}
    .panel-kicker{color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase}
    .camera-frame{position:relative;background:#0b1117;min-height:412px;flex:1;display:grid;place-items:center;overflow:hidden}
    .camera-frame img{width:100%;height:100%;object-fit:contain;display:block}
    .camera-overlay{position:absolute;left:14px;top:14px;display:flex;gap:8px;align-items:center;color:#fff;font-size:12px;font-weight:800;background:rgba(7,26,51,.88);border:1px solid rgba(255,255,255,.24);border-radius:999px;padding:8px 11px}
    .side{display:flex;flex-direction:column;gap:14px}.stats{display:grid;grid-template-columns:repeat(2,minmax(140px,1fr));gap:12px}
    .stat{padding:14px 16px;min-height:94px;border-left:4px solid var(--blue2)}
    .stat span{color:var(--muted);font-size:12px;text-transform:uppercase;font-weight:900}.stat strong{display:block;font-size:34px;line-height:1;margin-top:12px}
    .latest{flex:1;padding:16px}.latest-grid{display:grid;grid-template-columns:128px 1fr;gap:14px;align-items:center;margin-top:12px}
    .latest img{width:128px;height:82px;object-fit:cover;border-radius:7px;border:1px solid var(--line);background:#edf1f5}
    .latest-plate{font-size:26px;font-weight:900;line-height:1;color:var(--blue);margin-bottom:10px}
    .label{color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase}.value{font-size:14px;color:var(--ink);margin-top:4px}
    .section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:4px 0 12px}.section-head h2{margin:0;font-size:17px;line-height:1.2}
    .table-wrap{overflow:auto} table{width:100%;border-collapse:collapse;min-width:980px}
    th,td{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle;font-size:14px}
    th{color:var(--muted);font-size:12px;text-transform:uppercase;background:#f9fafb} tr:last-child td{border-bottom:0}
    .plate{font-size:20px;font-weight:900;white-space:nowrap;color:var(--blue)}.plate.pending{color:var(--muted);font-size:15px}
    .badge{display:inline-flex;align-items:center;min-height:28px;padding:0 10px;border-radius:999px;font-weight:900;font-size:12px;white-space:nowrap}
    .badge.queue{color:var(--queue);background:#fff7e6}.badge.garage{color:var(--garage);background:#e8f6ef}.badge.unknown{color:var(--danger);background:#fff0ee}
    .thumbs{display:flex;gap:8px;align-items:center}.capture{display:grid;gap:5px}.capture-label{color:var(--muted);font-size:11px;font-weight:900;text-transform:uppercase}
    img.thumb{width:112px;height:66px;object-fit:cover;border-radius:6px;border:1px solid var(--line);background:#eef1f5}
    button{border:1px solid #fff;background:#fff;color:var(--blue);min-height:36px;padding:0 13px;border-radius:6px;cursor:pointer;font-weight:900}
    button:hover{background:#e8f2ff;border-color:#e8f2ff}button.stop{background:transparent;color:#fff;border-color:rgba(255,255,255,.45)}button.stop:hover{background:rgba(255,255,255,.12)}
    .empty{padding:36px 16px;text-align:center;color:var(--muted)}.error{color:var(--danger);font-weight:800}
    @media(max-width:1080px){.overview{grid-template-columns:1fr}.camera{min-height:360px}}@media(max-width:720px){header,main{padding-left:16px;padding-right:16px}.topbar{align-items:flex-start;flex-direction:column}.meta{justify-content:flex-start}.brand{min-width:0}.emblem{width:66px;height:36px;font-size:22px}h1{font-size:20px}.stats{grid-template-columns:repeat(2,minmax(120px,1fr))}.latest-grid{grid-template-columns:1fr}.latest img{width:100%;height:150px}}
  </style>
</head>
<body>
  <header><div class="topbar"><div class="brand"><div class="emblem">H</div><div><div class="brand-title">HYUNDAI SERVICE</div><h1>ANPR Service Lane Monitor</h1></div></div>
    <div class="meta"><span><span class="live-dot" id="run-dot"></span><span id="run-label">Checking inference</span></span><span id="last-updated">Loading...</span>
      <div class="control-group"><button type="button" onclick="startInference()">Start Inference</button><button type="button" class="stop" onclick="stopInference()">Stop</button><button type="button" onclick="loadQueue()">Refresh</button></div>
    </div></div></header>
  <main>
    <section class="overview"><div class="panel camera"><div class="panel-head"><h2 class="panel-title">RTSP Service Lane Camera</h2><span class="panel-kicker" id="stream-status">Live inference stream</span></div>
      <div class="camera-frame"><img id="live-frame" src="/live-frame" alt="Live ANPR feed"><div class="camera-overlay"><span class="live-dot"></span>ANPR LIVE</div></div></div>
      <aside class="side"><div class="stats"><div class="stat"><span>Total Vehicles</span><strong id="total">0</strong></div><div class="stat"><span>In Queue</span><strong id="queue">0</strong></div><div class="stat"><span>In Garage</span><strong id="garage">0</strong></div><div class="stat"><span>Plates Read</span><strong id="plates">0</strong></div></div>
      <div class="panel latest"><h2 class="panel-title">Latest Vehicle</h2><div class="latest-grid" id="latest"><div class="empty">Waiting for detections...</div></div></div></aside></section>
    <div class="section-head"><h2>Service Lane Records</h2><span class="panel-kicker">Entry, exit and ANPR captures</span></div>
    <section class="table-wrap"><table><thead><tr><th>ID</th><th>Plate Number</th><th>Status</th><th>Entry Time</th><th>Exit Time</th><th>Dwell Time</th><th>Images</th></tr></thead><tbody id="rows"><tr><td colspan="7" class="empty">Loading...</td></tr></tbody></table></section>
  </main>
  <script>
    const rows=document.getElementById("rows"),latest=document.getElementById("latest"),liveFrame=document.getElementById("live-frame"),runDot=document.getElementById("run-dot"),runLabel=document.getElementById("run-label"),streamStatus=document.getElementById("stream-status");
    const fmt=v=>v&&String(v).trim()?v:"-"; const esc=v=>String(v??"").replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
    function statusClass(s){if(s==="In Queue")return"queue";if(s==="In Garage")return"garage";return"unknown"}
    function imgTag(src,label,cls="thumb"){return src?`<img class="${cls}" src="${src}&v=${Date.now()}" alt="${label}" onerror="this.style.display='none'">`:""}
    function captureTag(src,label){return src?`<div class="capture"><span class="capture-label">${label}</span>${imgTag(src,label)}</div>`:""}
    function imagePair(item){const v=captureTag(item.vehicle_image_url,"Vehicle"),p=captureTag(item.plate_image_url,"Plate");return v||p?`${v}${p}`:"-"}
    function refreshLiveFrame(){liveFrame.src=`/live-frame?v=${Date.now()}`}
    function setInferenceState(s){const running=Boolean(s&&s.running),configured=Boolean(s&&s.configured);runDot.style.background=running?"#16a34a":configured?"#f59e0b":"#ef4444";runLabel.textContent=running?`Inference running${s.pid?` · PID ${s.pid}`:""}`:configured?"Inference stopped":(s.message||"RTSP URL missing");streamStatus.textContent=running?"Live RTSP inference stream":configured?"Start inference to connect RTSP":"Set RTSP_URL in .env"}
    async function loadInferenceStatus(){try{const r=await fetch("/api/inference/status",{cache:"no-store"});const s=await r.json();setInferenceState(s);return s}catch{const s={running:false,configured:false,message:"API unavailable"};setInferenceState(s);return s}}
    async function startInference(){const r=await fetch("/api/inference/start",{method:"POST"});const s=await r.json();setInferenceState(s);if(r.ok)refreshLiveFrame()}
    async function stopInference(){const r=await fetch("/api/inference/stop",{method:"POST"});setInferenceState(await r.json())}
    function renderLatest(item){if(!item){latest.innerHTML=`<div class="empty">Waiting for detections...</div>`;return}latest.innerHTML=`${imgTag(item.vehicle_image_url||item.plate_image_url||item.image_url,"latest vehicle","")}<div><div class="latest-plate">${esc(item.platenumber)}</div><div class="label">Current Status</div><div class="value"><span class="badge ${statusClass(item.status)}">${esc(item.status)}</span></div><div class="label" style="margin-top:12px">Entry Time</div><div class="value">${esc(fmt(item.entry_time))}</div></div>`}
    async function loadQueue(){try{const r=await fetch("/api/queue",{cache:"no-store"});if(!r.ok)throw new Error(await r.text());const data=await r.json();document.getElementById("total").textContent=data.length;document.getElementById("queue").textContent=data.filter(x=>x.status==="In Queue").length;document.getElementById("garage").textContent=data.filter(x=>x.status==="In Garage").length;document.getElementById("plates").textContent=data.filter(x=>x.platenumber&&x.platenumber!=="Reading...").length;document.getElementById("last-updated").textContent=`Updated ${new Date().toLocaleTimeString()}`;renderLatest(data[0]);if(!data.length){rows.innerHTML=`<tr><td colspan="7" class="empty">No service lane records yet.</td></tr>`;return}rows.innerHTML=data.map(item=>`<tr><td>${esc(item.id)}</td><td class="plate ${item.plate_is_pending?"pending":""}">${esc(item.platenumber)}</td><td><span class="badge ${statusClass(item.status)}">${esc(item.status)}</span></td><td>${esc(fmt(item.entry_time))}</td><td>${esc(fmt(item.exit_time))}</td><td>${esc(fmt(item.duration))}</td><td><div class="thumbs">${imagePair(item)}</div></td></tr>`).join("")}catch(err){rows.innerHTML=`<tr><td colspan="7" class="empty error">Could not load ANPR data: ${esc(err.message)}</td></tr>`}}
    async function initializeDashboard(){loadQueue();const s=await loadInferenceStatus();if(s.configured&&!s.running)await startInference();else refreshLiveFrame()}
    liveFrame.onerror=()=>{streamStatus.textContent="Waiting for live frames"}; initializeDashboard(); setInterval(loadQueue,2000); setInterval(loadInferenceStatus,2500); setInterval(refreshLiveFrame,300);
  </script>
</body>
</html>
"""
