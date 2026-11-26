"""Shared camera engine for the phone pages (tools/phone_view.py drive mode, tools/phone_link): pick any of the
phone's cameras (front / wide / ultra wide / telephoto ...), zoom, and stream it as hardware H.264 (WebCodecs) with
capture times. Raw sensor-oriented frames go out as they are; the laptop rotates them upright (flags bits 1-2).

Page usage (HTML must have: <video id="me">, <div id="cams"></div>, <input id="zoom" type="range">, <span id="zoomv">,
<button id="rot">):
  const cam = CamEngine({video: $("me"), send: (cap, seq, flags, bytes) => ..., bitrate: 4e6});
  await cam.start();            // after a user tap (camera permission)
  cam.stats() -> {camera, label, enc_fps, encode_ms, rot}
"""

CAM_JS = r"""
function CamEngine(o){
  const v=o.video, nowMs=()=>performance.timeOrigin+performance.now();
  let track=null, enc=null, q=[], sq=0, FW=0, FH=0, ROT=0, rotUser=0, forceKey=false, devices=[], cur=null, nEnc=0, encMs=[];
  // settings survive a reload (this phone's browser storage): camera, zoom per camera, manual rotation
  const SK="camsettings", load=()=>{ try{ return JSON.parse(localStorage.getItem(SK))||{}; }catch(_){ return {}; } };
  const save=p=>{ try{ localStorage.setItem(SK,JSON.stringify({...load(),...p})); }catch(_){} };
  rotUser=(load().rot|0)&3;
  const short=l=>{ l=(l||"").toLowerCase();
    if(l.includes("front")) return "Front"; if(l.includes("ultra")) return "Ultra Wide"; if(l.includes("tele")) return "Tele";
    if(l.includes("triple")) return "Triple"; if(l.includes("dual wide")) return "Dual Wide"; if(l.includes("dual")) return "Dual";
    if(l.includes("back")) return "Wide"; return l||"Camera"; };
  function buttons(){ const el=document.getElementById("cams"); if(!el) return;
    el.innerHTML=devices.map(d=>`<button data-id="${d.deviceId}" class="cam${cur===d.deviceId?" on":""}">${short(d.label)}</button>`).join("");
    el.querySelectorAll("button").forEach(b=>b.onclick=()=>switchTo(b.dataset.id)); }
  function zoomUI(){ const z=document.getElementById("zoom"), zv=document.getElementById("zoomv"); if(!z) return;
    const c=track&&track.getCapabilities?track.getCapabilities():{};
    if(c.zoom&&c.zoom.max>c.zoom.min){ z.hidden=false; z.min=c.zoom.min; z.max=Math.min(c.zoom.max,10); z.step=0.1;
      const want=(load().zoom||{})[cur];          // this camera's saved zoom
      if(want&&want>=c.zoom.min&&want<=c.zoom.max) track.applyConstraints({advanced:[{zoom:want}]}).catch(()=>{});
      z.value=want||(track.getSettings().zoom||c.zoom.min); if(zv) zv.textContent=(+z.value).toFixed(1)+"×";
      z.oninput=()=>{ track.applyConstraints({advanced:[{zoom:+z.value}]}).catch(()=>{}); if(zv) zv.textContent=(+z.value).toFixed(1)+"×";
        save({zoom:{...(load().zoom||{}),[cur]:+z.value}}); };
    } else { z.hidden=true; if(zv) zv.textContent=""; } }
  async function configure(){
    if(!v.videoWidth) await new Promise(r=>v.onloadedmetadata=r);
    const probe=new VideoFrame(v,{timestamp:0}); FW=probe.displayWidth&~1; FH=probe.displayHeight&~1;
    // where Safari reports the frame's rotation use it, else infer: a landscape frame shown portrait = 1/4 turn
    const r=typeof probe.rotation==="number"?Math.round(probe.rotation/90)&3:(((FW>FH)!==(v.videoWidth>v.videoHeight))?1:0);
    probe.close(); ROT=(r+rotUser)&3;
    if(enc&&enc.state!=="closed") enc.close();
    q=[]; enc=new VideoEncoder({output:ch=>{ const c=q.shift()||{}; encMs.push(nowMs()-(c.t0||nowMs())); if(encMs.length>60)encMs.shift();
        const b=new Uint8Array(ch.byteLength); ch.copyTo(b); nEnc++; lastOut=nowMs(); o.send(c.cap||nowMs(), c.sq||0, (ch.type==="key"?1:0)|(ROT<<1), b); },
      error:e=>{ (o.onerror||console.error)("encoder: "+e.message); }});
    enc.configure({codec:"avc1.42E01F",width:FW,height:FH,bitrate:o.bitrate||4e6,framerate:30,latencyMode:"realtime",
                   hardwareAcceleration:"prefer-hardware",avc:{format:"annexb"}});
    forceKey=true; }
  async function switchTo(id){
    if(track) track.stop();
    const s=await navigator.mediaDevices.getUserMedia({video:{deviceId:id?{exact:id}:undefined,facingMode:id?undefined:"user",
      width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});
    track=s.getVideoTracks()[0]; cur=track.getSettings().deviceId||id; v.srcObject=new MediaStream([track]);
    if(id) save({cam:cur});
    // iOS pauses the camera in the background: mute -> unmute (keyframe), or it ends the track (restart it)
    track.onunmute=()=>{ forceKey=true; }; track.onended=()=>{ setTimeout(()=>ensure(),300); };
    v.style.transform=short(track.label)==="Front"?"scaleX(-1)":"none";   // preview mirrored only for the selfie camera
    await v.play().catch(()=>{}); await configure(); buttons(); zoomUI(); }
  // one frame loop only: each (re)start bumps loopGen and the old loop's callbacks stop (the watchdog used to add a
  // second loop beside a merely paused first one: every frame encoded twice = 60 fps, 2026-11-26)
  let loopGen=0;
  function loop(){ const g=++loopGen; const cb=(t,m)=>{ if(g!==loopGen) return; step(t,m); v.requestVideoFrameCallback(cb); }; v.requestVideoFrameCallback(cb); }
  function step(t,meta){ lastStep=nowMs(); if(enc&&enc.state==="configured"&&enc.encodeQueueSize<2){
      const cap=meta&&meta.captureTime?performance.timeOrigin+meta.captureTime:nowMs();
      try{ const f=new VideoFrame(v,{timestamp:Math.round((meta&&meta.mediaTime||t/1000)*1e6)});
        if((f.displayWidth&~1)!==FW||(f.displayHeight&~1)!==FH){ f.close(); configure(); }   // camera changed size
        else { sq++; q.push({cap,sq,t0:nowMs()}); enc.encode(f,{keyFrame:forceKey||sq%60===1}); forceKey=false; f.close(); }
      }catch(e){} } }
  let busy=false, lastOut=nowMs(), lastStep=nowMs(), restarts=0, lastKey=0;
  // watchdog: iOS's hardware encoder can stall (queue stuck, no output, no error) -- seen 2026-11-26 after a burst of
  // keyframe requests on bad Wi-Fi: the stream froze until a reload. Restart whatever stopped, within ~1.5 s.
  setInterval(()=>{ if(busy||!track||document.visibilityState!=="visible") return; const now=nowMs();
    if(now-lastStep>1500){ restarts++; lastStep=now; v.play().catch(()=>{}); loop(); }
    else if(now-lastOut>1500&&enc){ restarts++; lastOut=now; if(restarts%3===0) ensure(); else configure(); } },500);
  async function ensure(){   // after the page comes back: camera, encoder and frame loop alive? else restart them
    if(busy) return; busy=true;
    try{ if(!track||track.readyState!=="live"||!enc||enc.state==="closed"){ await switchTo(cur); }
         else if(v.paused){ await v.play().catch(()=>{}); }
         forceKey=true; } catch(e){ (o.onerror||console.error)("camera restart: "+(e.message||e)); }
    finally{ busy=false; } }
  return {
    ensure,
    async start(){ const sc=load().cam;
      if(sc){ try{ await switchTo(sc); }catch(_){ await switchTo(null); } } else await switchTo(null);   // last used camera
      devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==="videoinput");
      // front first, then back cameras from widest to longest
      const rank=d=>({"Front":0,"Ultra Wide":1,"Wide":2,"Tele":3,"Dual Wide":4,"Dual":5,"Triple":6}[short(d.label)]??9);
      devices.sort((a,b)=>rank(a)-rank(b)); buttons();
      const rb=document.getElementById("rot"); if(rb) rb.onclick=()=>{ rotUser=(rotUser+1)&3; save({rot:rotUser}); configure(); };
      loop(); },
    switchTo, key(){ const n=nowMs(); if(n-lastKey>500){ lastKey=n; forceKey=true; } },   // at most 2 keyframes/s
    restarts:()=>restarts,
    stats(){ const em=encMs.length?encMs.reduce((a,b)=>a+b)/encMs.length:null, r={camera:`${FW}x${FH}`,label:track?short(track.label):"",
             enc_fps:nEnc, encode_ms:em==null?null:+em.toFixed(1), rot:ROT*90}; nEnc=0; return r; },
  };
}
"""

CAM_CSS = """
#cams{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0} .cam{font:600 14px -apple-system,system-ui;border:0;border-radius:16px;
padding:8px 12px;background:#333;color:#ddd} .cam.on{background:#2e7d32;color:#fff}
#camrow{display:flex;align-items:center;gap:8px;flex-wrap:wrap} #zoom{flex:1;min-width:120px} #rot{font:600 14px -apple-system;
border:0;border-radius:16px;padding:8px 12px;background:#333;color:#ddd}
"""
