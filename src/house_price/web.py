"""Static demo page assets. Served as separate files so a strict CSP (no inline code) works."""

from __future__ import annotations

import json

N_FEATURES = 8

# name, slider min, max, step (inputs are standardised-scale values)
FIELDS = [
    ("Median income", -3, 3, 0.1),
    ("House age", -3, 3, 0.1),
    ("Avg rooms", -3, 3, 0.1),
    ("Avg bedrooms", -3, 3, 0.1),
    ("Population", -3, 3, 0.1),
    ("Avg occupancy", -3, 3, 0.1),
    ("Latitude", -3, 3, 0.1),
    ("Longitude", -3, 3, 0.1),
]

# 3x3 region grid (north to south, west to east): label, latitude, longitude
REGIONS = [
    (name, lat, lon)
    for name, lat, lon in [
        ("NW", 39.5, -122.5),
        ("N", 39.0, -121.0),
        ("NE", 39.0, -119.0),
        ("W", 37.5, -122.2),
        ("Central", 36.7, -119.8),
        ("E", 35.5, -118.0),
        ("SW", 34.2, -119.5),
        ("LA", 34.0, -118.2),
        ("SE", 33.0, -117.1),
    ]
]

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def _field(i: int, name: str, lo: float, hi: float, step: float) -> str:
    return (
        f'<div class="field"><label for="n{i}">{name}</label>'
        f'<input type="range" id="r{i}" data-i="{i}" min="{lo}" max="{hi}" step="{step}" '
        f'value="0" aria-label="{name} slider">'
        f'<input type="number" id="n{i}" data-i="{i}" step="any" name="f{i}" value="0" required>'
        "</div>"
    )


def _regions() -> str:
    buttons = "".join(
        f'<button type="button" class="region" data-lat="{lat}" data-lon="{lon}">{n}</button>'
        for n, lat, lon in REGIONS
    )
    return f'<fieldset><legend>Region</legend><div class="map">{buttons}</div></fieldset>'


PAGE = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>House price demo</title>
<link rel="stylesheet" href="/app.css">
</head><body><main>
<h1>House price demo</h1>
<form id="f" action="/predict" method="post">"""
    + _regions()
    + "".join(_field(i, *f) for i, f in enumerate(FIELDS))
    + """<button type="submit" class="go">Predict</button></form>
<section id="result" hidden>
<div id="price" aria-live="polite"></div>
<div class="bar" id="range-bar" role="img" aria-label="90% price range">
<span class="band" id="band"></span><i id="marker"></i></div>
<div id="range"></div>
<h2>Why this price</h2>
<svg id="waterfall" role="img" aria-label="Factor waterfall" viewBox="0 0 400 160"></svg>
<ul id="warnings"></ul>
</section>
<p id="error" role="alert"></p>
<script src="/app.js"></script>
</main></body></html>"""
)

CSS = """
:root{--bg:#f4f6fb;--card:#fff;--fg:#1c2333;--accent:#3b5bdb;--pos:#2b8a3e;--neg:#c92a2a}
@media (prefers-color-scheme:dark){
:root{--bg:#141824;--card:#1e2333;--fg:#e6e9f2;--accent:#748ffc;--pos:#69db7c;--neg:#ff8787}}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:2rem}
main{max-width:36rem;margin:auto}
form,#result{background:var(--card);padding:1.2rem;border-radius:12px;box-shadow:0 2px 8px #0001}
form{display:grid;gap:.8rem}
#result{margin-top:1.2rem}
fieldset{border:1px solid #8884;border-radius:8px}
.map{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem}
.region{padding:.6rem;border:1px solid #8886;border-radius:6px;background:transparent;
color:inherit;cursor:pointer}
.region[aria-pressed=true]{background:var(--accent);color:#fff}
.field{display:grid;grid-template-columns:8rem 1fr 5rem;gap:.5rem;align-items:center;
font-size:.85rem}
input[type=number]{padding:.4rem;border:1px solid #8886;border-radius:6px;font-size:1rem;
background:transparent;color:inherit}
.go{padding:.7rem;border:0;border-radius:8px;background:var(--accent);color:#fff;
font-size:1rem;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
#price{font-size:2.2rem;font-weight:700;font-variant-numeric:tabular-nums}
.bar{position:relative;height:10px;background:#8883;border-radius:5px;margin:1rem 0 .3rem}
.band{position:absolute;top:0;height:10px;background:var(--accent);opacity:.35;border-radius:5px}
.bar i{position:absolute;top:-3px;width:4px;height:16px;background:var(--accent);
border-radius:2px}
#waterfall{width:100%;height:auto}
#waterfall text{fill:var(--fg);font-size:11px}
.pos{fill:var(--pos)}.neg{fill:var(--neg)}
#error{color:var(--neg)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

_JS = """"use strict";
const $=id=>document.getElementById(id);
const form=$("f");
const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt=x=>x.toFixed(2);
const FEATURES=%s;

function setValue(i,v){
  form.querySelector('input[type=range][data-i="'+i+'"]').value=v;
  form.querySelector('input[type=number][data-i="'+i+'"]').value=v;
}
form.querySelectorAll("input[data-i]").forEach(el=>el.addEventListener("input",()=>
  setValue(el.dataset.i,el.value)));
form.querySelectorAll(".region").forEach(b=>b.addEventListener("click",()=>{
  form.querySelectorAll(".region").forEach(o=>o.setAttribute("aria-pressed",String(o===b)));
  setValue(6,b.dataset.lat);setValue(7,b.dataset.lon);
}));

function countUp(el,to){
  if(reduce){el.textContent=fmt(to);return}
  const t0=performance.now(),ms=700;
  const step=now=>{
    const p=Math.min(1,(now-t0)/ms);
    el.textContent=fmt(to*(1-Math.pow(1-p,3)));
    if(p<1)requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function rangeBar(d){
  const lo=d.interval_low,hi=d.interval_high,span=(hi-lo)||1;
  const pad=span*0.1,min=lo-pad,tot=span+2*pad;
  const pct=v=>(100*(v-min)/tot)+"%%";
  $("band").style.left=pct(lo);
  $("band").style.width=(100*span/tot)+"%%";
  $("marker").style.left=pct(d.price);
  $("range").textContent="90%% range: "+fmt(lo)+" to "+fmt(hi);
}

function svg(tag,attrs,text){
  const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs)e.setAttribute(k,attrs[k]);
  if(text!==undefined)e.textContent=text;
  return e;
}

function waterfall(d){
  const root=$("waterfall");
  root.replaceChildren();
  const rows=d.factors.map(f=>[FEATURES[f.feature]||"Feature "+f.feature,f.effect]);
  const max=Math.max(...rows.map(r=>Math.abs(r[1])),1e-9);
  const mid=250,scale=120/max;
  root.appendChild(svg("line",{x1:mid,x2:mid,y1:0,y2:160,stroke:"currentColor",opacity:.3}));
  rows.forEach(([name,eff],k)=>{
    const w=Math.abs(eff)*scale,y=10+k*40;
    root.appendChild(svg("text",{x:5,y:y+16},name));
    root.appendChild(svg("rect",{x:eff>=0?mid:mid-w,y,width:w,height:24,
      class:eff>=0?"pos":"neg"}));
    root.appendChild(svg("text",{x:eff>=0?mid+w+4:mid-w-4,y:y+16,
      "text-anchor":eff>=0?"start":"end"},(eff>=0?"+":"")+fmt(eff)));
  });
}

form.addEventListener("submit",async e=>{
  e.preventDefault();
  const features=[...form.querySelectorAll("input[type=number]")].map(i=>Number(i.value));
  let d;
  try{
    const r=await fetch("/predict",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({features})});
    d=await r.json();
  }catch(err){d={error:"request failed"}}
  $("error").textContent=d.error||"";
  $("result").hidden=!!d.error;
  if(d.error)return;
  countUp($("price"),d.price);
  rangeBar(d);
  waterfall(d);
  $("warnings").replaceChildren(...d.warnings.map(w=>{
    const li=document.createElement("li");
    li.textContent=(FEATURES[w.feature]||"Feature "+w.feature)+" is "+w.z_score.toFixed(1)+
      " std devs from the training mean";
    return li;
  }));
});
"""

JS = _JS % json.dumps([f[0] for f in FIELDS])
