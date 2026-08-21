"""Generate a fully self-contained, interactive HTML graph file.

The produced ``interactive_graph.html`` embeds the synthetic graph data and
all JavaScript inline, so it can be opened directly in any browser
(file://) with NO server.  vis-network is loaded from a CDN.

Run:
    PYTHONPATH=. python static/make_standalone.py
"""

from __future__ import annotations

import json
import os

from src.synthetic_data import load_graph_data, load_ground_truth, load_conditions

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "interactive_graph.html")
OUT_ROOT = os.path.join(PROJECT_ROOT, "index.html")

NODE_STYLE = {
    "patient":     {"color": "#5fd36a", "shape": "dot",      "size": 18},
    "disease":     {"color": "#e06a6a", "shape": "ellipse",   "size": 16},
    "indicator":   {"color": "#e0a04a", "shape": "diamond",   "size": 14},
    "drug":        {"color": "#4a90d9", "shape": "box",       "size": 14},
    "intermediate":{"color": "#5a6a7a", "shape": "dot",       "size": 8},
}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient-Centered Graph Retrieval (standalone)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:#0f1620;color:#dfe6ee;font-size:13px}
header{background:linear-gradient(90deg,#1b2a3a,#243b55);padding:12px 18px;border-bottom:2px solid #2a4365}
header h1{font-size:18px;color:#fff;margin-bottom:2px}
.subtitle{color:#8aa6c1;font-size:12px}
.layout{display:grid;grid-template-columns:1.4fr 1fr;gap:10px;padding:10px}
.panel{background:#16202c;border:1px solid #283648;border-radius:6px;padding:8px}
.panel-head{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #283648;padding-bottom:5px;margin-bottom:8px}
.panel-head h2{font-size:13px;color:#6fa8dc;text-transform:uppercase;letter-spacing:.5px}
.canvas-box{width:100%;background:#0d141d;border:1px solid #1f2c39;border-radius:4px}
#fullGraph{height:500px}
#subGraph{height:420px}
#infoBar{font-size:11px;color:#7a8da0;padding:5px 2px;min-height:16px}
.controls{display:flex;flex-direction:column;gap:5px}
.controls label{display:flex;align-items:center;gap:6px;font-size:12px}
.controls select,.controls input[type=number],.controls input[type=range]{background:#0f1822;color:#dfe6ee;border:1px solid #2a3b50;border-radius:3px;padding:3px;accent-color:#4a90d9}
.cond-row{display:flex;align-items:center;gap:6px;background:#1c2837;padding:5px 7px;border-radius:4px;border:1px solid #243648;font-size:12px}
.cond-row .anchor{font-weight:700;flex:1}
.cond-row input[type=checkbox]{accent-color:#4a90d9}
.cond-row select,.cond-row input[type=number]{width:auto;background:#0f1822;color:#dfe6ee;border:1px solid #2a3b50;border-radius:3px;padding:2px}
.cond-row input[type=number]{width:50px}
button{background:#2a3b50;color:#dfe6ee;border:1px solid #3a5070;border-radius:4px;padding:6px 12px;cursor:pointer;font-size:12px}
button:hover{background:#34507a}
button.primary{background:#2c6e9f;border-color:#3a82bf;font-weight:700}
.table-wrap{max-height:200px;overflow:auto;border:1px solid #1f2c39;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:4px 7px;text-align:left;border-bottom:1px solid #1f2c39;white-space:nowrap}
th{background:#1c2837;color:#8aa6c1;position:sticky;top:0}
tr.prow{cursor:pointer}
tr.prow:hover{background:#1f3550}
tr.prow.active{background:#2c5a3a}
.match{color:#5fd36a;font-weight:700}
.partial{color:#e0c34a}
.nomatch{color:#e06a6a}
#explanation{font-size:12px;max-height:200px;overflow:auto}
.pblock{background:#1c2837;border-left:3px solid #4a90d9;padding:6px 8px;margin-bottom:6px;border-radius:3px}
.pblock.fail{border-left-color:#e06a6a}
.pblock .ph{font-weight:700;margin-bottom:3px}
.pblock .chain{color:#b9c8d8;font-family:monospace;font-size:11px;word-break:break-all}
.pblock .meta{color:#7a8da0;font-size:11px;margin-top:2px}
.filter-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px}
.fcard{background:#1c2837;padding:5px 8px;border-radius:4px;font-size:11px;border-left:3px solid #4a90d9}
.fcard.fail{border-left-color:#e06a6a}
.fcard .fa{font-weight:700}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;padding:4px 0}
.legend .item{display:flex;align-items:center;gap:4px}
.swatch{width:12px;height:12px;border-radius:2px;display:inline-block}
.full-row{grid-column:1/-1}
</style>
</head>
<body>
<header>
  <h1>Patient-Centered Graph Retrieval</h1>
  <span class="subtitle">Standalone interactive demo &middot; 8 synthetic patients &middot; BFS / multi-hop / numeric hard-filter</span>
</header>

<div class="layout">
  <!-- Full graph -->
  <section class="panel">
    <div class="panel-head"><h2>A &middot; Full Synthetic Graph</h2>
      <span><button id="btnFit">Fit</button></span></div>
    <div id="fullGraph" class="canvas-box"></div>
    <div id="infoBar">Hover a node/edge for details. Click a <b>patient</b> node to select it.</div>
    <div class="legend" id="legend"></div>
  </section>

  <!-- Controls + table -->
  <section class="panel">
    <div class="panel-head"><h2>B &middot; Query &amp; Retrieve</h2></div>
    <div class="controls" id="queryPanel"></div>
    <div style="margin:8px 0"><button id="btnRun" class="primary">&#9654; Run Retrieval</button></div>
    <div class="panel-head"><h2>C &middot; Candidate Patients</h2></div>
    <div class="table-wrap">
      <table id="ptable"><thead><tr><th>Patient</th><th>Coverage</th><th>Match</th></tr></thead><tbody></tbody></table>
    </div>
  </section>

  <!-- Subgraph -->
  <section class="panel">
    <div class="panel-head"><h2>D &middot; Patient-Centered Subgraph</h2>
      <span>
        <select id="patientSel" style="background:#0f1822;color:#dfe6ee;border:1px solid #2a3b50;border-radius:3px;padding:2px"></select>
        <label style="font-size:11px"><input type="checkbox" id="showExcluded" checked> excl.</label>
      </span></div>
    <div id="subGraph" class="canvas-box"></div>
    <div id="subLegend" style="font-size:11px;color:#7a8da0;padding:5px 2px;min-height:14px"></div>
  </section>

  <!-- Path explanation -->
  <section class="panel">
    <div class="panel-head"><h2>E &middot; Path Explanation</h2></div>
    <div id="explanation">Click a patient row (or patient node) to see per-condition paths.</div>
  </section>

  <!-- Filter report -->
  <section class="panel full-row">
    <div class="panel-head"><h2>F &middot; Numeric Edge Hard-Filter Report</h2></div>
    <div id="filterReport" class="filter-grid"></div>
  </section>
</div>

<script>
"use strict";
const GRAPH = __GRAPH__;
const CONDS = __CONDS__;
const GT    = __GT__;
const STYLE = __STYLE__;

const COND_COLORS = {C1:"#e06a6a", C2:"#e0a04a", C3:"#4a90d9", C4:"#9a6ae0", C5:"#5fd36a"};

let fullNet=null, fullNodes=null, fullEdges=null;
let subNet=null;
let currentAcc=null;
let currentPid=null;

/* ---------- helpers ---------- */
function nodeById(id){ return GRAPH.nodes.find(n=>n.id===id) || {id:id, type:"intermediate"}; }
function edgeSatisfies(e,c){
  if(c.kind!=="numeric") return true;
  if(e.value===undefined) return false;
  var v=parseFloat(e.value), t=c.value;
  switch(c.operator){
    case ">":  return v>t;
    case ">=": return v>=t;
    case "<":  return v<t;
    case "<=": return v<=t;
  }
  return false;
}
function buildFilteredEdges(conds){
  var bad=new Set();
  GRAPH.edges.forEach(function(e,i){
    var c=conds.find(function(c){ return c.kind==="numeric" && c.anchor===e.source; });
    if(c && !edgeSatisfies(e,c)) bad.add(i);
  });
  return bad;
}
/* BFS from each condition anchor -> patients, within maxHops */
function runBFS(conds, maxHops){
  var bad=buildFilteredEdges(conds);
  var adj={};
  GRAPH.edges.forEach(function(e,i){
    if(bad.has(i)) return;
    if(!adj[e.source]) adj[e.source]=[];
    adj[e.source].push(e.target);
  });
  var acc={};
  GRAPH.nodes.filter(function(n){return n.type==="patient";}).forEach(function(p){ acc[p.id]={}; });
  conds.forEach(function(c){
    var queue=[[c.anchor,0,[c.anchor]]];
    var visited=new Set([c.anchor]);
    while(queue.length){
      var item=queue.shift();
      var node=item[0], dist=item[1], path=item[2];
      if(dist>=maxHops) continue;
      var nbrs=adj[node]||[];
      for(var k=0;k<nbrs.length;k++){
        var nb=nbrs[k];
        if(path.indexOf(nb)>=0) continue;
        var np=path.concat([nb]);
        var nd=dist+1;
        var ndata=nodeById(nb);
        if(ndata.type==="patient"){
          if(!acc[nb][c.id] || acc[nb][c.id].distance>nd){
            acc[nb][c.id]={matched:true, distance:nd, path:np, reason:"MATCH"};
          }
        }
        queue.push([nb,nd,np]);
      }
    }
    /* mark unmatched */
    GRAPH.nodes.filter(function(n){return n.type==="patient";}).forEach(function(p){
      if(!acc[p.id][c.id]){
        acc[p.id][c.id]={matched:false, distance:null, path:[], reason:"NOT REACHABLE"};
      }
    });
  });
  return acc;
}
function collectConds(){
  return CONDS.map(function(c){
    var cb=document.getElementById("cb_"+c.id);
    if(cb && !cb.checked) return null;
    if(c.kind==="numeric"){
      var sel=document.getElementById("op_"+c.id);
      var inp=document.getElementById("vl_"+c.id);
      var op=sel.options[sel.selectedIndex].text;
      return Object.assign({}, c, {operator:op, value:parseFloat(inp.value)});
    }
    return c;
  }).filter(function(x){return x;});
}
function coverageOf(pa){
  return CONDS.filter(function(c){return pa[c.id] && pa[c.id].matched;}).length;
}

/* ---------- full graph ---------- */
function nodeVis(n){
  var s=STYLE[n.type]||STYLE.intermediate;
  return {
    id:n.id, label:n.id, group:n.type,
    color:{background:s.color, border:s.color, highlight:{background:s.color,border:s.color}},
    shape:s.shape, size:s.size,
    font:{color:"#fff", size: s.type==="intermediate"?10:13},
    title:"id: "+n.id+"\ntype: "+n.type+"\ndegree: "+n.degree
  };
}
function edgeVis(e){
  var label=e.relation_type;
  if(e.value!==undefined) label+=" = "+e.value+(e.unit?" "+e.unit:"");
  return {
    id:e.source+"|"+e.target+"|"+e.relation_type,
    from:e.source, to:e.target,
    label:label, arrows:"to",
    color:{color:"#3a5070", highlight:"#6fa8dc", hover:"#6fa8dc"},
    font:{size:9, color:"#8aa6c1"},
    title:"relation: "+e.relation_type+
      (e.value!==undefined? "\nvalue: "+e.value+" "+(e.unit||""):"")+
      (e.time? "\ntime: "+e.time:"")
  };
}
function initFull(){
  fullNodes=new vis.DataSet(GRAPH.nodes.map(nodeVis));
  fullEdges=new vis.DataSet(GRAPH.edges.map(edgeVis));
  fullNet=new vis.Network(
    document.getElementById("fullGraph"),
    {nodes:fullNodes, edges:fullEdges},
    {
      edges:{smooth:{type:"continuous", roundness:0.15}},
      physics:{solver:"forceAtlas2Based", stabilization:{iterations:120}},
      interaction:{hover:true, tooltipDelay:80, navigationButtons:false}
    }
  );
  fullNet.on("selectNode", function(params){
    if(!params.nodes.length) return;
    var id=params.nodes[0];
    var n=nodeById(id);
    document.getElementById("infoBar").innerHTML=
      "<b>"+id+"</b> ("+n.type+", degree "+n.degree+")";
    if(n.type==="patient") selectPatient(id);
  });
  fullNet.on("hoverNode", function(p){
    var n=nodeById(p.node);
    document.getElementById("infoBar").innerHTML=
      "<b>"+n.id+"</b> &middot; "+n.type+" &middot; degree "+n.degree;
  });
}

/* ---------- query panel ---------- */
function buildQueryPanel(){
  var qp=document.getElementById("queryPanel");
  qp.innerHTML="";
  CONDS.forEach(function(c){
    var row=document.createElement("div");
    row.className="cond-row";
    if(c.kind==="numeric"){
      row.innerHTML=
        '<input type="checkbox" id="cb_'+c.id+'" checked>'+
        '<span class="anchor">'+c.anchor+'</span>'+
        '<select id="op_'+c.id+'">'+
          '<option '+('>'===c.operator?'selected':'')+'>&gt;</option>'+
          '<option '+('>='===c.operator?'selected':'')+'>&gt;=</option>'+
          '<option '+('<'===c.operator?'selected':'')+'>&lt;</option>'+
          '<option '+('&lt;='===c.operator?'selected':'')+'>&lt;=</option>'+
        '</select>'+
        '<input type="number" step="0.1" value="'+c.value+'" id="vl_'+c.id+'">';
    } else {
      row.innerHTML=
        '<input type="checkbox" id="cb_'+c.id+'" checked>'+
        '<span class="anchor">'+c.anchor+'</span>'+
        '<span style="color:#7a8da0">categorical</span>';
    }
    qp.appendChild(row);
  });
  // max hops slider
  var mh=document.createElement("div");
  mh.className="cond-row";
  mh.innerHTML='MAX_HOPS: <input type="range" id="maxHops" min="1" max="4" value="4" style="flex:1;max-width:140px"><b id="mhVal">4</b>';
  qp.appendChild(mh);
  // mode
  var md=document.createElement("div");
  md.className="cond-row";
  md.innerHTML='Mode: <select id="modeSel"><option value="strict">Strict (all)</option><option value="soft">Soft &ge;0.8</option></select>';
  qp.appendChild(md);
  document.getElementById("maxHops").oninput=function(e){
    document.getElementById("mhVal").textContent=e.target.value;
  };
}

/* ---------- run retrieval ---------- */
function runRetrieval(){
  var conds=collectConds();
  var mh=parseInt(document.getElementById("maxHops").value,10);
  var mode=document.getElementById("modeSel").value;
  currentAcc=runBFS(conds, mh);
  var strict=(mode==="strict");
  renderFilterReport(conds);
  var tb=document.querySelector("#ptable tbody");
  tb.innerHTML="";
  var order=Object.keys(currentAcc).sort(function(a,b){
    return coverageOf(currentAcc[b])-coverageOf(currentAcc[a]);
  });
  /* populate dropdown */
  var sel=document.getElementById("patientSel");
  sel.innerHTML="";
  order.forEach(function(pid){
    var cov=coverageOf(currentAcc[pid]);
    var o=document.createElement("option");
    o.value=pid; o.textContent=pid+"  ("+cov+"/"+CONDS.length+")";
    sel.appendChild(o);
  });
  order.forEach(function(pid){
    var pa=currentAcc[pid];
    var cov=coverageOf(pa);
    var m=CONDS.length;
    var full=(cov===m);
    var pass= strict ? full : (cov/m>=0.8);
    var cls= full?"match":(cov>=4?"partial":"nomatch");
    var tr=document.createElement("tr");
    tr.className="prow";
    tr.setAttribute("data-pid",pid);
    tr.innerHTML=
      "<td><b>"+pid+"</b></td>"+
      '<td class="'+cls+'">'+cov+"/"+m+"</td>"+
      '<td class="'+(pass?"match":"nomatch")+'">'+(pass?"Match":"-")+"</td>";
    tr.onclick=function(){ selectPatient(pid); };
    tb.appendChild(tr);
  });
  if(order.length) selectPatient(order[0]);
}

/* ---------- select patient + subgraph ---------- */
function selectPatient(pid){
  currentPid=pid;
  document.querySelectorAll(".prow").forEach(function(r){
    r.classList.toggle("active", r.getAttribute("data-pid")===pid);
  });
  var sel=document.getElementById("patientSel");
  if(sel) sel.value=pid;
  renderSubgraph(pid);
  renderExplanation(pid);
  if(fullNet) fullNet.selectNodes([pid]);
}

function renderSubgraph(pid){
  if(!currentAcc) return;
  var conds=collectConds();
  var pa=currentAcc[pid];
  var showExcl=document.getElementById("showExcluded").checked;
  var nodeSet=new Set([pid]);
  var edgeList=[];   // {source,target,data,status,condId}
  var seenEdge=new Set();

  /* 1) matched paths (colored by condition) */
  CONDS.forEach(function(c){
    var cp=pa[c.id];
    if(cp && cp.matched && cp.path.length>1){
      cp.path.forEach(function(n){ nodeSet.add(n); });
      for(var i=0;i<cp.path.length-1;i++){
        var u=cp.path[i], v=cp.path[i+1], ek=u+"|"+v;
        if(seenEdge.has(ek)) continue;
        seenEdge.add(ek);
        var ed=GRAPH.edges.find(function(e){return e.source===u && e.target===v;});
        edgeList.push({source:u, target:v, data:ed||{}, status:"match", condId:c.id});
      }
    }
  });

  /* 2) excluded conditions: filtered numeric edges anchor->pid (red dashed),
        plus not-reachable categorical anchors (red node) */
  var exclInfo=[];
  if(showExcl){
    CONDS.forEach(function(c){
      var cp=pa[c.id];
      if(!cp || cp.matched) return;
      if(c.kind==="numeric"){
        var found=false;
        GRAPH.edges.forEach(function(e){
          if(e.source===c.anchor && e.target===pid && !edgeSatisfies(e,c)){
            nodeSet.add(c.anchor);
            var ek=e.source+"|"+e.target+"|excl";
            if(!seenEdge.has(ek)){
              seenEdge.add(ek);
              edgeList.push({source:c.anchor, target:pid, data:e, status:"excluded", condId:c.id});
            }
            found=true;
          }
        });
        if(found){
          exclInfo.push(c.id+": "+c.anchor+" "+c.operator+" "+c.value+" FAIL");
        } else {
          nodeSet.add(c.anchor);
          exclInfo.push(c.id+": "+c.anchor+" NOT REACHABLE");
        }
      } else {
        nodeSet.add(c.anchor);
        exclInfo.push(c.id+": "+c.anchor+" NOT REACHABLE");
      }
    });
  }

  var anchorSet=new Set(CONDS.map(function(c){return c.anchor;}));
  var matchedAnchors=new Set(CONDS.filter(function(c){return pa[c.id]&&pa[c.id].matched;}).map(function(c){return c.anchor;}));
  var snodes=Array.from(nodeSet).map(function(id){
    var nd=nodeById(id);
    var s=STYLE[nd.type]||STYLE.intermediate;
    var style;
    if(id===pid){
      style={size:30,shape:"dot",color:{background:"#5fd36a",border:"#fff",highlight:{background:"#5fd36a",border:"#fff"}},borderWidth:3,font:{color:"#fff",size:15}};
    } else if(anchorSet.has(id) && matchedAnchors.has(id)){
      style={size:s.size+8,shape:s.shape,color:{background:s.color,border:"#ffe14a",highlight:{background:s.color,border:"#ffe14a"}},borderWidth:3,font:{color:"#fff",size:13}};
    } else if(anchorSet.has(id)){
      style={size:s.size+6,shape:s.shape,color:{background:"#3a2030",border:"#e06a6a",highlight:{background:"#5a3040",border:"#e06a6a"}},borderWidth:2,font:{color:"#e06a6a",size:12}};
    } else {
      style={size:12,shape:"dot",color:{background:"#4a5a6a",border:"#3a4a5a",highlight:{background:"#6a7a8a",border:"#5a6a7a"}},font:{color:"#b9c8d8",size:10}};
    }
    return Object.assign({id:id,label:id,title:id+" ("+nd.type+")"},style);
  });

  var sedges=edgeList.map(function(e,i){
    var d=e.data;
    var cond=CONDS.find(function(c){return c.id===e.condId;})||{};
    var color, dashes, fontColor, label, titleParts=[];
    if(d.relation_type) titleParts.push("relation: "+d.relation_type);
    if(d.value!==undefined) titleParts.push("value: "+d.value+" "+(d.unit||""));
    if(d.time) titleParts.push("time: "+d.time);
    if(e.status==="excluded"){
      color={color:"#e06a6a",highlight:"#e06a6a",opacity:0.9};
      dashes=true; fontColor="#e06a6a";
      label=(d.relation_type||"")+"\n"+d.value+" "+(d.unit||"")+"\nFAIL "+d.value+cond.operator+cond.value;
      titleParts.push("STATUS: EXCLUDED ("+d.value+" "+cond.operator+" "+cond.value+" = false)");
    } else {
      var cc=COND_COLORS[e.condId]||"#4a90d9";
      color={color:cc,highlight:cc}; dashes=false; fontColor="#e0c34a";
      if(d.value!==undefined){
        label=(d.relation_type||"")+"\n"+d.value+" "+(d.unit||"");
        if(d.time) label+=" ("+d.time+")";
      } else {
        label=d.relation_type||"";
      }
      titleParts.push("STATUS: MATCH ("+cond.id+")");
    }
    return {
      id:"se"+i, from:e.source, to:e.target,
      label:label, arrows:{to:{enabled:true, scaleFactor:0.6}},
      color:color, dashes:dashes,
      font:{size:11, color:fontColor, strokeWidth:0},
      title:titleParts.join("\n")
    };
  });

  if(subNet) subNet.destroy();
  subNet=new vis.Network(
    document.getElementById("subGraph"),
    {nodes:new vis.DataSet(snodes), edges:new vis.DataSet(sedges)},
    {
      layout:{hierarchical:{enabled:true, direction:"LR", sortMethod:"directed", levelSeparation:150, nodeSpacing:50}},
      edges:{smooth:{type:"cubicBezier", roundness:0.4}},
      physics:{enabled:false},
      interaction:{hover:true, tooltipDelay:80}
    }
  );

  /* sub-legend summary */
  var matched=CONDS.filter(function(c){return pa[c.id]&&pa[c.id].matched;}).map(function(c){return c.id;});
  var leg=document.getElementById("subLegend");
  leg.innerHTML="<b style='color:#5fd36a'>"+pid+"</b> &mdash; "+
    "<b>"+matched.length+"/"+CONDS.length+"</b> matched ("+matched.join(", ")+")"+
    (exclInfo.length? " &nbsp;|&nbsp; <b style='color:#e06a6a'>"+exclInfo.length+" excluded</b> ("+exclInfo.join("; ")+")" : "");
}

/* ---------- path explanation ---------- */
function renderExplanation(pid){
  if(!currentAcc) return;
  var pa=currentAcc[pid];
  var box=document.getElementById("explanation");
  var cov=coverageOf(pa);
  box.innerHTML='<div class="pblock"><span class="ph">Patient '+pid+' &mdash; coverage '+cov+'/'+CONDS.length+'</span></div>';
  CONDS.forEach(function(c){
    var cp=pa[c.id]||{matched:false};
    var b=document.createElement("div");
    b.className="pblock"+(cp.matched?"":" fail");
    var chain=(cp.path && cp.path.length)? cp.path.join(" \u2192 ") : "\u2014";
    var meta="";
    if(c.kind==="numeric") meta+="constraint: "+c.operator+" "+c.value;
    if(cp.distance) meta+=(meta?" | ":"")+cp.distance+" hop(s)";
    if(cp.reason) meta+=(meta?" | ":"")+cp.reason;
    var status=cp.matched?"MATCH":(cp.reason||"NOT MATCH");
    b.innerHTML=
      '<div class="ph">'+c.id+" \u00b7 "+c.anchor+
        (c.kind==="numeric"?" "+c.operator+" "+c.value:"")+
        " &mdash; "+status+'</div>'+
      '<div class="chain">'+chain+'</div>'+
      '<div class="meta">'+meta+'</div>';
    box.appendChild(b);
  });
}

/* ---------- filter report ---------- */
function renderFilterReport(conds){
  var grid=document.getElementById("filterReport");
  grid.innerHTML="";
  var numConds=conds.filter(function(c){return c.kind==="numeric";});
  if(!numConds.length){ grid.innerHTML='<span style="color:#7a8da0">No numeric conditions.</span>'; return; }
  GRAPH.edges.forEach(function(e){
    var c=numConds.find(function(c){return c.anchor===e.source;});
    if(!c) return;
    var ok=edgeSatisfies(e,c);
    var card=document.createElement("div");
    card.className="fcard"+(ok?"":" fail");
    card.innerHTML=
      '<span class="fa">'+e.source+'</span> \u2192 <b>'+e.target+'</b>'+
      '<br>value: <b>'+e.value+'</b> '+(e.unit||"")+
      ' &nbsp; '+c.operator+' '+c.value+
      '<br>'+(ok?'<span class="match">MATCH</span>':'<span class="nomatch">FAIL</span>')+
      (ok?'':' ('+e.value+' '+c.operator+' '+c.value+' = false)');
    grid.appendChild(card);
  });
}

/* ---------- legend ---------- */
function legend(){
  var l=document.getElementById("legend");
  l.innerHTML=Object.keys(STYLE).map(function(k){
    return '<span class="item"><span class="swatch" style="background:'+STYLE[k].color+'"></span>'+k+'</span>';
  }).join("");
}

/* ---------- init ---------- */
window.addEventListener("DOMContentLoaded", function(){
  buildQueryPanel();
  legend();
  initFull();
  document.getElementById("btnRun").onclick=runRetrieval;
  document.getElementById("btnFit").onclick=function(){ if(fullNet) fullNet.fit({animation:true}); };
  document.getElementById("patientSel").onchange=function(e){ selectPatient(e.target.value); };
  document.getElementById("showExcluded").onchange=function(){ if(currentPid) renderSubgraph(currentPid); };
  runRetrieval();
});
</script>
</body>
</html>
"""


def build() -> str:
    g = load_graph_data()
    gt = load_ground_truth()
    conds = load_conditions()
    from collections import Counter
    deg = Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    for n in g["nodes"]:
        n["degree"] = deg.get(n["id"], 0)
    html = (HTML_TEMPLATE
            .replace("__GRAPH__", json.dumps(g, ensure_ascii=False))
            .replace("__CONDS__", json.dumps(conds, ensure_ascii=False))
            .replace("__GT__", json.dumps(gt, ensure_ascii=False))
            .replace("__STYLE__", json.dumps(NODE_STYLE, ensure_ascii=False)))
    return html


if __name__ == "__main__":
    content = build()
    for path in (OUT, OUT_ROOT):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_ROOT}  (root index for `python -m http.server`)")
    print(f"Open:  http://localhost:1001/")
