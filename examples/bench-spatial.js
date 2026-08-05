// Benchmark: spatial join, 1M points against 200 zones.
//
//   node examples/bench-spatial.js
//
// Build first:
//   node bin/smp.js build examples/spatial-join.js --out examples/build --memory 400

import { assignZones } from "./spatial-join.js";
const { load } = await import(new URL("./build/spatial-join.js", import.meta.url));

const N = 1_000_000, NPOLY = 200, NV = 40;
let a = 424242 >>> 0;
const rng = () => { a|=0; a=(a+0x6d2b79f5)|0; let t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296; };

// 200 irregular zones, ~40 vertices each -- the shape of real administrative
// boundaries rather than convex blobs.
const verts=[], polyOff=new Int32Array(NPOLY), polyLen=new Int32Array(NPOLY), bbox=new Float64Array(NPOLY*4);
for (let p=0;p<NPOLY;p++){
  const cx=rng()*360-180, cy=rng()*140-70, r=0.5+rng()*3;
  polyOff[p]=verts.length; polyLen[p]=NV;
  let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  for(let v=0;v<NV;v++){
    const th=2*Math.PI*v/NV, rr=r*(0.6+0.8*rng());
    const x=cx+rr*Math.cos(th), y=cy+rr*Math.sin(th);
    verts.push(x,y); x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);
  }
  bbox[p*4]=x0;bbox[p*4+1]=x1;bbox[p*4+2]=y0;bbox[p*4+3]=y1;
}
const V=Float64Array.from(verts);
const px=new Float64Array(N), py=new Float64Array(N);
for(let i=0;i<N;i++){
  if(rng()<0.6){ const p=(rng()*NPOLY)|0; px[i]=bbox[p*4]+rng()*(bbox[p*4+1]-bbox[p*4]); py[i]=bbox[p*4+2]+rng()*(bbox[p*4+3]-bbox[p*4+2]); }
  else { px[i]=rng()*360-180; py[i]=rng()*140-70; }
}

const med=x=>[...x].sort((u,v)=>u-v)[x.length>>1];
const time=(f,w=1,it=5)=>{for(let i=0;i<w;i++)f();const s=[];for(let i=0;i<it;i++){const t=performance.now();f();s.push(performance.now()-t);}return med(s);};

const outJs=new Int32Array(N);
assignZones(px,py,N,V,polyOff,polyLen,bbox,NPOLY,outJs);
const hits=outJs.reduce((c,v)=>c+(v>=0?1:0),0);
console.log(`=== spatial join: ${N.toLocaleString()} points vs ${NPOLY} zones (~${NV} vertices each) ===`);
console.log(`${hits} points landed inside a zone (${(100*hits/N).toFixed(1)}%)\n`);
const tJs=time(()=>assignZones(px,py,N,V,polyOff,polyLen,bbox,NPOLY,outJs));

const rows=[];
for(const threads of [0,2,4,8]){
  const mod=await load(threads>1?{threads}:{});
  const _px=mod.alloc.f64(N),_py=mod.alloc.f64(N),_v=mod.alloc.f64(V.length);
  const _off=mod.alloc.i32(NPOLY),_len=mod.alloc.i32(NPOLY),_bb=mod.alloc.f64(NPOLY*4),_out=mod.alloc.i32(N);
  _px.view.set(px);_py.view.set(py);_v.view.set(V);_off.view.set(polyOff);_len.view.set(polyLen);_bb.view.set(bbox);
  _out.view.fill(-999);
  const call=threads>1
    ? ()=>mod.parallel.assignZones(N,_px.ptr,_py.ptr,N,_v.ptr,_off.ptr,_len.ptr,_bb.ptr,NPOLY,_out.ptr)
    : ()=>mod.kernels.assignZones(_px.ptr,_py.ptr,N,_v.ptr,_off.ptr,_len.ptr,_bb.ptr,NPOLY,_out.ptr);
  call();
  let bad=0; for(let i=0;i<N;i++) if(_out.view[i]!==outJs[i]) bad++;
  rows.push({label:threads>1?`${threads}t`:'1t',ms:time(call),bad});
  await mod.destroy();
}
console.log(`  source as plain JS   ${tJs.toFixed(0).padStart(5)} ms`);
for(const r of rows) console.log(`  smp.js wasm ${r.label.padEnd(3)}      ${r.ms.toFixed(0).padStart(5)} ms   ${(tJs/r.ms).toFixed(2)}x vs JS   ${r.bad===0?'identical':'MISMATCH '+r.bad}`);
