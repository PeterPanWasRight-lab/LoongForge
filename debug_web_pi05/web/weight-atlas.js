/* All tensor thumbnails, grouped into the same architecture modules. */
(()=>{
let report=null;
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const limit=g=>g?Math.max(0,...g.flat().filter(finite).map(Math.abs)):0;
const fmt=v=>v===0?'0':v.toExponential(2);
function grids(r){const a=r.thumbnail?.a,b=r.thumbnail?.b;return {a,b,delta:r.status==='matched'&&a&&b?a.map((row,i)=>row.map((v,j)=>finite(v)&&finite(b[i][j])?v-b[i][j]:null)):null};}
document.addEventListener('weight-report',e=>{report=e.detail;render();});
for(const id of ['atlas-view','atlas-scale','atlas-size'])$(id).onchange=render;
function render(){
 const root=$('weight-atlas');root.replaceChildren();if(!report)return;
 root.classList.toggle('atlas-ultra',$('atlas-size').value==='ultra');
 root.classList.toggle('atlas-single',$('atlas-view').value!=='all');
 root.classList.toggle('atlas-large',$('atlas-size').value==='normal');
 const view=$('atlas-view').value,uniform=$('atlas-scale').value==='module';
 const keys=view==='all'?['a','b','delta']:[view],labels={a:'A',b:'B',delta:'A−B'};
 $('atlas-count').textContent=`${report.modules.length} 个模块 · ${report.rows.length} 个参数 · 全部展示`;
 for(const module of report.modules){
  const rows=report.rows.filter(r=>r.group===module.key).sort((a,b)=>a.key.localeCompare(b.key,undefined,{numeric:true}));
  const maps=rows.map(grids);let common=0,change=0;
  if(uniform)for(const g of maps){common=Math.max(common,limit(g.a),limit(g.b));change=Math.max(change,limit(g.delta));}
  const section=node('details','atlas-module');section.open=true;
  section.append(node('summary','',$('atlas-size').value==='ultra'?module.label:`${module.label} · ${rows.length} 个参数`));const tiles=node('div','atlas-grid');
  rows.forEach((r,i)=>{const g=maps[i],max=uniform?common:Math.max(limit(g.a),limit(g.b)),diff=uniform?change:limit(g.delta);
   const tile=node('button','atlas-tile');tile.type='button';tile.setAttribute('aria-label',r.key+'，点击查看详细热图');tile.disabled=!r.a;tile.title=`${r.key}\n形状 ${(r.shape||[]).join(' × ')}\nA/B 色阶 ±${fmt(max)}；A−B 色阶 ±${fmt(diff)}\n点击打开参数大图`;
   const name=r.key.replace(/^paligemma_with_expert\.(paligemma\.model\.|gemma_expert\.model\.)/,'').replace(/^vision_tower\.vision_model\./,'');
   tile.append(node('span','atlas-name',name));const strip=node('div','atlas-strip');
   for(const key of keys){const holder=node('div','atlas-panel');holder.append(node('small','',labels[key]));const grid=g[key];
    if(!grid?.length){holder.append(node('span','atlas-missing',key==='delta'&&r.status==='shape_mismatch'?'形状不同':r.error?'不支持':'无数据'));}
    else {const canvas=node('canvas');canvas.width=48;canvas.height=48;canvas.setAttribute('aria-label',`${r.key} ${labels[key]} 热图`);const ctx=canvas.getContext('2d'),bound=key==='delta'?diff:max;
     grid.forEach((row,y)=>row.forEach((v,x)=>{const t=finite(v)&&bound?v/bound:0,n=Math.min(1,Math.abs(t));ctx.fillStyle=!finite(v)?'#ad7cad':t>=0?`rgb(${Math.round(80+165*n)},${Math.round(90-35*n)},${Math.round(105-45*n)})`:`rgb(${Math.round(80-35*n)},${Math.round(90+65*n)},${Math.round(105+140*n)})`;ctx.fillRect(x*48/row.length,y*48/grid.length,48/row.length+.1,48/grid.length+.1);}));holder.append(canvas);}
    strip.append(holder);
   }
   tile.append(strip,node('small','atlas-range',`权重 ±${fmt(max)} · Δ ±${fmt(diff)}`));tile.onclick=()=>document.dispatchEvent(new CustomEvent('weight-open-tensor',{detail:r.key}));tiles.append(tile);
  });section.append(tiles);root.append(section);
 }
}
})();
