/* Summary consumes the existing analysis; no additional weight reads. */
(()=>{
let report=null;
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const fmt=v=>!finite(v)?'—':v===0?'0':Math.abs(v)<.001?v.toExponential(3):v.toFixed(5);
const pct=v=>finite(v)?`${(100*v).toFixed(3)}%`:'—';
const count=n=>Number(n||0).toLocaleString();
$('summary-configure').onclick=()=>document.querySelector('[data-tab="weights"]').click();
$('summary-metric').onchange=render;
document.addEventListener('weight-report',event=>{report=event.detail;render();});
function render(){
 $('summary-empty').classList.toggle('hidden',!!report);$('summary-content').classList.toggle('hidden',!report);
 if(!report)return;
 const hasB=!!report.b;
 for(const o of $('summary-metric').options)o.disabled=!hasB&&o.value!=='rms';
 if(!hasB)$('summary-metric').value='rms';
 const paired=report.rows.filter(r=>r.status==='matched'&&r.paired>0);
 const n=paired.reduce((s,r)=>s+r.numel,0);
 const dd=paired.reduce((s,r)=>s+r.dd*r.numel,0),bb=paired.reduce((s,r)=>s+r.bb*r.numel,0);
 const missing=report.rows.filter(r=>!['matched','single'].includes(r.status)).length;
 const nonfinite=report.modules.reduce((s,m)=>s+m.nonfinite_samples,0);
 const ar=report.rows.filter(r=>finite(r.a?.rms));const an=ar.reduce((s,r)=>s+r.numel,0);
 const arms=an?Math.sqrt(ar.reduce((s,r)=>s+r.a.rms**2*r.numel,0)/an):null;
 $('summary-paths').textContent=`A：${report.a.path}\nB：${report.b?.path||'未选择（仅分析 A）'}`;
 $('summary-totals').replaceChildren(tile('A 总参数量（去重）',count(report.a.parameters)),tile('整体 A RMS',fmt(arms)),tile('对齐覆盖 · A',hasB?pct(report.a.parameters?n/report.a.parameters:null):'—'),tile('整体变化 RMS',fmt(n?Math.sqrt(dd/n):null)),tile('整体相对变化',pct(n&&bb?Math.sqrt(dd/bb):null)),tile('未对齐 / 异常采样',`${missing} / ${nonfinite}`));
 const ranked=report.modules.filter(m=>finite(m.relative)).sort((a,b)=>b.relative-a.relative);
 $('summary-insight').textContent=(hasB&&ranked.length?`相对变化最大：${ranked[0].label}（${pct(ranked[0].relative)}）。`:'当前仅显示 A 的权重规模与分布。')+` ${!hasB?'选择 B 后可查看对齐和变化统计；':missing?'有 '+missing+' 项未对齐，详见下表；':'所有可比参数已对齐；'}NaN/Inf 采样 ${nonfinite} 个。整体值由参数统计加权计算，并非模块均值的平均。`;
 const key=$('summary-metric').value;
 const modules=[...report.modules].sort((a,b)=>(b[key]??-1)-(a[key]??-1));
 const max=Math.max(0,...modules.map(m=>finite(m[key])?m[key]:0));
 $('summary-bars').replaceChildren();
 for(const m of modules){const row=node('div','summary-bar-row');const track=node('div','summary-bar-track'),fill=node('div','summary-bar-fill');const v=m[key],t=finite(v)&&max?v/max:0;fill.style.width=`${t*100}%`;fill.style.background=`hsl(${170-130*t} 65% 57%)`;track.append(fill);row.append(node('span','',m.label),track,node('b','',key==='relative'?pct(v):fmt(v)));$('summary-bars').append(row);}
 const table=$('summary-table');table.replaceChildren();const head=node('tr');['模块','A 参数量 / 张量','A RMS','变化 RMS','相对 B 变化','余弦相似度','对齐参数量','未对齐 / 非有限值'].forEach(t=>head.append(node('th','',t)));table.append(head);
 for(const m of report.modules){const tr=node('tr');[m.label,`${count(m.parameters_a)} / ${m.tensors}`,fmt(m.rms),fmt(m.diff_rms),pct(m.relative),finite(m.cosine)?m.cosine.toFixed(8):'—',hasB?count(m.matched_parameters):'—',`${m.unmatched} / ${m.nonfinite_samples}`].forEach(t=>tr.append(node('td','',t)));if(m.unmatched||m.nonfinite_samples)tr.className='summary-attention';table.append(tr);}
}
})();
