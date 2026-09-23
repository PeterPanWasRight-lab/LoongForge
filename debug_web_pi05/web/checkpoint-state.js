(()=>{
const format=x=>typeof x==='number'?(x===0?'0':Number.isInteger(x)?x.toLocaleString():x.toExponential(3)):'—';
document.querySelector('[data-tab="checkpoint-state"]').addEventListener('click',()=>{if(!$('state-path').value)$('state-path').value=inventory.checkpoints?.find(x=>x.resumable)?.path||inventory.checkpoints?.find(x=>x.step!=null)?.path||$('weight-a').value||inventory.checkpoints?.[0]?.path||'';});
$('state-load').onclick=async()=>{const b=$('state-load');b.disabled=true;$('state-result').replaceChildren();$('state-message').textContent='正在只读检查状态文件…';try{
 const d=await api('/api/checkpoint-state',{path:$('state-path').value.trim()}),root=$('state-result'),meta=d.meta['resume_meta.json']||{};
 root.append(node('p','path','实际检查目录：'+d.path));
 const totals=node('div','summary-grid');totals.append(tile('Completed step',meta.completed_steps??'未记录'),tile('Epoch',meta.epoch??'未记录'),tile('保存 world size',meta.world_size??'未记录'),tile('状态文件数',d.files.length));root.append(totals);
 if(!d.meta['resume_meta.json'])root.append(node('p','notice','此目录没有 resume_meta.json，无法确认训练步数或 epoch。目录名称不作为真实进度；请选择原生 checkpoint。'));
 if(!d.files.length){const candidates=(inventory.checkpoints||[]).filter(x=>x.path!==d.path&&(x.resumable||x.step!=null));if(candidates.length){const card=node('div','card');card.append(node('h2','','其他原生 checkpoint（独立目录，不代表当前权重的来源）'));for(const c of candidates){const button=node('button','run',`${c.name} · 已记录 ${c.step??'未知'} 步 · ${c.resumable?'发现续训文件，需检查内容':'有进度元数据'}`);button.append(node('small','',c.path));button.onclick=()=>{$('state-path').value=c.path;$('state-load').click();};card.append(button);}root.append(card);}}
 for(const note of d.notes)root.append(node('p','notice',note));
 const files=node('div','card');files.append(node('h2','','保存状态清单'));for(const f of d.files)files.append(node('p','path',`${f.name} · ${(f.bytes/1048576).toFixed(2)} MiB`));if(!d.files.length)files.append(node('p','','未发现 Optimizer / Scheduler / RNG / DataLoader 状态；不能据此恢复完整训练状态。'));root.append(files);
 const moments=node('div','card');moments.append(node('h2','','Adam / Master Parameters · 抽样统计'));
 if(d.moments.length){const wrap=node('div','weight-table-wrap'),table=node('table'),head=node('tr');for(const text of ['状态路径（可能为参数 ID）','类型','形状 / 精度','RMS','最大绝对值','零值比例','非有限采样点'])head.append(node('th','',text));table.append(head);for(const r of d.moments){const tr=node('tr');for(const v of [r.key,r.kind,`${r.shape.join('×')} / ${r.dtype}`,format(r.rms),format(r.max_abs),r.zero_fraction==null?'—':(100*r.zero_fraction).toFixed(2)+'%',String(r.nonfinite)])tr.append(node('td','',v));table.append(tr);}wrap.append(table);moments.append(wrap);}else moments.append(node('p','','没有可读取的 Adam / Master Parameters 数值，不能解释为零，也不能从权重反推。'));root.append(moments);
 for(const [name,value] of Object.entries({...d.meta,...d.sections,'Optimizer step（按状态路径）':d.optimizer_steps})){const card=node('details','card');card.append(node('summary','',name+' · 状态详情'));const pre=node('pre','',JSON.stringify(value,null,2));pre.style.whiteSpace='pre-wrap';pre.style.overflowWrap='anywhere';card.append(pre);root.append(card);}
 $('state-message').textContent=d.files.length?'检查完成 · 详见各文件读取结果。':'检查完成：未发现优化器等训练状态文件；不是加载出零值。';
 }catch(e){$('state-message').textContent=e.message;}finally{b.disabled=false;}};
})();
