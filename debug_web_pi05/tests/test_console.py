import json,os,struct,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
_runtime=tempfile.TemporaryDirectory(prefix='pi05-tests-')
os.environ['PI05_CONSOLE_RUNTIME']=_runtime.name
from app.settings import settings
from app.plans import build_plan
from app.catalog import run_detail,run_log,weight_layout,permitted
from app.api import clean,dispatch
from app.jobs import Job,_status_from_log

class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.repo=self.root/'repo'; self.outputs=self.root/'outputs'; self.artifacts=self.root/'artifacts'
        for name in ['loongforge/embodied/train.py','examples/embodied/pi05/run_pi05_ddp_finetune.sh','examples/embodied/pi05/run_pi05_ddp_zero1_finetune.sh','examples/embodied/pi05/run_pi05_fsdp_finetune.sh']:
            p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        self.outputs.mkdir(); self.checkpoint=self.artifacts/'pi05_base';self.checkpoint.mkdir(parents=True)
        header=json.dumps({'model.test':{'dtype':'F32','shape':[1],'data_offsets':[0,4]},'pi05.test':{'dtype':'F32','shape':[1],'data_offsets':[4,8]}}).encode()
        (self.checkpoint/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*8)
        self.dataset=self.artifacts/'libero_10';meta=self.dataset/'meta';meta.mkdir(parents=True)
        (meta/'info.json').write_text(json.dumps({'codebase_version':'v3.0','features':{'action':{'shape':[7]},'observation.state':{'shape':[8]},'observation.images.image':{},'observation.images.wrist_image':{}}}))
        (meta/'stats.json').write_text(json.dumps({k:{'q01':[-1]*n,'q99':[1]*n} for k,n in [('action',7),('observation.state',8)]}))
        self.token=self.root/'token'; self.token.mkdir();(self.token/'tokenizer.json').write_text('{}')
        self.libero=self.root/'libero';self.libero.mkdir();(self.libero/'config.yaml').touch()
        self.patcher=patch.dict(settings._data,{'repo_root':str(self.repo),'python':sys.executable,'benchmark_python':sys.executable,'artifacts_root':str(self.artifacts),'output_root':str(self.outputs),'tokenizer':str(self.token),'libero_config':str(self.libero)})
        self.patcher.start(); self.params={'checkpoint':str(self.checkpoint),'dataset':str(self.dataset),'output':str(self.outputs/'new'),'gpus':'0','steps':100}
    def tearDown(self): self.patcher.stop();self.tmp.cleanup()
    def test_train_plan(self):
        plan=build_plan('train',{**self.params,'gpus':'0,1','batch':3,'accum':4})
        self.assertTrue(plan['ok']);self.assertEqual(plan['env']['GPUS_PER_NODE'],'2');self.assertIn('model.state_dim=8',plan['argv']);self.assertFalse((self.outputs/'new').exists())
    def test_eval_plan(self):
        plan=build_plan('eval',self.params);self.assertTrue(plan['ok'])
        self.assertEqual(plan['config']['model']['state_encoding'],'lib_eef_axis_angle');self.assertFalse(plan['config']['server']['use_bf16'])
        self.assertNotEqual(plan['config']['server']['port'],plan['config']['server']['health_port'])
    def test_invalid_inputs(self):
        for patch_ in ({'gpus':'0,0'},{'steps':1.5},{'lr':float('nan')},{'output':'/tmp/escape'},{'resume':'yes'}):
            with self.subTest(patch=patch_),self.assertRaises(ValueError):build_plan('train',{**self.params,**patch_})
    def test_existing_output_and_pi0(self):
        out=Path(self.params['output']);out.write_text('keep')
        self.assertFalse(build_plan('train',self.params)['ok']);self.assertEqual(out.read_text(),'keep')
        out.unlink();(self.checkpoint/'config.json').write_text('{"type":"pi0"}')
        self.assertFalse(build_plan('train',self.params)['ok'])
    def test_resume_requires_state_and_larger_target(self):
        checkpoint=Path(self.params['output'])/'checkpoints/steps_80';checkpoint.mkdir(parents=True)
        (checkpoint/'training_state.pt').touch()
        self.assertFalse(build_plan('train',{**self.params,'resume':True})['ok'])
        (checkpoint/'resume_meta.json').write_text('{"completed_steps":80}')
        self.assertTrue(build_plan('train',{**self.params,'resume':True})['ok'])
        self.assertFalse(build_plan('train',{**self.params,'resume':True,'steps':80})['ok'])
    def test_resume_does_not_fall_back_to_older_checkpoint(self):
        parent=Path(self.params['output'])/'checkpoints'
        old=parent/'steps_10';old.mkdir(parents=True);(old/'training_state.pt').touch();(old/'resume_meta.json').write_text('{"completed_steps":10}')
        latest=parent/'steps_20';latest.mkdir()
        self.assertFalse(build_plan('train',{**self.params,'resume':True})['ok'])
        (latest/'zero_optimizer').mkdir();(latest/'zero_optimizer/metadata.json').write_text('{}');(latest/'resume_meta.json').write_text('{"completed_steps":20}')
        self.assertTrue(build_plan('train',{**self.params,'resume':True})['ok'])

    def test_stats_and_camera_validation(self):
        path=self.dataset/'meta/stats.json';path.write_text('{"action":{"q01":[NaN],"q99":[1]}}')
        self.assertFalse(build_plan('eval',self.params)['ok'])
    def test_metrics_results_log_and_containment(self):
        run=self.outputs/'run';run.mkdir()
        (run/'metrics.jsonl').write_text('{"step":1,"action_loss":NaN}\npartial')
        records=[{'episode_id':'a','task_id':0,'success':0},{'episode_id':'a','task_id':0,'success':1},{'episode_id':'b','task_id':1,'success':0}]
        (run/'results.jsonl').write_text('\n'.join(json.dumps(x) for x in records))
        (run/'policy_server.log').write_text('hello\n')
        detail=clean(run_detail(run));self.assertEqual(detail['completed'],2);self.assertEqual(detail['successes'],1);self.assertIsNone(detail['latest']['action_loss'])
        self.assertEqual(run_log(run)['text'],'hello\n')
        link=run/'outside.json';link.symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):permitted(link)
        self.assertEqual(dispatch('GET','/api/runlog','path='+str(run),None).status,200)
    def test_native_gif_replay_is_listed_and_served(self):
        run=self.outputs/'replay';run.mkdir();gif=run/'replay_fail.gif';gif.write_bytes(b'GIF89a')
        detail=run_detail(run)
        self.assertIn(str(gif),[item['path'] for item in detail['files']])
        response=dispatch('GET','/api/file','path='+str(gif),None)
        self.assertEqual(response.status,200);self.assertEqual(response.content_type,'image/gif')

    def test_runner_success_failure_stop(self):
        for code in [0,7]:
            job=Job('train',[],cwd=str(self.root))
            manifest=job.dir/'launch.json';manifest.write_text(json.dumps({'argv':[sys.executable,'-c',f'print("hello",flush=True);raise SystemExit({code})'],'cwd':str(self.root),'env':{}}))
            job.argv=[sys.executable,str(ROOT/'workers/runner.py'),str(manifest)];job.start()
            deadline=time.time()+10
            while job.status in ('pending','running') and time.time()<deadline:time.sleep(.05)
            self.assertEqual(job.status,'succeeded' if code==0 else 'failed');self.assertEqual(_status_from_log(job)[1],code)
        job=Job('train',[sys.executable,'-c','import time; print("running",flush=True);time.sleep(60)'],cwd=str(self.root));job.start();time.sleep(.2);job.stop(timeout=1)
        self.assertEqual(job.status,'stopped')
    def test_runner_single_signal_cleanup(self):
        child=self.root/'child.py'
        child.write_text('import signal,time\ndef stop(s,f):\n print("cleanup-start",flush=True)\n time.sleep(.3)\n print("cleanup-complete",flush=True)\n raise SystemExit(130)\nsignal.signal(signal.SIGINT,stop)\nprint("ready",flush=True)\ntime.sleep(60)\n')
        job=Job('eval',[],cwd=str(self.root))
        manifest=job.dir/'launch.json';manifest.write_text(json.dumps({'argv':[sys.executable,str(child)],'cwd':str(self.root),'env':{}}))
        job.argv=[sys.executable,str(ROOT/'workers/runner.py'),str(manifest)];job.start()
        deadline=time.time()+5
        while 'ready' not in job.log_path.read_text() and time.time()<deadline:time.sleep(.05)
        job.stop(timeout=2)
        self.assertIn('cleanup-complete',job.log_path.read_text())
        self.assertEqual(job.log_path.read_text().count('cleanup-start'),1)

    def test_eval_pins_weights_before_slow_model_loading(self):
        from workers.eval_entry import prepare_checkpoint
        checkpoint=self.root/'pi05_raw';checkpoint.mkdir()
        header=json.dumps({'test':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        original=checkpoint/'model.safetensors';original.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
        raw=original.read_bytes();pin=self.root/'pin';pin.mkdir()
        config={'server':{'ckpt_path':str(checkpoint)}}
        prepare_checkpoint(config,sys.executable,pin)
        original.unlink();checkpoint.rmdir()
        self.assertEqual((Path(config['server']['ckpt_path'])/'model.safetensors').read_bytes(),raw)

    def test_state_mapping(self):
        from workers.eval_entry import encode_libero_state
        import numpy as np
        result=encode_libero_state({'state_raw':{'eef_pos':[1,2,3],'eef_quat':[0,0,1,0],'gripper_qpos':[.1,.2]}})
        np.testing.assert_allclose(result,[1,2,3,0,0,np.pi,.1,.2],rtol=1e-6)
        with self.assertRaises(ValueError):encode_libero_state({})

if __name__=='__main__':unittest.main()
