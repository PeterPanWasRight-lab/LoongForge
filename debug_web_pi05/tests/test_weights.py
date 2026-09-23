import json,os,struct,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.settings import settings
from app import weights

def save(path, tensors):
    entries={};data=b''
    for name,(array,dtype) in tensors.items():
        array=np.asarray(array)
        raw=(array.astype('<f4').view('<u4')>>16).astype('<u2').tobytes() if dtype=='BF16' else array.astype(weights.DTYPES[dtype]).tobytes()
        entries[name]={'dtype':dtype,'shape':list(array.shape),'data_offsets':[len(data),len(data)+len(raw)]};data+=raw
    head=json.dumps(entries).encode();path.write_bytes(struct.pack('<Q',len(head))+head+data)

class WeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.patcher=patch.dict(settings._data,{'artifacts_root':str(self.root),'output_root':str(self.root)});self.patcher.start()
    def tearDown(self):self.patcher.stop();self.tmp.cleanup()
    def test_bf16_and_aliases_and_group(self):
        p=self.root/'a.safetensors';arr=np.array([[1,-2],[0,4]],dtype=np.float32)
        save(p,{'model.action_in_proj.weight':(arr,'BF16'),'pi05.action_in_proj.weight':(arr,'BF16')})
        with weights.Weights(p) as w:
            self.assertEqual(w.info()['aliases_removed'],1);self.assertEqual(w.info()['parameters'],4)
            np.testing.assert_equal(w.take('action_in_proj.weight',np.arange(4)),arr.ravel())
        self.assertEqual(weights.group('paligemma_with_expert.paligemma.model.vision_tower.x'),'vision')
        self.assertEqual(weights.group('paligemma_with_expert.gemma_expert.model.x'),'expert')
    def test_differences_zero_baseline_and_nonfinite(self):
        a=np.array([1.,2.,3.]);self.assertEqual(weights.compare(a,a)['diff_rms'],0)
        c=weights.compare(2*a,a);self.assertAlmostEqual(c['relative'],1);self.assertAlmostEqual(c['cosine'],1)
        self.assertIsNone(weights.compare(a,np.zeros(3))['relative'])
        self.assertEqual(weights.stats(np.array([1.,float('nan'),float('inf')]))['finite'],1)
    def test_window_coordinates_and_mismatches(self):
        a=self.root/'a.safetensors';b=self.root/'b.safetensors';arr=np.arange(30).reshape(5,6)
        save(a,{'action_in_proj.weight':(arr,'F32'),'time_mlp_in.weight':(np.ones(2),'F32')})
        save(b,{'model.action_in_proj.weight':(arr-1,'F32'),'time_mlp_in.weight':(np.ones(3),'F32'),'extra':(np.zeros(1),'F32')})
        result=weights.heatmap({'a':str(a),'b':str(b),'key':'action_in_proj.weight','rows':[1,3],'cols':[2,5]})
        self.assertEqual(result['a'],[[8.,9.,10.],[14.,15.,16.]]);self.assertEqual(result['delta'],[[1.]*3]*2)
        with weights.Weights(a) as wa,weights.Weights(b) as wb:
            self.assertEqual(weights.row_for(wa,wb,'time_mlp_in.weight')['status'],'shape_mismatch')
            self.assertEqual(weights.row_for(wa,wb,'extra')['status'],'only_b')
        with self.assertRaises(ValueError):weights.heatmap({'a':str(a),'key':'action_in_proj.weight','rows':[0,7]})
    def test_thumbnail_grid_and_single_sided_rows(self):
        a=self.root/'a.safetensors';b=self.root/'b.safetensors'
        array=np.arange(40*30,dtype=np.float32).reshape(40,30)
        save(a,{'x':(array,'F32')});save(b,{'x':(array-2,'F32'),'only_b':(np.arange(3),'F32')})
        with weights.Weights(a) as wa,weights.Weights(b) as wb:
            row=weights.row_for(wa,wb,'x');ga=row['thumbnail']['a'];gb=row['thumbnail']['b']
            self.assertEqual((len(ga),len(ga[0])),(16,16))
            self.assertEqual(ga[0][0],0);self.assertEqual(ga[-1][-1],1199)
            self.assertTrue(all(x-y==2 for ar,br in zip(ga,gb) for x,y in zip(ar,br)))
            missing=weights.row_for(wa,wb,'only_b')
            self.assertIsNone(missing['thumbnail']['a']);self.assertEqual(missing['thumbnail']['b'],[[0,1,2]])

    def test_file_deletion_after_open(self):
        p=self.root/'a.safetensors';save(p,{'x':(np.arange(5),'F32')})
        with weights.Weights(p) as w:
            p.unlink();np.testing.assert_equal(w.take('x',np.arange(5)),np.arange(5))
    def test_path_and_corrupt_header(self):
        with self.assertRaises(ValueError):weights.Weights('/etc/passwd')
        p=self.root/'bad.safetensors';p.write_bytes(struct.pack('<Q',1000))
        with self.assertRaises(ValueError):weights.Weights(p)
    def test_bounded_sampling(self):
        indices=weights.sample_indices(1_000_000_000)
        self.assertEqual(len(indices),4096);self.assertEqual(indices[0],0);self.assertEqual(indices[-1],999999999)
    def test_async_analysis_and_module_weighting(self):
        a=self.root/'a.safetensors';b=self.root/'b.safetensors'
        save(a,{'action_in_proj.bias':(np.ones(4)*2,'F32'),'action_out_proj.bias':(np.ones(12)*4,'F32')})
        save(b,{'action_in_proj.bias':(np.ones(4),'F32'),'action_out_proj.bias':(np.ones(12)*2,'F32')})
        ident=weights.launch({'a':str(a),'b':str(b)})['id'];deadline=time.time()+5
        while (d:=weights.status(ident))['status']=='running' and time.time()<deadline:time.sleep(.01)
        self.assertEqual(d['status'],'done');m=d['result']['modules'][0]
        self.assertAlmostEqual(m['relative'],1);self.assertAlmostEqual(m['rms'],np.sqrt(13))
        json.dumps(d,allow_nan=False)
    def test_cancellation_releases_reader(self):
        a=self.root/'a.safetensors';save(a,{f'x{i}':(np.ones(2),'F32') for i in range(20)})
        original=weights.row_for
        def slower(*args):time.sleep(.01);return original(*args)
        with patch.object(weights,'row_for',slower):
            ident=weights.launch({'a':str(a)})['id'];weights.status(ident,cancel=True)
            deadline=time.time()+5
            while (d:=weights.status(ident))['status'] in ('running','cancelling') and time.time()<deadline:time.sleep(.01)
            self.assertEqual(d['status'],'cancelled')

if __name__=='__main__':unittest.main()
