import json,tempfile,unittest
from pathlib import Path
from app.state_reader import inspect
class StateReaderTest(unittest.TestCase):
 def test_weights_only(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'resume_meta.json').write_text(json.dumps({'completed_steps':24000,'epoch':1}))
   r=inspect(d)
   self.assertEqual(r['meta']['resume_meta.json']['completed_steps'],24000)
   self.assertFalse(r['moments']);self.assertFalse(r['files']);self.assertTrue(r['notes'])
 def test_adam_and_scheduler(self):
  import torch
  with tempfile.TemporaryDirectory() as d:
   torch.save({'optimizer':{'state':{0:{'exp_avg':torch.tensor([3.,4.]),'exp_avg_sq':torch.tensor([0.,4.]),'step':torch.tensor(7)}},'param_groups':[{'lr':0.001}]},'scheduler':{'last_epoch':7},'epoch':1},Path(d,'training_state.pt'))
   r=inspect(d)
   self.assertEqual(len(r['moments']),2)
   self.assertAlmostEqual(r['moments'][0]['rms'],(12.5)**.5)
   self.assertEqual(r['moments'][1]['zero_fraction'],.5)
   self.assertEqual(r['sections']['training_state.pt']['scheduler']['last_epoch'],7)
