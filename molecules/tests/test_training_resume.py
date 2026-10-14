"""Same-run epoch-boundary recovery must match uninterrupted AdamW training."""
from pathlib import Path
import importlib
import sys
import torch


def test_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    config = repo / 'configs/cpu-smoke.yaml'
    def command(module, *args):
        monkeypatch.setattr(sys, 'argv', [module, *map(str, args)])
        importlib.import_module('etflow.cli.'+module).main()
    data = tmp_path/'data';a=tmp_path/'interrupted';b=tmp_path/'continuous'
    command('example','--output-dir',data)
    args=['--config',config,'--data-dir',data,'--cpu']
    command('train',*args,'--run-dir',a,'--stop-after-epoch',1)
    command('train',*args,'--run-dir',a,'--resume',a/'checkpoints/last.ckpt')
    command('train',*args,'--run-dir',b)
    cks=[torch.load(p/'checkpoints/final.ckpt',map_location='cpu',weights_only=False) for p in [a,b]]
    assert cks[0]['global_step']==cks[1]['global_step']==4
    assert cks[0]['lr_schedulers']==cks[1]['lr_schedulers']
    for k in cks[0]['state_dict']:
        torch.testing.assert_close(cks[0]['state_dict'][k],cks[1]['state_dict'][k],rtol=0,atol=0)
    for index, values in cks[0]['optimizer_states'][0]['state'].items():
        for k,v in values.items():
            torch.testing.assert_close(v,cks[1]['optimizer_states'][0]['state'][index][k],rtol=0,atol=0)
