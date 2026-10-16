#!/usr/bin/env python3
"""One train QA gradient smoke for each architecture, never a benchmark result."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import time

import torch

from run import choose_gpus
from vcrm.codec import SlotCodec, VersionMarkers
from vcrm.data import query_view
from vcrm.features import check, load
from vcrm.io import file_digest, read_json, write_json
from vcrm.reader import FrozenReader
from vcrm.training import temporal_pairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--gpu',default='4')
    parser.add_argument('--output',default='runs/architecture_v5_suite/gpu_smoke.json')
    args=parser.parse_args()
    gpu=choose_gpus(args.gpu)
    if len(gpu)!=1:raise ValueError('Exactly one idle GPU required')
    lock=open(f'/tmp/vcrm_icae_gpu{gpu[0]}.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu[0])
    cfg=read_json(args.config)|{'feature_root':'/share/latent2/vcrm_icae/features/paper_v1_official'}
    check(cfg,'train')
    episodes=read_json(Path(cfg['data_root'])/'prepared/templama_train.json')
    pairs=temporal_pairs(episodes,cfg['keyframe_interval'],'final_current_snapshot_v1')
    pair=pairs[0]
    episode=next(e for e in episodes if any(event['id']==pair['old']['id'] for event in e['events']))
    reader=FrozenReader(cfg)
    old,current=[load(cfg,'train',episode['id'],pair[k]).to(reader.device).float() for k in ('old','current')]
    definitions={'C0':{'conditioned':False},'D0':{'conditioned':True},
                 'E1':{'conditioned':True,'decode_conditioned':False},
                 'E2':{'conditioned':True,'decode_conditioned':False,'variant':'salience'},
                 'E3':{'conditioned':False,'value_path':'full'},
                 'FinalFullSalience':{'conditioned':True,'decode_conditioned':False,'variant':'salience','value_path':'full'}}
    results={}
    for name,options in definitions.items():
        torch.manual_seed(42)
        module=SlotCodec(**options).to(reader.device)
        markers=VersionMarkers().to(reader.device)
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        recovered=module(old,current)
        bundle=torch.cat((markers(current,'current',pair['current']['time']),
                          markers(recovered,'historical',pair['old']['time'])))
        loss=reader.qa_loss(query_view(pair['historical']),pair['historical']['answers'][0],bundle)
        loss.backward()
        parameters=list(module.parameters())+list(markers.parameters())
        valid=all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        nonzero=sum(float(p.grad.abs().sum()) for p in parameters)>0 if valid else False
        if not torch.isfinite(loss) or not valid or not nonzero:
            raise ValueError(f'Architecture QA gradients invalid: {name}')
        torch.cuda.synchronize()
        results[name]={'loss':float(loss.detach()),'codec_parameters':module.parameter_count(),
                       'finite_nonzero_gradients':True,'read_slots':int(bundle.shape[0]),
                       'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-started}
        print(name,results[name],flush=True)
        del module,markers,loss,bundle,recovered,parameters
        gc.collect();torch.cuda.empty_cache()
    write_json(args.output,{'complete':True,'scope':'one_training_QA_gradient_smoke_not_evaluation',
              'gpu':gpu[0],'reader_sha256':file_digest('vcrm/reader.py'),
              'results':results,'encoder_decoder_frozen':not any(p.requires_grad for p in reader.net.parameters())})


if __name__=='__main__':main()
