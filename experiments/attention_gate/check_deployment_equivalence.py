"""Synthetic checks for false acceptance, aggregate metric weighting and EP fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import deployment_equivalence as gates


def logits(pred):
    return np.stack([pred==0,pred==1],axis=0)[None].astype(np.float32)


def main(a):
    assert not a.output.exists()
    checks=[]
    zeros=np.zeros((2,2),dtype=np.int64)
    mixed=np.array([[1,1],[0,0]],dtype=np.int64)
    exact=gates.Agreement()
    exact.add(logits(zeros),logits(zeros),zeros,'empty')
    exact.add(logits(mixed),logits(mixed),mixed,'mixed')
    s=exact.summary()
    assert s['status']=='pass' and s['metrics']['pytorch']['boundary_excluded_images']==1
    assert s['metrics']['pytorch']['sidewalk_iou_pct']==100.
    checks.append('exact_logits_and_empty_boundary_exclusion')
    failing=gates.Agreement()
    failing.add(logits(mixed),logits(zeros),mixed,'wrong')
    s=failing.summary()
    assert s['status']=='fail' and not any(s['checks'].values())
    checks.append('wrong_labels_fail_all_three_gates')
    different=gates.Agreement()
    different.add(logits(mixed),logits(mixed)+.01,mixed,'same_argmax_wrong_logits')
    assert different.summary()['status']=='fail' and different.summary()['argmax_disagreement_fraction']==0
    checks.append('argmax_identity_does_not_hide_logit_failure')
    weighted=gates.Agreement()
    weighted.add(logits(mixed),logits(mixed),mixed,'first')
    weighted.add(logits(mixed),logits(mixed),zeros,'second')
    assert weighted.summary()['metrics']['pytorch']['confusion']==[[4,2],[0,2]]
    assert weighted.summary()['metrics']['pytorch']['sidewalk_iou_pct']==50.
    checks.append('global_confusion_aggregation_not_mean_image_iou')
    for name,value in [('nan',np.nan),('inf',np.inf)]:
        bad=logits(mixed); bad[0,0,0,0]=value
        try: gates.Agreement().add(logits(mixed),bad,mixed,name)
        except ValueError: checks.append(name+'_rejected')
        else: raise AssertionError('Nonfinite accepted')
    try: exact.add(logits(zeros),logits(zeros),zeros,'empty')
    except ValueError: checks.append('duplicate_image_rejected')
    else: raise AssertionError('Duplicate image accepted')
    good=[dict(cat='Node',name='conv',args=dict(provider='CUDAExecutionProvider'))]
    assert gates.provider_evidence(good,'CUDAExecutionProvider')['node_executions']=={'CUDAExecutionProvider':1}
    for events in ([],good+[dict(cat='Node',name='fallback',args=dict(provider='CPUExecutionProvider'))]):
        try: gates.provider_evidence(events,'CUDAExecutionProvider')
        except ValueError: checks.append('missing_or_fallback_provider_rejected')
        else: raise AssertionError('Incorrect provider accepted')
    a.output.mkdir(parents=True)
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(gates.__file__)]}
    (a.output/'verification.json').write_text(json.dumps(dict(status='pass',checks=checks,source_hashes=sources,
        real_model_validation=False,full500_verified=False,GPU_used=False),indent=2)+'\n',encoding='utf-8')
    print('PASS',len(checks),'synthetic acceptance and rejection checks')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
