"""FP32 agreement gates and provider evidence, independent of training or timing."""
import collections
import numpy as np
from metrics_v1 import confusion_matrix_numpy, scores_from_confusion, boundary_iou_author

LIMITS=dict(logit_rtol=1e-3,logit_atol=1e-4,argmax_disagreement_fraction=.001,metric_difference_pp=.05)


def provider_evidence(events,expected):
    nodes=[e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
    if not nodes: raise ValueError('Profile contains no actual node-provider evidence')
    counts=collections.Counter(e['args']['provider'] for e in nodes)
    if set(counts)!={expected}: raise ValueError('Unexpected execution provider: '+str(dict(counts)))
    return dict(provider=expected,node_executions=dict(counts),unique_node_names=len({e['name'] for e in nodes}))


class Agreement:
    def __init__(self):
        self.confusion={k:np.zeros((2,2),dtype=np.int64) for k in ('pytorch','onnx')}
        self.boundary={k:[] for k in self.confusion}
        self.records=[]

    def add(self,reference,exported,target,image_id):
        if reference.shape!=exported.shape or reference.ndim!=4 or reference.shape[:2]!=(1,2):
            raise ValueError('Expected matching NCHW binary logits, batch1')
        if target.shape!=reference.shape[2:]: raise ValueError('Target shape mismatch')
        if not np.isin(target,[0,1]).all(): raise ValueError('Core target must be binary')
        if not np.isfinite(reference).all() or not np.isfinite(exported).all(): raise ValueError('Nonfinite logits')
        if any(r['image_id']==image_id for r in self.records): raise ValueError('Repeated image ID')
        predictions={'pytorch':reference.argmax(1)[0],'onnx':exported.argmax(1)[0]}
        item=dict(image_id=image_id,pixels=target.size,
            maximum_absolute_logit_error=float(np.max(np.abs(reference-exported))),
            logits_allclose=bool(np.allclose(exported,reference,rtol=LIMITS['logit_rtol'],atol=LIMITS['logit_atol'])),
            argmax_disagreement_pixels=int(np.count_nonzero(predictions['pytorch']!=predictions['onnx'])))
        for backend,prediction in predictions.items():
            matrix=confusion_matrix_numpy(prediction,target)
            value=boundary_iou_author(target==1,prediction==1,width_pixels=26)
            self.confusion[backend]+=matrix
            if value is not None: self.boundary[backend].append(value)
            item[backend]=dict(confusion=matrix.tolist(),boundary_iou=value)
        self.records.append(item)

    def summary(self):
        if not self.records: raise ValueError('No evaluated images')
        metrics={}
        for backend,matrix in self.confusion.items():
            values=self.boundary[backend]
            metrics[backend]=dict(confusion=matrix.tolist(),sidewalk_iou_pct=100*scores_from_confusion(matrix)['iou'][1],
                boundary_iou_pct=100*float(np.mean(values)) if values else None,boundary_valid_images=len(values),
                boundary_excluded_images=len(self.records)-len(values))
        changes={}
        for metric in ('sidewalk_iou_pct','boundary_iou_pct'):
            reference,actual=metrics['pytorch'][metric],metrics['onnx'][metric]
            changes[metric]=abs(actual-reference) if actual is not None and reference is not None else (0. if reference is actual is None else None)
        fraction=sum(r['argmax_disagreement_pixels'] for r in self.records)/sum(r['pixels'] for r in self.records)
        checks=dict(logits_allclose=all(r['logits_allclose'] for r in self.records),argmax_disagreement=fraction<=LIMITS['argmax_disagreement_fraction'],
            regional_metrics=all(v is not None and v<=LIMITS['metric_difference_pp'] for v in changes.values()))
        return dict(status='pass' if all(checks.values()) else 'fail',images=len(self.records),limits=LIMITS,
            checks=checks,metrics=metrics,absolute_metric_difference_pp=changes,
            argmax_disagreement_fraction=fraction,maximum_absolute_logit_error=max(r['maximum_absolute_logit_error'] for r in self.records))
