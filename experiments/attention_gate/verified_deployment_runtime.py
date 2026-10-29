"""Reuse accepted FP32 execution settings in the demo, with an exclusive GPU lease."""
import hashlib
import json
import msvcrt
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from deployment_equivalence import provider_evidence
from deployment_pipeline import HEIGHT, WIDTH, MEAN, STD

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VerifiedRuntime:
    def __init__(self, manifest_path, backend, evidence_dir):
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest_hash=digest(self.manifest_path)
        self.config=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        assert (self.config['height'],self.config['width'])==(HEIGHT,WIDTH)
        assert self.config['mean']==MEAN and self.config['std']==STD
        assert self.config['execution']==dict(disabled_optimizers=['ConvBNFusion','ConvActivationFusion'])
        self.models=self.config['models']; self.backend=backend
        self.provider='CUDAExecutionProvider' if backend=='cuda' else 'CPUExecutionProvider'
        assert self.provider in ort.get_available_providers()
        self.evidence_dir=Path(evidence_dir).resolve()
        self.sessions={}; self.proofs={}; self.lock=None
        accepted=json.loads((ROOT/'results/deployment_cuda_fp32_preserved_bn.json').read_text(encoding='utf-8'))
        assert accepted['status']=='pass' and accepted['full500_acceptance_passed']
        for item in self.models:
            row=next(r for r in accepted['results'] if r['variant']==item['variant'])
            assert item['seed']==42 and item['selected_epoch']==row['selected_epoch']
            assert digest(item['onnx'])==item['onnx_sha256']==row['onnx_sha256']
            assert item['weights_sha256']==row['weights_sha256']
        torch.set_num_threads(1)

    def __enter__(self):
        if self.backend=='cuda':
            self.lock=(HERE/'seed42_training.lock').open('a+b')
            try:
                self.lock.seek(0); msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            except Exception:
                self.lock.close(); self.lock=None; raise
        try:
            self.evidence_dir.mkdir(parents=True,exist_ok=False)
        except Exception:
            self.__exit__(None,None,None); raise
        return self

    def session_for(self,index):
        if index not in self.sessions:
            item=self.models[index]
            opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
            opts.enable_profiling=True; opts.profile_file_prefix=str(self.evidence_dir/item['name'])
            if self.backend=='cuda': opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
            providers=[(self.provider,{'use_tf32':0})] if self.backend=='cuda' else [self.provider]
            runtime=ort.InferenceSession(item['onnx'],sess_options=opts,providers=providers,
                disabled_optimizers=self.config['execution']['disabled_optimizers'])
            runtime.disable_fallback()
            assert runtime.get_providers()[0]==self.provider
            if self.backend=='cuda': assert runtime.get_provider_options()[self.provider]['use_tf32']=='0'
            assert runtime.get_inputs()[0].shape==[1,3,HEIGHT,WIDTH]
            assert runtime.get_outputs()[0].shape==[1,2,HEIGHT,WIDTH]
            assert runtime.get_inputs()[0].type==runtime.get_outputs()[0].type=='tensor(float)'
            self.sessions[index]=runtime
        return self.sessions[index]

    def infer(self,index,tensor):
        if tensor.shape!=(1,3,HEIGHT,WIDTH) or tensor.dtype!=np.float32 or not tensor.flags.c_contiguous:
            raise ValueError('Unexpected deployment input')
        session=self.session_for(index)
        output=session.run(['logits'],{'image':tensor})[0]
        if not np.isfinite(output).all(): raise ValueError('Nonfinite inference output')
        if index not in self.proofs:
            profile=Path(session.end_profiling())
            provider=provider_evidence(json.loads(profile.read_text(encoding='utf-8')),self.provider)
            self.proofs[index]=dict(provider_evidence=provider,profile_sha256=digest(profile),
                provider_options=session.get_provider_options(),model=self.models[index],
                disabled_optimizers=self.config['execution']['disabled_optimizers'])
        return output

    def __exit__(self,*args):
        try:
            assert digest(self.manifest_path)==self.manifest_hash
            for index in self.sessions:
                item=self.models[index]; assert digest(item['onnx'])==item['onnx_sha256']
        finally:
            self.sessions.clear()
            if self.lock is not None:
                self.lock.seek(0); msvcrt.locking(self.lock.fileno(),msvcrt.LK_UNLCK,1)
                self.lock.close(); self.lock=None
