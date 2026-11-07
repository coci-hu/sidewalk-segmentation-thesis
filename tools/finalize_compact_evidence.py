"""Collect verified compact-model outcomes and produce a reproducible thesis figure."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from export_followup_summary import sanitize

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/attention_gate'
sys.path.insert(0,str(HERE))
from deployment_equivalence import provider_evidence

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    assert not p.exists(),p
    p.write_bytes((json.dumps(sanitize(v),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    demo=HERE/'step_13_2026-11-07/compact_demo_acceptance_001'
    proof=read(demo/'verification.json')
    assert proof['status']=='pass' and proof['GPU_lock_rejected_before_inference']
    assert proof['constructed_video']['frames']==32 and proof['constructed_video']['all_output_frames_decoded']
    assert proof['model_parameter_counts']=={'A':263640,'L0':161688}
    sources={str((demo/'verification.json').relative_to(ROOT)):sha(demo/'verification.json')}
    for p,h in proof['source_hashes'].items():assert sha(ROOT/p)==h
    media=[]
    for record in proof['image_cases']+[proof['constructed_video']]:
        path=Path(record['output']);side=path.with_suffix(path.suffix+'.json')
        assert sha(side)==record['report_sha256']
        report=read(side);assert report['speed_is_not_a_controlled_benchmark']
        for index,execution in report['execution_verified'].items():
            profiles=list(path.with_suffix(path.suffix+'.evidence').glob(execution['model']['name']+'*.json'))
            assert len(profiles)==1 and sha(profiles[0])==execution['profile_sha256']
            assert provider_evidence(read(profiles[0]),'CUDAExecutionProvider')==execution['provider_evidence']
            sources[str(profiles[0].relative_to(ROOT))]=sha(profiles[0])
        sources[str(side.relative_to(ROOT))]=sha(side)
        media.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),frames=len(report['frames'])))
    application=dict(status='pass',verification=proof,media=media,sources=sources,
        acceptance_scope='Real CLI image/video and GPU-lock rejection. No new GUI, natural-video accuracy, or L0 long-duration stability claim.')
    dump(ROOT/'results/compact_application_20261107.json',application)

    acc=read(ROOT/'results/l0_three_seed_review_20261107.json')
    timing=read(ROOT/'results/compact_benchmark_20261107.json')
    profile=read(ROOT/'results/lightweight_profile_20261101_v2.json')['results']
    assert acc['status']==timing['status']=='pass'
    a,l=timing['summary']['A'],timing['summary']['L0']
    changes=dict(params_reduction_pct=100*(1-profile['L0']['params']/profile['A']['params']),
        conv_linear_flops_reduction_pct=100*(1-profile['L0']['gflops_conv_linear']/profile['A']['gflops_conv_linear']),
        model_latency_reduction_pct=100*(1-l['model_only']['mean_ms']/a['model_only']['mean_ms']),
        model_fps_increase_pct=100*(l['model_only']['actual_fps']/a['model_only']['actual_fps']-1),
        png_latency_increase_pct=100*(l['end_to_end']['mean_ms']/a['end_to_end']['mean_ms']-1),
        png_fps_change_pct=100*(l['end_to_end']['actual_fps']/a['end_to_end']['actual_fps']-1),
        observed_gpu_memory_reduction_pct=100*(1-l['max_observed_dedicated_MiB']/a['max_observed_dedicated_MiB']),
        accuracy_changes_pp=acc['mean_changes_pp']['A'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(11.5,7.2))
    colors=['#4D6478','#C76B36'];labels=['A','L0']
    items=[('Parameters (thousands)',[profile[v]['params']/1000 for v in labels],None),
        ('Conv / Linear GFLOPs',[profile[v]['gflops_conv_linear'] for v in labels],None),
        ('Sidewalk IoU (%)',[acc['summary'][v]['sidewalk']['mean_pct'] for v in labels],[acc['summary'][v]['sidewalk']['sample_std_pct'] for v in labels]),
        ('Boundary IoU (%)',[acc['summary'][v]['boundary']['mean_pct'] for v in labels],[acc['summary'][v]['boundary']['sample_std_pct'] for v in labels]),
        ('Model-only mean latency (ms)',[timing['summary'][v]['model_only']['mean_ms'] for v in labels],None),
        ('Complete PNG mean latency (ms)',[timing['summary'][v]['end_to_end']['mean_ms'] for v in labels],None)]
    for axis,(title,values,errors) in zip(axes.flat,items):
        axis.bar(labels,values,color=colors,width=.52,yerr=errors,capsize=4)
        axis.set_title(title,pad=11);axis.set_ylim(0,max(values)*1.28)
        axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
        for i,value in enumerate(values):axis.text(i,value+(errors[i] if errors else 0)+max(values)*.025,f'{value:.2f}',ha='center')
    fig.suptitle('L0 compression trades some accuracy for smaller size and faster model inference',fontsize=13,y=.99)
    fig.text(.04,.055,'Accuracy: all three seeds, mean ± sample SD; validation reused for selection.  Timing: fixed seed42, four paired sessions per model.',fontsize=9)
    fig.text(.04,.028,'1×3×768×1024; 1 MAC = 2 FLOPs; Conv/ConvTranspose/Linear only.  RTX 4060 Laptop, AC, power-saver; PNG pipeline excludes saving/UI.',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.955))
    directory=ROOT/'results/figures';directory.mkdir(exist_ok=True)
    outputs=[directory/'compact_tradeoff_20261107.png',directory/'compact_tradeoff_20261107.pdf']
    for p in outputs:
        assert not p.exists();fig.savefig(p,dpi=220)
    plt.close(fig)
    inputs=['results/l0_three_seed_review_20261107.json','results/compact_benchmark_20261107.json',
        'results/compact_deployment_validation_20261107.json','results/compact_application_20261107.json',
        'results/lightweight_profile_20261101_v2.json','results/lightweight_operator_coverage_20261101_v2.json']
    dump(ROOT/'results/compact_final_effect_20261107.json',dict(status='pass',changes_vs_A=changes,
        accuracy_summary=acc['summary'],timing_summary=timing['summary'],
        conclusions=['Parameters and counted FLOPs reduced substantially.',
            'All L0 seeds retained; lower mean accuracy and higher seed variability.',
            'Model-only latency lower in all four paired rounds; complete PNG mean latency higher in all four.',
            'A/L0 full500 export acceptance and CLI media/lock checks passed.'],
        limits=['Partial operator FLOPs, not full operator cost.', 'Memory counters are snapshots, not true peaks.',
            'PNG throughput did not improve; no causal stage profiling or optimization claimed.',
            'Fixed tensor model-only FPS is not complete application FPS or embedded-board evidence.',
            'No natural-video temporal/generalization test or L0 long-duration stability test.',
            'No new operator or universal attention-failure claim; L1 seed42 exploratory only.'],
        inputs_sha256={p:sha(ROOT/p) for p in inputs},figures={str(p.relative_to(ROOT)):sha(p) for p in outputs},
        exporter_sha256=sha(Path(__file__))))
    print(json.dumps(changes,indent=2))


if __name__=='__main__':main()
