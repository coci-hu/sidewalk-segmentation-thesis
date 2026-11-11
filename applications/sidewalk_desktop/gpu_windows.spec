# Build from the validated Windows CUDA environment; artifacts stay outside Git.
from pathlib import Path
ROOT=Path(SPECPATH).parents[1]
APP=ROOT/'applications/sidewalk_desktop'
paths=[APP/'app.py',APP/'worker.py',APP/'model_catalog.py']
paths += list((APP/'models').glob('*'))
paths += [ROOT/'experiments/attention_gate'/p for p in (
    'verified_compact_runtime.py','reference_deployment_pipeline.py',
    'deployment_pipeline.py','deployment_equivalence.py','metrics_v1.py')]
paths += [ROOT/'results'/p for p in ('compact_final_effect_20261107.json',
    'lightweight_profile_20261101_v2.json','compact_deployment_validation_20261107.json')]
data=[(str(p),str(Path('project')/p.relative_to(ROOT).parent)) for p in paths]
a=Analysis([str(APP/'package_entry.py')],pathex=[str(APP)],datas=data,
    hiddenimports=['PIL.Image','PIL.ImageTk','tkinter','tkinter.ttk','tkinter.filedialog',
        'tkinter.messagebox','cv2','numpy','torch','torchvision','torchvision.transforms',
        'onnxruntime','msvcrt'],
    excludes=['matplotlib','pandas','scipy','IPython','notebook','pytest','tensorboard',
        'transformers','tensorflow','keras','sklearn','huggingface_hub','torchaudio'],
    noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='SidewalkGPU',debug=False,
    bootloader_ignore_signals=False,strip=False,upx=False,console=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='SidewalkGPU')
