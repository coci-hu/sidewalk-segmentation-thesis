"""A/L0 demo reusing the unchanged reference image/video processing path."""
import argparse
from pathlib import Path
from sidewalk_demo_reference import process
from verified_compact_runtime import VerifiedCompactRuntime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-index',type=int,default=1)
    p.add_argument('--backend',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--max-frames',type=int)
    p.add_argument('--source-note',default='User-provided media')
    a=p.parse_args()
    with VerifiedCompactRuntime(a.models,a.backend,a.output.with_suffix(a.output.suffix+'.evidence')) as runtime:
        process(a,runtime)


if __name__=='__main__':main()
