"""Verify the phase2 freeze at its original commit after later application work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REV='39f89ffaaaea77832ec6a5a945e5f2329a2223fa'
SNAPSHOT='results/compact_experiment_freeze_20261107.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');a=p.parse_args()
    raw=subprocess.check_output(['git','show',REV+':'+SNAPSHOT],cwd=ROOT)
    assert raw==(ROOT/SNAPSHOT).read_bytes(),'Original freeze manifest changed'
    snapshot=json.loads(raw)
    specs='\n'.join(REV+':'+r['path'] for r in snapshot['repository_files'])+'\n'
    process=subprocess.run(['git','cat-file','--batch'],input=specs.encode('utf-8'),stdout=subprocess.PIPE,check=True,cwd=ROOT)
    data=process.stdout;offset=0
    for row in snapshot['repository_files']:
        end=data.index(b'\n',offset);header=data[offset:end].split();assert header[1]==b'blob'
        size=int(header[2]);body=data[end+1:end+1+size];offset=end+size+2
        assert size==row['bytes'] and hashlib.sha256(body).hexdigest()==row['sha256'],row['path']
    assert offset==len(data)
    if a.local:
        for row in snapshot['local_artifacts']:
            path=(ROOT/row['path']).resolve();assert path.is_relative_to(ROOT)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            assert path.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
    print('PASS phase2 original revision:',len(snapshot['repository_files']),'repository files;',len(snapshot['local_artifacts']) if a.local else 0,'local artifacts')

if __name__=='__main__':main()
