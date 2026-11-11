"""Frozen launcher: retain the verified source layout inside the application bundle."""
import os
from pathlib import Path
import runpy
import sys

def main():
    root=Path(sys._MEIPASS)/'project'
    app=root/'applications/sidewalk_desktop'
    sys.path[:0]=[str(app),str(root/'experiments/attention_gate')]
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        sys.argv=[str(app/'worker.py')]+sys.argv[2:]
        runpy.run_path(str(app/'worker.py'),run_name='__main__')
    elif len(sys.argv)>1 and sys.argv[1]=='--check-window':
        import tkinter as tk
        from app import App
        window=tk.Tk();window.withdraw();view=App(window);window.update()
        assert '161,688' in view.algorithm_text.get()
        if len(sys.argv)>3:
            import time,json
            view.input_path.set(sys.argv[3]);view.start();deadline=time.monotonic()+240
            while view.process and view.process.poll() is None or view.log_handles:
                window.update();time.sleep(.05)
                assert time.monotonic()<deadline
            result=json.loads((view.job/'result.json').read_text(encoding='utf-8'))
            assert result['state']=='complete' and result['model']=='L0'
        window.destroy()
        Path(sys.argv[2]).write_text('PASS frozen Tk window and algorithm data',encoding='utf-8')
    else:
        runpy.run_path(str(app/'app.py'),run_name='__main__')

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
