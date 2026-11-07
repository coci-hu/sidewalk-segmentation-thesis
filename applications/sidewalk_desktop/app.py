"""Chinese desktop interface; inference runs in an isolated child process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import uuid
from datetime import datetime
from PIL import Image,ImageTk

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


class App:
    def __init__(self,window):
        self.window=window;self.process=None;self.job=None;self.log_handles=[];self.preview_version=None
        self.closing=False;self.photos=[];self.poll_id=None
        self.input_path=tk.StringVar();self.model=tk.StringVar(value='L0 轻量模型')
        self.status=tk.StringVar(value='选择一张图片或一段视频开始')
        self.detail=tk.StringVar(value='所有文件在本机处理，结果保存在独立目录')
        window.title('城市人行道识别 · 轻量化分割应用');window.geometry('1160x790');window.minsize(920,650)
        window.configure(bg='#f3f5f8')
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f3f5f8')
        style.configure('TButton',padding=(14,9));style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'))
        style.configure('Accent.TButton',background='#176b62',foreground='white')
        outer=ttk.Frame(window,padding=24);outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='城市人行道识别',style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer,text='图片与视频分割 · 原图 / 识别结果对照',foreground='#536274').pack(anchor='w',pady=(5,20))
        controls=ttk.Frame(outer);controls.pack(fill='x')
        self.entry=ttk.Entry(controls,textvariable=self.input_path);self.entry.pack(side='left',fill='x',expand=True)
        self.choose_button=ttk.Button(controls,text='选择文件',command=self.choose);self.choose_button.pack(side='left',padx=(10,0))
        row=ttk.Frame(outer);row.pack(fill='x',pady=12)
        ttk.Label(row,text='识别模型').pack(side='left',padx=(0,10))
        self.combo=ttk.Combobox(row,textvariable=self.model,values=['L0 轻量模型','A 基线模型'],state='readonly',width=19);self.combo.pack(side='left')
        self.start_button=ttk.Button(row,text='开始识别',style='Accent.TButton',command=self.start);self.start_button.pack(side='left',padx=12)
        self.cancel_button=ttk.Button(row,text='取消处理',command=self.cancel,state='disabled');self.cancel_button.pack(side='left')
        self.folder_button=ttk.Button(row,text='打开结果文件夹',command=self.open_folder,state='disabled');self.folder_button.pack(side='right')
        self.preview_frame=ttk.Frame(outer);self.preview_frame.pack(fill='both',expand=True,pady=(6,12))
        self.preview_frame.columnconfigure((0,1),weight=1);self.preview_frame.rowconfigure(1,weight=1)
        ttk.Label(self.preview_frame,text='原始画面').grid(row=0,column=0,sticky='w',pady=(0,8))
        ttk.Label(self.preview_frame,text='识别结果 · 绿色为预测人行道').grid(row=0,column=1,sticky='w',padx=(12,0),pady=(0,8))
        self.left=tk.Label(self.preview_frame,text='选择图片或视频',bg='#e3e8ee',fg='#536274',font=('Microsoft YaHei UI',12))
        self.right=tk.Label(self.preview_frame,text='结果将在此显示',bg='#e3e8ee',fg='#536274',font=('Microsoft YaHei UI',12))
        self.left.grid(row=1,column=0,sticky='nsew',padx=(0,6));self.right.grid(row=1,column=1,sticky='nsew',padx=(6,0))
        self.progress=ttk.Progressbar(outer,mode='determinate',maximum=100);self.progress.pack(fill='x',pady=(0,10))
        ttk.Label(outer,textvariable=self.status,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
        ttk.Label(outer,textvariable=self.detail,foreground='#536274',wraplength=1080).pack(anchor='w',pady=(5,10))
        ttk.Separator(outer).pack(fill='x',pady=8)
        ttk.Label(outer,text='处理时间包含当前运行开销，首帧通常较慢；页面计时不等于论文正式测速。',foreground='#637083').pack(anchor='w')
        ttk.Label(outer,text='用于人行道语义识别研究，结果不代表安全通行或避障结论。',foreground='#637083').pack(anchor='w',pady=(4,0))
        window.protocol('WM_DELETE_WINDOW',self.close)

    def choose(self):
        path=filedialog.askopenfilename(title='选择图片或视频',filetypes=[('图片与视频','*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.m4v'),('所有文件','*.*')])
        if path:self.input_path.set(path)

    def busy(self,value):
        for widget in (self.entry,self.choose_button,self.start_button):widget.configure(state='disabled' if value else 'normal')
        self.combo.configure(state='disabled' if value else 'readonly')
        self.cancel_button.configure(state='normal' if value else 'disabled')

    def start(self):
        if self.process and self.process.poll() is None:return
        source=Path(self.input_path.get().strip())
        if not source.is_file():messagebox.showerror('无法开始','请先选择一个存在的图片或视频文件。');return
        output=ROOT/'experiments/attention_gate/desktop_outputs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        output.mkdir(parents=True);self.job=output;self.preview_version=None
        self.photos=[]
        for widget in (self.left,self.right):widget.configure(image='',text='正在准备…')
        self.log_handles=[(output/'stdout.log').open('wb'),(output/'stderr.log').open('wb')]
        variant='L0' if self.model.get().startswith('L0') else 'A'
        command=[sys.executable,str(HERE/'worker.py'),'--input',str(source.resolve()),'--output',str(output),'--model',variant]
        try:
            self.process=subprocess.Popen(command,stdout=self.log_handles[0],stderr=self.log_handles[1],
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as error:
            for handle in self.log_handles:handle.close()
            self.log_handles=[];messagebox.showerror('启动失败',str(error));return
        self.busy(True);self.folder_button.configure(state='normal');self.progress['value']=0
        self.status.set('正在启动识别…');self.detail.set('首次加载需要一些时间，窗口仍可操作。')
        self.poll()

    def show_previews(self):
        if not self.job:return
        self.window.update_idletasks()
        for widget,name in [(self.left,'input_preview.jpg'),(self.right,'output_preview.jpg')]:
            p=self.job/name
            if p.exists():
                try:
                    with Image.open(p) as im:
                        im=im.convert('RGB');im.thumbnail((max(300,widget.winfo_width()-16),max(200,widget.winfo_height()-16)))
                        photo=ImageTk.PhotoImage(im)
                    widget.configure(image=photo,text='');widget.image=photo
                except OSError:pass

    def poll(self):
        self.poll_id=None
        status={}
        try:
            if self.job and (self.job/'status.json').exists():status=json.loads((self.job/'status.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):pass
        if status:
            frames=status.get('frames',0);total=status.get('total',0)
            self.status.set(status.get('message','正在处理'))
            if total:self.progress['value']=min(100,100*frames/total)
            if status.get('state')=='running':
                self.detail.set(f'已处理 {frames} / {total or "未知"} 帧 · 当前推理 {status.get("inference_ms",0):.1f} ms · 当前画面人行道比例 {status.get("foreground_percent",0):.1f}%')
            version=status.get('preview_version')
            if version!=self.preview_version or status.get('state')=='complete':
                self.show_previews();self.preview_version=version
        if self.process and self.process.poll() is None:
            self.poll_id=self.window.after(250,self.poll);return
        for handle in self.log_handles:handle.close()
        self.log_handles=[];self.busy(False)
        if status.get('state')=='complete':
            self.progress['value']=100;self.detail.set(f'共处理 {status["frames"]} 帧；结果已保存。点击“打开结果文件夹”查看叠加图、掩码或视频。')
        elif status.get('state')=='cancelled':self.detail.set('已完成的其他任务不受影响；本次未完成视频不会标为完成结果。')
        else:
            self.status.set('处理未完成');self.detail.set(status.get('message','程序退出，请打开结果文件夹查看错误日志。'))
        if self.closing:self.window.destroy()

    def cancel(self):
        if self.job and self.process and self.process.poll() is None:
            (self.job/'CANCEL_REQUESTED').touch();self.cancel_button.configure(state='disabled')
            self.status.set('已请求取消，当前帧结束后停止')

    def open_folder(self):
        if self.job:os.startfile(str(self.job))

    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno('任务仍在运行','取消当前处理并在安全退出后关闭窗口？'):return
            self.closing=True;self.cancel();return
        if self.poll_id:self.window.after_cancel(self.poll_id)
        self.window.destroy()


if __name__=='__main__':
    window=tk.Tk();App(window);window.mainloop()
