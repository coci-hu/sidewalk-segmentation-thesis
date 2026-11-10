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
from model_catalog import algorithm,default_models

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


class App:
    def __init__(self,window):
        self.window=window;self.process=None;self.job=None;self.log_handles=[];self.preview_version=None
        self.closing=False;self.photos=[];self.poll_id=None
        self.input_path=tk.StringVar();self.model=tk.StringVar(value='L0 轻量模型')
        self.result_view=tk.StringVar(value='人行道叠加');self.algorithm_text=tk.StringVar();self.algorithm_note=tk.StringVar()
        self.output_label=tk.StringVar(value='算法输出')
        self.status=tk.StringVar(value='选择一张图片或一段视频开始')
        self.detail=tk.StringVar(value='所有文件在本机处理，结果保存在独立目录')
        window.title('轻量化人行道分割 · 算法展示');window.geometry('1200x900');window.minsize(1060,820)
        window.configure(bg='#f3f5f8')
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f3f5f8')
        style.configure('TButton',padding=(14,9));style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'))
        style.configure('Accent.TButton',background='#176b62',foreground='white')
        outer=ttk.Frame(window,padding=24);outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='轻量化人行道分割',style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer,text='ENet 通道缩减 · A / L0 算法对照 · 固定输入 768 × 1024',foreground='#536274').pack(anchor='w',pady=(5,12))
        panel=ttk.LabelFrame(outer,text='算法与已验证结果',padding=(12,8));panel.pack(fill='x',pady=(0,12))
        ttk.Label(panel,textvariable=self.algorithm_text,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
        ttk.Label(panel,textvariable=self.algorithm_note,foreground='#536274').pack(anchor='w',pady=(5,0))
        ttk.Label(panel,text='精度为固定验证集三个种子均值；计算量仅计卷积/转置卷积/线性层。当前输入不显示准确率。',foreground='#637083').pack(anchor='w',pady=(4,0))
        controls=ttk.Frame(outer);controls.pack(fill='x')
        self.entry=ttk.Entry(controls,textvariable=self.input_path);self.entry.pack(side='left',fill='x',expand=True)
        self.choose_button=ttk.Button(controls,text='选择文件',command=self.choose);self.choose_button.pack(side='left',padx=(10,0))
        row=ttk.Frame(outer);row.pack(fill='x',pady=12)
        ttk.Label(row,text='识别模型').pack(side='left',padx=(0,10))
        self.combo=ttk.Combobox(row,textvariable=self.model,values=['L0 轻量模型','A 基线模型'],state='readonly',width=19);self.combo.pack(side='left')
        self.combo.bind('<<ComboboxSelected>>',lambda e:self.update_algorithm())
        ttk.Button(row,text='算法对照表',command=self.compare).pack(side='left',padx=(8,0))
        self.start_button=ttk.Button(row,text='开始识别',style='Accent.TButton',command=self.start);self.start_button.pack(side='left',padx=12)
        self.cancel_button=ttk.Button(row,text='取消处理',command=self.cancel,state='disabled');self.cancel_button.pack(side='left')
        self.folder_button=ttk.Button(row,text='打开结果文件夹',command=self.open_folder,state='disabled');self.folder_button.pack(side='right')
        self.preview_frame=ttk.Frame(outer);self.preview_frame.pack(fill='both',expand=True,pady=(6,12))
        self.preview_frame.columnconfigure((0,1),weight=1);self.preview_frame.rowconfigure(1,weight=1)
        ttk.Label(self.preview_frame,text='原始画面').grid(row=0,column=0,sticky='w',pady=(0,8))
        view=ttk.Frame(self.preview_frame);view.grid(row=0,column=1,sticky='ew',padx=(12,0),pady=(0,8))
        ttk.Label(view,textvariable=self.output_label).pack(side='left')
        choice=ttk.Combobox(view,textvariable=self.result_view,values=['人行道叠加','二值分割掩码'],state='readonly',width=16)
        choice.pack(side='right');choice.bind('<<ComboboxSelected>>',lambda e:self.show_previews())
        self.left=tk.Label(self.preview_frame,text='选择图片或视频',bg='#e3e8ee',fg='#536274',font=('Microsoft YaHei UI',12))
        self.right=tk.Label(self.preview_frame,text='结果将在此显示',bg='#e3e8ee',fg='#536274',font=('Microsoft YaHei UI',12))
        self.left.grid(row=1,column=0,sticky='nsew',padx=(0,6));self.right.grid(row=1,column=1,sticky='nsew',padx=(6,0))
        self.progress=ttk.Progressbar(outer,mode='determinate',maximum=100);self.progress.pack(fill='x',pady=(0,10))
        ttk.Label(outer,textvariable=self.status,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w')
        ttk.Label(outer,textvariable=self.detail,foreground='#536274',wraplength=1080).pack(anchor='w',pady=(5,10))
        ttk.Separator(outer).pack(fill='x',pady=8)
        ttk.Label(outer,text='流程：图像归一化 → ENet 编码 / 解码 → 二分类掩码 → 恢复原尺寸与可视化。',foreground='#637083').pack(anchor='w')
        ttk.Label(outer,text='页面计时含首帧初始化开销；完整图片流程未加速，不能将纯模型 FPS 当成应用帧率。',foreground='#637083').pack(anchor='w',pady=(4,0))
        ttk.Label(outer,text='用于人行道语义识别研究，结果不代表安全通行或避障结论。',foreground='#637083').pack(anchor='w',pady=(4,0))
        window.protocol('WM_DELETE_WINDOW',self.close)
        self.update_algorithm()

    def update_algorithm(self):
        variant='L0' if self.model.get().startswith('L0') else 'A';a=algorithm(variant)
        self.algorithm_text.set(f'{a["name"]}   |   参数 {a["params"]:,}   |   {a["gflops"]:.3f} GFLOPs   |   人行道 IoU {a["iou"]:.2f}% ± {a["iou_sd"]:.2f}%')
        self.algorithm_note.set(f'通道：{a["channels"]}   ·   本机纯推理均值 {a["latency"]:.2f} ms   ·   训练后权重固定')

    def compare(self):
        dialog=tk.Toplevel(self.window);dialog.title('算法对照 · 冻结实验结果');dialog.geometry('740x470')
        box=ttk.Frame(dialog,padding=20);box.pack(fill='both',expand=True)
        ttk.Label(box,text='同输入、同训练协议的算法取舍',font=('Microsoft YaHei UI',15,'bold')).pack(anchor='w',pady=(0,12))
        tree=ttk.Treeview(box,columns=('metric','A','L0'),show='headings',height=8)
        for col,text,width in [('metric','指标',240),('A','A 基线',200),('L0','L0 轻量模型',200)]:
            tree.heading(col,text=text);tree.column(col,width=width,anchor='w')
        a,b=algorithm('A'),algorithm('L0')
        for label,key,unit in [('参数量','params',''),('统计计算量','gflops',' GFLOPs'),('人行道 IoU 均值','iou','%'),('边界 IoU 均值','boundary','%'),('纯推理平均耗时','latency',' ms'),('完整 PNG 平均耗时','full_latency',' ms')]:
            values=[f'{x[key]:,}' if key=='params' else f'{x[key]:.3f}{unit}' for x in (a,b)]
            tree.insert('', 'end',values=(label,*values))
        tree.pack(fill='x')
        ttk.Label(box,text='L0：参数减少 38.67%，统计计算量减少 28.01%；精度略降且种子波动增加。\n纯推理延迟降低 8.62%，完整 PNG 流程延迟增加 3.56%。\n精度为三个种子均值；测速为固定种子42、四轮配对。非当前图片的准确率。',wraplength=680).pack(anchor='w',pady=16)
        return dialog

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
        if not default_models().is_file():messagebox.showerror('缺少推理模型','请按 README 的模型获取说明准备 A/L0 模型包。');return
        output=ROOT/'experiments/attention_gate/desktop_outputs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        output.mkdir(parents=True);self.job=output;self.preview_version=None
        self.photos=[]
        for widget in (self.left,self.right):widget.configure(image='',text='正在准备…')
        self.log_handles=[(output/'stdout.log').open('wb'),(output/'stderr.log').open('wb')]
        variant='L0' if self.model.get().startswith('L0') else 'A'
        self.output_label.set('算法输出 · '+variant)
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
        result='mask_preview.jpg' if self.result_view.get()=='二值分割掩码' else 'output_preview.jpg'
        for widget,name in [(self.left,'input_preview.jpg'),(self.right,result)]:
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
