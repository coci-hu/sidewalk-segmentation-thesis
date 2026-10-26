"""Maintain the requested single Word journal, preserving existing paragraphs.

Use the bundled document runtime. Experimental machine records remain separate
because checkpoints and integrity verification depend on them.
"""
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
JOURNAL = Path(r'C:\Users\13699\Desktop\毕设\工作记录.docx')


def paragraph(doc, label, text):
    p = doc.add_paragraph()
    p.add_run(label + '：').bold = True
    p.add_run(text)
    if label == '复用依据':
        p.paragraph_format.keep_with_next = True
    return p


def event(doc, name, action, result, change, next_step, evidence=''):
    doc.add_heading(name, level=2)
    paragraph(doc, '操作', action)
    paragraph(doc, '结果', result)
    paragraph(doc, '变化与判断', change)
    if evidence:
        paragraph(doc, '复用依据', evidence)
    paragraph(doc, '下一步', next_step)


def setup():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(2)
    section.footer_distance = Cm(0.8)
    for name, size, font in [('Normal', 10.5, '宋体'), ('Title', 18, '黑体'), ('Heading 1', 14, '黑体'), ('Heading 2', 12, '黑体')]:
        style = doc.styles[name]
        style.font.name = 'Times New Roman'
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), font)
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25
        if name.startswith('Heading'):
            style.paragraph_format.space_before = Pt(9)
            style.paragraph_format.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    r = footer.add_run('胡可怡  工作记录  第 ')
    r.font.size = Pt(9)
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'PAGE')
    footer._p.append(fld)
    footer.add_run(' 页').font.size = Pt(9)
    doc.core_properties.author = '胡可怡'
    doc.core_properties.title = '毕业设计实验与论文工作记录'
    return doc


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    for i, text in enumerate(headers):
        t.rows[0].cells[i].text = str(text)
    for values in rows:
        cells = t.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = str(value)
    for ri, row in enumerate(t.rows):
        for ci, cell in enumerate(row.cells):
            if widths:
                cell.width = Cm(widths[ci])
            tcpr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for side in ('top', 'left', 'bottom', 'right'):
                edge = OxmlElement('w:' + side)
                for k, v in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    edge.set(qn('w:' + k), v)
                borders.append(edge)
            tcpr.append(borders)
            if ri == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'EDEDED')
                tcpr.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.1
                for r in p.runs:
                    r.font.size = Pt(9.5)
                    r.bold = ri == 0
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def atomic_save(doc):
    # A locked Word document stays untouched; the task reports a pending update.
    for border in list(doc.styles['Title']._element.xpath('./w:pPr/w:pBdr')):
        border.getparent().remove(border)
    for p in doc.paragraphs:
        pp = p._p.get_or_add_pPr()
        if pp.find(qn('w:kinsoku')) is None:
            node = OxmlElement('w:kinsoku')
            node.set(qn('w:val'), '1')
            pp.append(node)
        if p.text.startswith('复用依据：'):
            p.paragraph_format.keep_with_next = True
    temporary = JOURNAL.with_name('.工作记录更新.tmp.docx')
    doc.save(temporary)
    os.replace(temporary, JOURNAL)


def initialize():
    if JOURNAL.exists() and JOURNAL.stat().st_size > 0:
        raise FileExistsError('Journal already has content; append instead of rebuilding')
    doc = setup()
    doc.add_paragraph('毕业设计实验与论文工作记录', 'Title')
    doc.add_paragraph('胡可怡  学号202324120204  软件工程2班')
    doc.add_paragraph('课题  基于轻量化语义分割的城市可通行区域识别与边缘部署研究')
    doc.add_paragraph('本记录汇总已核实的工作、结果、调整及下一步。2026年10月17日首次集中补录，历史条目按实际操作日期编排；未保存精确时刻的操作不补造时间。此后各次操作追加在本文末尾。')
    doc.add_heading('研究范围与当前口径', level=1)
    doc.add_paragraph('任务为Cityscapes标签ID 8的人行道二分类代理识别，非直接安全通行判断。主要研究轻量网络注意力的适配条件和部署取舍，不预先宣称CA失效。现有RTX 4060 Laptop 8GB用于训练和本机部署；硬件与云服务以0元为目标，授权预算上限50元，未租服务器或上传数据。')
    doc.add_paragraph('统一训练为768×1024、训练batch4、保留尾批3张、FP32、100轮、AdamW学习率1e-4与权重衰减0.01、余弦T_max100与最低学习率1e-6、损失0.4CE加0.6Dice。使用42、43、44配对种子；主对照不加增强、不早停。其他原标签并入背景的规则保留，并作为局限说明。')
    doc.add_paragraph('实验根目录为D:\\毕业设计\\blind_road_project\\experiments\\attention_gate。下文依据路径相对此目录；原始数据、权重、逐轮指标、检查点、源码快照与恢复指纹属于实验资料，保留于本地，不与重复说明文档混为一类。')
    doc.add_heading('逐次工作记录', level=1)
    event(doc, '01  2026年10月13日至14日  文件整理',
        '清点D盘实验文件，按代码、数据、权重、结果和历史文档整理位置，按新位置修正路径入口。',
        '25155个原文件建立路径、大小和SHA256对照；原始代码、权重和实验结果保留，整理完成后复核通过。',
        '整理位置与复用入口，不重训或改写历史指标。桌面材料当时没有移动。',
        '固定问题与评价协议。', 'D:\\毕业设计\\_organization_records\\2026-10-13保留原路径映射、源备份及恢复工具。')
    event(doc, '02  2026年10月14日  第一步协议复核',
        '检查历史训练参数、数据配对和边界指标定义，新增独立协议与正确指标。',
        '181个核心旧文件指纹一致；Cityscapes训练2975张、验证500张，缺失标签0；10项人工掩码检查通过。',
        '发现旧ENet的权重衰减0.01而CA为0.0001，相差100倍；旧mIoU是batch平均，旧名为official的边界公式与作者定义不一致。因此旧CA差值不能直接归因于模块。',
        '统一评估已有权重，再做相同训练条件对照。', 'protocol_v1.json；metrics_v1.py；step_01_2026-10-14/run_002。')
    event(doc, '03  2026年10月14日  第二步已有权重统一评估',
        '对15个已完成模型各评估500张验证图，共7500次预测，使用全局混淆矩阵和作者前景Boundary IoU。',
        'ENet全局mIoU81.519%、人行道IoU65.134%、边界IoU54.092%；旧ENet加CA分别81.185%、64.500%、53.276%。DeepLabv3+全局mIoU83.311%，参数40.347M。',
        '这些仍是历史权重的统一重评，不是统一训练结果。DeepLabv3+已经完成；U-Net未完成，不进入主表，不为主线重跑所有模型。',
        '检查并统一训练入口。', 'step_02_2026-10-14/run_002/main_table.csv保留全部15模型及权重哈希。')
    event(doc, '04  2026年10月14日  第三步统一训练入口',
        '新增A为ENet、B为现有ReLU CA的统一训练器，复制共同骨干状态并独立初始化CA，固定样本生成器与完整随机状态。',
        '14项检查通过，包括三个种子的共同初值、CA旁路整网一致、矩形公式与梯度、全分辨率batch4训练以及独立进程断点恢复。A约0.263640M参数，B约0.266992M，8GB显卡可执行。',
        '严格确定性模式下原CUDA交叉熵mean算子报错，采用数学等价的log_softmax与gather求均值；CPU和CUDA数值、梯度核验通过，单列v1_1执行修订。小规模smoke不当作论文精度。',
        '只对既有权重开展固定样本注意力诊断。', 'common_runner.py；protocol_v1_1.json；step_03_2026-10-14/run_003。')
    event(doc, '05  2026年10月14日  第四步注意力诊断',
        '18个训练城市各固定3图共54图，4个CA模型10种条件及2个基线，共2268条预测记录；不训练、不用验证图调参。',
        '平均门控约0.235至0.242，L2范数保留约25%至26%，近零门控占比不到1%；ENet加CA在能量匹配的空间打乱下，人行道IoU下降32.59和34.07个百分点。',
        '原门控空间分布有实际作用，不能称注意力完全无效。对旧权重直接改成弱混合门控均退化，但不代表从头训练的候选必然无效；热图平均方法另行纠正。',
        '先做可靠A/B重复训练，按结果决定新模块。', 'step_04_2026-10-14/run_001保留逐图计数、门控数组、固定例图和核验。')
    event(doc, '06  2026年10月14日  路线文档核对与第五步首轮耗时',
        '只读提取codex思路.doc，核对12步路线；A/B种子42各完成一轮全部训练2975图和验证500图，并保存744次更新的完整检查点。',
        'A训练加验证14.64分钟，B13.79分钟；共享骨干、样本排列、优化器计数、调度器和轮末随机状态检查通过。',
        '数据等待占训练时间很大。旧142小时是单读取进程对六次100轮的外推，不作为优化后预计耗时。两组02:49训练完成后因系统空闲睡眠，收尾中午恢复；训练计时未混入等待，父进程唤醒保护已补。',
        '先验证读取加速等价，再继续长训。', 'step_05_2026-10-14/benchmark_001；document_read_001保留原文档只读提取。')
    event(doc, '07  2026年10月14日  统一数据读取优化',
        '比较1、2、4个读取进程，检查跨18城市图像张量、标签与顺序，以及A/B小规模连续更新的模型、优化器、调度器和随机状态。',
        '一致性及新入口独立进程恢复检查通过；定常数据供给3.73提升至17.20图每秒。统一选择4进程，首个优化完整ENet轮次训练4.48分钟、验证1.62分钟，合计6.10分钟。',
        '只改变读取进程数，数据供给4.61倍不等于训练4.61倍。新增v1_2及适配器，旧首轮只读保留，迁移副本只改执行元数据，所有训练状态与历史保持一致。',
        '种子42的A从第2轮续至100，之后B续至100。', 'protocol_v1_2.json；common_runner_v2.py；loader_optimization_001、loader_migration_001、loader_v2_integration_001。')
    event(doc, '08  2026年10月14日  启动第五步正式对照',
        '后台顺序启动种子42的A/B，不自动开始43、44及C/D/E；逐轮保存检查点、指标、样本顺序和随机状态。',
        '初期第2轮核验通过，随后持续训练。后来确认A完整保存到60轮，14日19:26的残留进度停在第61轮训练中。',
        '旧任务001的running后来已不再表示真实运行。中断原因缺乏直接证据；后续系统重启不能证明最早停止原因。',
        '恢复时使用最新完整检查点和全新目录，未完成轮次从头执行。', 'step_05_2026-10-14/seed42_training_001。')
    event(doc, '09  2026年10月14日  开题报告初版',
        '检查学校空表及老师在他人报告正文中的红字备注，按本人课题、姓名胡可怡、学号202324120204和2班填写新报告，保留原件。',
        '形成开题阶段的拟研究写法，依据约942字、现状约972字、16篇英文文献、两项拟创新、三个研究节点与六个月进度；导师及指导小组意见留空。',
        '不将已做实验伪写为开题已有成果。学院、专业、导师沿用参考表，日期2026年10月5日。现有报告后来已有用户修改，本次不覆盖或删除本人版本及其副本。',
        '按实际实验结果调整最终论文，不提前承诺注意力提升。', '桌面本科毕业论文开题报告表_胡可怡_202324120204.docx；参考文献与生成脚本保留。')
    event(doc, '10  2026年10月14日  第六步提前准备',
        '准备C为作者形式h-swish CA，保留本地轴布局，仅替换ReLU；固定作者源码提交a3f3490ceba779a8443a231d0f792866bd24436d。',
        '五组CPU检查通过：三种子共同初值、非方形输入与作者公式的前反向及BN对应、恢复ReLU后整网逐位复现B、小型状态恢复后下一次更新一致。',
        '本地分支命名与作者方向相反，需按轴交换权重后比较，不能仅凭变量名认定轴错误。C参数量与B相同；h-swish对照本身不是原创算法。首个GPU隐藏检查失败后已用显式-1修正，成功证据在checks_002。',
        '待显卡空闲做CUDA全尺寸、新入口和独立进程恢复检查；尚未正式训练C。', 'ca_hswish_control.py；step_06_2026-10-14/preparation_001/checks_002。')
    event(doc, '11  2026年10月17日约14时  第一次中断恢复',
        '确认任务001进程已退出，从A第60轮恢复至新任务002，B仍使用已核验的第1轮状态。',
        '恢复后第61轮训练和验证完成，约6.27分钟；完整历史、样本随机顺序、优化器45384次更新、BN计数及学习率核验通过，随后进入62轮。',
        '181个原文件、196个前序产物和37个耗时测试文件指纹一致。新增进度检查入口同时核验进程与状态文件时效，不再只看running。',
        '继续A至100，再B至100；完成本种子不等于三种子研究完成。', 'step_05_2026-10-17/recovery_001/first_resumed_epoch_check.json；inspect_step05_progress.ps1。')
    event(doc, '12  2026年10月17日约17时  再次检查与后台方式调整',
        '检查发现任务002于14:10后失去进程，完整保存61轮，第62轮训练部分结束但验证及检查点未完成。改用当前用户的一次性系统后台任务托管原训练流程。',
        '新任务003从A第61轮检查点继续，第62轮重做；B仍从第2轮继续。系统任务名ThesisSeed42_20261017_1705，无开机或周期触发，最长执行窗口48小时。',
        '重复中断原因尚未证实，新托管方式减少依赖命令会话。原训练算法、数据、冻结协议及数值核验不变；仅新增托管入口，结尾可关闭额外Markdown说明并自动追加本文。',
        '观察首次完整恢复轮次及检查点，再维持当前配对训练。', 'run_seed42_managed.py；step_05_2026-10-14/seed42_training_003；current_seed42_job.json。')
    event(doc, '13  2026年10月17日  统一工作记录与清理规则',
        '按用户要求，将历史操作和以后各次变化集中写入本Word；减少新增人工说明文件，准备清理重复摘要、废弃排版草稿、内部预览及已解压安装包。',
        '原工作记录为0字节空文件，已从已核实实验资料补录。机器可读配置、逐轮日志、检查点和指纹证明继续保留，受保护说明暂不删以免破坏恢复核验。',
        '以后每次操作记录目的、实际做法、结果、与此前的变化及下一步；清理的实际名单和数量在完成后追加。不会清除原数据、权重、训练结果或用户修改的开题报告。',
        '完成本次安全清理和Word逐页排版检查。')
    doc.add_heading('当前主线与继续条件', level=1)
    table(doc, ['步骤', '状态', '完成条件'], [
        ['1至4', '协议、统一评估、入口检查、诊断完成', '证据已保留，相关摘要见前文'],
        ['5', '种子42的A/B继续中', '各100轮核验后审阅，再安排43与44'],
        ['6', 'C模型与CPU预检已备好', 'CUDA与恢复检查后，按计划开展正式对照'],
        ['7至9', '门控与附加实验未开展', '由A/B/C证据决定，不默认堆模块'],
        ['10至12', '部署、原型与最终论文未完成', '模型导出一致性、实测性能、场景与写作闭环'],
    ], [2, 7, 8])
    doc.add_paragraph('运行时保持接电，避免手动休眠、关机或另启GPU训练。可以随时要求检查进度。种子42的A/B完成且核验通过后，先审阅本种子结果并安排剩余重复；不能把单个种子写成稳定增益或普遍失效。')
    atomic_save(doc)


def append_cleanup(plan_file):
    doc = Document(JOURNAL)
    data = json.loads(Path(plan_file).read_text(encoding='utf-8'))
    removed = [x for x in data['targets'] if not Path(x['path']).exists()]
    assert len(removed) == len(data['targets']), 'Cleanup not finished'
    total = sum(x['bytes'] for x in removed)
    count = sum(x['files'] for x in removed)
    doc.add_heading('14  2026年10月17日  清理完成', level=2)
    paragraph(doc, '操作与结果', f'删除下列已汇总或可重新生成的内容，共{count}个文件，约{total / 1024 / 1024:.1f}MiB。训练数据、权重、源码和关键核验记录未列入删除范围。')
    for x in removed:
        doc.add_paragraph(x['display'] + '；' + x['reason'])
    paragraph(doc, '保留原因', '前序实验受哈希保护的说明和机器记录仍参与恢复或来源证明，删除会使严格核验失败；本轮不修改其内容。已解压的本地排版依赖保留，避免以后反复下载；安装包和旧预览删除。')
    paragraph(doc, '后续记录方式', '人工工作说明集中追加本文，实验目录保留程序所需数据及一个简短入口。后台A/B完成后自动追加数值结果；若Word正在占用导致写入失败，任务标记待补写，不覆盖现有文档。')
    atomic_save(doc)


def append_status(job):
    state = json.loads((job / 'job_status.json').read_text(encoding='utf-8'))
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    if 'stage1' in plan and plan.get('scope', '').startswith('C seed42'):
        if state.get('status') == 'complete' and state.get('verified'):
            return append_step06(job, 'completion')
        doc = Document(JOURNAL)
        doc.add_heading('第六步训练进度检查  ' + datetime.now().strftime('%Y年%m月%d日 %H时%M分'), level=2)
        live = state.get('training') or {}
        paragraph(doc, '状态', f'C种子42，任务状态{state["status"]}，已完整保存{state.get("saved_completed_epoch", 0)}轮；当前第{live.get("epoch", "待初始化")}轮、阶段{live.get("phase", "初始化")}。第五步A/B三种子此前已完成，C仍是单种子探索。')
        paragraph(doc, '下一步', 'C42完成100轮并核验后停止，先审阅再决定重复或新方案，不自动训练C43/44或D/E。')
        atomic_save(doc)
        return
    if state.get('status') == 'complete' and state.get('verified'):
        return append_repeat_completion(job) if 'queue' in plan else append_completion(job)
    doc = Document(JOURNAL)
    if 'queue' in plan:
        now = datetime.now().strftime('%Y年%m月%d日 %H时%M分')
        doc.add_heading('训练进度检查  ' + now, level=2)
        seed, variant = state.get('seed'), state.get('variant')
        live = state.get('training') or {}
        saved = Path(state['variant_folder']) / 'status.json' if state.get('variant_folder') else None
        epoch = json.loads(saved.read_text(encoding='utf-8')).get('completed_epoch', 0) if saved and saved.exists() else 0
        current = f'第{live["epoch"]}轮{live.get("phase", "执行")}，批次{live.get("batches_completed", 0)}/{live.get("batches_total", "待更新")}' if live.get('epoch') else '初始化或轮次切换，等待批次进度'
        paragraph(doc, '核实进度', f'任务状态{state["status"]}，种子{seed}、模型{variant}，已完整保存{epoch}轮；{current}。当前队列已完成{state.get("completed_runs", 0)}/4次训练，种子42此前已完成2次。进程存活与状态文件时效由进度检查入口另行核验。')
        if saved:
            first = saved.parent / 'first_epoch_verification.json'
            if first.exists():
                assert json.loads(first.read_text(encoding='utf-8'))['status'] == 'pass'
                paragraph(doc, '首轮核验', '744次更新、2975张训练与500张验证、样本顺序及生成器状态、骨干初值和数值检查通过。')
        paragraph(doc, '下一步', '按A43、B43、A44、B44顺序完成后核验三种子结果并停止，不自动启动新模块。若当前状态异常，保留完整检查点，先查原因再恢复。')
        atomic_save(doc)
        return
    folder = Path(plan['variant_folders']['A'])
    saved = folder / 'status.json'
    epoch = json.loads(saved.read_text(encoding='utf-8'))['completed_epoch'] if saved.exists() else 61
    now = datetime.now().strftime('%Y年%m月%d日 %H时%M分')
    doc.add_heading('本次结束检查  ' + now, level=2)
    paragraph(doc, '真实进度', f'当前任务{job.name}，状态文件为{state["status"]}；A已完整保存{epoch}轮，B长训尚待顺序运行。以进程存活、文件时效和检查点共同判断，不凭日历天数推断完成。')
    proof = job / 'first_resumed_epoch_check.json'
    if proof.exists():
        p = json.loads(proof.read_text(encoding='utf-8'))
        paragraph(doc, '续训核验', f'第{p["completed_epoch"]}轮恢复检查通过，历史、学习率、样本排列和随机状态延续，参数检查通过。')
    paragraph(doc, '下一步', '继续种子42两组至100轮并自动核验，再审阅并补齐剩余配对种子。现在不追加新模块，不据未完成曲线下结论。')
    atomic_save(doc)


def append_completion(job):
    doc = Document(JOURNAL)
    marker = '后台完成核验  ' + job.name
    if any(p.text == marker for p in doc.paragraphs):
        return
    proof = json.loads((job / 'final_verification.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'pass'
    data = json.loads((job / 'seed42_results.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=1)
    paragraph(doc, '完成时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '结果', '种子42的ENet与现有ReLU CA两组各100轮完成，74400次更新、样本配对、轮末随机状态、最优权重与旧文件完整性核验通过。该记录由本次托管程序在核验后自动追加。')
    table(doc, ['模型', '最优轮', '全局mIoU%', '人行道IoU%', '边界IoU%'], [[r['variant'], r['best_epoch'], f"{r['best_global_miou_pct']:.3f}", f"{r['best_sidewalk_iou_pct']:.3f}", f"{r['best_boundary_iou_pct']:.3f}"] for r in data['results']], [3, 2, 4, 4, 4])
    paragraph(doc, '变化与下一步', '这只完成一个配对种子，43、44和C正式训练尚未自动开始。先审阅逐轮结果、最优轮及配对差值，再决定剩余重复安排；不能据一个种子宣布稳定增益或失效。')
    paragraph(doc, '复用依据', str(job / 'seed42_results.json') + '；' + str(job / 'final_verification.json'))
    atomic_save(doc)


def append_project_review(job):
    doc = Document(JOURNAL)
    marker = '全流程复核与剩余配对训练  ' + job.name
    if any(p.text == marker for p in doc.paragraphs):
        return
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    state = json.loads((job / 'job_status.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=1)
    paragraph(doc, '操作时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '目的与结论', '对照codex思路.doc中的12步路线，复核实验文件、统一训练、部署与论文现状。主线进入第5步的剩余重复验证，尚未完成整个第5步；已完成种子42的2次训练，占计划6次A/B训练的2/6，不把这个比例当作整个毕设完成率。')
    table(doc, ['步骤', '核实状态', '下一项或完成条件'], [
        ['1', '研究问题、数据和指标协议已固定', '保持版本，不随结果改口径'],
        ['2', '15份旧权重统一重评已完成', '历史结果与新训练分开解释'],
        ['3', 'A/B入口与恢复检查已通过', '继续使用已冻结v1_2入口'],
        ['4', '固定训练样本的门控诊断已完成', '仅作机制线索，不单凭幅值判断有害'],
        ['5', '种子42的A/B各100轮已核验', '补齐43、44共4次，再汇总均值、样本标准差及配对差值'],
        ['6', 'h-swish CA代码与CPU预检已备好', 'CUDA全尺寸及独立进程恢复检查后才可正式训练C'],
        ['7', '固定及可学习混合门控未训练', '依据A/B/C结果决定是否启动D/E'],
        ['8', '新训练模型的机制与性能联合分析未完成', '用固定诊断样本复核，区分观察与解释'],
        ['9', '第二骨干和CBR必要消融尚未扩展', '有收益证据再扩展，控制本科工作量'],
        ['10', '有历史部署脚本、CSV及归档ONNX', '当前新权重仍需导出一致性和实际后端核验'],
        ['11', '有本地照片置信度统计等探索记录', '尚不构成标注精度验证，原型和持续运行测试未闭环'],
        ['12', '开题报告已形成，最终论文未完成', '实验结论、工程证据、正文与答辩材料待完成'],
    ], [1.1, 7.1, 8.8])
    paragraph(doc, '种子42审阅', 'ENet与CA全局mIoU为81.033%与81.620%，人行道IoU为64.223%与65.309%，边界IoU为53.043%与53.567%；CA分别提高0.587、1.086、0.524个百分点。最优轮分别92与84，均按事先固定的全局验证mIoU选取。本次结果支持当前条件下的单种子收益，不能再将CA失效作为预设结论，也不能由此认定稳定收益。')
    paragraph(doc, '历史工程证据的限度', '旧deploy_benchmark.csv记录过PyTorch、ONNX FP32与INT8的精度和耗时，归档还保留BiSeNetV2旧命名ONNX。但当前A/B新权重没有对应导出身份与一致性证据；旧表的CPU动态INT8速度亦低于CPU FP32，不能预设量化必然加速。CPU也是有效的实测部署平台，旧脚本中否定CPU测试价值的注释不作为论文依据。海口照片现有前景比例与置信度不是人工真值精度，不能当作跨域IoU。')
    paragraph(doc, '本次代码变化', '新增剩余配对训练托管脚本，按A43、B43、A44、B44顺序执行；复用原CUDA训练器。仅将数值核验函数的种子参数从固定42推广为默认42，保留旧调用行为；进度入口增加新任务指针和动态种子识别。训练算法、损失、分辨率、权重衰减、数据划分及原有结果均不改变。')
    paragraph(doc, '启动前核验', '冻结源文件与种子42一致，数据身份、协议、运行环境一致；43与44的A/B共同骨干初始张量逐项相同，三个种子的初始骨干彼此不同，43与44样本顺序不同，种子42最终检查点指纹一致。每个新运行的首轮另查744次更新、完整2975/500样本与顺序状态，完成每个配对后严格核验，再生成三种子汇总。')
    live = state.get('training') or {}
    progress = f'第{live["epoch"]}轮的{live.get("phase", "执行")}阶段' if live.get('epoch') else '初始化阶段，尚未保存首轮'
    paragraph(doc, '启动状态', f'任务{job.name}当前状态{state["status"]}，当前种子{state.get("seed", "尚未开始")}、模型{state.get("variant", "尚未开始")}；最新进度为{progress}。完整轮次均保存检查点，首轮通过情况见对应核验文件。')
    paragraph(doc, '预计耗时', f'按种子42两种模型各最后20轮的实测均值估计，剩余4次训练约{plan["estimated_training_hours"]:.1f}小时，不含中断、其他软件竞争及核验余量。不是承诺完成时刻。一次性后台任务无开机或周期触发；无自动重试，逐轮保存，保持电脑接电且不手动休眠。')
    paragraph(doc, '资源与停止范围', '沿用现有4060和本地数据，不租服务、不新增硬件。完成43、44后停止，不自动开始C/D/E，不修改现有Word内容，仅追加。每个配对及最终汇总会尝试写入本记录，Word占用时标记待补写。')
    paragraph(doc, '本科范围与待改进点', '现阶段优先做可靠对照、一个有依据的改进或简化，以及可运行的部署原型，不扩大到另一个大型数据集或补跑U-Net；DeepLabv3+已有历史结果。论文应围绕注意力收益条件与工程取舍组织证据，实际课题名称不在本次擅自改动。忽略区域处理和拉伸缩放的敏感性仍需独立检查，不能夹进本轮对照；验证集参与最优轮选择，须明确其不是独立测试集。部署先做当前权重FP32导出与精度一致性，再测同一硬件同一分辨率的模型延迟、端到端延迟、P95和内存，最后决定是否值得做FP16或INT8。')
    paragraph(doc, '下一步', '四次训练完成并通过核验后，可以让我检查第五步三种子结果。届时先判断CA收益方向和波动，再进行C的CUDA与恢复检查；先不增加新模块。在训练期间可只读检查进度和准备论文结构，不同时开启另一项GPU训练。')
    paragraph(doc, '复用依据', str(job / 'job_plan.json') + '；preflight_verification.json；inspect_step05_progress.ps1；历史deploy_benchmark.csv。')
    atomic_save(doc)


def append_recovery(job, check_only=False):
    doc = Document(JOURNAL)
    marker = ('恢复首轮核验完成  ' if check_only else '恢复训练与进度写入容错  ') + job.name
    if any(p.text == marker for p in doc.paragraphs):
        return
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    state = json.loads((job / 'job_status.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=2)
    paragraph(doc, '记录时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    if check_only:
        proof = json.loads((job / 'first_resumed_epoch_check.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass' and proof['completed_epoch'] == 94
        paragraph(doc, '结果', '第94轮已完整保存并通过CPU核验。前93轮历史、设置与冻结源文件、下一轮样本排列及生成器状态核验通过。训练2975张、验证500张，优化器和BN计数69936次，参数有限值检查通过。')
        paragraph(doc, '结论边界', '本项确认恢复连续性；完整A/B结论仍待各组100轮后核验。')
        paragraph(doc, '下一步', '继续既定A/B队列，完成后核验三种子结果并停止，不启动新模块。')
    else:
        proof = json.loads(Path(plan['io_verification']).read_text(encoding='utf-8'))
        assert proof['status'] == 'pass'
        paragraph(doc, '用户授权', '用户在10月20日明确要求继续，撤销上一日等待续跑的暂停要求。本次使用新恢复目录和新的一次性后台任务；旧任务保持禁用，旧暂停标记、报错记录和检查点完整保留。')
        paragraph(doc, '修复内容', '冻结的common_runner.py、common_runner_v2.py、模型、协议和原检查点均未修改。新增独立的操作适配层，仅在运行进程中替换监控JSON写入函数：使用进程内唯一临时文件，遇到Windows共享或权限错误最多重试12次；持续被占用时只跳过training_progress.json与epoch_timing.json的当次刷新并记日志。关键任务状态、权重和检查点写入失败仍报错，序列化、磁盘及其他错误不被忽略。操作适配器的源码指纹和证据另行固定，未静默改变科学指纹。')
        paragraph(doc, '验证结果', '五组检查通过：用真实Windows文件句柄复现原PermissionError并检查非关键进度跳过；关键状态与无效JSON错误仍拒绝；短暂占用后重试成功；从相同第93轮检查点以768×1024、batch4进行CUDA一步更新，在原写入与进度持续被锁住的容错路径中，损失、模型参数、BN、优化器、调度器及完整随机状态逐位一致；冻结源文件不变且自有临时文件清理通过。该检查不计作正式精度实验。')
        paragraph(doc, '恢复位置', '第93轮检查点完整验证通过，数据身份、协议、运行环境和检查点设置与旧任务完全一致；从第94轮开始，保留前93轮历史和最优权重，恢复模型、优化器、学习率调度器和Python、NumPy、Torch及数据生成器状态。A43剩7轮，B43、A44、B44各100轮，共剩307轮，不从头重跑。')
        live = state.get('training') or {}
        progress = f'第{live["epoch"]}轮{live.get("phase", "执行")}阶段' if live.get('epoch') else '恢复初始化阶段'
        paragraph(doc, '本次启动', f'任务状态为{state["status"]}，当前{progress}；是否完整保存第94轮及核验通过，以随后追加的恢复首轮记录为准。恢复目录为{job.name}。')
        paragraph(doc, '耗时与范围', f'按既有实测折算剩余训练约{plan["estimated_training_hours"]:.1f}小时，另有核验和负载波动余量。仍使用本机4060，不新增硬件或付费服务，无开机或周期触发、无自动重试。完成三种子A/B后停止，不自动开始C/D/E。保持电脑接电、避免休眠；如要打游戏，可再次要求暂停。')
        paragraph(doc, '下一步', '保存第94轮后自动用独立CPU进程核对完整历史、下一轮样本排列与生成器状态、学习率、优化器及BN步数、参数有限值和检查点来源，再维持剩余配对训练。')
    paragraph(doc, '复用依据', '上文恢复目录中的first_resumed_epoch_check.json。' if check_only else str(job / 'job_plan.json') + '；进度写入容错核验证据；恢复首轮完成后生成独立核验文件。')
    atomic_save(doc)


def append_step06(job, mode):
    doc = Document(JOURNAL)
    labels = {'review': '第五步结果审阅与第六步启动', 'check': '第六步完整数据恢复核验', 'completion': '第六步种子42探索完成'}
    marker = labels[mode] + '  ' + job.name
    if any(p.text == marker for p in doc.paragraphs):
        return
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=1 if mode != 'check' else 2)
    paragraph(doc, '记录时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    if mode == 'review':
        ab_job = Path(plan['ab_job'])
        data = json.loads((ab_job / 'three_seed_results.json').read_text(encoding='utf-8'))
        proof = json.loads((ab_job / 'final_verification.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass' and proof['three_seed_study_complete']
        paragraph(doc, '第五步完成', 'A/B在42、43、44三个配对种子下共6次训练，各100轮，于10月21日18时05分全部核验完成。每参数74400次更新，最优权重、样本排列和轮末随机状态配对及旧文件完整性通过；全部结果已自动追加到本工作记录。本次再次审阅完整结果与六份最终检查点身份。')
        fields = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
        table(doc, ['模型', '全局mIoU%', '人行道IoU%', '边界IoU%'], [[v, *(f"{data['summary'][v][k]['mean']:.3f} ± {data['summary'][v][k]['sample_std_ddof1']:.3f}" for k in fields)] for v in ('A', 'B')], [2, 5, 5, 5])
        paragraph(doc, '结果判断', 'A为ENet，B为原ReLU CA，上表为均值和样本标准差。CA在三个种子的三项指标中均为正收益：人行道IoU差值为1.086、0.958、1.269个百分点，均值1.104；边界IoU平均提高0.571个百分点，全局mIoU平均提高0.584个百分点。支持本协议下可重复的CA收益，不支持将CA失效作为预设结论；三个种子仍不足以声称普遍规律或统计显著。')
        paragraph(doc, '对课题构思的调整', '保留实际课题名称，机制研究按注意力收益条件、实现细节和部署取舍组织。历史负收益没有在统一训练中复现，旧比较存在权重衰减、指标与初始化等混杂，不能把原因直接归结为某一个参数。第六步继续检验ReLU与h-swish这一实现差异；h-swish对照本身不当作原创方法。暂不叠加门控、CBR或第二数据集。')
        paragraph(doc, '本次新增代码', '新增独立C训练入口和单种子托管脚本，复用原数据、损失、优化器、指标和训练函数，保留本地轴布局，仅将CA激活改为h-swish。C在设置、进度、初始化和检查点中均明确标为C，并使用独立检查点格式与源文件指纹；恢复A/B、不同种子或设置的检查点会在创建输出目录前拒绝。C只从配对初值开始，不继承训练过的B权重。原A/B代码、协议、指标与结果不改写。')
        paragraph(doc, 'GPU与入口核验', '四组检查通过：六份A/B最终检查点指纹及三种子B/C初始张量一致；768×1024、batch4下恢复ReLU后逐位复现B损失、BN状态和所有参数梯度；C在同一生产代码路径上进行7张训练、3张验证的两轮短跑，连续训练与独立进程恢复的完整检查点逐位一致，CA参数确有更新；拒绝A/B和不同种子检查点。短跑不计作正式精度结果，也不冒充完整2975/500图恢复验证。')
        paragraph(doc, '检查过程修正', '首批三次短跑均完成，完整恢复对比已通过，但检查脚本收尾读取UTF-8状态文件时因Windows默认GBK编码报错，未生成最终通过标记。修正检查脚本的显式UTF-8读取后，在新核验目录重新检查共同初值和GPU还原，并按C设置与源文件指纹复核原短跑产物、补齐检查点拒绝测试；保留首次失败记录，不重复执行已完成的三次短跑，不修改C模型或训练源码。')
        state = json.loads((job / 'job_status.json').read_text(encoding='utf-8'))
        paragraph(doc, '正式执行安排', f'当前任务状态{state["status"]}。先以正式设置完成C42的完整第1轮2975张训练与500张验证，保存后退出并核验；随后独立进程从C第1轮继续到100轮，在第2轮再次核对完整数据恢复连续性。约13至15小时仅为A/B耗时外推，C实际耗时需首轮实测。各阶段和完成结果自动尝试追加本记录，完成后停止，不自动开始C43/44或D/E。')
        paragraph(doc, '主线剩余工作', '第1至5步已完成；第6步进入单种子探索，第7至9步按证据决定是否开展。最终仍需当前模型的FP32导出一致性、同平台模型及端到端延迟与P95、内存、原型和真实场景真值验证。开题报告已有，最终论文与答辩材料未完成。原有忽略区域和拉伸缩放的敏感性须独立检查，不能混入本核心对照；验证集参与选最优轮，不能称独立测试集。')
        paragraph(doc, '资源与继续条件', '继续使用已有4060及本地数据，不租服务器或新增硬件，不上传模型和数据。一次性任务无开机或周期触发、无自动重试；暂停时在完整轮次边界保存，用户可随时要求暂停。C42完成核验后可让我审阅第六步结果，再决定是否补齐C的配对重复；正式C收益结论须有相应重复。')
    elif mode == 'check':
        proof = json.loads((job / 'formal_resume_verification.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass'
        paragraph(doc, '结果', 'C种子42已完成正式第1轮并独立进程恢复完成第2轮。每轮训练2975张、验证500张，前一轮历史、下一轮样本排列、来源与C身份保持一致；优化器和BN计数1488次，参数及学习率核验通过。完整数据恢复检查已补齐，不再只有小数据短跑证据。')
        paragraph(doc, '下一步', '继续C42至100轮并自动核验，单种子只作探索，不提前判断激活优劣。')
    else:
        proof = json.loads((job / 'final_verification.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass' and proof['C_seed42_complete']
        data = json.loads((job / 'c_seed42_results.json').read_text(encoding='utf-8'))
        paragraph(doc, '完成结果', 'C种子42各100轮完成；74400次更新、完整数据断点恢复、与B42的全部样本排列及最终随机状态、最优权重严格加载和旧产物完整性检查通过。以下A/B/C均为种子42各自按全局验证mIoU选择的最优轮，不与A/B三种子均值混列。')
        table(doc, ['模型', '最优轮', '全局mIoU%', '人行道IoU%', '边界IoU%'], [[r['variant'], r['best_epoch'], f"{r['best_global_miou_pct']:.3f}", f"{r['best_sidewalk_iou_pct']:.3f}", f"{r['best_boundary_iou_pct']:.3f}"] for r in data['results']], [2, 2, 4.3, 4.3, 4.4])
        paragraph(doc, '结论边界', 'C只完成一个种子，不能宣称实现改变带来稳定收益或失效。正式比较需补齐C43、44；后续候选须与合理的标准实现对照，不能只挑较弱基线。')
        paragraph(doc, '下一步', '本任务已停止。下次审阅C对A、B的各项差值和成本，再决定C的重复与必要的固定/可学习门控对照；部署与真实场景验证继续作为工业应用闭环。')
    paragraph(doc, '复用依据', str(job / 'job_plan.json') + '；GPU核验证据、完整首轮及恢复核验、完成后的C42结果均在本任务目录。')
    atomic_save(doc)


def append_repeat_pair(job, seed):
    doc = Document(JOURNAL)
    marker = f'剩余配对训练完成  种子{seed}  {job.name}'
    if any(p.text == marker for p in doc.paragraphs):
        return
    proof = json.loads((job / f'seed{seed}_final_verification.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'pass'
    data = json.loads((job / f'seed{seed}_results.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=2)
    paragraph(doc, '完成时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '结果', '本种子的A/B各100轮完成。每参数74400次更新、样本排列与轮末随机状态配对、最优权重严格加载及历史产物完整性核验通过。')
    table(doc, ['模型', '最优轮', '全局mIoU%', '人行道IoU%', '边界IoU%'], [[r['variant'], r['best_epoch'], f"{r['best_global_miou_pct']:.3f}", f"{r['best_sidewalk_iou_pct']:.3f}", f"{r['best_boundary_iou_pct']:.3f}"] for r in data['results']], [3, 2, 4, 4, 4])
    paragraph(doc, '下一步', '继续种子44；完成后汇总42、43、44的全部结果，暂不启动C/D/E。' if seed == 43 else '汇总42、43、44的全部结果并核验，暂不启动C/D/E。')
    atomic_save(doc)


def append_repeat_completion(job):
    doc = Document(JOURNAL)
    marker = '第五步三种子全部核验完成  ' + job.name
    if any(p.text == marker for p in doc.paragraphs):
        return
    proof = json.loads((job / 'final_verification.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'pass' and proof['three_seed_study_complete']
    data = json.loads((job / 'three_seed_results.json').read_text(encoding='utf-8'))
    doc.add_heading(marker, level=1)
    paragraph(doc, '完成时间', datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '结果', '种子42、43、44下的A/B共6次训练各100轮完成，配对与完整性核验通过。以下标准差为三个种子间的样本标准差，ddof=1。')
    fields = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
    table(doc, ['种子', '模型', '最优轮', '全局mIoU%', '人行道IoU%', '边界IoU%'], [[r['seed'], r['variant'], r['best_epoch'], *(f'{r[k]:.3f}' for k in fields)] for r in data['results']], [1.5, 1.5, 2, 4, 4, 4])
    table(doc, ['汇总', '全局mIoU', '人行道IoU', '边界IoU'], [[v, *(f"{data['summary'][v][k]['mean']:.3f} ± {data['summary'][v][k]['sample_std_ddof1']:.3f}" for k in fields)] for v in ('A', 'B')], [2, 5, 5, 5])
    delta = data['paired_CA_minus_ENet_percentage_points']
    paragraph(doc, '配对变化', 'CA减ENet的差值，单位为百分点：' + '；'.join(f'{name}平均{delta[k]["mean"]:.3f}，样本标准差{delta[k]["sample_std_ddof1"]:.3f}，种子42、43、44分别为' + '、'.join(f'{delta[k]["per_seed"][s]:.3f}' for s in ('42', '43', '44')) for k, name in zip(fields, ('全局mIoU', '人行道IoU', '边界IoU'))) + '。')
    paragraph(doc, '结论边界与下一步', '三个种子只支持本任务与固定协议下的收益方向和波动，不据此宣称普遍失效、普遍收益或统计显著。任务已停止，不自动训练新模型。下次先审阅三种子的各项配对差值，再开展C的CUDA全尺寸与独立进程恢复检查。')
    paragraph(doc, '复用依据', str(job / 'three_seed_results.json') + '；three_seed_results.csv；final_verification.json。')
    atomic_save(doc)


def append_sensitivity_status(job):
    state = json.loads((job / 'job_status.json').read_text(encoding='utf-8'))
    doc = Document(JOURNAL)
    doc.add_heading('固定权重敏感性评价  ' + job.name, level=2)
    doc.add_paragraph('记录时间：' + datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    if state['status'] == 'complete':
        proof = json.loads((job / 'final_verification.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass' and proof['full_core_metrics_reproduced']
        rows = json.loads((job / 'results.json').read_text(encoding='utf-8'))['results']
        paragraph(doc, '完成结果', '9组A/B/C最佳权重各重新推理500张验证图，原核心mIoU、人行道IoU及26像素边界指标均在核验容差内复现。'
            '另计算排除官方忽略像素的区域指标，以及6、13、26像素边界指标；逐图与逐城市的混淆矩阵、精确率、召回率和误检漏检数据均保存。没有重新训练、重选权重或覆盖旧结果。')
        table(doc, ['模型与种子', '原人行道IoU', '排除忽略区IoU', '原边界IoU'],
            [[r['variant'] + str(r['seed']), f'{r["policies"]["core"]["sidewalk_iou_pct"]:.3f}',
              f'{r["policies"]["valid_pixels"]["sidewalk_iou_pct"]:.3f}', f'{r["core_boundary"]["26"]["mean_pct"]:.3f}'] for r in rows], [3, 4.5, 5, 4.5])
        paragraph(doc, '解释边界与后续', '本次为相同权重、不同评价政策的敏感性检查，不代表采用新标签政策重新训练。'
            '验证集参与过模型选择，仍不能称独立测试；未把忽略区域人为边界当成官方边界指标。'
            '下一步核对三种子的排序及场景差异，结合固定训练样本的机制诊断进入D/E候选，并继续部署验证。用户已授权后续工作，无需重新审批普通实验步骤。')
    else:
        paragraph(doc, '当前状态', state['status'] + '。已生成的文件保留，未宣称完整评价完成。')
        if state.get('traceback'):
            paragraph(doc, '需处理的问题', state['traceback'][-1800:])
    paragraph(doc, '复用依据', str(job) + '；evaluation_plan.json、results.json、逐图与汇总文件和final_verification.json。')
    atomic_save(doc)


def append_c_repeats(job, phase, seed=None):
    plan = json.loads((job / 'job_plan.json').read_text(encoding='utf-8'))
    doc = Document(JOURNAL)
    entry_start = len(doc.paragraphs)
    title = f'C重复实验 {phase} 种子{seed if seed is not None else "43和44"} {job.name}'
    if any(p.text == title for p in doc.paragraphs):
        return
    doc.add_heading(title, level=2)
    doc.add_paragraph('记录时间：' + datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    if phase == 'start':
        paragraph(doc, '授权与范围', '用户明确要求自行完成后续计划并补齐C的两个种子。先执行C43、C44各100轮，C42及A/B结果继续复用。'
            '后续按已记录的评价、诊断、候选门控、部署和原型验收条件推进；实际街景采集与人工标注审核仍需用户参与。')
        paragraph(doc, '本次变化与检查', '新增种子感知的任务管理和核验脚本，冻结C训练源文件及科学协议不变。'
            '使用C42真实第1、2、100轮回归检查，与原审计结果一致；错误种子、错误模型身份、缺失样本及错误优化器步数均被拒绝；C43/44初始张量与各自B精确配对。'
            '沿用已通过的全尺寸GPU和独立进程恢复证据；每个新种子仍执行正式完整第1轮、独立进程恢复第2轮及最终第100轮审计。')
        paragraph(doc, '运行安排', '种子43完成并核验后顺序开始44；预计两次合计26至30小时，实际以日志为准。'
            '每个种子首轮训练2975张、验证500张，每轮744次更新。后台任务无开机或周期触发、无自动重试；失败保留检查点，用户要求暂停时按轮次边界保存。'
            '完成43、44及三种子汇总时自动追加本工作记录，Word被占用则标记待补记。训练期间接电并避免休眠，暂不并行开展GPU性能测量。')
        paragraph(doc, '已接续的下一阶段', '新增固定权重评价程序，先用人工可核对的小矩阵验证忽略像素计数，用空前景和完美预测验证三种边界宽度，再在A42、C42各两张真实验证图上通过试运行。'
            '本次一次性后台流水线在C43/44全部核验完成后自动启动9组最佳权重各500图评价，复核原核心指标，再输出忽略标签及6、13、26像素边界敏感性。'
            '试运行并不等于完整500图复现；完整结果通过后自动追加工作记录。训练失败或用户暂停时不进入后续评价。该流水线只自动执行已验收的训练和评价程序，未验证的D/E长训练尚未排入。')
        candidate_proof = ROOT / 'step_07_2026-10-24/candidate_cpu_checks_001/verification.json'
        export_proof = ROOT / 'step_10_2026-10-24/export_cpu_preflight_001/verification.json'
        demo_proof = ROOT / 'step_10_2026-10-24/prototype_check_002/verification.json'
        if candidate_proof.exists():
            assert json.loads(candidate_proof.read_text(encoding='utf-8'))['status'] == 'pass'
            paragraph(doc, '第七步并行准备', '新增D固定α和E可学习α组件，保留C的共享参数名、h-swish和轴方向。'
                '5项CPU检查通过：三个种子的共享张量与α初值一致，D有266992个参数、E只多1个；实际CA特征尺寸上D/E初始输出相同，α为0与1时分别精确恢复原特征和C门控，β及CA有非零有限梯度且β确实更新。'
                'E的β独立参数组权重衰减为0，共享权重仍为0.01。当前只通过组件CPU检查，没有训练精度结论；D/E正式入口、GPU全尺寸与独立进程恢复检查仍待完成。')
        if export_proof.exists() and demo_proof.exists():
            assert json.loads(export_proof.read_text(encoding='utf-8'))['status'] == 'pass'
            assert json.loads(demo_proof.read_text(encoding='utf-8'))['status'] == 'pass'
            paragraph(doc, '第十与十一步并行准备', 'A42与C42已导出固定batch1、768×1024的FP32 ONNX，onnx.checker通过。'
                '每个模型在同一张真实验证图上进行CPU对齐，最大logits绝对误差分别约5.33e-5和4.09e-5，argmax无差异，满足预设单图容差。'
                '这只是CPU单图导出预检，GPU数值、500图完整指标和正式速度测试均未完成。CPU导出不与正式GPU训练争用显卡。')
            paragraph(doc, '原型检查', '新增本地图片与视频推理原型，支持A/C模型选择、绿色人行道叠加、后端和耗时显示、结果保存，以及窗口按键切换。'
                '两张模型叠加输出及一个由3张验证图片合成的3帧视频读入、写出、重新解码检查通过，并调整了小画幅文字背景与换行；原始试运行输出保留。'
                '无界面模式已测，交互按键尚未经人工操作；3帧合成视频仅验证输入输出功能，不冒充真实街景视频或30分钟稳定性测试。显示耗时不是受控性能基准。'
                '原型脚本sidewalk_demo.py、导出模型models.json与结果位于本次step_10目录，可按后续部署验证继续复用。')
        paragraph(doc, '仍需补齐的条件', '已检查现有项目，旧海口脚本指向的dataset/haikou_blind目录当前不存在，尚未确认可直接用于独立验证的本地原图与人工审核掩码。'
            '继续推进不依赖它的训练、评价和工程工作；真实本地场景定量验证最终需要用户提供图像来源与复核标注。用户不需要逐项批准常规实验，但不能用生成掩码、未审核预测或公开验证图代替真实人工标注测试。')
    elif phase == 'resume':
        proof = json.loads((job / f'C_seed{seed}_resume_verification.json').read_text(encoding='utf-8'))
        assert proof['status'] == 'pass'
        paragraph(doc, '验证结果', f'C种子{seed}已从完整第1轮检查点由独立进程恢复并完成第2轮，1488次更新。'
            '全量训练与验证、历史记录、优化器与调度器、BN、配对样本顺序及检查点来源核验通过。继续到100轮。')
    elif phase == 'seed':
        row = json.loads((job / f'C_seed{seed}_result.json').read_text(encoding='utf-8'))
        paragraph(doc, '完成结果', f'C种子{seed}已完成100轮，74400次更新。最佳轮次{row["best_epoch"]}，'
            f'全局mIoU {row["best_global_miou_pct"]:.3f}%，人行道IoU {row["best_sidewalk_iou_pct"]:.3f}%，边界IoU {row["best_boundary_iou_pct"]:.3f}%。'
            '同种子B/C的全部样本顺序、最终随机状态、最佳权重及历史文件完整性核验通过。')
        paragraph(doc, '下一步', '开始C44并沿用相同验证条件。' if seed == 43 else '汇总A/B/C三种子结果，按计划进入固定权重评价补充。')
    elif phase == 'complete':
        data = json.loads((job / 'three_seed_ABC_results.json').read_text(encoding='utf-8'))
        assert json.loads((job / 'final_verification.json').read_text(encoding='utf-8'))['status'] == 'pass'
        fields = ('best_global_miou_pct', 'best_sidewalk_iou_pct', 'best_boundary_iou_pct')
        table(doc, ['模型', '全局mIoU', '人行道IoU', '边界IoU'],
            [[v, *(f'{data["summary"][v][k]["mean"]:.3f} ± {data["summary"][v][k]["sample_std_ddof1"]:.3f}' for k in fields)] for v in ('A', 'B', 'C')], [2, 5, 5, 5])
        paragraph(doc, '配对差值', 'C减B，单位为百分点：' + '；'.join(
            f'{name}平均{data["paired_percentage_points"]["C_minus_B"][k]["mean"]:.3f}，各个种子为' +
            '、'.join(f'{x:.3f}' for x in data['paired_percentage_points']['C_minus_B'][k]['per_seed'].values())
            for name, k in zip(('全局mIoU', '人行道IoU', '边界IoU'), fields)) + '。')
        paragraph(doc, '解释与后续', '至此A/B/C共9次100轮均已完成。均值与样本标准差来自三个配对种子，不宣称统计显著或独立测试集效果。'
            '后续先在相同最佳权重上检查忽略标签、边界宽度和逐城市误差，再依据固定训练样本诊断开展有界D/E探索；标准CA始终作为参照。')
    else:
        raise ValueError(phase)
    paragraph(doc, '复用依据', str(job) + '；job_plan.json、首轮与恢复核验证据、各种子结果和最终汇总。')
    if phase == 'start':
        for p in doc.paragraphs[entry_start:]:
            if not p.style.name.startswith('Heading'):
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(3)
    atomic_save(doc)


def append_study_review(job):
    evidence = json.loads((job / 'review_verification.json').read_text(encoding='utf-8'))
    labels = json.loads((job / 'label_audit.json').read_text(encoding='utf-8'))
    results = json.loads((job / 'verified_results.json').read_text(encoding='utf-8'))['results']
    plan = json.loads((job / 'next_experiments.json').read_text(encoding='utf-8'))
    probe = json.loads((job / 'deployment_backend_probe.json').read_text(encoding='utf-8'))
    assert evidence['status'] == labels['status'] == 'pass'
    title = '第六步完成复核与后续实验设计  2026年10月23日'
    doc = Document(JOURNAL)
    if any(p.text == title for p in doc.paragraphs):
        raise RuntimeError('This review is already recorded; do not duplicate the entry')
    doc.add_heading(title, level=1)
    doc.add_paragraph('记录时间：' + datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '本次结论', 'C种子42于10月23日9时47分完成100轮，训练进程累计用时'
        f'{evidence["C_training_process_wall_hours"]:.2f}小时。A/B三个配对种子共6次训练及C42共7次完整训练均已完成；'
        '当前没有长训练运行。本次执行结果复核、全量标签审计、部署后端探测及后续设计，没有启动C43/44或D/E。第1至5步完成，第6步尚缺C的两个重复；候选改进、正式部署、原型和毕业论文尚未完成，不以训练次数换算整篇毕设完成百分比。')
    paragraph(doc, '复核内容', '重新核验7个最终检查点的SHA256、模型及种子身份、100轮状态、最佳轮次与纯权重一致性，并严格加载模型。'
        '重新执行C的全数据、74400次更新、样本顺序及随机状态配对核查，旧受保护产物的指纹检查通过。'
        '从同一最佳轮次保存的全局混淆矩阵补算精确率、召回率和Dice；这不是重新推理，也不是独立测试结果。')
    table(doc, ['种子42模型', '全局mIoU', '人行道IoU', '边界IoU', '精确率', '召回率'],
        [[{'A': 'A ENet', 'B': 'B ReLU CA', 'C': 'C h-swish CA'}[r['variant']],
          *(f'{r[k]:.3f}' for k in ('best_global_miou_pct', 'best_sidewalk_iou_pct',
            'best_boundary_iou_pct', 'sidewalk_precision_pct', 'sidewalk_recall_pct'))]
         for r in results if r['seed'] == 42], [3.7, 2.65, 2.65, 2.65, 2.65, 2.7])
    paragraph(doc, '结果解释', '表中数值单位为百分比。C42相对B42的全局mIoU、人行道IoU和边界IoU分别增加0.103、0.231、0.499个百分点。'
        '单次探索不能证明h-swish稳定更优。C42精确率比B42降低约1.276个百分点，召回率提高约1.704个百分点，说明当前权重在减少漏检的同时增加了误检，不能称所有方面均改善。'
        '后续保留精确率、召回率和误检像素报告，不因观察到这一取舍就事后改变核心分类阈值。A/B三个种子已有一致正收益，后续研究问题应表述为注意力的收益条件、实现差异和精度与部署开销取舍。h-swish来自作者实现，不能作为本课题原创算法；历史负收益也不能直接归因于权重衰减。')
    doc.add_heading('本次发现的实验规范缺口', level=2)
    tr, va = labels['splits']['train'], labels['splits']['val']
    paragraph(doc, '标签与划分', f'扫描全部2975张训练及500张验证原始标签，在核心768×1024尺寸下，官方标记忽略类别的像素分别占'
        f'{tr["official_ignored_pixels_pct_at_core_resolution"]:.3f}%和{va["official_ignored_pixels_pct_at_core_resolution"]:.3f}%；'
        f'人行道分别占{tr["sidewalk_pixels_pct"]:.3f}%和{va["sidewalk_pixels_pct"]:.3f}%。'
        f'训练{tr["empty_sidewalk_images_at_core_resolution"]}张、验证{va["empty_sidewalk_images_at_core_resolution"]}张无前景，必须保留以考察误检。'
        '样本标识及城市跨划分不重叠，标签文件哈希无跨划分完全重复；本次没有做RGB图像内容哈希或近重复检测，不能据此保证所有形式的数据泄漏均已排除。'
        '数据文件大小与修改时间清单与训练记录一致；此次新建标签内容哈希基线，不追溯宣称历史像素已逐一核验。')
    paragraph(doc, '评价补充', '既有核心协议将非8标签全部当背景；保留所有原指标，另用相同最佳权重在500张验证图上排除官方忽略像素，输出区域指标敏感性表。'
        '验证集已经参与选权重，不称独立测试集。先补人行道精确率、召回率及逐图和逐城市混淆矩阵；按城市汇总时保留像素累积口径，不用城市均值偷换总指标。'
        '边界主口径仍为26像素，另预先固定6、13像素作敏感性分析并全量报告；不能将忽略区置零后产生的人造边界称为官方Boundary IoU。双方均空的边界图像按原协议排除，有误检的空前景图像仍计入。')
    paragraph(doc, '尺度限制', '原图2048×1024直接变为1024×768，改变了纵横比。核心配对实验继续保持该处理以确保可比。'
        '如果补1024×512的同权重推理，只能称几何与分辨率联合敏感性检查，不当作等比例处理单因素因果验证，也不与重新训练的分辨率消融混淆。')
    paragraph(doc, '部署可用性', f'本机ONNX {probe["onnx_version"]}和ONNX Runtime GPU {probe["onnxruntime_version"]}已存在。'
        '本次禁止CPU算子回退执行小型Add图，输出精确一致，运行剖析确认计算节点在CUDA执行，环境探测通过。'
        '该结果只证明小型算子的GPU后端可用；真实ENet/CA的导出正确性、全量指标和延迟仍需实测。未安装或升级训练环境，也未租服务器。')
    doc.add_heading('后续执行顺序和结束条件', level=2)
    paragraph(doc, '第一项 补齐标准对照', '先完成C43、C44各100轮，按实测预计合计26至30小时；不重跑已有A/B及C42。'
        '现有单次托管器的审计函数写死种子42，因此先新增面向43/44的托管与核验逻辑，保持冻结训练代码不变。'
        '通过初始张量、配对样本顺序、全尺寸运行、断点状态和停止标记检查后，才提交串行训练。全部完成后报告各种子、均值、样本标准差和C减B的配对差值。C无论结果如何都是后续应保留的标准实现参照。')
    paragraph(doc, '第二项 补充评价与机制观察', '在冻结权重上完成忽略标签和边界宽度敏感性评估，整理按城市及前景面积分组的错误。'
        'B/C机制诊断复用既定54张训练图和干预设置，观察门控、前景边界特征、匹配能量的均匀缩放及空间打乱。'
        '热图、幅值减小或推理时替换效果只提供线索；不能代替训练消融。若标签政策改变排序，先披露并限定结论，再判断是否需要小范围、配对的新协议训练。')
    paragraph(doc, '第三项 有边界的候选改进', '仅探索y=x×[(1−α)+α×(a_h×a_w)]。D固定α=0.1；E令α=sigmoid(β)，'
        'β初值为log(0.1/0.9)，两组初始强度一致，共享h-swish CA和骨干初值。共享参数训练规则不变，β使用相同学习率与调度、权重衰减为0，避免衰减直接把α推向0.5；该设置明确记入新协议。'
        '先做端点、梯度、全尺寸和独立进程恢复检查，再完整运行D42和E42，预计26至30小时；不用短跑精度选结构，不扫描α。')
    paragraph(doc, '候选取舍规则', '在D/E结果产生前规定：至少一个候选相对C42在人行道IoU或边界IoU有提升且另一项不降低，才进入补充种子阶段。'
        '这是探索筛选规则，不是稳定改进证明。若继续，D/E均补43、44，保留后来退化的结果，明确42用于探索、43/44是事先保留的重复种子；仍不把同一验证集称为独立确认数据。'
        '若E没有一致优于D，不宣称可学习强度起关键作用；若均不通过，保留负结果、停止该分支。公式本身不自动构成新颖性，后续需对照相关原始文献。')
    paragraph(doc, '第四项 正式部署与取舍', '兼容性工作可以提前，精度和速度测试不与训练同时进行。以A42、C42及至多一个最终候选为对象，先导出固定输入、batch1的FP32 ONNX。'
        '预先规定项目验收容差：logits有限，rtol=1e-3且atol=1e-4；argmax差异不超过0.1%；500图的人行道及主边界IoU差值绝对值各不超过0.05个百分点。'
        '这些是待执行的工程验收线，不是已经通过或通用标准；失败时记录定位及协议修订，不事后静默放宽。通过后才尝试FP16，INT8暂不列必做。')
    paragraph(doc, '性能与原型验收', '固定设备、电源模式、尺寸、batch1和输入集，预热50次，开展3轮各500帧计时，交替模型顺序并进行GPU同步。'
        '报告平均、中位数、P95、实际处理帧率、内存及显存，区分模型前向与包含读帧、预处理、传输、推理和叠加的端到端耗时；保存或界面刷新另列。'
        '通过真实模型节点剖析核实后端，不能用参数量推断速度。原型支持图片与视频、模型切换、区域叠加、耗时显示及结果保存，再做预先规定的30分钟连续运行。'
        '明确平台为RTX4060 Laptop，不能写成Jetson等专用嵌入式实测；是否满足实时要求由数据判断，不预先承诺30帧每秒。')
    paragraph(doc, '第五项 本地场景验证', '建议采集40至50张、至少5个独立地点或序列的本地照片，推理前按地点整组分开，约10至15张开发、30至35张留作最终观察。'
        '先定义可见人行道、障碍物及不确定区的标注规则；人工逐张复核掩码，模糊区域设为忽略。最终集只在模型确定后使用，继续据其调参则失去独立测试资格。'
        '少量便利样本不代表所有城市，无标注照片仅作定性展示。需要用户提供普通街景和审核标注含义；本系统只验证视觉人行道区域，不推断安全通行、测距或导盲决策。')
    paragraph(doc, '第六项 论文交付', '按任务界定与相关工作、数据和协议、注意力诊断与方法、配对结果与敏感性、部署和原型、结论与局限组织正文。'
        '对应开题的两项拟创新：诊断引导的轻量适配、区域与边界及实际延迟联合选型；算法增益由实验决定，不保证正结果。'
        '旧DeepLab、SBNet和CBR放入历史探索背景，若要作公平主表需另行配对。U-Net、大数据集、第二骨干、边界新损失、蒸馏及专用硬件均非当前必做项。'
        '已核对较新的开题报告副本，两份报告均支持现有4060和视觉子模块范围；本次未获得学院最终评分细则、篇幅和最终模板，不能宣称已满足全部校方硬性要求。')
    paragraph(doc, '工作量与预算', 'C重复2次必做约26至30小时；D/E探索2次约26至30小时；若进入完整重复，再增加4次约52至60小时。'
        '从当前起最多新增8次长训练，估计104至120小时，分阶段决策，不自动排满。近期1至2周优先补标准对照和评价，之后2至3周完成受控候选及部署，随后进行本地原型、标注和初稿，余下时间留给导师反馈。'
        '新增硬件与云费用目标0元；电费尚未测量，若50元包含电费，应按整机墙插功率、运行时长和实际电价另计，不能把显卡功耗等同整机耗电。')
    paragraph(doc, '本次文件变化', '新增独立只读审计脚本及本次机器证据，记录标签内容哈希、7组结果复核、CUDA小图探测和后续实验规则。'
        '只扩展单一工作记录维护脚本并追加本节，未修改原始数据、已完成权重或冻结训练源文件，未删除历史结果，未改写开题报告。本次没有新开长训练。')
    paragraph(doc, '下一步与用户操作', '下一步实施C43/44托管器并启动串行重复，结束后审阅三种子的配对差值，再进入候选方法。'
        '现在无需手工改文件；可直接提出“按计划补齐C的两个种子”。有普通本地街景时可用于准备标注。暂不需要购买硬件、租服务器或提供云账号。')
    paragraph(doc, '复用依据', str(job) + '；next_experiments.json固定本次后续规则；review_verification.json、verified_results.json、label_audit.json、label_inventory.json、deployment_backend_probe.json保留核验数据。')
    paragraph(doc, '原始资料', 'Cityscapes官方标签定义 https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py；'
        'CA作者代码 https://github.com/houqb/CoordAttention/blob/main/coordatt.py；'
        'ONNX Runtime量化说明 https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html。参考核查日期2026年10月23日。')
    atomic_save(doc)


def append_literature_review(job):
    plan = json.loads((job / 'plan_addendum.json').read_text(encoding='utf-8'))
    proof = json.loads((job / 'preparation_verification.json').read_text(encoding='utf-8'))
    assert proof['status'] == 'pass' and proof['candidate_CPU_checks'] == 'pass'
    title = '进度复核、近期模型调研与D/E入口准备  2026年10月24日'
    doc = Document(JOURNAL)
    if any(p.text == title for p in doc.paragraphs):
        raise RuntimeError('Literature review already recorded')
    entry_start = len(doc.paragraphs)
    doc.add_heading(title, level=1)
    doc.add_paragraph('记录时间：' + datetime.now().strftime('%Y年%m月%d日 %H时%M分'))
    paragraph(doc, '执行范围', '按用户要求先检查进度，再阅读公开GitHub及原始论文，调整后续步骤并推进代码准备。'
        '本次未能连接内置浏览器控制入口，实际使用网页检索工具直接读取公开仓库、配置和作者文档；没有声称操作过可见浏览器标签页，也未下载执行第三方工程或改变训练环境。')
    paragraph(doc, '进度与新结果', 'A/B的三个配对种子共6次100轮已完成；C42、C43已完成，C44正在训练。'
        'C43于15时19分完成100轮，最佳轮次69，全局mIoU 81.235%、人行道IoU 64.613%、主边界IoU 53.164%。'
        '相对B43分别下降0.259、0.519、0.140个百分点；C42的三项差值此前为正。当前结果呈现种子差异，不能称h-swish稳定改善。'
        'C43结果和校验文件均保留，C44结束并通过审计后，现有队列将继续9组权重的500图评价。')
    paragraph(doc, '记录补写', '本轮发现C43完成条目的自动写入失败标记，15时27分已成功重试写入。'
        '原始自动失败没有保留足够错误信息，原因未确定，不将其直接归咎于WPS或文件锁。托管进程内的pending列表仍可能保留旧标记，须以Word条目及补写证据核对；未修改正在运行的托管源码或状态。')
    doc.add_heading('近期模型与可行性判断', level=2)
    paragraph(doc, '调研口径', '检索截至2026年10月24日公开可读的一组相关项目，不称穷尽全部最新模型。'
        '区分论文年份、仓库维护状态、19类分割和本项目二分类，以及预训练、数据增强、输入尺寸与GPU差异。作者指标用于理解方案，未当作本机复现实测。')
    for ref in plan['references']:
        paragraph(doc, ref['name'] + '（' + ref['year'] + '）', ref['method'] + ref['assessment'])
    doc.add_heading('收敛后的研究路线', level=2)
    paragraph(doc, '创新表述', '将主线收敛为“轻量网络注意力适配条件的诊断、有界门控强度的配对验证、精度与实际部署开销的联合取舍”。'
        '注意力原公式、h-swish及标量残差门控都有前例，不作为首创；贡献是否成立取决于可复现结果与问题分析。'
        '同协议A/B已经显示CA有益，应撤回预设CA失效的叙述。当前证据也不能证明历史负收益一定由权重衰减导致。')
    paragraph(doc, '下一阶段顺序', 'C44完成→9模型固定权重评价→既定54张训练图B/C机制复核→D/E的GPU全尺寸与独立进程恢复验收→D42/E42各完整100轮→按原预声明决定是否补种子→正式部署和原型验收。'
        '本次只新增计划补充，未覆盖10月23日已冻结的计划和晋级规则。现有自动队列覆盖C44及评价，D/E正式训练尚未启动。')
    paragraph(doc, '增加的低成本分析', '预先固定按核心尺寸GT人行道面积占比分为0、(0,1%]、(1%,5%]、>5%四组，报告各组样本量、累积混淆矩阵及误检/漏检。'
        '该分析属于探索性分组，不将验证集变成独立测试集，也不据此事后调阈值。D/E统一同时报告相对A、B、C的结果；只超过C而落后B时不能称优于已有注意力。')
    paragraph(doc, '工业价值与工作量', '优先完成真实图片/视频处理、FP32导出全量一致性、FP16候选验证、模型与端到端耗时、P95和内存及30分钟稳定性。'
        '第二骨干或现代预训练参考最多选择一个，必须锁定版本、核验权重训练数据和标签映射，另表呈现；不现在重跑大模型。'
        'D/E若均无益，保留负结果并推进部署；仅在错误分析支持时再设计一个配对边界辅助监督实验。'
        '新增硬件与云服务支出仍为0元，电费未单独计量；不宣称已在Jetson/ARM上运行或可判断物理安全通行。')
    doc.add_heading('本轮实际完成及验收边界', level=2)
    paragraph(doc, '代码新增', '新增common_runner_mixed.py与check_mixed_runner_cpu.py。D固定α=0.1，E学习sigmoid(β)，共享初值和科学训练循环。'
        'E的β单独使用零权重衰减参数组；每轮保存α、完整优化器、调度器、随机状态及最佳权重。'
        '断点严格匹配模型、种子、协议、数据、运行环境和源码；正式训练入口要求GPU及机制复核验收，CPU证据不能解锁正式训练。')
    paragraph(doc, '检查结果', '7项CPU检查通过：D/E各自的独立进程恢复与连续运行全状态逐项完全一致；错误模型、种子、数据和源码及C检查点被拒绝；E参数恢复后仍以零衰减更新；'
        '完整768×1024前向输出有限，D/E初始logits完全一致；CPU证据无法冒充GPU验收。恢复实验采用128×256合成图，含4+3尾批，不能当作真实数据训练或GPU断点证明。'
        '首次检查脚本使用了错误的生成器键名sampler，检查在训练前失败；改为现有接口sample_order后另目录复测通过，失败日志保留。'
        '冻结源码、C队列及评价入口哈希复核通过，旧模型及实验结果未覆盖。')
    paragraph(doc, '尚未完成', 'D/E的GPU全尺寸、GPU独立恢复和正式训练未做；统一评价仍等待C44。已有ONNX与演示程序仅完成早期兼容性检查，不能称全量部署验收已完成。'
        '后续本地场景验证仍需用户采集或指出真实图片位置，并人工核对标签。论文最终正文、答辩及学校格式核对仍待推进。')
    paragraph(doc, '用户下一步', '当前无需手动启动任务。保持电源与正常散热，避免休眠；需要游戏时先要求按完整轮次暂停。'
        '按此前每100轮约13至15小时估算，C44预计10月25日凌晨至上午早段完成，具体受运行速度影响。可在明早询问“检查C44和统一评价是否完成，进入D/E验收”。')
    doc.add_heading('公开来源（访问日期2026年10月24日）', level=2)
    for ref in plan['references']:
        p = doc.add_paragraph(ref['name'] + '：' + ref['urls'][0])
        p.paragraph_format.space_after = Pt(3)
        for r in p.runs:
            r.font.size = Pt(9)
    paragraph(doc, '复用依据', str(job) + '；plan_addendum.json、preparation_verification.json；runner_cpu_checks_002/verification.json。')
    for p in doc.paragraphs[entry_start:]:
        if not p.style.name.startswith('Heading'):
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(4)
    atomic_save(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('initialize', 'cleanup', 'status', 'completion', 'project-review', 'repeat-pair', 'repeat-completion', 'recovery-start', 'recovery-check', 'step06-review', 'step06-check', 'step06-completion', 'study-review', 'c-repeats', 'sensitivity-status', 'literature-review'), required=True)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--seed', type=int, choices=(43, 44))
    parser.add_argument('--phase', choices=('start', 'resume', 'seed', 'complete'))
    parser.add_argument('--cleanup-json', type=Path)
    args = parser.parse_args()
    if args.mode == 'initialize':
        initialize()
    elif args.mode == 'cleanup':
        append_cleanup(args.cleanup_json)
    elif args.mode == 'status':
        append_status(args.job.resolve())
    elif args.mode == 'completion':
        append_completion(args.job.resolve())
    elif args.mode == 'project-review':
        append_project_review(args.job.resolve())
    elif args.mode == 'study-review':
        append_study_review(args.job.resolve())
    elif args.mode == 'c-repeats':
        append_c_repeats(args.job.resolve(), args.phase, args.seed)
    elif args.mode == 'sensitivity-status':
        append_sensitivity_status(args.job.resolve())
    elif args.mode == 'literature-review':
        append_literature_review(args.job.resolve())
    elif args.mode == 'repeat-pair':
        append_repeat_pair(args.job.resolve(), args.seed)
    elif args.mode in ('recovery-start', 'recovery-check'):
        append_recovery(args.job.resolve(), check_only=args.mode == 'recovery-check')
    elif args.mode.startswith('step06-'):
        append_step06(args.job.resolve(), args.mode.split('-', 1)[1])
    else:
        append_repeat_completion(args.job.resolve())
    print('Work record updated: ' + str(JOURNAL))
