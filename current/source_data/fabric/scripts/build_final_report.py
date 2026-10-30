"""Create an evidence-linked Chinese PDF/Markdown interpretation from frozen results."""
from pathlib import Path
import csv
import hashlib
import json
import math
from xml.sax.saxutils import escape
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

OUT=Path(__file__).resolve().parents[1]


def readcsv(name):
    with (OUT/'results'/name).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))


def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    verification=json.loads((OUT/'audit/verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed'
    summary=readcsv('main_metrics.csv');primary={r['method']:r for r in summary if r['study']=='primary355'}
    clean={r['method']:r for r in summary if r['study']=='clean353'}
    splits=json.loads((OUT/'protocol/photo_splits.json').read_text(encoding='utf-8'))
    classes=sorted({r['class_name'] for r in splits});qi=np.array([r['index'] for r in splits if r['role']=='test'])
    labels=np.array([splits[i]['label'] for i in qi])
    # Post-hoc descriptive paired contrasts; no model selection is changed.
    keys=['duplicate:a452+a456' if classes[int(label)] in ('a452','a456') else classes[int(label)] for label in labels]
    cluster_names=sorted(set(keys));groups=np.array([cluster_names.index(key) for key in keys]);assert len(cluster_names)==354
    weights=np.random.default_rng(20261029).multinomial(354,np.full(354,1/354),size=2000)
    den=np.bincount(groups,minlength=354)
    hits={}
    for method in primary:
        with np.load(OUT/'predictions/scores'/f'{method}.npz') as data:
            runs=[data[k] for k in data.files if k.startswith('test_')]
            hits[method]=np.mean([(np.argmax(score,axis=1)==labels).astype(float) for score in runs],axis=0)
    contrasts=[]
    for treatment,baseline in [('dinov2_center','resnet50_center'),('dinov2_rgb_fusion','dinov2_center'),
                                ('dinov2_lbp_fusion','dinov2_center'),('resnet50_full','resnet50_partial'),
                                ('dinov2_rgb_fusion','resnet50_full')]:
        values=hits[treatment]-hits[baseline]
        sums=np.bincount(groups,weights=values,minlength=354)
        draws=(weights@sums)/(weights@den);lo,hi=np.quantile(draws,[.025,.975])
        contrasts.append(dict(treatment=treatment,baseline=baseline,n_queries=1082,bootstrap_clusters=354,
                              difference_pp=float(values.mean()*100),ci_low_pp=float(lo*100),ci_high_pp=float(hi*100),
                              status='post-hoc descriptive contrast; no selection or multiplicity correction'))
    writecsv(OUT/'results/additional_descriptive_paired_contrasts.csv',contrasts)
    aliases={'rgb_center':'RGB直方图','gray_center':'灰度直方图','lbp_center':'LBP',
             'resnet50_center':'冻结ResNet50','dinov2_center':'冻结DINOv2','clip_center':'冻结CLIP',
             'resnet50_linear':'ResNet50线性头','resnet50_partial':'ResNet50末阶段微调','resnet50_full':'ResNet50全量微调',
             'dinov2_rgb_fusion':'DINOv2＋RGB','dinov2_lbp_fusion':'DINOv2＋LBP'}
    for name in ['resnet50','dinov2']:
        for n in [1,5,10,20,30]:aliases[f'{name}_A{n}']=('ResNet50' if name=='resnet50' else 'DINOv2')+f' A{n}'
    def f(value,digits=2):return f'{float(value):.{digits}f}'
    def interval(row):return f"{f(row['recall1'])} [{f(row['recall1_ci_low'])}, {f(row['recall1_ci_high'])}]"
    pages=[]
    def page(title):
        value=[('title',title)];pages.append(value);return value
    def para(p,text):p.append(('p',text))
    def heading(p,text):p.append(('h',text))
    def table(p,headers,rows,widths=None):p.append(('table',(headers,rows,widths)))

    p=page('T1K+ Fabric 外部检索实验结果报告')
    para(p,'完成日期：2026-10-29。训练、特征提取、最终测试和主结果独立复算均已完成。本报告只交付新增实验结果，未修改论文或补充材料，未进行投稿或数据上传。')
    heading(p,'核心结果')
    para(p,'单参考条件下，DINOv2＋RGB的Top-1点估计为97.60%，冻结DINOv2为95.75%；ResNet50末阶段与全量微调三种子平均分别为97.01%和97.04%。这些是本次固定原图划分与有限预算下的结果，不是对所有皮革或布料的普遍保证。')
    para(p,'颜色有用，但仅靠颜色不够：RGB为88.45%，明显低于冻结DINOv2；在相同727条查询上，增加第二张真实参考将DINOv2从95.60%提高到99.04%。真实参考数量与数字增强视图数量必须分开解释。')
    shown=['rgb_center','resnet50_center','dinov2_center','clip_center','dinov2_A30','resnet50_partial','resnet50_full','dinov2_rgb_fusion']
    table(p,['方法','Top-1与95%区间','Top-3','Top-5'],
          [[aliases[m],interval(primary[m]),f(primary[m]['recall3']),f(primary[m]['recall5'])] for m in shown],[155,190,75,75])
    heading(p,'数据角色与完整性')
    table(p,['集合','官方标签行数','实际不同照片数'],[['参考','355','354'],['验证','355','354'],['测试','1,082','1,079'],['总计','1,792','1,787']],[165,165,165])
    para(p,'所有原图哈希与下载清单一致，逐张完整解码。官方编号a452与a456的5张照片逐张字节相同；它们的集合角色已同步，没有完全重复文件或像素跨集合。主结果保留355类严格编号，另给出排除这两个编号及候选的353类敏感性结果。')
    para(p,'区间由2,000次来源身份簇重采样计算；重复编号对合为一个簇，共354簇。增强池或训练种子的结果先在每条查询内平均，再计算区间；种子SD另列。')

    p=page('全部方法与参考平均结果')
    para(p,'下表均在同一套355类、1,082条测试查询上评价。R@k为正确编号进入前k名的比例，MRR为正确编号排名倒数的平均值。A1/A5/A10/A20/A30表示同一真实参考照片的数字增强视图数；A1是随机增强，与确定性中心裁剪不同。')
    table(p,['方法','R@1','R@3','R@5','MRR','R@1 SD','353类R@1'],
          [[aliases[m],f(r['recall1']),f(r['recall3']),f(r['recall5']),f(r['mrr'],4),f(r['recall1_run_sd']),f(clean[m]['recall1'])] for m,r in primary.items()],
          [145,57,57,57,62,57,60])
    para(p,'冻结与融合方法只有1个确定性运行，SD列记0，不表示采集不确定性为零；其查询不确定性仍由来源簇区间描述。增强与训练方法的SD来自3个固定参考池或训练种子。')
    heading(p,'平均更稳定，不等于性能必然单调提高')
    para(p,'ResNet50的池间平均角度由随机N1的23.61°降至N30的4.46°，DINOv2由24.13°降至4.52°。但ResNet50的R@1在A20为95.53%、A30为95.47%，没有随视图数严格单调上升。')
    para(p,'相对确定性中心基线，ResNet50 A30提高0.83个百分点，配对95%区间[0.06,1.67]；DINOv2 A30提高1.05个百分点，区间[-0.09,2.30]。后者在355类主分析中跨零，不能将点估计优势写成稳定确定的提升。')

    p=page('训练预算、模型选择与配对解释')
    para(p,'共执行12次候选训练，保留9个选定模型：每种训练范围在种子42比较两个学习率，再以选定学习率分别训练种子43和44。每次均为10轮、3,550次图像呈现、230次更新，batch=16，Adam权重衰减0.0001。头使用参考初始化的余弦分类器，logit尺度20。')
    para(p,'线性头学习率候选0.001/0.01；末阶段及全量微调骨干候选0.00001/0.0001，头学习率为骨干10倍。BatchNorm运行统计固定。只用355条验证记录选择学习率和最早最佳轮次，测试不参与选择。12次运行累计记录训练时间约59.5分钟；该值不是端到端流水线墙钟耗时。')
    selection=json.loads((OUT/'results/adaptation_selection.json').read_text(encoding='utf-8'))
    table(p,['范围','种子','学习率','选中轮次','验证命中/355'],
          [[r['mode'],str(r['seed']),str(r['lr']),str(r['selected_epoch']),str(r['validation_correct'])] for r in selection],
          [120,70,115,90,100])
    heading(p,'与冻结ResNet50的配对比较')
    paired=readcsv('paired_recall1_differences.csv')
    rows=[]
    for r in paired:
        if r['study']=='primary355' and r['treatment'] in ['resnet50_linear','resnet50_partial','resnet50_full']:
            rows.append([aliases[r['treatment']],f"+{f(r['recall1_difference_pp'])}",f"[{f(r['ci_low_pp'])}, {f(r['ci_high_pp'])}]"])
    table(p,['方法','R@1差值/百分点','配对95%区间'],rows,[210,125,160])
    heading(p,'报告阶段的补充配对描述')
    table(p,['比较','差值/百分点','配对95%区间'],
          [[aliases[r['treatment']]+' - '+aliases[r['baseline']],f(r['difference_pp']),f"[{f(r['ci_low_pp'])}, {f(r['ci_high_pp'])}]"] for r in contrasts],
          [270,85,140])
    para(p,'补充比较在报告阶段计算，没有改变任何模型或权重选择，属于未做多重比较校正的探索性描述。全量与末阶段微调的平均R@1只差约0.03个百分点，不能据此认定全量训练更优，也不能由不显著推断等价。')

    p=page('真实参考预算、图库选择与未知实例')
    heading(p,'相同查询上的1张与2张真实参考')
    para(p,'第二张参考在比较前从原测试池移出，两种预算均使用同样的727条查询（725张不同照片），不拿不同测试集合的准确率作提升比较。')
    budget=readcsv('real_reference_budget.csv');bmap={r['method']:r for r in budget};bp={r['method']:r for r in readcsv('real_reference_budget_paired.csv')}
    table(p,['冻结方法','1参考R@1','2参考R@1','提升/百分点','配对95%区间'],
          [[aliases[m+'_center'],f(bmap[m+'_R1']['recall1']),f(bmap[m+'_R2']['recall1']),f(bp[m]['recall1_two_minus_one_pp']),f"[{f(bp[m]['ci_low_pp'])}, {f(bp[m]['ci_high_pp'])}]"] for m in ['rgb','gray','lbp','resnet50','dinov2','clip']],
          [135,75,75,85,125])
    heading(p,'五种替代图库的描述性敏感性')
    alternate=readcsv('reference_choice_sensitivity.csv');r=[]
    for m in ['rgb','resnet50','dinov2','clip']:
        v=[float(row['recall1']) for row in alternate if row['method']==m+'_center']
        r.append([aliases[m+'_center'],f(np.mean(v))+' ± '+f(np.std(v,ddof=1)),f(min(v))+' - '+f(max(v))])
    table(p,['方法','五图库R@1均值±SD','范围'],r,[165,165,165])
    para(p,'这些图库重用同一批照片，查询池包含除当前参考外的其他照片，属于替代协议；不能把其均值与主划分准确率直接视作同一测试集的改进，也不能称为五次独立采集。')
    heading(p,'未知Fabric编号的拒识诊断')
    para(p,'排除两个重复编号后，按类别分为282已知、35校准未知、36测试未知。阈值只由已知验证照片与校准未知照片选择；测试含855条已知查询和181张测试未知照片。仅评价冻结描述符，避免适配模型已经训练过这些未知编号。')
    opened=[r for r in readcsv('open_set_metrics.csv') if r['detector']=='maximum_cosine']
    table(p,['方法','AUROC','已知接受%','未知拒识%','平衡准确率%'],
          [[aliases[r['method']],f(r['auroc'],4),f(100*float(r['known_accept_rate'])),f(100*float(r['unknown_reject_rate'])),f(100*float(r['test_balanced_accuracy']))] for r in opened],
          [145,80,90,90,90])
    para(p,'接受率不等于正确识别率。ResNet50最大余弦的已知接受率90.18%，同时被接受且正确识别的比例为88.54%；当要求95%已知接受时，未知误接受率仍约24.86%。这些结果不能证明工厂中的相近未知皮革拒识已解决。')

    p=page('对审核问题的补充程度与证据边界')
    heading(p,'本次结果新增了哪些证据')
    mapping=[
        ['真实多查询','355类每类4-8张原图；1,082条留出测试记录','补充了外部布料的跨照片验证；不补回原皮革来源记录'],
        ['外部任务匹配','按具体表面编号检索，不是只按材质大类分类','更接近身份检索；布料仍不等于皮革复采'],
        ['现代表示基线','同像素比较ResNet50、DINOv2和CLIP','未实现RADAM、DTPNet、VORTEX等专门纹理读出'],
        ['训练预算公平','三范围匹配呈现量/更新数，验证集选择，三种子','10轮有限预算，不代表充分调参或最优训练'],
        ['参考构建机制','共享增强池、嵌套N、池间角度与配对查询效果','构建稳定不自动等于新会话鲁棒性'],
        ['未知材料','类别分离校准/测试未知布料编号','不是专家认证的相近未知皮革；阈值不可直接迁移']]
    table(p,['关注点','新增结果','仍需保留的边界'],mapping,[78,190,227])
    para(p,'上表对应用户指定的“询问投稿前是否调整格式”和“论文独立审核：顶刊差距与教授意见”中关于来源、重复真实查询、基线覆盖及适配预算的讨论。期刊格式要求没有参与实验参数选择，论文内容也未修改。')
    heading(p,'可以支持的结论')
    para(p,'在本次公开Fabric实例集合中，少量真实参考与预训练表示能实现较高的留出照片检索；颜色信息仍有互补价值，但颜色直方图不能代替全部学习表征。增加真实参考与参考平均分别带来不同证据，训练适配的收益依赖数据与预算。')
    heading(p,'不能据此支持的结论')
    para(p,'不能宣称原203类皮革的采集来源与独立性已解决；不能宣称跨设备、跨会话、制造批次或工厂部署已验证。EXIF与完全重复检查不排除近重复或同次拍摄相关性。353类只是评价敏感性，不能把它写成重新训练的干净353类实验。')
    heading(p,'复核与交付')
    para(p,'主结果独立复算了42行汇总、50,854条逐查询记录及约4,183万项候选分数排名，并核对12次训练的历史、9个所选检查点及SHA-256。4个主文/补充源文件与PDF、1,792个原图文件哈希均保持不变。复核范围不包括独立重训。')
    auxiliary=OUT/'audit/auxiliary_verification.json'
    if auxiliary.exists() and json.loads(auxiliary.read_text(encoding='utf-8')).get('status')=='passed':
        para(p,'参考预算、图库敏感性、参考池稳定性和开放集阈值/指标也通过独立数值复算，记录见audit/auxiliary_verification.json。')
    else:
        para(p,'辅助分析由固定评价脚本生成；独立数值复算记录以audit/auxiliary_verification.json的实际状态为准。')
    para(p,'核心文件：results/main_metrics.csv、per_run_metrics.csv、paired_recall1_differences.csv、real_reference_budget.csv、open_set_metrics.csv；预测在predictions/，划分在protocol/，审计在audit/，训练检查点在checkpoints/。结果包保留原始数表、分数、协议和复核脚本；大体积像素缓存及训练参数仍留在本地目录。')
    para(p,'数据来源：Cusano, Napoletano and Schettini (2021), T1K+: A Database for Benchmarking Color Texture Classification and Retrieval Methods. DOI: 10.3390/s21031010。数据仅用于非商业研究。')

    # Markdown and PDF share the same ordered content and numbers.
    md=[]
    for items in pages:
        for kind,value in items:
            if kind=='title':md.extend(['# '+value,''])
            elif kind=='h':md.extend(['## '+value,''])
            elif kind=='p':md.extend([value,''])
            elif kind=='table':
                headers,body,widths=value;md.append('| '+' | '.join(headers)+' |');md.append('| '+' | '.join(['---']*len(headers))+' |')
                md.extend('| '+' | '.join(map(str,row))+' |' for row in body);md.append('')
    md_path=OUT/'T1K_Fabric_实验结果与解释报告.md';md_path.write_text('\n'.join(md),encoding='utf-8')
    pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('YaHeiBold',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
    body=ParagraphStyle('body',fontName='YaHei',fontSize=9,leading=14,wordWrap='CJK',spaceAfter=7,textColor=colors.HexColor('#20252b'))
    title=ParagraphStyle('title',parent=body,fontName='YaHeiBold',fontSize=16,leading=23,spaceAfter=14)
    hstyle=ParagraphStyle('head',parent=body,fontName='YaHeiBold',fontSize=11,leading=16,spaceBefore=8,spaceAfter=6)
    cell=ParagraphStyle('cell',parent=body,fontSize=7.6,leading=11,spaceAfter=0)
    header=ParagraphStyle('header',parent=cell,fontName='YaHeiBold',textColor=colors.white)
    pdf_path=OUT/'T1K_Fabric_实验结果与解释报告.pdf'
    story=[]
    for pi,items in enumerate(pages):
        if pi:story.append(PageBreak())
        for kind,value in items:
            if kind=='title':story.append(Paragraph(escape(value),title))
            elif kind=='h':story.append(Paragraph(escape(value),hstyle))
            elif kind=='p':story.append(Paragraph(escape(value),body))
            else:
                headers,rows,widths=value
                factor=495/sum(widths)
                table_data=[[Paragraph(escape(str(x)),header) for x in headers]]+[[Paragraph(escape(str(x)),cell) for x in row] for row in rows]
                tab=Table(table_data,colWidths=[w*factor for w in widths],repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#263e53')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f4f7')]),
                    ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d4dbe2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                    ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.extend([tab,Spacer(1,8)])
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('YaHei',7);canvas.setFillColor(colors.HexColor('#68717c'))
        canvas.drawString(50,24,'T1K+ Fabric | 2026-10-29 | 固定原图划分与有限训练预算')
        canvas.drawRightString(A4[0]-50,24,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=38,bottomMargin=40,
                          title='T1K+ Fabric 外部检索实验结果报告',author='Experimental analysis')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('REPORT',md_path,pdf_path)
    print('POSTHOC',contrasts)


if __name__=='__main__':main()
