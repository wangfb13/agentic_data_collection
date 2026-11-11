"""Generate a report/plot from frozen offline experiment outputs; never fits models."""
import argparse
import json
from pathlib import Path
import numpy as np

def build(root, diagnosis):
    root,diagnosis=Path(root),Path(diagnosis)
    summary=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    lookup={(r['suite'],r['mode'],r['budget']):r for r in summary['aggregate']}
    suites=[('session72','整 session 留出（仅 RSI 为测试域）',72),('repeat72','同 session 内完整重复试验组留出',72),('leave_formal48','57条正式 session 整体留出',48)]
    modes=[('fixed','固定随机顺序'),('simple_feedback','简单反馈'),('agent_evidence','证据规则驱动')]
    lines=['# 多种子、多分组药瓶 a 轴离线对照结果（2026-11-06）','',
      '结论：扩大评测后，24条优于72条的反转没有稳定复现；现有证据不支持证据驱动补采普遍优于对照。',
      '所有代码、计算、图表和结果都在5090。未启动机器人，新增真实交互0。','',
      '## 设计与数据',
      '使用同一批122条合格A、同一ContactPolicy学习器、500epochs和原参数，不根据外部评测调参。',
      '57条来自一个正式A session，65条来自12个RSI session。三种方法每组共享相同初始24条、开发集、33条历史B开发反馈及训练预算。',
      '26种预先冻结的分组方案 × 2个采样种子 × 3个模型种子，共1296个模型检查点。不是1296份独立实物证据。',
      '主对照沿用原采样策略：fixed为不读反馈的随机顺序；agent_evidence为参考重复/反馈难点/覆盖的本地规则，不是本次调用LLM。',
      '',
      '|评测方式|划分数|每次测试A|开发A|训练预算|',
      '|---|---:|---:|---:|---|',
      '|整session留出|10|32–36|10–12|24/48/72|',
      '|同session内重复试验组留出|10|32–36|10–12|24/48/72|',
      '|正式大session整体留出|6|57|11–16|24/48|','',
      '72预算时无法把57条大session整体留出，因此另外报告最高48预算的反向测试，不悄悄改变各方法预算。',
      '重复试验组不是连续时间块；组内试验在原采集时可能交错。零点监测附属于同瓶最近前序重复组，避免监测记录独自跨集合。',
      '整session方案覆盖65条不同RSI测试记录；重复组方案覆盖117条不同记录；大session测试57条固定记录，仅开发/训练配置变化。',
      '模型种子会通过B预测影响后续采样路径，主实验包含这种交互；另以固定采样选择的10种子诊断隔离训练随机性。','',
      '## 全部预算点：平均原始几何预测 MAE（mm）',
      '这里包括因开发门禁未批准的模型，避免删除失败训练产生幸存者偏差；原始MAE不等于实际会执行的动作误差。','',
      '|评测方式|方法|24条|48条|72条|',
      '|---|---|---:|---:|---:|']
    for suite,label,high in suites:
        for mode,name in modes:
            vals=[f"{lookup[suite,mode,n]['mae_mean_mm']:.3f}" if (suite,mode,n) in lookup else '不适用' for n in (24,48,72)]
            lines.append('|'+label+'|'+name+'|'+'|'.join(vals)+'|')
    lines+=['','两类72预算方案中，三种方法各自在全部10个划分的种子平均结果上，72条都优于24条。具体到单个种子仍有例外，不能声称每次补采都改善。','',
      '## 方法差异是否稳定','',
      '|最高共同预算|证据规则减固定 MAE|胜出划分|证据规则减简单反馈 MAE|胜出划分|',
      '|---|---:|---:|---:|---:|']
    for suite,label,high in suites:
        p={r['contrast']:r for r in summary['paired_outer_split_summary'] if r['suite']==suite and r['budget']==high}
        a,b=p['agent_evidence_minus_fixed'],p['agent_evidence_minus_simple_feedback']
        lines.append(f"|{label}，{high}条|{a['mean_difference_mm']:+.4f}|{a['outer_split_wins']}/{a['n_outer_splits']}|{b['mean_difference_mm']:+.4f}|{b['outer_split_wins']}/{b['n_outer_splits']}|")
    lines+=['','负值表示证据规则较好。两类72预算里优势约0.001–0.013mm，胜负随划分变化。',
      '在大session留出的48预算测试中，简单反馈明显更低；这是同一药瓶任务的分布变化测试，不能称作跨任务实验。',
      '因此可报告“在部分评测设置下有小幅平均改善”，不能报告稳定、显著或普遍优势，也不能证明更少真机交互达到完整skill能力。',
      '72条已接近76–80条候选池上限，各方法平均共享66.9–68.53条训练样本，多种策略最终看到的大部分数据相同；应同时查看48条预算与selection_overlap.json。',
      '按session等权后结论相近。但正向a子区域证据规则不占优：整session72为1.241mm、简单反馈1.096mm；重复组72为1.010与0.932mm。总体均值不能代表所有偏移方向。','',
      '## 为什么此前24条比72条好',
      '固定原始选择和划分，额外完成117次MLP拟合及9次线性诊断；原seed9个模型的预测复现误差不超过1e-9。',
      '1. 原评测仅5条且来自同一session：+10mm一条、−5mm两条、−10mm两条，没有零点。唯一+10点主导净恶化；其他4条在三个方法中合计误差都下降。',
      '2. +10点的原始预测误差从24条时0.749mm，变为72条时固定5.072、简单反馈3.274、证据规则1.651mm。固定72的这一点实际被不确定性门禁拒绝，并未发生新机器人动作。',
      '3. 原seed的24结果在10种子中第2好，固定72结果却是最差。固定数据选择、换10个训练种子后的测试均值如下：','',
      '|方法|24条MAE均值±种子SD|72条MAE均值±种子SD|72更好的种子数|',
      '|---|---:|---:|---:|',
      '|固定|0.792±0.360|0.703±0.333|6/10|',
      '|简单反馈|0.792±0.360|0.591±0.128|5/10|',
      '|证据规则|0.792±0.360|0.607±0.141|6/10|','',
      '4. 唯一+10点没有简单地落到OOD半径之外；固定72各bootstrap成员也都含+10样本，但成员预测分歧明显。增加至1500epochs、固定24条时的归一化尺度均未一致修复三种方法，不能归结为单个超参数。',
      '5. 新增数据分布确实变化、不均衡，但没有证据确认错误标签或参考漂移。所有session的T0一致，不等于能证明物体始终没有微动；不能据测试误差删除记录。',
      '可支持的解释是：原5条评测的单点敏感性与训练随机性共同导致了不稳健的反转；目前没有理由将其解释为“数据越多越差”。详细诊断见旁边skill_budget_diagnosis_20261106目录。','',
      '## 边界、审计与复现',
      '所有外部测试记录在训练/开发之外；独立审计核验1296个模型的训练/验证索引、仅训练集归一化及48,708条外部预测。1248个模型通过原几何门禁，48个未通过模型仍保留；不代表已取得新skill部署证书。见audit.json、summary.json与subgroup_metrics.json。',
      '各划分重复使用有限的122条数据；分位数仅描述本次划分/种子变化，不是独立总体置信区间。没有把重复次数当新物理样本数，也没有报告不合适的显著性p值。',
      '历史B反馈属于旧控制器，不能反事实验证新策略动作；历史数据也可能受旧探索方法影响。这仍是回顾性的有限数据池回放，不是前瞻实机消融。',
      '新skill的调用/恢复/退出真实效用、方法记忆有无收益、多方向与跨任务迁移都不由本次几何误差实验证明。','',
      '复现（在远程仓库，使用新的输出目录，原目录拒绝覆盖）：',
      '```bash',
      'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python skill_repeated_benchmark.py --out outputs/skill_repeated_new --workers 8',
      '.venv/bin/python -m bottle_scan.skill_budget_diagnosis --out outputs/skill_diagnosis_new',
      '.venv/bin/python -m pytest tests/test_skill_repeated_benchmark.py tests/test_skill_budget_diagnosis.py tests/test_skill_offline.py tests/test_contact_policy.py tests/test_agentic_learning.py -q',
      '```',
      '协议/哈希：protocol.json、protocol_hash.json；所有检查点/逐点预测：runs/；配对结果：summary.json；分组平衡：subgroup_metrics.json；来源审计：audit.json；耗时：compute_cost.json。']
    (root/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'fixed':'#576175','simple_feedback':'#e69f00','agent_evidence':'#0072b2'}
        labels={'fixed':'Fixed order','simple_feedback':'Simple feedback','agent_evidence':'Evidence rule'}
        titles={'session72':'Whole sessions held out (RSI test domain)','repeat72':'Within-session repeat groups held out','leave_formal48':'Formal session held out (57 test trials)'}
        fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
        for ax,(suite,_,high) in zip(axs,suites):
            for mode,_ in modes:
                rs=[lookup[suite,mode,n] for n in (24,48,72) if (suite,mode,n) in lookup]
                ax.plot([r['budget'] for r in rs],[r['mae_mean_mm'] for r in rs],marker='o',lw=2,color=colors[mode],label=labels[mode])
            ax.set_title(titles[suite],fontsize=10)
            ax.set_xlabel('Selected A training trials')
            ax.set_xticks([24,48,72] if high==72 else [24,48])
            ax.grid(alpha=.2)
            ax.set_ylim(.45,1.3)
        axs[0].set_ylabel('Mean held-out geometric MAE (mm)')
        axs[1].legend(fontsize=8)
        fig.suptitle('Offline replay: more data helps on average; evidence-rule superiority is not consistent',fontsize=11)
        fig.text(.5,.01,'Same learner and matched budgets. Reused observations; curves are descriptive, not independent population estimates.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,.94])
        fig.savefig(root/'learning_curves.png',dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    return str(root/'REPORT_zh.md')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='outputs/skill_repeated_20261106')
    p.add_argument('--diagnosis',default='outputs/skill_budget_diagnosis_20261106')
    a=p.parse_args();print(build(a.out,a.diagnosis))
