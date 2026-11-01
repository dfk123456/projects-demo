# -*- coding: utf-8 -*-
# ============================================================
# gen_sample_contract.py —— 生成用于验收的超长「高风险」样例合同 PDF
#
# 验收标准：上传一份包含【免责条款】【高额违约金】的合同，Agent 应识别风险点。
# 用法：
#   python scripts/gen_sample_contract.py                # 默认生成约 40 页
#   python scripts/gen_sample_contract.py --pages 200    # 生成 200 页压力样本
# ============================================================
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors                                  # noqa: E402
from reportlab.lib.enums import TA_CENTER                         # noqa: E402
from reportlab.lib.pagesizes import A4                            # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm                                # noqa: E402
from reportlab.pdfbase import pdfmetrics                          # noqa: E402
from reportlab.pdfbase.cidfonts import UnicodeCIDFont             # noqa: E402
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")

TITLE = "技术服务与设备采购总包合同"
PARTIES = """甲方（委托方）：星辰智能科技有限公司
乙方（服务方）：宏图系统集成（北京）有限公司
签订地点：北京市海淀区    签订日期：2026 年 3 月 18 日"""

# 高风险条款：故意设置的坑（高额违约金 / 免责 / 期限不明 / 单方解除 / 管辖不利 等）
RISK_CLAUSES = [
    ("第 8.2 条 违约责任", "若甲方未按约定支付任何一笔款项，应按未付金额日千分之五向乙方支付违约金，"
     "且乙方有权立即停止全部服务；如甲方逾期超过 5 日，还应另行支付合同总金额 30% 的惩罚性违约金，"
     "该等违约金不得低于人民币伍佰万元。"),
    ("第 9.1 条 责任限制", "乙方因任何原因对甲方承担的赔偿责任总额，无论基于合同或侵权，"
     "均不超过甲方已支付价款的 5%；除该限额外，乙方不承担任何间接损失、利润损失、数据损失及第三方索赔。"),
    ("第 10.3 条 免责条款", "对于因乙方原因导致的人身损害、财产损失及任何形式的连带损失，"
     "乙方均不承担赔偿责任，亦不接受任何形式的追偿与索赔，甲方应自行承担全部风险。"),
    ("第 12.1 条 验收与期限", "乙方完成交付后，由甲方在合理期限内组织验收；"
     "验收异议期以乙方通知为准，逾期未提出视为验收合格，甲方不得再行主张任何质量异议。"),
    ("第 13.5 条 付款条件", "甲方应在收到乙方开具发票后付款，具体付款时间由乙方根据资金安排另行通知，"
     "乙方通知后甲方应无条件按时支付，不得抵扣或主张同时履行抗辩。"),
    ("第 15.2 条 单方变更", "乙方有权在提前 3 日通知后单方调整服务内容、交付标准及价款，"
     "甲方应予接受并按调整后的标准支付费用，否则视为甲方根本违约。"),
    ("第 18.4 条 知识产权", "本合同项下产生的一切工作成果、源代码、文档及其衍生作品的知识产权"
     "均归乙方所有，甲方仅在付清全部价款后享有不可转让的使用权，不得反向工程或自行维护。"),
    ("第 20.1 条 保密义务", "甲方对本合同内容及合作过程中知悉的信息负有永久保密义务，"
     "未经乙方书面同意不得向第三方披露，且不因合同终止而免除。"),
    ("第 22.6 条 不可抗力", "因不可抗力事件导致乙方无法履约的，乙方不承担责任且有权单方解除合同；"
     "甲方已支付款项概不退还，尚未支付款项甲方仍应支付至乙方通知之日止。"),
    ("第 25.3 条 争议解决", "因本合同产生的一切争议，均提交乙方所在地仲裁委员会按其规则仲裁，"
     "仲裁裁决为终局裁决，甲方承担全部仲裁费、律师费及维权费用。"),
    ("第 28.1 条 合同解除", "乙方可在任何情况下提前 15 日书面通知解除本合同，"
     "无需说明理由且不承担任何赔偿责任；甲方仅在乙方同意后方可解除。"),
    ("第 31.2 条 自动续约", "本合同期满后自动顺延三年，除非甲方在到期前 6 个月书面提出不再续约，"
     "续约期间服务费按本合同标准上浮不低于 20% 执行。"),
]

NORMAL_CLAUSES = [
    ("第 {n} 条 服务地点", "服务履行地点由双方在本合同附件二中约定，双方应各自指派项目负责人对接日常事项。"),
    ("第 {n} 条 人员管理", "乙方派驻现场的人员应遵守甲方的园区管理规定，双方按周核对考勤与工时记录。"),
    ("第 {n} 条 文档交付", "乙方应按季度提交运行维护月报、巡检记录与年度总结报告，内容应真实、完整。"),
    ("第 {n} 条 安全规范", "双方在作业过程中应严格遵守国家安全生产相关法律法规，建立健全安全管理制度。"),
    ("第 {n} 条 通知送达", "本合同项下的通知以书面形式作出，可通过专人送达、挂号信或双方确认的电子邮箱送达。"),
    ("第 {n} 条 可分割性", "本合同任一条款被认定无效或不可执行，不影响其余条款的效力。"),
    ("第 {n} 条 附件效力", "本合同附件为本合同不可分割的组成部分，与正文具有同等法律效力。"),
    ("第 {n} 条 文本份数", "本合同一式肆份，双方各执贰份，具有同等法律效力，自双方签字盖章之日起生效。"),
]


def build_pages(total_pages: int):
    """构造 PDF 内容流：开头放高风险条款，后续重复常规条款把文档撑到指定页数。"""
    from reportlab.platypus import Flowable

    styles = getSampleStyleSheet()
    cn = ParagraphStyle("cn", parent=styles["Normal"], fontName="STSong-Light", fontSize=10.5,
                        leading=17, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=cn, fontSize=20, alignment=TA_CENTER, leading=28, spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=cn, fontSize=13, leading=20, spaceBefore=8, spaceAfter=6)
    body = ParagraphStyle("body", parent=cn, textIndent=0)

    story = [Paragraph(TITLE, h1), Spacer(1, 6), Paragraph(PARTIES.replace("\n", "<br/>"), cn),
             Spacer(1, 10),
             Paragraph("鉴于条款", h2),
             Paragraph("甲乙双方本着平等互利、协商一致的原则，就技术服务与设备采购事宜达成一致，"
                       "共同订立本合同，以资信守。", body)]

    n = 0
    for title, text in RISK_CLAUSES:
        n += 1
        story += [Paragraph(title, h2), Paragraph(text, body)]

    # 用常规条款把页数撑到目标（实测每页约 4 条，接近真实合同的文字密度）
    per_page = 4
    target_items = max(0, (total_pages - 6) * per_page)
    for i in range(target_items):
        tpl_title, tpl_text = NORMAL_CLAUSES[i % len(NORMAL_CLAUSES)]
        num = 100 + i
        story.append(Paragraph(tpl_title.format(n=num), h2))
        story.append(Paragraph(tpl_text.format(n=num) if "{n}" in tpl_text else tpl_text, body))
        story.append(Paragraph(f"双方确认，本条第 {num} 款所述事项如需变更，应经双方书面协商一致并签署补充协议，"
                               f"任何单方变更均不发生效力；未尽事宜按相关法律法规执行。", body))
        if (i + 1) % per_page == 0:
            story.append(PageBreak())

    story += [Spacer(1, 20), Paragraph("甲方（盖章）：____________    授权代表：____________", body),
              Paragraph("乙方（盖章）：____________    授权代表：____________", body)]
    return story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40, help="生成页数（默认 40，200 页用于性能验收）")
    ap.add_argument("--out", default="", help="输出文件路径")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    out = args.out or os.path.join(OUT_DIR, f"样例高风险合同_{args.pages}页.pdf")

    doc = SimpleDocTemplate(out, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            title=TITLE, author="星辰智能科技")
    doc.build(build_pages(args.pages))
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"已生成样例合同：{out}（{args.pages} 页目标，{size_mb:.2f}MB）")


if __name__ == "__main__":
    main()
