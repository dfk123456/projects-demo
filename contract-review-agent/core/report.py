# ============================================================
# report.py —— 审查报告数据模型 + Word 导出
#
# 报告结构（前后端共享）：
# {
#   "task_id": "...", "filename": "...", "pages": 0, "chunks": 0,
#   "model": {"name": "...", "model": "..."},
#   "created_at": "...", "elapsed_sec": 0.0,
#   "executive_summary": "...", "key_advices": [...],
#   "overall_risk_level": "高/中/低",
#   "statistics": {"高":0,"中":0,"低":0,"total":0},
#   "risks": [ {id, clause, location, quote, risk_type, description,
#               suggestion, legal_basis, level, page, confidence} ]
# }
# ============================================================
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import config
from core.logging_setup import setup_logger

logger = setup_logger()

EMPTY_REPORT = {
    "task_id": "", "filename": "", "pages": 0, "chunks": 0,
    "model": {"id": "", "name": "", "model": ""},
    "created_at": "", "elapsed_sec": 0.0,
    "executive_summary": "", "key_advices": [], "overall_risk_level": "中",
    "statistics": {"高": 0, "中": 0, "低": 0, "total": 0},
    "risks": [],
}


# ------------------------------------------------------------
# 规范化 / 去重 / 统计
# ------------------------------------------------------------
def normalize_level(level: Any) -> str:
    """把任意输入归一到 高/中/低。"""
    text = str(level or "")
    for lv in config.RISK_LEVELS:
        if lv in text:
            return lv
    text_l = text.lower()
    if "high" in text_l or "severe" in text_l:
        return "高"
    if "low" in text_l:
        return "低"
    return "中"


def normalize_risk(raw: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """把模型/人工给出的风险点补齐为标准字段。"""
    page = raw.get("page")
    if page in (None, "", 0):
        m = re.search(r"第\s*(\d+)\s*页", str(raw.get("location", "")))
        page = int(m.group(1)) if m else 0
    return {
        "id": str(raw.get("id") or f"r{index + 1}"),
        "clause": str(raw.get("clause") or "未编号"),
        "location": str(raw.get("location") or ""),
        "quote": str(raw.get("quote") or ""),
        "risk_type": str(raw.get("risk_type") or "其他"),
        "description": str(raw.get("description") or ""),
        "suggestion": str(raw.get("suggestion") or ""),
        "legal_basis": str(raw.get("legal_basis") or ""),
        "level": normalize_level(raw.get("level")),
        "confidence": float(raw.get("confidence") or 0.0),
        "page": int(page or 0),
    }


def dedupe_risks(risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去重：同风险类型 + 原文摘录高度相似（或完全相同）视为同一条。"""
    def key_of(r):
        quote = re.sub(r"\s+", "", r.get("quote", ""))[:30]
        return (re.sub(r"\s+", "", r.get("risk_type", "")), quote)

    seen: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for r in risks:
        k = key_of(r)
        if k not in seen:
            seen[k] = r
            order.append(k)
        else:
            old = seen[k]
            # 保留信息量更大的一条，法条做并集
            if len(r.get("description", "")) > len(old.get("description", "")):
                seen[k] = r
                old = r
            bases = [x for x in [old.get("legal_basis", ""), r.get("legal_basis", "")] if x]
            old["legal_basis"] = "；".join(dict.fromkeys(bases))[:300]
    result = [seen[k] for k in order]
    result.sort(key=lambda x: (config.RISK_LEVEL_ORDER.get(x["level"], 9), x.get("page", 0)))
    for i, r in enumerate(result, start=1):
        r["id"] = f"r{i}"
    return result


def compute_statistics(risks: List[Dict[str, Any]]) -> Dict[str, int]:
    stat = {"高": 0, "中": 0, "低": 0, "total": len(risks)}
    for r in risks:
        stat[normalize_level(r.get("level"))] += 1
    return stat


def finalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """统一收口：标准化风险点 → 去重 → 排序 → 重算统计与整体等级。"""
    risks = [normalize_risk(r, i) for i, r in enumerate(report.get("risks", []))]
    risks = dedupe_risks(risks)
    report["risks"] = risks
    report["statistics"] = compute_statistics(risks)
    report["overall_risk_level"] = _overall_level(report["statistics"],
                                                  report.get("overall_risk_level"))
    report.setdefault("key_advices", [])
    report.setdefault("executive_summary", "")
    return report


def _overall_level(stat: Dict[str, int], original: str = "") -> str:
    """整体风险等级：存在高风险即为高；否则中为多则中；维持人工选择优先。"""
    if str(original) in config.RISK_LEVELS:
        return str(original)
    if stat.get("高", 0) >= 1:
        return "高"
    if stat.get("中", 0) >= 2:
        return "中"
    return "中" if stat.get("total", 0) else "低"


# ------------------------------------------------------------
# Word 导出
# ------------------------------------------------------------
def export_docx(report: Dict[str, Any], out_path: str) -> str:
    """根据报告生成 Word 文件，返回文件路径。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = Document()

    # 默认中文字体
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(11)

    title = doc.add_heading(config.DOCX_TITLE, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(f"生成时间：{report.get('created_at') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    "
                      f"审查模型：{(report.get('model') or {}).get('name', '-')}")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # 基本信息表
    info = doc.add_table(rows=0, cols=2)
    info.style = "Light Grid Accent 1"
    rows = [
        ("合同名称", report.get("filename", "-")),
        ("文档页数", f"{report.get('pages', 0)} 页"),
        ("分析片段", f"{report.get('chunks', 0)} 个"),
        ("审查耗时", f"{report.get('elapsed_sec', 0):.1f} 秒"),
        ("风险总数", f"{report.get('statistics', {}).get('total', 0)} 条（高 "
                   f"{report.get('statistics', {}).get('高', 0)} / 中 "
                   f"{report.get('statistics', {}).get('中', 0)} / 低 "
                   f"{report.get('statistics', {}).get('低', 0)}）"),
        ("整体风险等级", report.get("overall_risk_level", "-")),
    ]
    for k, v in rows:
        cells = info.add_row().cells
        cells[0].text = k
        cells[1].text = str(v)

    # 结论
    doc.add_heading("一、审查结论", level=1)
    doc.add_paragraph(report.get("executive_summary") or "（暂无）")

    advices = report.get("key_advices") or []
    if advices:
        doc.add_heading("二、整体处理建议", level=1)
        for i, a in enumerate(advices, start=1):
            doc.add_paragraph(f"{i}. {a}", style="List Number")

    # 风险明细
    doc.add_heading("三、风险条款明细", level=1)
    risks = report.get("risks") or []
    if not risks:
        doc.add_paragraph("本次审查未发现明显风险条款。")
    else:
        table = doc.add_table(rows=1, cols=5)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        for i, name in enumerate(["序号", "风险位置", "风险类型", "等级", "参考法条"]):
            hdr[i].text = name
        for idx, r in enumerate(risks, start=1):
            cells = table.add_row().cells
            cells[0].text = str(idx)
            cells[1].text = f"{r.get('location', '')}\n{r.get('clause', '')}".strip()
            cells[2].text = r.get("risk_type", "")
            cells[3].text = r.get("level", "")
            cells[4].text = r.get("legal_basis", "")

        doc.add_paragraph()
        for idx, r in enumerate(risks, start=1):
            doc.add_heading(f"{idx}. 【{r.get('level', '')}】{r.get('risk_type', '')} —— {r.get('clause', '')}",
                            level=2)
            para = doc.add_paragraph()
            para.add_run("风险位置：").bold = True
            para.add_run(str(r.get("location", "")))
            if r.get("quote"):
                q = doc.add_paragraph()
                q.add_run("原文摘录：").bold = True
                q.add_run(str(r.get("quote", "")))
            d = doc.add_paragraph()
            d.add_run("风险描述：").bold = True
            d.add_run(str(r.get("description", "")))
            s = doc.add_paragraph()
            s.add_run("修改建议：").bold = True
            s.add_run(str(r.get("suggestion", "")))
            lb = doc.add_paragraph()
            lb.add_run("参考法条：").bold = True
            lb.add_run(str(r.get("legal_basis") or "（未匹配）"))

    doc.add_paragraph()
    foot = doc.add_paragraph()
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = foot.add_run(config.DOCX_COMPANY + " · 本报告由 AI 辅助生成，最终以执业律师复核意见为准")
    frun.font.size = Pt(9)
    frun.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    doc.save(out_path)
    logger.info("Word 报告已导出：%s", out_path)
    return out_path


def export_markdown(report: Dict[str, Any]) -> str:
    """生成 Markdown 版本报告文本（便于下载/预览）。"""
    lines = [f"# {config.DOCX_TITLE}", "",
             f"- 合同名称：{report.get('filename', '-')}",
             f"- 页数/片段：{report.get('pages', 0)} 页 / {report.get('chunks', 0)} 段",
             f"- 整体风险等级：{report.get('overall_risk_level', '-')}",
             f"- 风险总数：{report.get('statistics', {}).get('total', 0)}", ""]
    if report.get("executive_summary"):
        lines += ["## 审查结论", report["executive_summary"], ""]
    advices = report.get("key_advices") or []
    if advices:
        lines += ["## 整体处理建议"] + [f"{i}. {a}" for i, a in enumerate(advices, 1)] + [""]
    lines.append("## 风险条款明细")
    for idx, r in enumerate(report.get("risks") or [], 1):
        lines += [f"### {idx}. 【{r.get('level')}】{r.get('risk_type')}",
                  f"- 位置：{r.get('location')}",
                  f"- 摘录：{r.get('quote')}",
                  f"- 描述：{r.get('description')}",
                  f"- 建议：{r.get('suggestion')}",
                  f"- 法条：{r.get('legal_basis') or '（未匹配）'}", ""]
    return "\n".join(lines)
