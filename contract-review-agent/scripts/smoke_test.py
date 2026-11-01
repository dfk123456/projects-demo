# -*- coding: utf-8 -*-
# ============================================================
# smoke_test.py —— 端到端冒烟测试（不依赖 Web 服务）
#
# 覆盖：文档解析 → 切分 → LangGraph 循环 Agent → RAG 法条 → 汇总 → Word 导出
# 用法：
#   python scripts/smoke_test.py                          # 用 40 页样例合同
#   python scripts/smoke_test.py --file data/samples/xx.pdf --model mimo-flash
#   python scripts/smoke_test.py --pages 200              # 性能：200 页 ≤120s
# ============================================================
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                     # noqa: E402
from core.logging_setup import setup_logger       # noqa: E402
from core.rag import LawKnowledgeBase             # noqa: E402
from core.report import export_docx, export_markdown  # noqa: E402
from graph.builder import run_review              # noqa: E402

logger = setup_logger()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_sample(pages: int) -> str:
    path = os.path.join(ROOT, "data", "samples", f"样例高风险合同_{pages}页.pdf")
    if not os.path.exists(path):
        os.system(f'"{sys.executable}" "{os.path.join(ROOT, "scripts", "gen_sample_contract.py")}" --pages {pages}')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="", help="待审查的合同文件路径")
    ap.add_argument("--pages", type=int, default=40, help="样例合同页数")
    ap.add_argument("--model", default=config.DEFAULT_MODEL, help="模型 id")
    ap.add_argument("--embed", default=config.DEFAULT_EMBED_MODEL, help="向量模型 id")
    ap.add_argument("--no-kb", action="store_true", help="跳过知识库自动建库")
    args = ap.parse_args()

    path = args.file or ensure_sample(args.pages)
    print(f"\n待审查文档：{path}")
    assert os.path.exists(path), "文件不存在"

    if not args.no_kb:
        kb = LawKnowledgeBase(embed_id=args.embed)
        if kb.count() <= 0:
            print("知识库为空，先建库…")
            print(kb.build(rebuild=False))
        else:
            print(f"法条知识库就绪：{kb.count()} 条")

    task_id = time.strftime("%Y%m%d%H%M%S") + "-smoke"
    print(f"\n开始审查（模型={args.model}，预算={config.TIME_BUDGET_SEC}s，"
          f"每轮批量={config.BATCH_SIZE}，并发={config.MAX_WORKERS}）")

    t0 = time.time()

    def cb(event):
        if event.get("type") == "log":
            print("  ·", event["message"])
        elif event.get("type") == "progress":
            print(f"    [{int(event['progress'] * 100):>3}%] {event['stage']}")

    report = run_review(task_id, path, os.path.basename(path),
                        model_id=args.model, embed_id=args.embed, progress_cb=cb)
    # 注意：run_review 通过 storage 读取文件，这里把本地文件注册进去
    total = time.time() - t0

    stat = report.get("statistics", {})
    print(f"\n===== 审查完成（{total:.1f}s）=====")
    print(f"页数 {report.get('pages')}｜片段 {report.get('chunks')}｜"
          f"大模型调用 {report.get('llm_calls')} 次｜法条命中 {report.get('rag_hits')} 条")
    print(f"风险合计 {stat.get('total')} 条：高 {stat.get('高')} / 中 {stat.get('中')} / 低 {stat.get('低')}")
    print("\n结论：", report.get("executive_summary", "")[:400])
    for i, r in enumerate(report.get("risks", [])[:10], 1):
        print(f"\n{i}. 【{r['level']}】{r['risk_type']}（{r['location']}）")
        print("   摘录：", r["quote"][:60])
        print("   描述：", r["description"][:120])
        print("   法条：", r["legal_basis"][:120])

    out_dir = os.path.join(config.EXPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, f"冒烟测试报告_{task_id}.docx")
    export_docx(report, docx_path)
    with open(os.path.join(out_dir, f"冒烟测试报告_{task_id}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nWord 已导出：{docx_path}")
    print("提示：Markdown 导出长度 =", len(export_markdown(report)))


if __name__ == "__main__":
    main()
