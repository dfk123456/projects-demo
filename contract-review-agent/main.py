# -*- coding: utf-8 -*-
# ============================================================
# main.py —— 项目启动入口
#   python main.py                 # 启动 Web 服务（默认 http://127.0.0.1:8000）
#   python main.py --port 8080     # 指定端口
#   python main.py --model mimo-flash   # 指定默认模型
# ============================================================
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_web(args):
    import config
    from api.startup import bootstrap
    from core.logging_setup import setup_logger

    if args.model:
        config.DEFAULT_MODEL = args.model
    if args.embed:
        config.DEFAULT_EMBED_MODEL = args.embed
    config.WEB_PORT = args.port
    config.WEB_HOST = args.host

    logger = setup_logger()
    bootstrap()
    logger.info("服务启动：http://%s:%s", config.WEB_HOST, config.WEB_PORT)

    import uvicorn
    from api.main_api import app
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level="info")


def cmd_review(args):
    """命令行直接审查一份合同。"""
    import config
    from graph.builder import run_review
    from core.logging_setup import setup_logger

    setup_logger()
    report = run_review("cli-" + str(int(__import__("time").time())), args.file,
                        os.path.basename(args.file), model_id=args.model or config.DEFAULT_MODEL)
    print("\n===== 审查报告 =====")
    print("风险合计：", report.get("statistics"))
    for i, r in enumerate(report.get("risks", []), 1):
        print(f"{i}. 【{r['level']}】{r['risk_type']}｜{r['location']}\n   {r['description'][:120]}\n   法条：{r['legal_basis'][:100]}")


def cmd_ingest(args):
    import config
    from core.rag import LawKnowledgeBase
    from core.logging_setup import setup_logger

    setup_logger()
    kb = LawKnowledgeBase(embed_id=args.embed or config.DEFAULT_EMBED_MODEL)
    print(kb.build(rebuild=args.rebuild))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="长文档深度分析 Agent · 合同审查系统")
    sub = parser.add_subparsers(dest="cmd")

    w = sub.add_parser("web", help="启动 Web 服务")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--model", default="")
    w.add_argument("--embed", default="")
    w.set_defaults(func=cmd_web)

    r = sub.add_parser("review", help="命令行审查合同")
    r.add_argument("--file", required=True)
    r.add_argument("--model", default="")
    r.set_defaults(func=cmd_review)

    i = sub.add_parser("ingest", help="构建法条知识库")
    i.add_argument("--rebuild", action="store_true")
    i.add_argument("--embed", default="")
    i.set_defaults(func=cmd_ingest)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        args = parser.parse_args(["web"])
    args.func(args)
