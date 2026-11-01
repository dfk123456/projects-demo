# -*- coding: utf-8 -*-
# ============================================================
# ingest_laws.py —— 构建法律法规 RAG 知识库
# 用法：
#   python scripts/ingest_laws.py               # 增量入库
#   python scripts/ingest_laws.py --rebuild     # 清空重建
#   python scripts/ingest_laws.py --embed bge-m3-siliconflow
# ============================================================
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                   # noqa: E402
from core.rag import LawKnowledgeBase           # noqa: E402
from core.logging_setup import setup_logger     # noqa: E402

logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(description="法律法规知识库入库")
    parser.add_argument("--rebuild", action="store_true", help="清空后重建")
    parser.add_argument("--embed", default=config.DEFAULT_EMBED_MODEL, help="向量模型 id")
    parser.add_argument("--dir", default=config.KNOWLEDGE_DIR, help="知识文档目录")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    t0 = time.time()
    kb = LawKnowledgeBase(embed_id=args.embed)
    result = kb.build(knowledge_dir=args.dir, rebuild=args.rebuild)
    print(f"\n入库完成：文件 {result['files']} 个，新增 {result['added']} 条，"
          f"知识库共 {result['total']} 条，耗时 {time.time() - t0:.2f}s")

    # 检索自测
    for q in ["违约金过高怎么办", "免责条款免除了对方全部责任", "付款期限没有约定"]:
        refs = kb.retrieve(q, top_k=2)
        print(f"\n[检索测试] {q}")
        for r in refs:
            print(f"  - {r['reference']}（score={r['score']}）{r['excerpt'][:40]}…")


if __name__ == "__main__":
    main()
