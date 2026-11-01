# ============================================================
# nodes.py —— LangGraph 循环工作流的全部节点
#
# 循环结构体现在两处：
#   ① 片段循环：analyze_batch → route_next_batch →（回到）analyze_batch，
#      每次推进 cursor += BATCH_SIZE，直到所有大片段处理完或时间预算耗尽；
#   ② 复核循环：enrich → route_after_enrich → recheck →（回到）enrich，
#      对缺失法条/低置信度的风险点做定向复核，最多 MAX_RECHECK 轮。
# ============================================================
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import config
from core import prompts
from core.chunker import chunk_pages, tail_summary
from core.doc_parser import parse_document
from core.llm_client import get_llm_client
from core.logging_setup import setup_logger
from core.rag import LawKnowledgeBase
from core.report import EMPTY_REPORT, finalize_report, normalize_risk
from core.storage import get_storage
from graph.state import ReviewState, elapsed, push_log, set_progress, time_left

logger = setup_logger()

_TMP_ROOT = os.path.join(config.DATA_DIR, "tmp")


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def materialize(state: ReviewState) -> str:
    """把待审查文件落到本地可读路径并返回。

    支持三种输入：① 本地绝对路径（脚本直连测试）；② 存储中的 key（Web 上传）。
    """
    key = state.get("file_path", "")
    if os.path.exists(key) and not key.startswith(config.MINIO_BUCKET_CONTRACT):
        return key                                      # 本地路径（脚本 / 命令行直接审查）
    filename = state.get("filename") or os.path.basename(key)
    local_dir = os.path.join(_TMP_ROOT, state.get("task_id", "default"))
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    data = get_storage().read(key)
    if not data:
        raise FileNotFoundError(f"文件不存在或为空：{key}")
    with open(local_path, "wb") as fh:
        fh.write(data)
    return local_path


def _risk_payload_for_prompt(risks: List[Dict[str, Any]], limit: int = 40) -> str:
    """压缩风险点，喂给汇总/复核模型（控制 token）。"""
    slim = []
    for r in risks[:limit]:
        slim.append({
            "id": r.get("id"), "clause": r.get("clause", ""),
            "location": r.get("location", ""),
            "quote": str(r.get("quote", ""))[:60],
            "risk_type": r.get("risk_type", ""),
            "description": str(r.get("description", ""))[:160],
            "suggestion": str(r.get("suggestion", ""))[:160],
            "legal_basis": str(r.get("legal_basis", ""))[:120],
            "level": r.get("level", "中"), "confidence": r.get("confidence", 0.0),
        })
    return json.dumps(slim, ensure_ascii=False)


# ------------------------------------------------------------
# 节点 1：文档解析 + 切分
# ------------------------------------------------------------
def node_parse(state: ReviewState) -> dict:
    set_progress(state, "文档解析与切片", 0.06)
    local_path = materialize(state)
    t0 = time.time()
    pages, engine = parse_document(local_path)
    push_log(state, f"文档解析完成（引擎：{engine}），共 {len(pages)} 页，耗时 {time.time() - t0:.1f}s")

    chunks = chunk_pages(pages)
    if not chunks:
        state["error"] = "未能从文档中提取到有效文本（可能是扫描件图片型 PDF，需 OCR）"
        push_log(state, "❌ " + state["error"])
        return {"pages": len(pages), "chunks": [], "parser": engine, "error": state["error"]}

    push_log(state,
             f"切分完成：{len(chunks)} 个大片段（每片约 {config.CHUNK_SIZE} 字，"
             f"重叠 {config.CHUNK_OVERLAP} 字，跨片段携带上下文摘要）")
    return {
        "pages": len(pages), "chunks": chunks, "cursor": 0, "prev_summary": "",
        "parser": engine, "started_at": time.time(),
        "deadline": time.time() + config.TIME_BUDGET_SEC,
    }


# ------------------------------------------------------------
# 节点 2：批次分析（循环体，内部并发调用大模型）
# ------------------------------------------------------------
def _analyze_one(chunk: Dict[str, Any], seq: int, total: int, prev_summary: str,
                 model_id: str) -> Tuple[List[Dict[str, Any]], str]:
    """分析单个大片段，返回 (风险点列表, 本片段摘要)。"""
    llm = get_llm_client()
    prev_hint = (f"【上一片段上下文摘要】{prev_summary}\n" if prev_summary and config.CARRY_PREV_SUMMARY else "")
    prompt = prompts.render(prompts.RISK_USER, page_start=chunk.get("page_start"),
                            page_end=chunk.get("page_end"), seq=seq, total=total,
                            prev_hint=prev_hint, chunk=chunk["text"])
    system = prompts.render(prompts.RISK_SYSTEM, max_risks=config.MAX_RISK_PER_CHUNK)
    data = llm.chat_json(prompt, fallback={"risks": [], "summary": ""},
                         model_id=model_id, system=system)
    raw_risks = data.get("risks") or []
    risks = []
    for k, item in enumerate(raw_risks[:config.MAX_RISK_PER_CHUNK]):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item["id"] = f"c{chunk.get('index')}-{k}"
        item["page"] = chunk.get("page_start")
        item["chunk_index"] = chunk.get("index")
        risks.append(normalize_risk(item, k))
    summary = str(data.get("summary") or "").strip() or tail_summary(chunk["text"])
    return risks, summary


def node_analyze_batch(state: ReviewState) -> dict:
    chunks = state.get("chunks") or []
    cursor = int(state.get("cursor") or 0)
    batch = chunks[cursor: cursor + config.BATCH_SIZE]
    if not batch:
        return {}
    total = len(chunks)
    set_progress(state, f"风险识别中（第 {cursor + 1}-{cursor + len(batch)} / {total} 片段）",
                 0.12 + 0.6 * cursor / max(total, 1))

    results: List[Tuple[List[Dict], str]] = []
    workers = max(1, min(config.MAX_WORKERS, len(batch)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analyzer") as pool:
        futures = [pool.submit(_analyze_one, chunk, cursor + i + 1, total,
                               state.get("prev_summary", ""), state.get("model_id", ""))
                   for i, chunk in enumerate(batch)]
        for fu in futures:
            try:
                results.append(fu.result())
            except Exception as exc:
                logger.warning("片段分析失败：%s", str(exc)[:160])
                results.append(([], ""))

    new_risks: List[Dict[str, Any]] = []
    summaries = []
    for risks, summary in results:
        new_risks.extend(risks)
        if summary:
            summaries.append(summary)
    push_log(state,
             f"片段 {cursor + 1}-{cursor + len(batch)} 分析完成：新增 {len(new_risks)} 个风险点，"
             f"累计 {len(state.get('candidate_risks') or []) + len(new_risks)} 个"
             f"（剩余预算 {max(0, time_left(state)):.0f}s）")

    return {
        "cursor": cursor + len(batch),
        "candidate_risks": (state.get("candidate_risks") or []) + new_risks,
        "prev_summary": summaries[-1] if summaries else state.get("prev_summary", ""),
        "llm_calls": int(state.get("llm_calls") or 0) + len(batch),
    }


def route_next_batch(state: ReviewState) -> str:
    """循环控制①：还有片段且时间预算充足 → 继续循环分析；否则进入法条补充。"""
    chunks = state.get("chunks") or []
    cursor = int(state.get("cursor") or 0)
    if state.get("error"):
        return "aggregate"
    if cursor < len(chunks):
        if time_left(state) > 3:
            return "analyze_batch"
        state["truncated"] = True
        push_log(state, f"⏱ 时间预算（{config.TIME_BUDGET_SEC}s）耗尽，剩余 {len(chunks) - cursor} 个片段未处理")
        return "enrich"
    return "enrich"


# ------------------------------------------------------------
# 节点 3：RAG 法条补充
# ------------------------------------------------------------
def node_enrich(state: ReviewState) -> dict:
    set_progress(state, "检索法律法规依据", 0.78)
    kb = LawKnowledgeBase(embed_id=state.get("embed_id", ""))
    if kb.count() == 0:
        push_log(state, "⚠ 法条知识库为空，跳过法条检索（可运行 scripts/ingest_laws.py 建库）")
        return {}
    risks = state.get("candidate_risks") or []
    enriched = set(state.get("enriched_ids") or [])
    for r in risks:
        if str(r.get("legal_basis") or "").strip():
            continue
        refs = kb.enrich(r)
        if refs:
            r["legal_basis"] = "；".join(x["reference"] for x in refs)
            r["legal_excerpt"] = refs[0].get("excerpt", "")
            enriched.add(str(r.get("id")))
    push_log(state, f"法条检索完成：{len(enriched)} 个风险点补充了法律依据")
    return {"candidate_risks": risks, "rag_hits": len(enriched), "enriched_ids": sorted(enriched)}


def route_after_enrich(state: ReviewState) -> str:
    """循环控制②：存在缺法条/低置信的无把握风险点 → 进入复核循环（最多 MAX_RECHECK 轮）。"""
    if not config.ENABLE_RECHECK:
        return "aggregate"
    if state.get("error"):
        return "aggregate"
    if int(state.get("recheck_count") or 0) >= config.MAX_RECHECK:
        return "aggregate"
    risks = state.get("candidate_risks") or []
    suspicious = [r for r in risks
                  if (not str(r.get("legal_basis") or "").strip())
                  or float(r.get("confidence") or 0) < 0.7]
    if not suspicious:
        return "aggregate"
    return "recheck"


# ------------------------------------------------------------
# 节点 4：定向复核（第二轮循环体）
# ------------------------------------------------------------
def node_recheck(state: ReviewState) -> dict:
    round_no = int(state.get("recheck_count") or 0) + 1
    set_progress(state, f"风险点复核（第 {round_no} 轮）", 0.85)
    risks = state.get("candidate_risks") or []
    suspicious = [r for r in risks
                  if (not str(r.get("legal_basis") or "").strip())
                  or float(r.get("confidence") or 0) < 0.7][:15]
    if not suspicious:
        return {"recheck_count": round_no}

    llm = get_llm_client()
    kept = 0
    try:
        prompt = prompts.render(prompts.RECHECK_USER, items=_risk_payload_for_prompt(suspicious, 15))
        data = llm.chat_json(prompt, fallback={"items": []}, model_id=state.get("model_id", ""),
                             system=prompts.RECHECK_SYSTEM)
        if isinstance(data, list):                 # 部分模型直接返回数组而非对象
            data = {"items": data}
        items = data.get("items") or []
        by_id = {str(r.get("id")): r for r in suspicious}
        for item in items:
            if not isinstance(item, dict):
                continue
            target = by_id.get(str(item.get("id")))
            if not target:
                continue
            if not item.get("keep", True):
                risks.remove(target)
                continue
            for field in ("risk_type", "description", "suggestion", "level", "legal_basis"):
                val = item.get(field)
                if val:
                    target[field] = str(val)
            target["confidence"] = min(1.0, float(target.get("confidence") or 0) + 0.25)
            kept += 1
    except Exception as exc:
        logger.warning("复核调用失败：%s", str(exc)[:160])

    push_log(state, f"复核第 {round_no} 轮完成：保留/修正 {kept} 条，剩余风险点 {len(risks)} 条")
    return {"candidate_risks": risks, "recheck_count": round_no,
            "llm_calls": int(state.get("llm_calls") or 0) + 1}


# ------------------------------------------------------------
# 节点 5：汇总生成报告
# ------------------------------------------------------------
def node_aggregate(state: ReviewState) -> dict:
    set_progress(state, "汇总生成审查报告", 0.94)
    risks = state.get("candidate_risks") or []
    model_meta = {}
    for m in config.MODELS:
        if m.get("id") == state.get("model_id"):
            model_meta = {"id": m.get("id"), "name": m.get("name"), "model": m.get("model")}
    report = dict(EMPTY_REPORT)
    report.update({
        "task_id": state.get("task_id", ""), "filename": state.get("filename", ""),
        "pages": int(state.get("pages") or 0), "chunks": len(state.get("chunks") or []),
        "model": model_meta, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed(state), 2),
        "llm_calls": int(state.get("llm_calls") or 0),
        "rag_hits": int(state.get("rag_hits") or 0),
        "truncated": bool(state.get("truncated")),
        "risks": risks,
    })

    stat_line = (f"大模型调用 {report['llm_calls']} 次，法条命中 {report['rag_hits']} 条，"
                 f"耗时 {report['elapsed_sec']}s"
                 + ("（时间预算内未完成全部片段）" if report["truncated"] else ""))

    if state.get("error"):
        report["executive_summary"] = "审查未完成：" + state["error"]
    elif risks:
        llm = get_llm_client()
        prompt = prompts.render(prompts.AGGREGATE_USER, filename=report["filename"],
                                pages=report["pages"], chunks=report["chunks"], count=len(risks),
                                stat=stat_line, risks=_risk_payload_for_prompt(risks))
        data = llm.chat_json(prompt, fallback={"overall_risk_level": "中", "executive_summary": "",
                                               "key_advices": [], "risks": risks},
                             model_id=state.get("model_id", ""), system=prompts.AGGREGATE_SYSTEM)
        if isinstance(data, list):                 # 部分模型直接返回风险数组而非对象
            data = {"risks": data}
        merged = {str(r.get("id")): r for r in risks}
        for item in data.get("risks") or []:
            if isinstance(item, dict) and str(item.get("id")) in merged:
                target = merged[str(item.get("id"))]
                for field in ("risk_type", "description", "suggestion", "level", "legal_basis"):
                    if item.get(field):
                        target[field] = item[field]
        report["executive_summary"] = str(data.get("executive_summary") or "")
        report["key_advices"] = [str(x) for x in (data.get("key_advices") or [])][:6]
        report["overall_risk_level"] = str(data.get("overall_risk_level") or "")

    # 风险过多时按等级排序截断，保证报告可读性
    if len(report.get("risks", [])) > config.MAX_RISK_IN_REPORT:
        order = {"高": 0, "中": 1, "低": 2}
        report["risks"].sort(key=lambda x: (order.get(x.get("level"), 9), x.get("page", 0)))
        dropped = len(report["risks"]) - config.MAX_RISK_IN_REPORT
        report["risks"] = report["risks"][:config.MAX_RISK_IN_REPORT]
        push_log(state, f"风险点过多，按等级保留前 {config.MAX_RISK_IN_REPORT} 条（截断 {dropped} 条）")
    else:
        report["executive_summary"] = (
            f"本次共处理 {report['pages']} 页 / {report['chunks']} 个片段，"
            f"未发现明显风险条款。{stat_line}")
        report["overall_risk_level"] = "低"

    finalize_report(report)
    push_log(state, f"✅ 审查完成：共 {report['statistics']['total']} 条风险"
                    f"（高 {report['statistics']['高']} / 中 {report['statistics']['中']} / "
                    f"低 {report['statistics']['低']}），总耗时 {report['elapsed_sec']}s")
    set_progress(state, "已完成", 1.0)
    return {"final_report": report, "elapsed_sec": report["elapsed_sec"]}
