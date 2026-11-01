# ============================================================
# chunker.py —— 长文档切分为「大片段」
#
# 对策（需求第 10 条：超长文档分片造成上下文割裂导致漏检）：
#   * 大片段 + 较大重叠（默认 3500 字 / 重叠 500 字）；
#   * 每个片段携带上一片段的尾部摘要（CARRY_PREV_SUMMARY）；
#   * 尽量在条款边界（第 X 条 / 句号）切，避免拦腰截断条款。
# ============================================================
import re
from typing import List

import config
from core.doc_parser import PageBlock


class Chunk(dict):
    """{'index':int, 'page_start':int, 'page_end':int, 'text':str}"""


_SPLIT_RE = re.compile(r"(?<=[。；！？\n])")


def _find_break(text: str, target: int, window: int = 400) -> int:
    """在 target 附近寻找条款/句子边界，找不到就返回 target。"""
    seg = text[target - window: target + window]
    candidates = [m.end() for m in _SPLIT_RE.finditer(seg)]
    if not candidates:
        return min(target, len(text))
    mid = len(seg) // 2
    best = min(candidates, key=lambda c: abs(c - mid) + (0 if abs(c - mid) < 50 else 5000))
    return max(1, min(len(text), target - window + best))


def chunk_pages(pages: List[PageBlock],
                size: int = None,
                overlap: int = None) -> List[Chunk]:
    """把分页文本切成带重叠的大片段，并记录页码区间。"""
    size = int(size or config.CHUNK_SIZE)
    overlap = int(overlap or config.CHUNK_OVERLAP)

    flat, marks = [], []           # marks: [(起始位置, 页码)]
    pos = 0
    for p in pages:
        text = re.sub(r"\s+\n", "\n", p["text"]).strip()
        if not text:
            continue
        flat.append(text)
        marks.append((pos, int(p["page"])))
        pos += len(text) + 2
    merged = "\n\n".join(flat)
    total = len(merged)
    if total == 0:
        return []

    def page_at(idx: int) -> int:
        page = marks[0][1] if marks else 1
        for start, pg in marks:
            if start <= idx:
                page = pg
            else:
                break
        return page

    chunks, start, idx = [], 0, 0
    while start < total:
        end = min(total, start + size)
        if end < total:
            end = _find_break(merged, end)
        piece = merged[start:end].strip()
        if piece:
            chunks.append(Chunk(index=idx, page_start=page_at(start),
                                page_end=page_at(end - 1), text=piece))
            idx += 1
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return chunks


def tail_summary(text: str, chars: int = None) -> str:
    """取片段尾部若干字作为上下文摘要，供下一片段携带。"""
    chars = int(chars or config.SUMMARY_CHARS)
    tail = text[-chars:].replace("\n", " ")
    return re.sub(r"\s{2,}", " ", tail).strip()
