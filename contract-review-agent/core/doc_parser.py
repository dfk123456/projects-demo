# ============================================================
# doc_parser.py —— 长文档解析（上百页 PDF 合同）
#
# 引擎优先级（可用/config.DOC_PARSER 指定）：
#   1) unstructured  —— 技术约束指定，版面还原最好；
#   2) pdfplumber    —— 纯 Python，表格/排版保持较好；
#   3) pypdf         —— 最轻量，纯依赖兜底。
# 统一输出：List[PageBlock] = [{page:int, text:str}]
# ============================================================
import os
import re
from typing import List

import config
from core.logging_setup import setup_logger

logger = setup_logger()


class PageBlock(dict):
    """单页文本块：{'page': int, 'text': str}。"""


def _available(name: str) -> bool:
    try:
        if name == "unstructured":
            from unstructured.partition.pdf import partition_pdf  # noqa: F401
        elif name == "pdfplumber":
            import pdfplumber  # noqa: F401
        elif name == "pypdf":
            import pypdf  # noqa: F401
        else:
            return False
        return True
    except Exception:
        return False


def _parse_by_unstructured(path: str) -> List[PageBlock]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=path, strategy="fast")
    pages: dict = {}
    last_page = 1
    for el in elements:
        meta = getattr(el, "metadata", None)
        page_no = getattr(meta, "page_number", None) if meta else None
        page_no = int(page_no) if page_no else last_page
        last_page = page_no
        text = (el.text or "").strip()
        if text:
            pages.setdefault(page_no, []).append(text)
    total = max(pages.keys()) if pages else 0
    return [PageBlock(page=i, text="\n".join(pages.get(i, []))) for i in range(1, total + 1)]


def _parse_by_pdfplumber(path: str) -> List[PageBlock]:
    import pdfplumber

    blocks = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            blocks.append(PageBlock(page=idx, text=_clean(text)))
    return blocks


def _parse_by_pypdf(path: str) -> List[PageBlock]:
    import pypdf

    blocks = []
    reader = pypdf.PdfReader(path)
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        blocks.append(PageBlock(page=idx, text=_clean(text)))
    return blocks


def _clean(text: str) -> str:
    """清掉多余空白/页眉页脚噪声字符。"""
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_document(path: str, parser: str = None) -> tuple:
    """解析文档，返回 (pages, parser_name)。pages 为 List[PageBlock]。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
        # 无分页信息的纯文本：按 3000 字虚拟分页，保证后续逻辑统一
        size = 3000
        pages = [PageBlock(page=i // size + 1, text=raw[i:i + size])
                 for i in range(0, max(len(raw), 1), size)]
        return pages, "text"

    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(path)
            raw = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [PageBlock(page=1, text=_clean(raw))], "python-docx"
        except Exception as exc:
            raise RuntimeError(f"docx 解析失败：{str(exc)[:160]}")

    want = (parser or config.DOC_PARSER or "auto").lower()
    order = ["unstructured", "pdfplumber", "pypdf"] if want == "auto" else [want]
    errors = []
    for name in order:
        if not _available(name):
            errors.append(f"{name} 未安装")
            continue
        try:
            pages = {"unstructured": _parse_by_unstructured,
                     "pdfplumber": _parse_by_pdfplumber,
                     "pypdf": _parse_by_pypdf}[name](path)
            pages = [p for p in pages if p.get("text")] or pages
            logger.info("文档解析完成：engine=%s，%s 页", name, len(pages))
            return pages, name
        except Exception as exc:
            errors.append(f"{name} 失败：{str(exc)[:120]}")
            logger.warning("解析引擎 %s 失败：%s", name, str(exc)[:160])
    raise RuntimeError("文档解析失败（" + "；".join(errors) + "）")


def document_text(pages: List[PageBlock]) -> str:
    """把分页内容拼成带页码标记的完整文本。"""
    parts = []
    for p in pages:
        parts.append(f"[第{p['page']}页]\n{p['text']}")
    return "\n\n".join(parts)
