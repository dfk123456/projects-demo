# ============================================================
# tools/calc_tool.py —— 离线公式演算工具（@tool 注册）
# 功能：纯本地计算，不依赖任何网络/大模型，可稳定离线执行：
#   1) 卷积输出尺寸：H_out=(H+2P-d*(K-1)-1)/S+1 及卷积核参数量
#   2) 矩阵运算：两矩阵乘法（含向量点积）
#   3) 概率计算：二项分布 P(X=k)=C(n,k)·p^k·(1-p)^(n-k)
# 解析采用关键词+正则的容错设计；解析失败时返回用法说明，
# 提示调用方按标准短语重新组织输入，绝不抛出异常。
# ============================================================
import ast
import math
import re
from typing import List

from langchain_core.tools import tool

# ------------------------------------------------------------
# 工具可识别的中英文参数关键词
# ------------------------------------------------------------
_KEYS_INPUT = ["输入", "input", "input_size"]
_KEYS_KERNEL = ["卷积核", "核大小", "核尺寸", "kernel", "filter"]
_KEYS_STRIDE = ["步长", "stride"]
_KEYS_PADDING = ["填充", "padding"]
_KEYS_DILATION = ["空洞", "膨胀", "dilation"]


def _first_number(text: str, keys: List[str]) -> int:
    """在 text 中定位第一个关键词，取其后的第一个数字；找不到返回 None。"""
    for key in keys:
        idx = text.lower().find(key)
        if idx == -1:
            continue
        tail = text[idx:]
        m = re.search(r"\d+", tail)
        if m:
            return int(m.group())
    return None


# ------------------------------------------------------------
# 子运算：卷积尺寸
# ------------------------------------------------------------
def _conv_calc(text: str) -> str:
    """解析并计算卷积输出尺寸与参数量。"""
    # 输入尺寸：优先匹配 "输入32x32"，其次单个数字默认视为正方形
    m = re.search(r"输入[^\d]{0,8}(\d+)\s*[x×X*]\s*(\d+)", text)
    if m:
        h_in = w_in = int(m.group(1))
        w_in = int(m.group(2))
        square_hint = ""
    else:
        single = _first_number(text, _KEYS_INPUT)
        if single is None:
            return ("卷积尺寸计算缺少「输入」尺寸。"
                    "请按示例描述：卷积尺寸：输入 32x32，卷积核 3，填充 1，步长 1")
        h_in = w_in = single
        square_hint = "（未说明宽高，按方形输入估算）"

    k = _first_number(text, _KEYS_KERNEL) or 3          # 卷积核默认3
    stride = _first_number(text, _KEYS_STRIDE) or 1     # 步长默认1
    padding = _first_number(text, _KEYS_PADDING) or 0   # 填充默认0
    dilation = _first_number(text, _KEYS_DILATION) or 1  # 空洞率默认1
    eff_k = (k - 1) * dilation + 1                        # 等效核大小

    def out_size(size: int) -> int:
        return math.floor((size + 2 * padding - eff_k) / stride) + 1

    h_out, w_out = out_size(h_in), out_size(w_in)
    params = k * k                                         # 忽略通道数时的核参数量

    lines = [
        "【卷积输出尺寸计算】",
        f"输入 {h_in}×{w_in}，卷积核 {k}×{k}（等效 {eff_k}，dilation={dilation}），"
        f"padding={padding}，stride={stride}",
        "公式：H_out = floor((H_in + 2P - (K-1)*D - 1)/S) + 1",
        f"输出尺寸 = {h_out} × {w_out}",
        f"单卷积核参数量 = {k}×{k} = {params}（不含偏置；含通道数 C_in 则为 {k}×{k}×C_in）",
    ]
    if square_hint:
        lines.append("注：" + square_hint)
    return "\n".join(lines)


# ------------------------------------------------------------
# 子运算：矩阵乘法 / 向量点积
# ------------------------------------------------------------
def _parse_matrix_groups(text: str) -> List[list]:
    """抽取文本中形如 [[1,2],[3,4]] 的矩阵字面量（安全使用 ast.literal_eval）。"""
    groups = []
    for matched in re.findall(r"\[\[.*?\]\]", text, flags=re.S):
        try:
            mat = ast.literal_eval(matched)
            if isinstance(mat, list) and len(mat) > 0:
                groups.append(mat)
        except Exception:
            continue
    return groups


def _parse_vector_groups(text: str) -> List[list]:
    """抽取形如 [1,2,3] 的一维数组（排除已被矩阵抽取覆盖的嵌套结构）。"""
    groups = []
    for matched in re.findall(r"(?<!\[)\[[^\[\]]+\](?!\])", text):
        try:
            vec = ast.literal_eval(matched)
            if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], (int, float)):
                groups.append(vec)
        except Exception:
            continue
    return groups


def _mat_calc(text: str) -> str:
    """执行矩阵乘法或向量点积。"""
    # 情况1：两个矩阵字面量 -> 矩阵乘法
    mats = _parse_matrix_groups(text)
    if len(mats) >= 2:
        a, b = mats[0], mats[1]
        if not all(isinstance(r, list) for r in a) or not all(isinstance(r, list) for r in b):
            return ("矩阵格式有误：需要如 [[1,2],[3,4]] 的标准二维列表。")
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        if cols_a != rows_b:
            return (f"矩阵无法相乘：A 为 {rows_a}×{cols_a}，B 为 {rows_b}×{cols_b}，"
                    f"需满足 A 列数 == B 行数。")
        # 小规模矩阵保护（避免超大文本展开）
        if rows_a * cols_b * cols_a > 200_000:
            return "矩阵规模过大，离线工具暂不计算，请拆分为小矩阵。"
        out = [[sum(a[i][t] * b[t][j] for t in range(cols_a))
                for j in range(cols_b)] for i in range(rows_a)]
        fmt = "\n".join("  " + str(row) for row in out)
        return (f"【矩阵乘法结果】\nA({rows_a}×{cols_a}) × B({rows_b}×{cols_b})"
                f" = C({rows_a}×{cols_b}):\n{fmt}")

    # 情况2：两个一维向量 -> 点积
    vecs = _parse_vector_groups(text)
    if len(vecs) >= 2:
        u, v = vecs[0], vecs[1]
        if len(u) != len(v):
            return f"向量点积要求等长，实际长度分别为 {len(u)} 与 {len(v)}。"
        dot = sum(x * y for x, y in zip(u, v))
        return f"【向量点积】u·v = {dot}"
    return ("未识别到矩阵/向量。请按示例描述：矩阵乘法：A=[[1,2],[3,4]]，B=[[5,6],[7,8]]；"
            "或向量点积：u=[1,2,3]，v=[4,5,6]")


# ------------------------------------------------------------
# 子运算：二项分布概率
# ------------------------------------------------------------
def _prob_calc(text: str) -> str:
    """解析 n/k/p 后计算二项分布概率。"""
    def num_after(key: str):
        idx = text.lower().find(key.lower())
        if idx == -1:
            return None
        m = re.search(r"[-+]?\d*\.?\d+", text[idx + len(key):])
        return float(m.group()) if m else None

    n = num_after("n=") or num_after("n =") or num_after("总次数")
    k = num_after("k=") or num_after("k =") or num_after("成功")
    p = num_after("p=") or num_after("p =") or num_after("概率")
    if p is not None and p > 1:      # 常见误用：把 50% 写成 50
        p = p / 100.0
    if n is None or k is None or p is None or not (0 <= p <= 1):
        return ("二项概率需要 n/k/p 参数。请按示例描述："
                "概率：二项分布 n=10，k=6，p=0.5")
    if k > n:
        return f"k（{int(k)}）不能大于 n（{int(n)}）。"
    comb = math.comb(int(n), int(k))
    prob = comb * (p ** k) * ((1 - p) ** (n - k))
    lines = [
        "【二项分布概率计算】",
        f"P(X={int(k)}) = C({int(n)},{int(k)}) × {p}^{int(k)} × (1-{p})^{int(n - k)}",
        f"组合数 C({int(n)},{int(k)}) = {comb}",
        f"概率 ≈ {prob:.6f}（约 {prob * 100:.2f}%）",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------
# 工具主函数（@tool 注册，供图编排作为工具调用）
# ------------------------------------------------------------
def offline_formula_calc(text: str) -> str:
    """离线公式演算。支持三类：
    1) 卷积输出尺寸，例如："卷积尺寸：输入32x32，卷积核3，步长1，填充1"
    2) 矩阵乘法 / 向量点积，例如："矩阵乘法：A=[[1,2],[3,4]]，B=[[5,6],[7,8]]"、
       "向量点积：u=[1,2,3]，v=[4,5,6]"
    3) 二项分布概率，例如："概率：二项分布 n=10，k=6，p=0.5"
    纯本地计算，无网络请求，失败时返回用法提示而不抛异常。
    """
    text = (text or "").strip()
    if not text:
        return "请提供需要计算的数学问题文本。"

    # 1) 意图分流：优先识别运算类型关键词
    if any(w in text for w in ("卷积", "conv", "核大小", "卷积核")):
        return _conv_calc(text)
    if any(w in text for w in ("矩阵", "向量", "点积", "matmul", "矩阵乘法")):
        return _mat_calc(text)
    if any(w in text for w in ("概率", "二项", "组合数", "binomial")):
        return _prob_calc(text)

    # 2) 没有明确关键词但包含明显数字表达式，尽量尝试矩阵/卷积
    if re.search(r"\[\[.*?\]\]", text, flags=re.S):
        return _mat_calc(text)
    return ("无法识别计算类型。本离线公式工具仅支持三类计算，请补充关键词：\n"
            "  卷积尺寸（含 卷积核/步长/填充）\n"
            "  矩阵乘法 / 向量点积（给出方括号数据）\n"
            "  二项分布概率（给出 n/k/p）")


# 注册为 LangChain Tool 对象（同时保留原函数便于直接调用）
calc_tool_obj = tool(offline_formula_calc)

if __name__ == "__main__":  # 允许单独调试
    print(offline_formula_calc("卷积尺寸：输入 32x32，卷积核 3，填充 1，步长 1"))
    print(offline_formula_calc("矩阵乘法：A=[[1,2],[3,4]]，B=[[5,6],[7,8]]"))
    print(offline_formula_calc("概率：二项分布 n=10，k=6，p=0.5"))
