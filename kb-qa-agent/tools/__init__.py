# ============================================================
# tools/__init__.py —— 工具子包入口
# 功能：集中导出系统可用工具，供图编排/汇总阶段引用。
# ============================================================
"""
tools 包 —— 系统可用工具集合：
  * offline_formula_calc ：离线公式演算（卷积尺寸 / 矩阵 / 概率）
  * online_tech_search   ：可选联网检索（仅限 AI / 深度学习前沿技术）
"""

from tools.calc_tool import calc_tool_obj, offline_formula_calc
from tools.online_search_tool import online_search_tool_obj, online_tech_search

__all__ = ["offline_formula_calc", "calc_tool_obj",
           "online_tech_search", "online_search_tool_obj"]
