# ============================================================
# agents/__init__.py —— 智能体子包入口
# 功能：统一导出四个智能体，供 graph_builder 实例化编排。
# ============================================================
"""
agents 包 —— 多智能体集合：
  * dispatcher_agent ：中控调度（话题过滤/多轮记忆/问题分类/工具决策/联网判断）
  * tech_rag_agent   ：大模型技术知识库 RAG 检索
  * math_rag_agent   ：深度学习数学知识库 RAG 检索
  * summary_agent    ：结果汇总 & 幻觉校验（标注来源，禁止编造）
"""
from agents.dispatcher_agent import DispatcherAgent
from agents.math_rag_agent import MathRAGAgent
from agents.summary_agent import SummaryAgent
from agents.tech_rag_agent import TechRAGAgent

__all__ = ["DispatcherAgent", "TechRAGAgent", "MathRAGAgent", "SummaryAgent"]
