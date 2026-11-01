# 基于 LangGraph 的长文档合同风险审查 Agent

> 面向律所与法务团队的合同审查助手：上传上百页 PDF 合同，系统自动识别风险点并通过 RAG 回填相关法条。把传统人工逐页审读变为「AI 初筛 + 律师复核」模式，显著降低漏检率与审查人力成本。

## 功能亮点

- **LangGraph 双循环工作流**：`parse → analyze_batch → enrich → recheck → aggregate` 五节点编排
  - **片段循环**：长文档分片分批送入大模型识别风险，解决超长文本无法一次性入模的行业痛点
  - **复核循环**：对缺失法条或置信度偏低的风险点定向复审，误报直接剔除，回填准确法条
- **长文档切分策略**：3500 字大片段 + 500 字重叠窗口 + 跨片段上下文摘要，规避分片导致的上下文割裂漏检
- **法条 RAG**：自建核心法规知识库（民法典 / 民诉 / 仲裁 / 消保等），向量检索为每个风险点回填 `legal_basis`
- **模型弹性**：审查模型支持 DeepSeek / MiMo / 通义 / 本机 Ollama 热切换；向量模型多源切换，在线不可用自动降级为本地词法向量
- **工程韧性**：Milvus 向量库与 MinIO 对象存储双后端自动探测，缺失时回退本地实现，部署无需改代码

## 技术架构（LangGraph 状态机）

```
PDF → [parse 解析] → [analyze_batch 分片循环识别风险]
                    → [enrich 法条RAG回填]
                    → [recheck 定向复核/误报剔除]
                    → [aggregate 汇总成文] → 结构化报告
```

## 技术栈

`Python` · `LangGraph` · `DeepSeek API` · `Milvus` · `MinIO` · `Unstructured` · `FastAPI` · `SSE`

## 性能指标

| 指标 | 数值 |
|---|---|
| 390 页合同审查耗时 | 61.5 秒（满足 200 页 ≤ 120 秒指标） |
| 196 页合同审查耗时 | 约 35 秒 |
| 并发线程 | 16 线程 + 110 秒全局时间预算 |

## 快速开始

```bash
# 1. 准备环境
pip install -r requirements.txt

# 2. 配置
cp config.example.py config.py
# 编辑 config.py 填入模型密钥（或设置环境变量）

# 3. 启动
python main.py
```

## 目录结构

```
contract-review-agent/
├── api/        # Web 接口层
├── core/       # 核心：RAG / 向量库 / 模型调用 / 存储
├── graph/      # LangGraph 节点与状态机
├── scripts/    # 辅助脚本
├── static/     # 前端页面
├── config.py   # 配置（已 gitignore，从 example 复制）
└── main.py     # 启动入口
```

## 说明

- 报告含风险位置、描述、参考法条、等级，支持 Web 在线编辑与 Word 导出
- 内置本地词法向量 fallback，无在线向量模型时仍可运行
- 本项目仅用于技术展示，测试合同不包含真实业务数据
