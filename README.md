# Python & AI Agent Projects

> 个人 AI 应用开发作品集 —— 基于 LangGraph 的多智能体系统实践

本仓库收录两个可独立运行的完整 AI 应用项目，覆盖 **RAG 知识库问答** 与 **长文档智能审查** 两大 LLM 落地场景。所有项目均为个人独立开发，从需求设计、架构编排、编码实现到部署测试全流程完成。

## 项目一览

| 项目 | 定位 | 核心技术 | 关键指标 |
| :--- | :--- | :--- | :--- |
| [contract-review-agent](./contract-review-agent) | 长文档合同风险审查 Agent | LangGraph · DeepSeek · Milvus · MinIO | 390 页合同 **61.5 秒**审完 |
| [kb-qa-agent](./kb-qa-agent) | 离线知识库问答系统 | LangGraph · Ollama · Chroma · SSE | 单轮响应 **20~50 秒** |

---

## 1. contract-review-agent

**长文档合同风险审查 Agent** —— 面向律所与法务团队。

上传上百页 PDF 合同，系统自动识别风险点并通过 RAG 回填相关法条，把传统人工逐页审读变为「AI 初筛 + 律师复核」模式，显著降低漏检率与审查人力成本。

**核心设计**

- **LangGraph 双循环工作流**：`parse → analyze_batch → enrich → recheck → aggregate` 五节点编排
  - **片段循环**：长文档分片分批送入大模型识别风险，解决超长文本无法一次性入模的行业痛点
  - **复核循环**：对缺失法条 / 低置信风险点定向复审，剔除误报并回填准确法条
- **长文档切分策略**：3500 字大片段 + 500 字重叠窗口 + 跨片段上下文摘要，规避分片导致的上下文割裂漏检
- **法条 RAG**：自建核心法规知识库，向量检索为每个风险点回填 `legal_basis`
- **模型弹性**：DeepSeek / MiMo / 通义 / 本机 Ollama 热切换，在线向量模型不可用时自动降级本地词法向量
- **工程韧性**：Milvus 向量库与 MinIO 对象存储双后端自动探测，缺失时回退本地实现

**性能**：390 页合同 **61.5 秒**完成审查（满足 200 页 ≤ 120 秒指标），196 页约 35 秒。

**技术栈**：`Python` `LangGraph` `DeepSeek API` `Milvus` `MinIO` `Unstructured` `FastAPI` `SSE`

---

## 2. kb-qa-agent

**多智能体离线知识库问答系统** —— 面向企业内网的私有化问答助手。

用户上传 PDF/TXT 课件构建专属问答库，全程本地 Ollama 推理、**数据不出内网**，解决企业敏感文档上云的安全顾虑。

**核心设计**

- **5 个 Agent 分工状态机**（LangGraph 编排）
  - **调度拦截**：闲聊 / 娱乐 / 生活类问题第一层拦截，不调用大模型，节省推理成本
  - **分类校正**：区分「技术（tech）」与「数学（math）」类问题，携带多轮记忆识别指代
  - **双库 RAG**：技术库 / 数学库独立持久化，章节感知切分 + 实体一致性过滤，保证问 A 不答 B
  - **充分性判定**：相似度阈值评估资料充足度，纯计算题触发离线公式演算，结果不经过大模型
  - **汇总校验**：强约束 Prompt 强制资料范围内作答，幻觉校验后输出
- **防幻觉与溯源**：答案强制标注【本地课件】/【计算结果】/【外部补充资料】来源
- **增量入库**：PDF/TXT 按 SHA-256 指纹判重，避免全量重建向量库
- **离线优先**：仅当资料不足且判定为前沿技术时才允许联网补充

**性能**：GPU 加速后单轮响应由 100+ 秒降至 **20~50 秒**。

**技术栈**：`Python` `LangGraph` `Ollama` `Qwen2.5-7B` `bge-m3` `Chroma` `FastAPI` `SSE`

---

## 技术栈总览

- **大模型应用**：LangGraph 多智能体编排、RAG 全流程、Prompt Engineering、防幻觉与溯源
- **模型与向量**：Ollama / DeepSeek / 通义 / MiMo、bge-m3、Chroma、Milvus
- **工程能力**：FastAPI、SSE 流式输出、Linux 部署、多后端自动降级

---

## 快速开始

每个子项目目录下均有独立 README，包含环境准备、配置与运行说明：

```bash
git clone https://github.com/dfk123456/projects-demo.git
cd projects-demo/<项目名>
pip install -r requirements.txt
cp config.example.py config.py   # 填入自己的模型密钥
```

> 配置文件中的密钥均通过环境变量读取，仓库不含任何真实凭证。

---

## 关于我

计算机专业背景，专注 AI 应用开发与大模型落地，具备「上层应用开发 + 底层系统管理」完整技术栈。

**求职方向**：AI 应用开发工程师 / 大模型应用开发（期望城市：杭州）

📧 3061263533@qq.com
