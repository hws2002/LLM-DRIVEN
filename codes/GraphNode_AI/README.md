# GraphNode AI

> **由大语言模型（LLM）驱动的知识结构化引擎**
>
> 本仓库为 GraphNode 项目的 AI 服务端，是《大模型驱动的软件开发》课程的源代码提交。
> 它将用户的 AI 对话与个人笔记自动转化为可导航的知识图谱（Knowledge Graph），
> 并基于图谱提供检索增强生成（RAG）问答。

本 README 用于源码导览，说明当前代码目录结构、核心模块职责、主要入口文件，以及它们与最终报告中系统设计的对应关系。
具体内容请参考根目录的`最终报告.pdf`。

## 1. 目录结构

```text
GraphNode_AI/
├── add_node/          # 增量图谱更新流水线
├── dto/               # 请求与响应数据模型
├── infra/             # 数据库与存储层适配器
├── macro/             # 全局 Macro Graph 生成流水线
├── microscope/        # 单篇文档图谱抽取与 GraphRAG
├── server/            # HTTP 服务与 SQS Worker 入口
├── shared/            # 公共配置、LLM Provider、文本处理、成本计算等
├── .env.example       # 密钥模板；
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. 核心模块

| 模块 | 职责 | 主要入口 |
|---|---|---|
| `server/` | 接收 HTTP/SQS 任务，路由请求并上报进度 | `server/worker.py`, `server/main.py` |
| `macro/` | 基于全部对话与笔记生成全局知识图谱 | `macro/src/run_pipeline.py` |
| `microscope/` | 对单篇文档抽取概念-关系图谱，并支持文档级分析 | `microscope/call.py` |
| `microscope/rag/` | 构建 GraphRAG 上下文与回答提示词 | `microscope/rag/retrieval_strategies.py`, `microscope/rag/prompt_builder.py` |
| `microscope/services/` | RAG 服务层 API 逻辑 | `microscope/services/rag_service.py` |
| `add_node/` | 将新的对话或笔记加入已有 Macro Graph | `add_node/call.py` |
| `infra/` | 封装 Neo4j、ChromaDB、MongoDB 与统一图谱仓储接口 | `infra/repositories/` |
| `shared/` | LLM Provider 抽象、配置、文本清洗、价格计算与日志等公共能力 | `shared/api_provider.py`, `shared/config.py`, `shared/text_core.py` |
| `dto/` | Server 与各流水线之间传递的请求/响应数据结构 | `dto/server_dto.py`, `dto/microscope_dto.py` |


## 3. 详细目录结构

```text
add_node/
├── call.py                         # 批量与单项 add-node 流水线入口
├── readme.md
├── analyze/                        # 对话加载与 Q-A 解析辅助模块
├── steps/                          # Q-A 抽取、关键词、聚类、边生成等分步逻辑
├── stop_words/                     # 停用词资源
└── utils/                          # 嵌入、聚类、IO、提示词、相似度等工具

dto/
├── server_dto.py                   # HTTP/SQS 请求模型
└── microscope_dto.py               # Microscope 流水线上下文模型

infra/
├── README.md
└── repositories/
    ├── graph/                      # 统一图谱仓储封装
    ├── mongodb/                    # MongoDB 访问
    ├── neo4j/                      # Neo4j 访问
    └── vectordb/                   # ChromaDB 节点与 chunk 存储

macro/
├── README.md
├── config.yaml
└── src/
    ├── run_pipeline.py             # Macro 流水线编排
    ├── extract_features.py         # 嵌入与关键词提取
    ├── cluster_with_llm.py         # LLM 辅助大聚类生成与节点分配
    ├── build_edges.py              # 语义边构建
    ├── build_subclusters.py        # 子聚类构建
    ├── merge_graph.py              # 图谱合并与后处理
    ├── insights/                   # 图谱摘要、用户模式发现、向量索引
    └── util/                       # 文件加载、图工具、Notion 与原始文件支持

microscope/
├── README.md
├── SERVICE_OVERVIEW.md
├── call.py                         # 文档载入与图谱抽取主入口
├── block/                          # Block View 切分、排序与组装
├── graph_generation/               # 实体与关系抽取编排
├── prompts/                        # Prompt 模板与工厂
├── rag/                            # 检索、上下文构建、答案生成
├── schema/                         # 本体 schema 与类型映射 JSON
├── services/                       # RAG 服务层
├── tools/                          # 本地图谱可视化工具
└── utils/                          # 文档、IO、LLM 工具函数

server/
├── worker.py                       # SQS Worker 与生产任务路由
└── main.py                         # 轻量 HTTP API 服务

shared/
├── api_provider.py                 # 统一 LLM API 封装
├── config.py                       # 公开运行配置
├── env_loader.py                   # 环境变量加载
├── text_core.py                    # 关键词归一化与多语言文本清洗
├── text_rules/                     # 停用词与关键词清洗规则
├── cost_calculator.py              # token 与 API 成本计算
├── token_usage.py
├── llm_model_aliases.json
├── llm_pricing.json
└── tools/mem_check.py              # 本地内存分析工具
```


## 4. 代码导览

| 想查看的内容 | 直接打开 |
|---|---|
| SQS Worker 路由 | `server/worker.py` |
| HTTP 请求数据结构 | `dto/server_dto.py` |
| Macro 流水线编排 | `macro/src/run_pipeline.py` |
| Macro 特征提取 | `macro/src/extract_features.py` |
| LLM 聚类分配与解析 | `macro/src/cluster_with_llm.py` |
| 图谱摘要与用户模式发现 | `macro/src/insights/discovery/graph_summarizer.py` |
| Microscope 主流程 | `microscope/call.py` |
| 实体-关系抽取 Prompt | `microscope/prompts/entity_relation_prompt.py` |
| Block View 切分 | `microscope/block/segmenter.py` |
| GraphRAG 检索 | `microscope/rag/retrieval_strategies.py` |
| RAG 服务入口 | `microscope/services/rag_service.py` |
| 增量 add-node 流水线 | `add_node/call.py` |
| 图数据库 / 向量库统一仓储 | `infra/repositories/graph/graphnode_repository.py` |
| LLM Provider 抽象 | `shared/api_provider.py` |
| 公共文本归一化 | `shared/text_core.py` |


## 5. 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.11+ |
| LLM Provider | OpenAI / Groq / Z.AI / OpenRouter，通过 `shared/api_provider.py` 统一封装 |
| 嵌入模型 | `sentence-transformers` |
| 关键词抽取 | KeyBERT |
| 图与聚类 | NetworkX、余弦相似度、Louvain 风格社区发现 |
| 图数据库 | Neo4j |
| 向量数据库 | ChromaDB |
| 文档数据库 | MongoDB |
| 消息队列 | AWS SQS |


## 6. 本地运行方式

在实际系统中，本 AI 服务并不是单独运行的命令行程序，而是与 GraphNode 的 backend / frontend 配合使用：frontend 触发任务，backend 将任务写入 AWS SQS，本服务作为 SQS Worker 消费任务并执行相应的 Macro、Microscope 或 RAG 流水线。

当前代码中的 SQS 队列、数据库与云端资源均属于个人开发环境，外部用户无法直接访问。因此，如果希望完整复现线上调用链，需要自行准备：

- 可访问的 AWS SQS 队列地址；
- Neo4j、ChromaDB、MongoDB 等外部服务；
- LLM API 密钥；
- 与 `dto/server_dto.py` 中请求结构匹配的输入消息。

准备环境的基本步骤如下：

```bash
pip install -r requirements.txt
cp .env.example .env
python -m server.worker --dev
```

真实密钥只应写入 `.env`。模型名、chunk 大小、默认服务配置等公开配置集中在 `shared/config.py`。

如果不通过 SQS，也可以直接从各模块的单独入口进行本地调试或源码审阅：

- Macro Graph：`macro/src/run_pipeline.py`
- Microscope：`microscope/call.py`
- RAG：`microscope/services/rag_service.py`
- add_node：`add_node/call.py`


## 7. SQS 任务类型

| `taskType` | 流水线 | 说明 |
|---|---|---|
| `ADD_NODE_REQUEST` | `add_node` | 增量添加对话或笔记 |
| `GRAPH_GENERATION_REQUEST` | `macro` | 生成全局 Macro Graph |
| `GRAPH_SUMMARY_REQUEST` | `macro` | 生成图谱摘要与用户洞察 |
| `MICROSCOPE_INGEST_FROM_NODE_REQUEST` | `microscope` | 将单个源节点载入 Microscope |
| `MICROSCOPE_QUERY_REQUEST` | `microscope/rag` | 基于图谱上下文进行问答 |
| `MICROSCOPE_SYNTHESIZE_REQUEST` | `microscope/rag` | 基于召回图谱上下文进行主题综合 |
| `MICROSCOPE_RELATED_QUESTIONS_REQUEST` | `microscope/rag` | 生成相关追问 |
