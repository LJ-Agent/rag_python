# RAG-PYTHON — AI 计算核心 架构设计文档

## 一、概述

RAG-PYTHON 是整个 RAG 知识库系统的 AI 计算核心，基于 **Python 3.11+** 构建。它通过 gRPC 提供同步的向量检索和 LLM 生成服务，通过 Kafka 消费异步的文档处理任务（解析/分块/嵌入/删除），是 Java 业务控制中心的 AI 计算后端。

**核心定位**：AI 计算引擎 — Python 负责 NLP、向量操作、ML 推理；Java 负责事务、用户、状态机编排。

## 二、技术栈

| 组件 | 技术 |
|------|------|
| 向量数据库 | Milvus 2.x (pymilvus) |
| 关键词检索 | BM25 (rank-bm25 + jieba 分词) |
| 文档解析 | PyPDF2, python-docx, Markdown |
| 向量嵌入 | sentence-transformers (BAAI/bge-base-zh-v1.5) + OpenAI 兼容 API |
| LLM 调用 | OpenAI 兼容接口（流式/非流式） |
| 对象存储 | MinIO (S3 兼容) |
| 缓存 | Redis |
| 通信 | gRPC (同步 RPC) + Kafka (异步任务) |

## 三、分层架构

```
┌─────────────────────────────────────────────────────┐
│  communication/       外部通信层                      │
│  ├── grpc_server/     gRPC 服务 (:50051 + :50052)    │
│  ├── kafka_consumer/  Kafka 消费者 (4 个 topic)      │
│  └── kafka_producer/  Kafka 生产者 (状态回写)        │
├─────────────────────────────────────────────────────┤
│  ai_core/             AI 核心算法 (无外部通信依赖)    │
│  ├── parser/          文档解析 → 标准化 Markdown      │
│  ├── chunker/         智能分块 (6 种策略)             │
│  ├── embedder/        向量嵌入 + Milvus 存储          │
│  ├── retrieval/       混合检索 (向量+BM25, RRF 融合) │
│  └── generation/      LLM 问答生成 (流式/非流式)      │
├─────────────────────────────────────────────────────┤
│  infrastructure/      基础设施适配器                   │
│  ├── milvus/          Milvus 向量数据库客户端         │
│  ├── bm25/            BM25 关键词检索引擎             │
│  ├── llm/             LLM 适配层                      │
│  ├── minio/           对象存储客户端                  │
│  └── redis_utils/     Redis 缓存                      │
├─────────────────────────────────────────────────────┤
│  task_scheduler/      任务调度器                      │
│  ├── task_dispatcher  并发控制 + 重试 + 回调          │
│  └── task_handlers    4 个任务处理器                  │
├─────────────────────────────────────────────────────┤
│  common/              共享模块                        │
│  ├── config_loader    YAML + ${ENV:default}          │
│  ├── enums            DocumentStatus 状态机           │
│  ├── exceptions       分级异常体系                    │
│  └── result           统一 Result<T> 封装             │
└─────────────────────────────────────────────────────┘
```

## 四、异步任务流水线（Kafka 驱动）

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: FILE_PROCESS                                  │
│  MinIO 下载原始文件 → 解析 (PDF/MD/TXT/DOCX)            │
│  → 清洗 → 回写 cleaned.md 到 MinIO                      │
│  → status: PENDING_REVIEW                               │
├─────────────────────────────────────────────────────────┤
│  Stage 2: CHUNK_PROCESS                                 │
│  读取 cleaned.md → 智能分块 (6 种策略可选)               │
│  → 返回块列表给 Java 入库 → 人工审核                     │
│  → status: CHUNK_REVIEW (或自动通过)                    │
├─────────────────────────────────────────────────────────┤
│  Stage 3: EMBED_PROCESS                                 │
│  审核通过的块 → 向量嵌入 (batch=20)                     │
│  → 写入 Milvus 向量 + BM25 索引                         │
│  → status: COMPLETED                                    │
├─────────────────────────────────────────────────────────┤
│  Stage 4: DOCUMENT_DELETE                               │
│  清理 Milvus 向量 + BM25 索引                           │
└─────────────────────────────────────────────────────────┘
```

**Topic 通信**:
| Topic | 方向 | 用途 |
|-------|------|------|
| `rag-file-process` | Java → Python | 文件解析 |
| `rag-chunk-process` | Java → Python | 文档分块 |
| `rag-embed-process` | Java → Python | 向量嵌入 |
| `rag-document-delete` | Java → Python | 文档删除 |
| `rag-task-complete` | Python → Java | 任务完成回调 |
| `rag-task-failed` | Python → Java | 任务失败回调 |

**消息格式**:
```json
{
  "taskId": "task-20260525-embed-8",
  "taskType": "EMBED_PROCESS",
  "documentId": 8,
  "kbId": 2,
  "data": { "chunks": [...], "fileName": "doc.md" },
  "createdAt": "2026-05-25T09:30:00"
}
```

## 五、任务调度器

- **文件**: `task_scheduler/task_dispatcher.py`
- **并发控制**: `ThreadPoolExecutor` (max_concurrent=5)
- **缓冲队列**: 当所有 worker 忙碌时，新任务进入 deque 等待
- **重试策略**: 指数退避（基础延迟 2s，最大 3 次）
- **回调机制**: `on_complete` → 提交 Kafka offset + 发送完成状态; `on_failed` → 发送失败状态
- **Offset 管理**: 任务完成后才提交 offset，保证至少一次投递

## 六、AI 核心模块

### 6.1 文档解析器 (`parser/document_parser.py`)

- **支持格式**: PDF (PyPDF2), DOCX (python-docx), Markdown, TXT (UTF-8/GBK)
- **输出**: 标准化 Markdown（去除格式噪音，保留结构）
- **异常**: 解析失败抛 `AIComputeException`

### 6.2 智能分块器 (`chunker/smart_chunker.py`)

- **6 种策略**:
  | 策略 | 原理 | 适用场景 |
  |------|------|----------|
  | `fixed` | 固定大小 + 重叠 + 句边界感知 | 通用 |
  | `recursive` | 优先级分隔符递归拆分 | 代码/日志 |
  | `hierarchical` | 按 Markdown 标题层级拆分 | 结构化文档 |
  | `semantic` | 按段落拆分 + 短段落合并 | 自然语言 |
  | `topic` | 字符集 Jaccard 相似度检测主题偏移 | 多主题文档 |
  | `hybrid` | 先层级拆分 + 固定大小细化 | 大文档 |
- **约束**: 最大 65535 字符（Milvus VarChar 限制），最小可配置
- **输出**: `Chunk { chunk_id, document_id, content, chunk_index, level, parent_id, metadata }`

### 6.3 嵌入服务 (`embedder/embedding_service.py`)

- **默认**: 本地 `sentence-transformers` (BAAI/bge-base-zh-v1.5, 768 维) — 零 API 成本
- **Fallback**: OpenAI 兼容远程 API
- **批处理**: 每批 20 个块
- **Milvus Schema**:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 (auto) | 主键 |
| chunk_id | VARCHAR | 块标识 |
| document_id | INT64 | 文档 ID |
| kb_id | INT64 | 知识库 ID（分区键） |
| chunk_index | INT32 | 块序号 |
| content | VARCHAR(65535) | 块内容 |
| embedding | FLOAT_VECTOR(768) | 向量 |

### 6.4 混合检索 (`retrieval/hybrid_retrieval.py`)

```
用户 Query
    │
    ├──► [向量检索] jieba 分词 → embedding → Milvus ANN (top_k*2)
    │         │
    │         └── L2 距离 → 相似度 (1 - distance/2)
    │
    ├──► [BM25 检索] jieba 分词 → 关键词匹配 → BM25 排序 (top_k*2)
    │         │
    │         └── 分数归一化 (除以 top score)
    │
    └──► [RRF 融合] score(f) = Σ weight / (60 + rank(f))
              │
              ├── vector_weight = 0.7
              ├── bm25_weight = 0.3
              └── k = 60
              │
              ▼
          排序 + 截断 → DocumentChunk[]
```

### 6.5 问答生成器 (`generation/qa_generator.py`)

- **System Prompt**: 中文指令，要求严格基于上下文回答，标注来源片段编号
- **消息结构**: `[system: contexts] + [user: query]`
- **流式**: 通过 `asyncio` + `queue.Queue` 桥接异步 LLM 流到 gRPC 同步流
- **Fallback**: 无上下文或 LLM 调用失败时返回礼貌的"无法回答"

## 七、基础设施

### 7.1 Milvus 客户端 (`infrastructure/milvus/`)

- **集合**: `rag_collection`, 自动创建 schema
- **索引**: IVF_FLAT, L2 距离度量
- **查询**: 支持 `kb_id` 过滤表达式
- **操作**: 批量插入、相似搜索、按 doc_id 删除

### 7.2 BM25 引擎 (`infrastructure/bm25/`)

- **实现**: 全内存 BM25 (k1=1.5, b=0.75)
- **分词**: jieba 中文分词
- **生命周期**: 单例，在服务整个生命周期中维护
- **限制**: 受限于可用 RAM（适合中小规模语料）

### 7.3 LLM 适配器 (`infrastructure/llm/`)

- **双模式**: 本地 embedding (sentence-transformers) + 远程 chat (OpenAI 兼容)
- **流式**: async generate_stream 桥接到 openai async client

## 八、gRPC 服务

### RetrievalService (:50051)

```protobuf
rpc Retrieve(RetrievalRequest) returns (RetrievalResponse);
// Request: query, kb_ids[], top_k, score_threshold
// Response: DocumentChunk[] + total_count + latency_ms
```

### GenerationService (:50052)

```protobuf
rpc Generate(GenerationRequest) returns (GenerationResponse);         // 非流式
rpc GenerateStream(GenerationRequest) returns (stream GenerationResponse);  // 流式
// Response: content, is_end, token_count, finish_reason
```

## 九、关键架构决策

| 决策 | 选择 | 理由 |
|------|------|------|
| AI 计算语言 | Python | NLP/ML 生态优势 |
| 异步任务 | Kafka | 解耦 + 重试 + 削峰 |
| 同步 RPC | gRPC | 性能 + 强类型 + 流式支持 |
| 检索融合 | RRF (Reciprocal Rank Fusion) | 处理向量/BM25 分数量纲不一致 |
| 默认嵌入 | 本地 BGE 模型 | 零 API 成本，中文优化 |
| BM25 存储 | 全内存 | 检索速度优先，中小规模适用 |
| 任务调度 | ThreadPoolExecutor + 缓冲队列 | 简单可靠，不拒绝任务 |
| 分块策略 | 6 种可配 | 覆盖结构化/非结构化各种文档 |
| 人工审核 | 双阶段 (内容+分块) | 质量门控，分别校验解析和分块 |
| Kafka offset | 手动提交 | 至少一次投递保证 |

## 十、配置要点

```yaml
grpc.port: 50051/50052                    # 两个 gRPC 端口
kafka.topics: 4 consume + 2 produce       # 6 个 topic
milvus.dimension: 768                     # 向量维度
chunk.default_size: 500                   # 默认块大小
chunk.overlap: 50                         # 重叠字符数
retrieval.vector_weight: 0.7              # 向量检索权重
retrieval.bm25_weight: 0.3                # BM25 权重
task.max_concurrent: 5                    # 最大并发任务
task.retry_max: 3                         # 最大重试次数
llm.use_local_embedding: true             # 本地嵌入开关
```
