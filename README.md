# RAG Python AI Service

RAG知识库系统 Python AI 计算服务 —— 作为 RAG 系统的 AI 计算核心、内容处理引擎、检索生成中枢。

## 核心定位

严格遵循 **Java稳态业务、Python敏态AI计算** 的分层设计原则。

| 职责 | 说明 |
|------|------|
| ✅ AI计算 | 文档解析清洗、智能分块、向量化、多路检索、大模型生成 |
| ✅ 文件处理 | 多格式文档解析（PDF/MD/Word/TXT）、文本标准化 |
| ✅ 向量检索 | 向量检索 + BM25关键词检索 + 混合检索 + 重排序 |
| ✅ 流式生成 | LLM流式问答、上下文组装、答案兜底 |
| ❌ 业务逻辑 | 不做用户权限、知识库管理、文档状态持久化 |
| ❌ 数据存储 | 不操作MySQL/PostgreSQL，不管理文件上传 |

所有业务交互、状态管理统一由 Java 服务（`rag-server`）实现，Python 仅被动消费任务、执行 AI 计算、返回标准化结果。

## 整体架构

```
┌──────────────────────────────────────────────────────────┐
│ 通信接入层（对外统一入口）                                │
│  gRPC服务端(50051/50052)、Kafka消费者、请求校验          │
├──────────────────────────────────────────────────────────┤
│ 任务调度层（任务治理核心）                                │
│  任务分发、并发控制、指数退避重试、状态上报               │
├──────────────────────────────────────────────────────────┤
│ AI核心业务层（计算能力核心）                              │
│  文档解析清洗、智能分块、向量计算、混合检索、LLM生成      │
├──────────────────────────────────────────────────────────┤
│ 基础设施层（底层能力支撑）                                │
│  MinIO、Milvus、LLM API、Redis、BM25检索引擎             │
├──────────────────────────────────────────────────────────┤
│ 公共支撑层（全局通用能力）                                │
│  常量枚举、统一响应、自定义异常、日志规范、工具类         │
└──────────────────────────────────────────────────────────┘
```

## 项目结构

```
RAG-PYTHON/
├── proto/                              # gRPC协议定义（完全对齐Java）
│   ├── retrieval.proto                 # 检索服务：Retrieve RPC
│   └── generation.proto               # 生成服务：Generate + GenerateStream RPC
├── config/
│   ├── settings.yaml                   # 全局配置（中间件地址/模型参数/检索阈值）
│   └── .env                            # 敏感环境变量
├── scripts/
│   └── compile_proto.py                # Proto编译脚本
├── src/
│   ├── main.py                         # 主入口：启动gRPC+Kafka+优雅关闭
│   ├── common/                         # 公共支撑层
│   │   ├── config_loader.py            # YAML + 环境变量配置加载
│   │   ├── constant/
│   │   │   └── kafka_constants.py      # Kafka Topic/ConsumerGroup常量
│   │   ├── enums/
│   │   │   └── status_enums.py         # DocumentStatus(13状态机)/TaskType/ReviewResult
│   │   ├── result/
│   │   │   └── result.py               # Result<T>统一封装 + ResultCodeEnum错误码
│   │   ├── exception/
│   │   │   └── exceptions.py           # RAGException/TaskException/AIComputeException
│   │   └── util/
│   │       ├── logger.py               # Loguru日志（支持task_id链路追踪）
│   │       └── utils.py                # MD5校验/JSON序列化/文本清洗
│   ├── infrastructure/                 # 基础设施层
│   │   ├── minio/
│   │   │   └── minio_client.py         # MinIO只读客户端（文件下载/流式读取/预签名URL）
│   │   ├── milvus/
│   │   │   └── milvus_client.py        # Milvus向量库（插入/搜索/删除/知识库隔离）
│   │   ├── llm/
│   │   │   └── llm_adapter.py          # LLM适配器（Embedding+Chat+流式+本地sentence-transformers）
│   │   ├── redis_utils/
│   │   │   └── redis_client.py         # Redis缓存（连接池/JSON序列化/TTL策略）
│   │   └── bm25/
│   │       └── bm25_engine.py          # BM25关键词检索引擎（jieba分词+BM25算法）
│   ├── ai_core/                        # AI核心业务层
│   │   ├── parser/
│   │   │   └── document_parser.py      # 多格式文档解析+文本清洗(PDF/MD/DOCX/TXT)
│   │   ├── chunker/
│   │   │   └── smart_chunker.py        # 智能分块(固定长度/标题层级/语义段落三种策略)
│   │   ├── embedder/
│   │   │   └── embedding_service.py    # 批量向量化+Milvus入库/更新/删除
│   │   ├── retrieval/
│   │   │   └── hybrid_retrieval.py     # 混合检索(向量+BM25 RRF加权融合+重排序)
│   │   └── generation/
│   │       └── qa_generator.py         # Prompt组装+流式/非流式LLM生成+答案兜底
│   ├── task_scheduler/                 # 任务调度层
│   │   ├── task_dispatcher.py          # ThreadPoolExecutor并发控制+指数退避重试(3次)
│   │   └── task_handlers.py            # FILE_PROCESS/CHUNK_PROCESS任务处理器
│   └── communication/                  # 通信接入层
│       ├── grpc_server/
│       │   ├── retrieval_service.py    # RetrievalService gRPC服务端(端口50051)
│       │   └── generation_service.py   # GenerationService gRPC服务端(端口50052)
│       ├── kafka_consumer/
│       │   └── task_consumer.py        # Kafka消费者(rag-file-process/rag-chunk-process)
│       └── kafka_producer/
│           └── status_producer.py      # Kafka生产者(rag-task-complete/rag-task-failed)
├── docker/
│   ├── Dockerfile                      # Python 3.11-slim镜像
│   └── docker-compose.yml              # 本地开发编排
├── pyproject.toml                       # 项目依赖管理
└── .gitignore
```

## 与Java服务协同关系

```
┌──────────────┐     gRPC(同步)      ┌──────────────┐
│              │◄───────────────────►│              │
│   Java       │   RetrievalService  │   Python     │
│   rag-server │   GenerationService │   AI Service │
│   (业务)     │                     │   (计算)     │
│              │◄──── Kafka ───────►│              │
│  端口:8080   │   rag-file-process  │  端口:50051  │
│              │   rag-chunk-process │  端口:50052  │
│  MySQL/Redis │   rag-task-complete │  Milvus/MinIO│
└──────────────┘   rag-task-failed   └──────────────┘
```

### 通信协议

| 通道 | 方向 | 说明 |
|------|------|------|
| `RetrievalService` gRPC | Java → Python | 同步检索请求（端口50051） |
| `GenerationService` gRPC | Java → Python | 同步/流式生成请求（端口50052） |
| `rag-file-process` Kafka | Java → Python | 文件解析清洗异步任务 |
| `rag-chunk-process` Kafka | Java → Python | 分块向量化异步任务 |
| `rag-task-complete` Kafka | Python → Java | 任务完成通知+结果回传 |
| `rag-task-failed` Kafka | Python → Java | 任务失败通知+错误信息 |

### 数据流转规则

- **业务数据**：不读取MySQL，所有参数由Java通过消息/请求透传
- **向量数据**：仅存储在Milvus，由Python全权管理
- **文件数据**：存储在MinIO，Java负责上传，Python负责读取处理
- **关联键**：以 `document_id`、`kb_id` 为唯一关联键打通Java与Python数据

## 核心任务流程

### 文档处理离线流程（异步Kafka）

```
Java上传文件 → 保存元数据 → 推送rag-file-process任务
  → Python消费 → 读取MinIO文件 → 解析清洗 → 生成标准MD
  → 上报rag-task-complete → Java更新为待审核
  → 人工审核通过 → Java推送rag-chunk-process任务
  → Python分块 → 向量化 → 向量入库 → BM25索引
  → 上报rag-task-complete → Java更新为COMPLETED
```

### 问答实时流程（同步gRPC）

```
前端提问 → Java校验鉴权 → gRPC调用RetrievalService
  → Python混合检索(向量+BM25) → 重排序 → 返回上下文
  → Java调用GenerationService → Python流式生成
  → 逐token返回Java → Java推送前端SSE + 保存问答记录
```

## 快速开始

### 环境要求

- Python >= 3.11
- 中间件：Kafka、MinIO、Milvus、Redis（与Java服务共用）

### 安装

```bash
cd RAG-PYTHON
pip install -e .
```

### 编译Proto

```bash
python scripts/compile_proto.py
```

### 配置

编辑 `config/settings.yaml` 和 `config/.env`，确保以下中间件地址正确：

- Kafka `bootstrap_servers`
- MinIO `endpoint` + 认证
- Milvus `host:port`
- Redis `host:port`
- LLM `api_key` + `base_url`

### 启动

```bash
PYTHONPATH=src python -m main
```

服务将启动：
- gRPC RetrievalService → `0.0.0.0:50051`
- gRPC GenerationService → `0.0.0.0:50052`
- Kafka Consumer → 监听 `rag-file-process`、`rag-chunk-process`

### Docker部署

```bash
docker compose -f docker/docker-compose.yml up -d
```

## 配置参数说明

| 分类 | 关键参数 | 默认值 | 说明 |
|------|----------|--------|------|
| gRPC | `retrieval.port` | 50051 | 检索服务端口 |
| gRPC | `generation.port` | 50052 | 生成服务端口 |
| gRPC | `timeout_seconds` | 10 | 调用超时 |
| 分块 | `chunk.default_size` | 500 | 默认分块大小（字符） |
| 分块 | `chunk.strategy` | semantic | 分块策略：fixed/hierarchical/semantic |
| 检索 | `retrieval.default_top_k` | 5 | 默认返回条数 |
| 检索 | `retrieval.score_threshold` | 0.5 | 相似度阈值 |
| 检索 | `retrieval.vector_weight` | 0.7 | 混合检索向量权重 |
| 检索 | `retrieval.bm25_weight` | 0.3 | 混合检索BM25权重 |
| 任务 | `task.max_concurrent` | 5 | 最大并发任务数 |
| 任务 | `task.retry_max` | 3 | 最大重试次数 |
| LLM | `llm.chat_model` | gpt-4o | 对话模型 |
| LLM | `llm.embedding_model` | text-embedding-3-small | Embedding模型 |
| LLM | `llm.temperature` | 0.3 | 生成温度 |

## 异常体系

| 异常类 | 错误码 | 说明 |
|--------|--------|------|
| `RAGException` | 500 | 基础异常 |
| `TaskException` | 500 | 任务执行异常（可重试） |
| `AIComputeException` | 10501 | AI计算异常（模型调用/Embedding失败） |
| `RetrievalException` | 10502 | 检索服务异常 |
| `GenerationException` | 10503 | 生成服务异常 |
| `ResourceException` | 500 | 基础设施资源异常（MinIO/Milvus/Redis不可用） |
| `ValidationException` | 400 | 请求参数校验异常 |

完全对齐 Java 端 `ResultCodeEnum` 错误码体系（0-10602）。

## 文档状态机

```
UPLOADED → PARSING → CLEANING → PENDING_REVIEW → APPROVED → CHUNKING → EMBEDDING → COMPLETED
              ↓          ↓                            ↑
         PARSING_FAILED  CLEANING_FAILED          REJECTED
                                                ↓
              ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←↵
```

13状态文档生命周期，与Java端 `DocumentStatus` 状态机完全一致。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式化
ruff format src/

# 类型检查
mypy src/
```
