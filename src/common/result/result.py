"""Unified response wrapper — fully aligned with Java Result<T> format."""

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Unified API response, identical structure to com.rag.common.result.Result."""

    code: int = 0
    message: str = "success"
    data: T | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    @staticmethod
    def success(data: T | None = None) -> "Result[T]":
        return Result(code=0, message="success", data=data)

    @staticmethod
    def fail(code: int, message: str) -> "Result":
        return Result(code=code, message=message, data=None)


@dataclass
class ResultCode:
    """Error code marker, aligned with Java ResultCode interface."""

    code: int
    message: str


class ResultCodeEnum:
    """All error codes aligned with Java ResultCodeEnum."""

    SUCCESS = ResultCode(0, "成功")
    PARAM_ERROR = ResultCode(400, "参数错误")
    UNAUTHORIZED = ResultCode(401, "未授权")
    FORBIDDEN = ResultCode(403, "无权限")
    NOT_FOUND = ResultCode(404, "资源不存在")
    METHOD_NOT_ALLOWED = ResultCode(405, "请求方法不支持")
    CONFLICT = ResultCode(409, "资源冲突")
    RATE_LIMIT = ResultCode(429, "请求过于频繁")
    SYSTEM_ERROR = ResultCode(500, "系统内部错误")
    SERVICE_UNAVAILABLE = ResultCode(503, "服务不可用")

    # Business error codes (10000+)
    FILE_UPLOAD_ERROR = ResultCode(10001, "文件上传失败")
    FILE_NOT_FOUND = ResultCode(10002, "文件不存在")
    FILE_DUPLICATE = ResultCode(10003, "文件已存在")
    FILE_TYPE_NOT_SUPPORTED = ResultCode(10004, "不支持的文件类型")
    FILE_SIZE_EXCEEDED = ResultCode(10005, "文件大小超出限制")
    DOCUMENT_NOT_FOUND = ResultCode(10101, "文档不存在")
    DOCUMENT_STATUS_ERROR = ResultCode(10102, "文档状态异常")
    KB_NOT_FOUND = ResultCode(10201, "知识库不存在")
    KB_NAME_DUPLICATE = ResultCode(10202, "知识库名称已存在")
    REVIEW_NOT_FOUND = ResultCode(10301, "审核记录不存在")
    REVIEW_ALREADY_HANDLED = ResultCode(10302, "审核已处理")
    REVIEW_PERMISSION_DENIED = ResultCode(10303, "无审核权限")
    USER_NOT_FOUND = ResultCode(10401, "用户不存在")
    USERNAME_DUPLICATE = ResultCode(10402, "用户名已存在")
    PASSWORD_ERROR = ResultCode(10403, "密码错误")
    TOKEN_EXPIRED = ResultCode(10404, "Token已过期")
    TOKEN_INVALID = ResultCode(10405, "Token无效")
    ACCOUNT_DISABLED = ResultCode(10406, "账号已被禁用")
    QA_SERVICE_ERROR = ResultCode(10501, "问答服务异常")
    QA_RETRIEVAL_ERROR = ResultCode(10502, "检索服务异常")
    QA_GENERATION_ERROR = ResultCode(10503, "生成服务异常")
    GRPC_CALL_ERROR = ResultCode(10601, "gRPC调用失败")
    KAFKA_SEND_ERROR = ResultCode(10602, "Kafka消息发送失败")
