"""MinIO object storage client, aligned with Java MinioConfig."""
from minio import Minio
from minio.error import S3Error
from common.config_loader import get_config
from common.exception.exceptions import ResourceException
from common.util.logger import get_logger

logger = get_logger()


class MinioClient:
    """Read-only MinIO client for retrieving files uploaded by Java service."""

    def __init__(self):
        cfg = get_config()["minio"]
        self._endpoint = cfg["endpoint"]
        self._bucket = cfg["bucket_name"]
        self._client = Minio(
            endpoint=cfg["endpoint"],
            access_key=cfg["access_key"],
            secret_key=cfg["secret_key"],
            secure=cfg.get("secure", False),
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self._client.bucket_exists(self._bucket):
            raise ResourceException(f"MinIO bucket not found: {self._bucket}")

    def _strip_bucket_prefix(self, object_path: str) -> str:
        """Strip bucket name prefix if present (Java may include it in originalFileUrl)."""
        prefix = self._bucket + "/"
        if object_path.startswith(prefix):
            return object_path[len(prefix):]
        return object_path

    def get_object(self, object_path: str) -> bytes:
        """Read an object's full content from MinIO by path."""
        clean_path = self._strip_bucket_prefix(object_path)
        try:
            response = self._client.get_object(self._bucket, clean_path)
            data = response.read()
            response.close()
            response.release_conn()
            logger.info(f"MinIO read: {clean_path} ({len(data)} bytes)")
            return data
        except S3Error as e:
            raise ResourceException(f"MinIO read failed: {clean_path} — {e}")

    def put_object(self, object_path: str, data: bytes, content_type: str = "text/plain"):
        """Write an object to MinIO."""
        import io
        clean_path = self._strip_bucket_prefix(object_path)
        try:
            self._client.put_object(
                self._bucket, clean_path,
                io.BytesIO(data), len(data),
                content_type=content_type,
            )
            logger.info(f"MinIO write: {clean_path} ({len(data)} bytes)")
        except S3Error as e:
            raise ResourceException(f"MinIO write failed: {clean_path} — {e}")

    def get_object_stream(self, object_path: str, chunk_size: int = 8192):
        """Stream read an object, yielding chunks."""
        clean_path = self._strip_bucket_prefix(object_path)
        try:
            response = self._client.get_object(self._bucket, clean_path)
            for chunk in response.stream(chunk_size):
                yield chunk
            response.close()
            response.release_conn()
        except S3Error as e:
            raise ResourceException(f"MinIO stream failed: {clean_path} — {e}")

    def object_exists(self, object_path: str) -> bool:
        """Check if an object exists in the bucket."""
        clean_path = self._strip_bucket_prefix(object_path)
        try:
            self._client.stat_object(self._bucket, clean_path)
            return True
        except S3Error:
            return False

    def get_presigned_url(self, object_path: str, expires_seconds: int = 3600) -> str:
        """Generate a temporary download URL (7-day expiry matching Java config)."""
        clean_path = self._strip_bucket_prefix(object_path)
        try:
            return self._client.presigned_get_object(self._bucket, clean_path, expires_seconds)
        except S3Error as e:
            raise ResourceException(f"MinIO presigned URL failed: {clean_path} — {e}")

    @property
    def bucket(self) -> str:
        return self._bucket


# Singleton
_minio_client: MinioClient | None = None


def get_minio_client() -> MinioClient:
    global _minio_client
    if _minio_client is None:
        _minio_client = MinioClient()
    return _minio_client
