"""Configuration center client — listens to Kafka config changes and hot-reloads.

All Python services (RAG-PYTHON, RAG-CLEANING, RAG-MEMORY, RAG-QUE) use this
module to subscribe to the `rag-config-change` Kafka topic. When a config value
changes, the in-memory cache is updated immediately, and registered reload
callbacks are invoked.

Usage:
    from common.config_center import get_config_center
    cc = get_config_center()
    size = cc.get_int("chunk.default_size", 500)
"""

import json
import logging
import os
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Kafka consumer config
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CONFIG_CHANGE_TOPIC = "rag-config-change"
CONSUMER_GROUP_PREFIX = "rag-config-consumer"


class ConfigCenter:
    """In-memory config cache with Kafka-driven hot reload."""

    def __init__(self, service_name: str, default_config: dict[str, Any] | None = None):
        """
        Args:
            service_name: Used to filter which config changes apply to this service.
            default_config: Fallback values (e.g. from settings.yaml).
        """
        self._service_name = service_name
        self._cache: dict[str, Any] = {}
        self._reload_callbacks: dict[str, list[Callable[[str, Any], None]]] = {}

        # Initialize with defaults
        if default_config:
            self._cache.update(self._flatten(default_config))

        self._running = False
        self._thread: threading.Thread | None = None

    # ─── Public API ──────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self._cache.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._cache.get(key, default))
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self._cache.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    def get_str(self, key: str, default: str = "") -> str:
        val = self._cache.get(key, default)
        return str(val) if val is not None else default

    def set_local(self, key: str, value: Any):
        """Manually update a config value in cache (no Kafka publish)."""
        self._cache[key] = value

    # ─── Reload callbacks ────────────────────────────────

    def on_change(self, key_prefix: str, callback: Callable[[str, Any], None]):
        """Register a callback invoked when any config with the given prefix changes.

        Example:
            cc.on_change("chunk.", lambda k, v: reload_chunker())
        """
        if key_prefix not in self._reload_callbacks:
            self._reload_callbacks[key_prefix] = []
        self._reload_callbacks[key_prefix].append(callback)

    # ─── Kafka consumer ──────────────────────────────────

    def start(self):
        """Start listening for config changes via Kafka."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="config-listener")
        self._thread.start()
        logger.info(f"ConfigCenter started for service={self._service_name}, topic={CONFIG_CHANGE_TOPIC}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                CONFIG_CHANGE_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=f"{CONSUMER_GROUP_PREFIX}-{self._service_name}",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=30000,
            )
            logger.info(f"ConfigCenter Kafka consumer connected: {KAFKA_BOOTSTRAP_SERVERS}")
            for msg in consumer:
                if not self._running:
                    break
                try:
                    self._handle_event(msg.value)
                except Exception as e:
                    logger.error(f"ConfigCenter event handler error: {e}")
            consumer.close()
        except ImportError:
            logger.warning("kafka-python not installed; ConfigCenter running without Kafka listener")
        except Exception as e:
            logger.warning(f"ConfigCenter Kafka consumer failed: {e}. Running with static config.")

    def _handle_event(self, event: dict):
        key = event.get("configKey", "")
        new_value = event.get("newValue")
        value_type = event.get("valueType", "STRING")
        target_services = event.get("targetServices", "")
        reload_strategy = event.get("reloadStrategy", "kafka")

        # Filter: only apply if this service is targeted
        if target_services and self._service_name not in target_services.split(","):
            return

        # Cast value
        old_value = self._cache.get(key)
        cast_value = self._cast(new_value, value_type)

        # Update cache
        self._cache[key] = cast_value
        logger.info(f"ConfigCenter updated: {key} = {cast_value} (strategy={reload_strategy}, old={old_value})")

        # Fire reload callbacks
        for prefix, callbacks in self._reload_callbacks.items():
            if key.startswith(prefix):
                for cb in callbacks:
                    try:
                        cb(key, cast_value)
                    except Exception as e:
                        logger.error(f"ConfigCenter callback '{prefix}' error: {e}")

        # Special: restart strategy
        if reload_strategy == "restart":
            logger.warning(
                f"Config '{key}' requires service restart to take effect. "
                f"New value '{cast_value}' saved but NOT applied yet."
            )

    @staticmethod
    def _cast(value: str, value_type: str) -> Any:
        if value_type in ("NUMBER",):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        elif value_type == "BOOLEAN":
            return value.lower() in ("true", "1", "yes")
        elif value_type == "JSON":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict to dot-notation keys matching DB config keys."""
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict) and not any(isinstance(v, t) for t in (list, tuple)):
                result.update(ConfigCenter._flatten(v, f"{full_key}."))
            else:
                result[full_key] = v
        return result


# ─── Global singleton ─────────────────────────────────

_config_center: ConfigCenter | None = None


def init_config_center(service_name: str, default_config: dict[str, Any] | None = None) -> ConfigCenter:
    """Initialize and start the global config center singleton."""
    global _config_center
    if _config_center is not None:
        return _config_center
    _config_center = ConfigCenter(service_name, default_config)
    _config_center.start()
    return _config_center


def get_config_center() -> ConfigCenter:
    """Get the global config center singleton (must call init_config_center first)."""
    global _config_center
    if _config_center is None:
        _config_center = ConfigCenter("unknown")
    return _config_center
