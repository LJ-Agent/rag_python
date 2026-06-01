"""Nacos service registration for Python microservices.
Called at startup to register the service with Nacos.
"""
import logging
import os
import socket
import requests
import threading
import time

logger = logging.getLogger(__name__)

NACOS_HOST = os.environ.get("NACOS_HOST", "nacos")
NACOS_PORT = os.environ.get("NACOS_PORT", "8848")
NACOS_NAMESPACE = os.environ.get("NACOS_NAMESPACE", "rag-prod")
NACOS_GROUP = os.environ.get("NACOS_GROUP", "RAG_GROUP")

def register_service(service_name: str, port: int, metadata: dict | None = None):
    """Register this service instance with Nacos."""
    ip = socket.gethostbyname(socket.gethostname())
    
    params = {
        "serviceName": service_name,
        "ip": ip,
        "port": port,
        "namespaceId": NACOS_NAMESPACE,
        "groupName": NACOS_GROUP,
        "enable": True,
        "healthy": True,
        "metadata": (metadata or {}).__str__(),
    }
    
    url = f"http://{NACOS_HOST}:{NACOS_PORT}/nacos/v1/ns/instance"
    
    def _register_loop():
        while True:
            try:
                resp = requests.post(url, params=params, timeout=10)
                if resp.status_code == 200 and resp.text == "ok":
                    logger.info(f"Nacos registered: {service_name} @ {ip}:{port}")
                else:
                    logger.warning(f"Nacos registration failed: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.warning(f"Nacos registration error: {e}")
            time.sleep(10)  # Heartbeat every 10s
    
    t = threading.Thread(target=_register_loop, daemon=True, name="nacos-registry")
    t.start()
    logger.info(f"Nacos registration started for {service_name}")
