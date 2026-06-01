"""Nacos service registration for Python microservices."""
import json
import os
import socket
import threading
import time
import requests

NACOS_HOST = os.environ.get("NACOS_HOST", "nacos")
NACOS_PORT = os.environ.get("NACOS_PORT", "8848")
NACOS_NAMESPACE = os.environ.get("NACOS_NAMESPACE", "")  # public namespace
NACOS_GROUP = os.environ.get("NACOS_GROUP", "RAG_GROUP")

def register_service(service_name: str, port: int, metadata: dict | None = None):
    """Register this service with Nacos via HTTP API."""
    ip = socket.gethostbyname(socket.gethostname())
    url = f"http://{NACOS_HOST}:{NACOS_PORT}/nacos/v1/ns/instance"
    meta_str = json.dumps(metadata or {})
    
    def _beat():
        while True:
            try:
                resp = requests.post(url, params={
                    "serviceName": service_name, "ip": ip, "port": port,
                    "namespaceId": NACOS_NAMESPACE, "groupName": NACOS_GROUP,
                    "enable": "true", "healthy": "true", "metadata": meta_str,
                }, timeout=10)
                if resp.status_code == 200:
                    print(f"[Nacos] {service_name} registered OK")
                else:
                    print(f"[Nacos] {service_name} failed: {resp.status_code} {resp.text[:100]}")
            except Exception as e:
                print(f"[Nacos] {service_name} error: {e}")
            time.sleep(10)
    
    t = threading.Thread(target=_beat, daemon=True, name=f"nacos-{service_name}")
    t.start()
    print(f"[Nacos] Registration started: {service_name} @ {ip}:{port}")
