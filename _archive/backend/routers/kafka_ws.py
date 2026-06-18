"""
Kafka alert bridge: SSE endpoint for the browser dashboard + command REST endpoint.

SSE replaces WebSocket to avoid the websockets library keepalive_ping AssertionError
that kills connections under load.

- GET  /kafka/stream   → text/event-stream  (browser EventSource)
- POST /kafka/command  { "machine": "pompe", "command": "Inner Race Fault" }
- GET  /kafka/status
"""
import json
import asyncio
import threading
import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kafka import KafkaConsumer, KafkaProducer

router = APIRouter()

KAFKA_BROKER  = "localhost:9092"
COMMAND_TOPIC = "sensor-command"
MACHINES      = ["pompe", "ventilateur", "compresseur"]
ALERT_TOPICS  = [f"alert-{m}" for m in MACHINES]


# ── Singleton KafkaBridge ──────────────────────────────────────────────────────
class KafkaBridge:
    """
    Single background thread that consumes all three alert topics and
    fans out to every connected SSE client via asyncio.Queue.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._clients: dict = {}        # client_id → (asyncio.Queue, asyncio.Loop)
        self._clients_lock = threading.Lock()
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def subscribe(self, client_id: int, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=200)
        with self._clients_lock:
            self._clients[client_id] = (q, loop)
        return q

    def unsubscribe(self, client_id: int):
        with self._clients_lock:
            self._clients.pop(client_id, None)

    def _consume_loop(self):
        print("[KafkaBridge] Starting — subscribing to:", ALERT_TOPICS, flush=True)
        while True:
            try:
                consumer = KafkaConsumer(
                    *ALERT_TOPICS,
                    bootstrap_servers=[KAFKA_BROKER],
                    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                    auto_offset_reset="latest",
                    group_id=None,
                )
                print("[KafkaBridge] Connected to Kafka.", flush=True)
                for message in consumer:
                    alert = message.value
                    with self._clients_lock:
                        snapshot = list(self._clients.items())
                    for client_id, (q, loop) in snapshot:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                q.put(alert), loop
                            )
                        except Exception:
                            pass
            except Exception as e:
                print(f"[KafkaBridge] Error: {e} — retrying in 3s", flush=True)
                time.sleep(3)


# ── SSE endpoint (replaces WebSocket) ─────────────────────────────────────────
@router.get("/kafka/stream")
async def kafka_stream_sse(request: Request):
    """
    Server-Sent Events stream of Kafka alerts.
    The browser uses EventSource('/kafka/stream') — no WebSocket needed.
    Each event: data: <JSON>\\n\\n
    """
    client_id = id(request)
    loop = asyncio.get_event_loop()

    bridge = KafkaBridge.get()
    queue  = bridge.subscribe(client_id, loop)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    alert = await asyncio.wait_for(queue.get(), timeout=2.0)
                    payload = {
                        "type":                 "alert",
                        "machine":              alert.get("machine", "pompe"),
                        "sensor_id":            alert.get("sensor_id", ""),
                        "timestamp":            alert.get("timestamp", 0),
                        "state":                alert.get("state", "Sain"),
                        "mse":                  alert.get("mse", 0.0),
                        "status":               alert.get("status", "NORMAL"),
                        "signal_original":      alert.get("signal_original", []),
                        "signal_reconstructed": alert.get("signal_reconstructed", []),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive comment — prevents browser from timing out
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            bridge.unsubscribe(client_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Keep WebSocket for backward compat (but SSE is preferred) ─────────────────
# Kept here so old browser tabs connecting to /ws/kafka-stream still work.
from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws/kafka-stream")
async def kafka_stream_ws(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    loop = asyncio.get_event_loop()

    bridge = KafkaBridge.get()
    queue  = bridge.subscribe(client_id, loop)

    try:
        await websocket.send_json({"type": "connected"})
    except Exception:
        bridge.unsubscribe(client_id)
        return

    try:
        while True:
            try:
                alert = await asyncio.wait_for(queue.get(), timeout=2.0)
                payload = {
                    "type":                 "alert",
                    "machine":              alert.get("machine", "pompe"),
                    "sensor_id":            alert.get("sensor_id", ""),
                    "timestamp":            alert.get("timestamp", 0),
                    "state":                alert.get("state", "Sain"),
                    "mse":                  alert.get("mse", 0.0),
                    "status":               alert.get("status", "NORMAL"),
                    "signal_original":      alert.get("signal_original", []),
                    "signal_reconstructed": alert.get("signal_reconstructed", []),
                }
                await websocket.send_json(payload)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[KafkaBridge] WS {client_id} error: {e}", flush=True)
    finally:
        bridge.unsubscribe(client_id)


# ── Command endpoint ───────────────────────────────────────────────────────────
class KafkaCommandRequest(BaseModel):
    machine: str   # "pompe" | "ventilateur" | "compresseur"
    command: str   # "Sain" | "Inner Race Fault" | "Ball Defect"

_cmd_producer = None
_cmd_lock = threading.Lock()

def _get_cmd_producer():
    global _cmd_producer
    with _cmd_lock:
        if _cmd_producer is None:
            try:
                _cmd_producer = KafkaProducer(
                    bootstrap_servers=[KAFKA_BROKER],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
            except Exception as e:
                print(f"[KafkaBridge] Cannot create command producer: {e}", flush=True)
    return _cmd_producer


@router.post("/kafka/command")
def send_kafka_command(body: KafkaCommandRequest):
    """
    Sends a targeted state-switch command for ONE machine only.
    """
    machine = body.machine.lower()
    if machine not in MACHINES:
        return {"status": "error", "message": f"Unknown machine: {machine!r}. Must be one of {MACHINES}"}

    producer = _get_cmd_producer()
    if producer is None:
        return {"status": "error", "message": "Kafka producer unavailable"}

    try:
        producer.send(COMMAND_TOPIC, {"machine": machine, "command": body.command})
        producer.flush()
        return {"status": "ok", "machine": machine, "command": body.command}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/kafka/status")
def get_kafka_status():
    return {
        "machines":     MACHINES,
        "vib_topics":   {m: f"vib-{m}"   for m in MACHINES},
        "alert_topics": {m: f"alert-{m}" for m in MACHINES},
    }
