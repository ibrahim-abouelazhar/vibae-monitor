"""
Multi-machine MLOps Consumer - one thread per machine topic.

Consumes:
  vib-pompe        -> alert-pompe
  vib-ventilateur  -> alert-ventilateur
  vib-compresseur  -> alert-compresseur

Each alert carries "machine" so the WebSocket bridge can route it to the
correct machine card in the dashboard.
"""
import json
import sys
import time
import threading

import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# Configuration
KAFKA_BROKER = "localhost:9092"
BACKEND_ANALYZE_URL = "http://127.0.0.1:8000/analyze"  # FIXED P5
WINDOW_SIZE = 2048  # FIXED P2 P5

MACHINES = ["pompe", "ventilateur", "compresseur"]
VIB_TOPICS = {m: f"vib-{m}" for m in MACHINES}
ALERT_TOPICS = {m: f"alert-{m}" for m in MACHINES}


def ensure_alert_topics():
    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
        existing = set(admin.list_topics())
        to_create = [
            NewTopic(name=t, num_partitions=1, replication_factor=1)
            for t in ALERT_TOPICS.values()
            if t not in existing
        ]
        if to_create:
            admin.create_topics(to_create)
        admin.close()
    except (TopicAlreadyExistsError, Exception):
        pass


ensure_alert_topics()

try:
    alert_producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print("[Consumer] Alert producer connected.", flush=True)
except Exception as e:
    print(f"[Consumer] Kafka producer error: {e}", flush=True)
    sys.exit(1)


def analyze_window(window):
    response = requests.post(BACKEND_ANALYZE_URL, json={"window": window}, timeout=10)  # FIXED P5
    response.raise_for_status()  # FIXED P5
    return response.json()  # FIXED P5


def consume_machine(machine: str):
    vib_topic = VIB_TOPICS[machine]
    alert_topic = ALERT_TOPICS[machine]
    print(f"[Consumer:{machine}] Subscribed to '{vib_topic}' -> '{alert_topic}'", flush=True)

    buffer = []  # FIXED P5

    while True:
        try:
            consumer = KafkaConsumer(
                vib_topic,
                bootstrap_servers=[KAFKA_BROKER],
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="latest",
                group_id=None,
            )
            for msg in consumer:
                try:
                    payload = msg.value
                    sensor_id = payload.get("sensor_id", machine)
                    timestamp = payload.get("timestamp", time.time())
                    vibration_data = payload.get("vibration_data", [])
                    state = payload.get("state", "Sain")

                    if not vibration_data:
                        continue

                    buffer.extend(float(x) for x in vibration_data)  # FIXED P5
                    if len(buffer) < WINDOW_SIZE:  # FIXED P5
                        continue  # FIXED P5
                    if len(buffer) > WINDOW_SIZE:  # FIXED P5
                        buffer = buffer[-WINDOW_SIZE:]  # FIXED P5

                    analysis = analyze_window(buffer)  # FIXED P5
                    mse = analysis.get("mse")  # FIXED P5
                    status = analysis.get("status", "NORMAL")  # FIXED P5
                    reconstructed = analysis.get("reconstructed", [])  # FIXED P5

                    chunk_len = len(vibration_data)
                    chunk_recon = reconstructed[-chunk_len:] if len(reconstructed) >= chunk_len else reconstructed  # FIXED P5

                    alert = {
                        "machine": machine,
                        "sensor_id": sensor_id,
                        "timestamp": timestamp,
                        "state": state,
                        "mse": mse,
                        "status": status,
                        "signal_original": vibration_data,
                        "signal_reconstructed": chunk_recon,
                    }
                    alert_producer.send(alert_topic, alert)
                    alert_producer.flush()
                    print(f"[Consumer:{machine}] MSE={mse:.5f} | {status} | state={state}", flush=True)

                except Exception as e:
                    print(f"[Consumer:{machine}] Processing error: {e}", flush=True)

        except Exception as e:
            print(f"[Consumer:{machine}] Kafka error: {e} - retrying in 3s", flush=True)
            time.sleep(3)


threads = []
for m in MACHINES:
    t = threading.Thread(target=consume_machine, args=(m,), daemon=True)
    t.start()
    threads.append(t)

print("[Consumer] All machine consumer threads started. Press Ctrl+C to stop.", flush=True)
try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("[Consumer] Stopped.", flush=True)
