"""
Multi-machine IoT Sensor Simulator — one thread & one topic per machine.

Topics produced:
  vib-pompe        ← Pompe      (machine_1)
  vib-ventilateur  ← Ventilateur (machine_2)
  vib-compresseur  ← Compresseur (machine_3)

Command topic consumed:
  sensor-command   ← {"machine": "pompe", "command": "Inner Race Fault"}

Each machine has FULLY INDEPENDENT state. Changing Pompe to "Inner Race Fault"
does NOT affect Ventilateur or Compresseur.
"""
import os
import sys
import time
import json
import threading
import asyncio
import numpy as np
import scipy.io as sio
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# ── Configuration ──────────────────────────────────────────────────────────────
KAFKA_BROKER   = "localhost:9092"
COMMAND_TOPIC  = "sensor-command"
CHUNK_SIZE     = 64
SAMPLE_RATE    = 12800
SLEEP_SEC      = 0.1    # 100ms per chunk → 640 samples/sec → oscilloscope buffer scrolls in ~4.7s (readable)
MIN_STATE_DURATION_SEC = 60

MACHINES = ["pompe", "ventilateur", "compresseur"]
TOPICS   = {m: f"vib-{m}" for m in MACHINES}

import glob
import random

# Scan data/raw dynamically for .mat files
raw_paths = glob.glob("data/raw/*.mat")
NORMAL_FILES = [os.path.basename(p) for p in raw_paths if "Normal" in os.path.basename(p)]
FAULT_FILES  = [os.path.basename(p) for p in raw_paths if os.path.basename(p) not in NORMAL_FILES]
ALL_FILES    = NORMAL_FILES + FAULT_FILES

if not NORMAL_FILES:
    print("[Producer] ERROR: No normal files found in data/raw/", flush=True)
    sys.exit(1)

def roll_random_file():
    # 95% random from NORMAL_FILES, 5% random from FAULT_FILES
    if random.random() < 0.95:
        return random.choice(NORMAL_FILES)
    else:
        if FAULT_FILES:
            return random.choice(FAULT_FILES)
        else:
            return random.choice(NORMAL_FILES)

# ── Pre-load signals ────────────────────────────────────────────────────────────
signals: dict[str, np.ndarray] = {}
for filename in ALL_FILES:
    path = f"data/raw/{filename}"
    if not os.path.exists(path):
        print(f"[Producer] ERROR: {path} not found.", flush=True)
        sys.exit(1)
    mat   = sio.loadmat(path)
    de_key = next((k for k in mat if "DE_time" in k), None)
    if de_key is None:
        print(f"[Producer] ERROR: No DE_time key in {path}", flush=True)
        sys.exit(1)
    signals[filename] = mat[de_key].flatten()
    print(f"[Producer] Loaded '{filename}' — {len(signals[filename])} samples", flush=True)

# ── Per-machine isolated state ─────────────────────────────────────────────────
machine_current_file: dict[str, str] = {m: roll_random_file() for m in MACHINES}
machine_mode:         dict[str, str] = {m: "auto" for m in MACHINES}
machine_cursors:      dict[str, int] = {m: 0       for m in MACHINES}
machine_state_started_at: dict[str, float] = {m: time.time() for m in MACHINES}
state_locks:          dict[str, threading.Lock] = {m: threading.Lock() for m in MACHINES}

# ── Ensure topics exist ────────────────────────────────────────────────────────
def ensure_topics():
    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
        existing = set(admin.list_topics())
        to_create = [
            NewTopic(name=topic, num_partitions=1, replication_factor=1)
            for topic in TOPICS.values()
            if topic not in existing
        ]
        if to_create:
            admin.create_topics(to_create)
            for t in to_create:
                print(f"[Producer] Created topic '{t.name}'", flush=True)
        admin.close()
    except TopicAlreadyExistsError:
        pass
    except Exception as e:
        print(f"[Producer] Warning: could not ensure topics ({e}). "
              "Topics will be auto-created on first produce.", flush=True)

ensure_topics()

# ── Kafka producer (shared, thread-safe) ───────────────────────────────────────
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"[Producer] Kafka producer connected to {KAFKA_BROKER}", flush=True)
except Exception as e:
    print(f"[Producer] Failed to connect: {e}", flush=True)
    sys.exit(1)

# ── Duration-gated auto re-rolling check ───────────────────────────────────────
def maybe_reroll(machine: str):
    if machine_mode[machine] != "auto":
        return  # never auto-reroll a manually-overridden machine
    
    elapsed = time.time() - machine_state_started_at[machine]
    if elapsed >= MIN_STATE_DURATION_SEC:
        chosen = roll_random_file()
        with state_locks[machine]:
            machine_current_file[machine] = chosen
            machine_state_started_at[machine] = time.time()
        print(f"[Producer] {machine.upper()} auto-rerolled to {chosen} (elapsed={elapsed:.1f}s)", flush=True)

# ── Command listener — sets state for ONE machine only ─────────────────────────
def command_listener():
    try:
        consumer = KafkaConsumer(
            COMMAND_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest",
        )
        print("[Producer] Command listener ready.", flush=True)
        for msg in consumer:
            cmd = msg.value
            machine = cmd.get("machine", "").lower() if cmd.get("machine") else None
            new_state = cmd.get("command", "")
            
            # If no machine specified (e.g. from global override legacy command)
            if not machine:
                file_map = {
                    "Sain": random.choice(NORMAL_FILES),
                    "Inner Race Fault": "IR014_1_175.mat",
                    "Ball Defect": "B014_1_190.mat"
                }
                target_file = file_map.get(new_state, new_state)
                if target_file in signals:
                    for m in MACHINES:
                        with state_locks[m]:
                            machine_current_file[m] = target_file
                            machine_mode[m] = "manual"
                            machine_state_started_at[m] = time.time()
                    print(f"[Producer] Global Override -> {target_file} for all machines", flush=True)
                continue

            if machine in machine_current_file:
                if new_state == "Reset Auto":
                    chosen = roll_random_file()
                    with state_locks[machine]:
                        machine_current_file[machine] = chosen
                        machine_mode[machine] = "auto"
                        machine_state_started_at[machine] = time.time()
                    print(f"[Producer] {machine.upper()} Reset Auto -> {chosen} (mode=auto)", flush=True)
                else:
                    file_map = {
                        "Sain": random.choice(NORMAL_FILES),
                        "Inner Race Fault": "IR014_1_175.mat",
                        "Ball Defect": "B014_1_190.mat"
                    }
                    target_file = file_map.get(new_state, new_state)
                    if target_file in signals:
                        with state_locks[machine]:
                            machine_current_file[machine] = target_file
                            machine_mode[machine] = "manual"
                            machine_state_started_at[machine] = time.time()
                        print(f"[Producer] {machine.upper()} -> {target_file} (mode=manual)", flush=True)
                    else:
                        print(f"[Producer] Unknown command/file for {machine.upper()}: {new_state!r}", flush=True)
            else:
                print(f"[Producer] Unknown machine: {machine!r}", flush=True)
    except Exception as e:
        print(f"[Producer] Command listener error: {e}", flush=True)

threading.Thread(target=command_listener, daemon=True).start()

# ── Per-machine producer loop (asyncio task) ───────────────────────────────────
async def produce_for_machine(machine: str):
    topic = TOPICS[machine]
    sensor_id = f"Sensor-{machine}"
    print(f"[Producer] Starting loop for {machine.upper()} -> topic '{topic}'", flush=True)

    while True:
        try:
            maybe_reroll(machine)

            with state_locks[machine]:
                state  = machine_current_file[machine]
                cursor = machine_cursors[machine]

            signal       = signals[state]
            total        = len(signal)
            end          = cursor + CHUNK_SIZE

            if end <= total:
                window = signal[cursor:end].tolist()
            else:
                window = np.concatenate([signal[cursor:], signal[:end - total]]).tolist()

            payload = {
                "machine":        machine,
                "sensor_id":      sensor_id,
                "timestamp":      time.time(),
                "state":          state,
                "vibration_data": window,
            }
            producer.send(topic, payload)

            new_cursor = (cursor + CHUNK_SIZE) % total
            with state_locks[machine]:
                machine_cursors[machine] = new_cursor

            await asyncio.sleep(SLEEP_SEC)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Producer:{machine}] Error: {e}", flush=True)
            await asyncio.sleep(1)

# ── Launch concurrent asyncio tasks ───────────────────────────────────────────
async def main():
    print("[Producer] Starting asyncio event loop...", flush=True)
    tasks = []
    for m in MACHINES:
        tasks.append(asyncio.create_task(produce_for_machine(m)))
    
    print("[Producer] All machine tasks started. Press Ctrl+C to stop.", flush=True)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Producer] Stopped.", flush=True)
