import time
import json
import requests
from kafka import KafkaConsumer

API_URL = "http://localhost:8000/kafka/command"

def run_test():
    print("Connecting to alert-pompe Kafka topic...", flush=True)
    consumer = KafkaConsumer(
        "alert-pompe",
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="latest"
    )
    
    # Wait for the first message to verify connection is active
    print("Waiting for active stream...", flush=True)
    next(consumer)
    print("Stream active.", flush=True)

    print("\n--- 1. Injecting 'Inner Race Fault' on Pompe ---", flush=True)
    res = requests.post(API_URL, json={"machine": "pompe", "command": "Inner Race Fault"})
    print("API Response:", res.json(), flush=True)

    print("Monitoring Pompe status for 5 seconds to ensure it holds steady (no auto-rerolls)...", flush=True)
    start_time = time.time()
    states_seen = []
    
    for message in consumer:
        alert = message.value
        state = alert.get("state", "")
        mse = alert.get("mse", 0.0)
        status = alert.get("status", "")
        # Filter out messages that might have been in flight before override command took effect
        if time.time() - start_time > 1.5: 
            states_seen.append(state)
        print(f"[{time.time() - start_time:.2f}s] Alert: state={state}, status={status}, mse={mse:.5f}", flush=True)
        
        if time.time() - start_time > 5.0:
            break

    states_set = set(states_seen)
    print(f"States observed during override (excluding first 1.5s of transition): {states_set}", flush=True)
    assert all(s == "IR014_1_175.mat" for s in states_seen), f"Pompe state changed during manual override! Seen: {states_set}"

    print("\n--- 2. Sending 'Reset Auto' for Pompe ---", flush=True)
    res = requests.post(API_URL, json={"machine": "pompe", "command": "Reset Auto"})
    print("API Response:", res.json(), flush=True)

    print("Monitoring Pompe status to confirm it switched back to normal/auto file...", flush=True)
    start_time = time.time()
    states_seen = []
    for message in consumer:
        alert = message.value
        state = alert.get("state", "")
        mse = alert.get("mse", 0.0)
        status = alert.get("status", "")
        if time.time() - start_time > 1.5:
            states_seen.append(state)
        print(f"[{time.time() - start_time:.2f}s] Alert: state={state}, status={status}, mse={mse:.5f}", flush=True)
        if time.time() - start_time > 5.0:
            break

    states_set = set(states_seen)
    print(f"States observed after reset (excluding first 1.5s of transition): {states_set}", flush=True)
    assert any("Normal" in s for s in states_seen), "Pompe did not return to normal auto mode!"
    print("\nVerification Test Completed Successfully!", flush=True)

if __name__ == "__main__":
    run_test()
