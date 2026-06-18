import json
import time
from kafka import KafkaProducer, KafkaConsumer

KAFKA_BROKER = "localhost:9092"
TOPIC = "test-topic"

try:
    print("Creating producer...")
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    print("Sending message...")
    future = producer.send(TOPIC, {"hello": "world", "time": time.time()})
    print("Flushing...")
    producer.flush()
    print("Message sent successfully. Future metadata:", future.get(timeout=5))

    print("Creating consumer...")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000
    )
    print("Reading message...")
    for msg in consumer:
        print("Received:", msg.value)
        break
    else:
        print("No messages received.")
except Exception as e:
    print("Error:", e)
