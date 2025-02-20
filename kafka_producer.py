from kafka import KafkaProducer
import json
import random
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",  # Update with Kafka server details
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
)

while True:
    data = {
        "features": [random.uniform(-2, 2) for _ in range(10)]  # Replace with actual data
    }
    producer.send("data_stream", value=data)
    print(f"Sent: {data}")
    time.sleep(1)
