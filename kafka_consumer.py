from kafka import KafkaConsumer
import json
import requests

consumer = KafkaConsumer(
    "data_stream",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)

for message in consumer:
    data = message.value
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    print(f"Received: {data} --> Prediction: {response.json()}")
