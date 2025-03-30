import requests

url = "http://127.0.0.1:8000/chat/"
params = {"query": "How do I return my order?"}

response = requests.post(url, params=params)
print(response.json())
