import requests

url = 'http://localhost:8000/predict'
r = requests.post(url,json={'quality_of_sleep':2, 'sleep_duration':9, 'stress_level':6,'physical_activity':2})

print(r.json())