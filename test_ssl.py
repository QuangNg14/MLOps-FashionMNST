import ssl
import certifi
import requests

print("Certifi CA Bundle:", certifi.where())

try:
    response = requests.get(
        "https://mlops-fashionmnst.s3.us-west-2.amazonaws.com", verify=certifi.where()
    )
    print("Success:", response.status_code)
except requests.exceptions.SSLError as e:
    print("SSL Error:", e)
