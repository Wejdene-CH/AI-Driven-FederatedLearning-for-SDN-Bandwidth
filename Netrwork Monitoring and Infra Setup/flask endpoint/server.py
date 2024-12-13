import requests

file_path = r"G:\Documents\model.pth"

upload_url = "http://192.168.85.144:5001/model.pth"

try:
    with open(file_path, 'rb') as file:
        response = requests.post(upload_url, files={'file': file})

    print(response.json())
except Exception as e:
    print(f"Error: {e}")
