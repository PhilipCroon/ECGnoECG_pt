# %%
import requests

API_URL = "http://localhost:8000/predict"
API_KEY = "CarDS_1@3*"
FILE_PATH = "/Users/philipcroon/PycharmProjects/Yale/ECGnoECG_docker/tests/data/test_ecg.dcm"  # or image
IMAGE_PATH = "/Users/philipcroon/PycharmProjects/Yale/ECGnoECG_docker/tests/data/test_image.png"

def send_file(file_path, return_render=False):
    with open(file_path, "rb") as f:
        files = {"file": f}
        params = {"return_render": str(return_render).lower()}
        headers = {"X-API-Key": API_KEY}
        response = requests.post(API_URL, files=files, params=params, headers=headers)
    if response.ok:
        print("✅ Success")
        print(response.json())
    else:
        print("❌ Error", response.status_code, response.text)

def send_image():
    print("Sending image file...")
    send_file(IMAGE_PATH, return_render=False)

if __name__ == "__main__":
    print("Sending DICOM file...")
    send_file(FILE_PATH, return_render=True)
    print("Sending image file...")
    send_file(IMAGE_PATH, return_render=False)