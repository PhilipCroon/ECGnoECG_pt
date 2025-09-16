# ECGnoECG Dockerized Service - README

## Introduction

This project provides a Dockerized service for ECG detection (ECGnoECG). The service exposes a REST API for detecting ECG signals in uploaded files. This README explains how to build, run, test, and use the service.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- (Optional) [curl](https://curl.se/) for command-line API usage.
- (Optional) Python 3.x for script-based usage.

## Authentication

The API is protected with an API key. You must include the header `X-API-Key` with the value `<your-api-key>` in all your requests to authenticate.

Example header:

```
X-API-Key: <your-api-key>
```

> **Note:** By default, the Dockerfile sets the API key to `CarDS_1@3*`. You can override this when running the container.

## Build & Run

### Build the Docker Image

Run the following command from the project root (where the Dockerfile is located):

```bash
docker build -t ecg-service .
```

### Run the Docker Container

```bash
docker run -d -p 8000:8000 --name ecgnoecg_container ecg-service
```
This command starts the service on port 8000.

## API Usage

The ECGnoECG service exposes a REST API via FastAPI.

### Endpoint: `/predict`

**Method:** `POST`

**Description:** Upload an ECG file (supported formats: DICOM `.dcm`, PNG `.png`, JPEG `.jpg`/`.jpeg`, PDF `.pdf`) to receive ECG detection results.

**Request:**
- `multipart/form-data` with a file field named `file`
- Header: `X-API-Key` with your API key

**Response:**
- JSON with detection results.

### Endpoint: `/predict_async`

**Method:** `POST`

**Description:** Upload an ECG file to submit an asynchronous detection job. The service responds immediately with a `task_id` that can be used to check the job status and results later.

**Request:**
- `multipart/form-data` with a file field named `file`
- Header: `X-API-Key` with your API key

**Response:**
- JSON containing a `task_id` string.

### Endpoint: `/result/{task_id}`

**Method:** `GET`

**Description:** Check the status and retrieve results of an asynchronous detection job using the `task_id`.

**Request:**
- URL path parameter: `task_id` (string)
- Header: `X-API-Key` with your API key

**Response:**
- JSON containing the job status (`PENDING`, `STARTED`, `SUCCESS`, `FAILURE`) and, if completed successfully, the detection results.

## Running Tests

The test suite includes tests for both synchronous and asynchronous endpoints. To run the test suite (from within the project directory):

```bash
docker run --rm ecg-service pytest
```

Or, if you want to run tests using the container interactively:

```bash
docker exec -it ecgnoecg_container pytest
```

## Examples

### Example: Using curl

Suppose you have a file called `sample_ecg.dcm`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@sample_ecg.dcm"
```

### Example: Using Python requests

```python
import requests

files = {'file': open('sample_ecg.dcm', 'rb')}
headers = {'X-API-Key': '<your-api-key>'}
response = requests.post('http://localhost:8000/predict', files=files, headers=headers)
print(response.json())
```

### Example: Using curl for `/predict_async` (submit async job)

```bash
curl -X POST "http://localhost:8000/predict_async" \
  -H "accept: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@sample_ecg.dcm"
```

### Example: Using curl for `/result/{task_id}` (check async job status)

```bash
curl -X GET "http://localhost:8000/result/<task_id>" \
  -H "accept: application/json" \
  -H "X-API-Key: <your-api-key>"
```

### Example: Using Python requests for `/predict_async` (submit async job)

```python
import requests

files = {'file': open('sample_ecg.dcm', 'rb')}
headers = {'X-API-Key': '<your-api-key>'}
response = requests.post('http://localhost:8000/predict_async', files=files, headers=headers)
print(response.json())  # Contains 'task_id'
```

### Example: Using Python requests for `/result/{task_id}` (check async job status)

```python
import requests

task_id = 'your-task-id-here'
headers = {'X-API-Key': '<your-api-key>'}
response = requests.get(f'http://localhost:8000/result/{task_id}', headers=headers)
print(response.json())
```

## Curl Examples

**Basic file upload (supported file types: DICOM, PNG, JPEG, PDF):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "X-API-Key: <your-api-key>" \
     -F "file=@path/to/your/file.dcm"
```

**With custom headers:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "X-API-Key: <your-api-key>" \
     -F "file=@path/to/your/file.png"
```

## Notes

- Adjust the port (`8000`) if you mapped a different port when starting the container.
- The API requires an API key set in the Dockerfile (default: `CarDS_1@3*`).
- See the codebase for further details on accepted file formats and API options.
- For development, you can mount your code into the container with `-v $(pwd):/app` (modify as needed).
