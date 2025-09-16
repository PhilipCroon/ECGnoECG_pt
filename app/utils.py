# app/utils.py
import io
from PIL import Image
import pydicom
from pdf2image import convert_from_bytes

def pil_from_bytes(content: bytes) -> Image.Image:
    """Convert raw image bytes (PNG/JPEG) to a PIL image."""
    return Image.open(io.BytesIO(content)).convert("RGB")

def dicom_to_pil(content: bytes):
    """Convert DICOM bytes to a PIL image (first frame if multiframe)."""
    dcm = pydicom.dcmread(io.BytesIO(content))
    arr = dcm.pixel_array

    # Handle multi-frame ECG/DICOM
    if arr.ndim == 3:
        arr = arr[0]

    img = Image.fromarray(arr)
    return img.convert("RGB"), "dicom"

def pdf_to_pil(content: bytes):
    """Convert first page of PDF bytes to a PIL image."""
    pages = convert_from_bytes(content, dpi=300)
    if not pages:
        raise ValueError("No pages found in PDF")
    return pages[0].convert("RGB"), "pdf"