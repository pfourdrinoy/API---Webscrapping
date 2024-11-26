import kagglehub
import shutil
import subprocess
import os
from fastapi import APIRouter
from src.schemas.message import MessageResponse

router = APIRouter()

@router.get("/data", name="Download route", response_model=MessageResponse)
def data() -> MessageResponse:
    download_path = "TP2 and  3\services\epf-flower-data-science\src\data"
    try:
        path = kagglehub.dataset_download("uciml/iris")
        destination_path = os.path.join(download_path, os.path.basename(path))
        shutil.move(path, destination_path)
        return MessageResponse(message=f"Data successfully downloaded and moved to: {destination_path}")
    except Exception as e:
        return MessageResponse(message=f"Error during download or file move: {e}")
