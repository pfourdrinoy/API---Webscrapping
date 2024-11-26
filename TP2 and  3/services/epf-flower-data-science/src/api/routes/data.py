import kagglehub
import shutil
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from fastapi import APIRouter
from src.schemas.message import MessageResponse

router = APIRouter()

@router.get("/data", name="Download route", response_model=MessageResponse)
def data() -> MessageResponse:
    destination_path = "TP2 and  3\services\epf-flower-data-science\src\data"
    target_path = r"C:\Users\paulf\.cache\kagglehub\datasets\uciml\iris\versions\2\iris.csv"
    
    try:
        path = kagglehub.dataset_download("uciml/iris")
        shutil.move(target_path, destination_path)
        return MessageResponse(message=f"Data successfully downloaded and moved to: {destination_path}")
    except Exception as e:
        return MessageResponse(message=f"Error during download or file move: {e}")
    
@router.get("/load_iris", name="Load Iris Dataset", response_model=MessageResponse)
def load_iris() -> MessageResponse:
    download_path = r"TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    
    try:
        iris_df = pd.read_csv(download_path)
        iris_json = iris_df.to_dict(orient='records')
        iris_json_str = json.dumps(iris_json)
        return MessageResponse(message=iris_json_str)
    
    except Exception as e:
        return MessageResponse(message=f"Error during loading the dataset: {e}")
    
@router.get("/process_iris", name="Process Iris Dataset", response_model=MessageResponse)
def process_iris() -> MessageResponse:
    path = r"TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    
    try:
        iris_df = pd.read_csv(path)
        iris_df = iris_df.dropna()
        if 'Id' in iris_df.columns:
            iris_df = iris_df.drop(columns=["Id"])
        iris_df.to_csv(path, index=False)
        return MessageResponse(message="Data processed successfully, ready for model training.")
    
    except Exception as e:
        return MessageResponse(message=f"Error during data processing: {e}")

@router.get("/split_iris", name="Split Iris Dataset", response_model=MessageResponse)
def process_iris() -> MessageResponse:
    path = r"TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    short_path = r"TP2 and  3\services\epf-flower-data-science\src\data"
    
    try:
        iris_df = pd.read_csv(path)
        iris_df = iris_df.dropna()
        X = iris_df.drop(columns=["Species"])
        y = iris_df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        datasets = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        for key, value in datasets.items():
            file_path = os.path.join(short_path, f"{key}.csv")
            value.to_csv(file_path, index=False)

        datasets_json = {
            key: json.dumps(value.values.tolist()) if isinstance(value, pd.DataFrame) else json.dumps(value.tolist())
            for key, value in datasets.items()
        }
        datasets_json_str = json.dumps(datasets_json)
        return MessageResponse(message=datasets_json_str)
    
    except Exception as e:
        return MessageResponse(message=f"Error during data spliting: {e}")
