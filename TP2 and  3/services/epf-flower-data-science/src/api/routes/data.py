import kagglehub
import shutil
import pandas as pd
import numpy as np
import json
import os
import joblib
import firebase_admin
from sklearn.model_selection import train_test_split
from fastapi import APIRouter
from src.schemas.message import MessageResponse
from sklearn.linear_model import LogisticRegression
from google.cloud import firestore
from firebase_admin import credentials, firestore
from fastapi import HTTPException

router = APIRouter()

from fastapi import HTTPException

@router.get("/data", name="Download dataset from Kaggle", response_model=MessageResponse)
def data() -> MessageResponse:
    """
    Downloads the Iris dataset from Kaggle using `kagglehub`, 
    and moves it to the specified destination directory.

    Returns:
        MessageResponse: A success message if the dataset was downloaded and moved successfully, 
        or an error message otherwise.
    """
    destination_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data"
    target_path = r"C:\Users\paulf\.cache\kagglehub\datasets\uciml\iris\versions\2\iris.csv"
    
    try:
        path = kagglehub.dataset_download("uciml/iris")
        shutil.move(target_path, destination_path)
        return MessageResponse(message=f"Data successfully downloaded and moved to: {destination_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during download or file move: {e}")


@router.get("/load_iris", name="Load Iris Dataset", response_model=MessageResponse)
def load_iris() -> MessageResponse:
    """
    Loads the Iris dataset from a local CSV file and returns it as JSON.

    Returns:
        MessageResponse: JSON representation of the Iris dataset, 
        or an error message if loading fails.
    """
    download_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    
    try:
        iris_df = pd.read_csv(download_path)
        iris_json = iris_df.to_dict(orient='records')
        iris_json_str = json.dumps(iris_json)
        return MessageResponse(message=iris_json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during loading the dataset: {e}")


@router.get("/process_iris", name="Process Iris Dataset", response_model=MessageResponse)
def process_iris() -> MessageResponse:
    """
    Processes the Iris dataset by removing missing values and unnecessary columns.
    Saves the cleaned dataset back to the original path.

    Returns:
        MessageResponse: A success message if processing is successful, 
        or an error message otherwise.
    """
    path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    
    try:
        iris_df = pd.read_csv(path)
        iris_df = iris_df.dropna()
        if 'Id' in iris_df.columns:
            iris_df = iris_df.drop(columns=["Id"])
        iris_df.to_csv(path, index=False)
        return MessageResponse(message="Data processed successfully, ready for model training.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during data processing: {e}")


@router.get("/split_iris", name="Split Iris Dataset", response_model=MessageResponse)
def split_iris() -> MessageResponse:
    """
    Splits the Iris dataset into training and testing datasets.
    Saves these splits as separate CSV files and returns them as JSON.

    Returns:
        MessageResponse: JSON representation of the split datasets, 
        or an error message if splitting fails.
    """
    path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data\iris.csv"
    short_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data"
    
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
        raise HTTPException(status_code=500, detail=f"Error during data splitting: {e}")


@router.get("/train_model", name="Train the Logistic Regression Model", response_model=MessageResponse)
def train_model() -> MessageResponse:
    """
    Trains a Logistic Regression model on the Iris dataset.
    The model is saved to a specified path.

    Returns:
        MessageResponse: A message containing model hyperparameters and training score, 
        or an error message if training fails.
    """
    short_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data"
    model_path = r'C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\models\logistic_regression_model.joblib'

    try:
        X_train = pd.read_csv(short_path + "\X_train.csv")
        y_train = pd.read_csv(short_path + "\y_train.csv")
        with open('config/model_parameters.json', 'r') as file:
            model_params = json.load(file)["LogisticRegression"]

        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)

        return MessageResponse(message=f"Model hyperparameters: {model.get_params()} | Training score: {model.score(X_train, y_train)} | Successfully saved.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training or saving: {e}")


@router.get("/predict_model/{prediction}", name="Make a Prediction with the Model", response_model=MessageResponse)
def predict_model(prediction: str) -> MessageResponse:
    """
    Makes a prediction with the trained Logistic Regression model 
    using input features provided as a comma-separated string.

    Args:
        prediction (str): A comma-separated list of feature values (e.g., "5.1,3.5,1.4,0.2").

    Returns:
        MessageResponse: The predicted class label, 
        or an error message if prediction fails.
    """
    model_path = r'C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\models\logistic_regression_model.joblib'

    try:
        features = list(map(float, prediction.split(',')))
        if len(features) != 4:
            raise HTTPException(status_code=400, detail="The features list must contain exactly 4 values: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input. Please provide a comma-separated list of numeric values: {e}")
    
    try:
        features = np.array(features).reshape(1, -1)
        model = joblib.load(model_path)
        prediction = model.predict(features)
        return MessageResponse(message=f"The model predicted: {prediction[0]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
@router.get("/retrieve_parameters", name="Retrieve parameters from Firestore", response_model=MessageResponse)
def retrieve_parameters() -> MessageResponse:
    """Retrieve parameters from Firestore.

    This endpoint retrieves the parameters stored in the Firestore 'parameters' document.
    If the parameters document does not exist, a 404 error is raised.

    Returns:
        MessageResponse: A response message containing the parameters from Firestore if found,
        or an error message if the parameters document is not found.
    """
    key_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\api-fourdrinoy-7d86e5dd6927.json"

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        parameters_ref = db.collection('parameters').document('parameters')
        parameters = parameters_ref.get()

        if parameters.exists:
            params_dict = parameters.to_dict()
            return MessageResponse(message=json.dumps(params_dict))
        else:
            raise HTTPException(status_code=404, detail="Parameters document not found in Firestore.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parameters: {e}")


@router.post("/add_parameters", name="Add parameters to Firestore", response_model=MessageResponse)
def add_parameters(parameters: dict) -> MessageResponse:
    """Add parameters to Firestore.

    This endpoint adds the provided parameters to the Firestore 'parameters' document.
    If the document already exists, the parameters will be merged.

    Args:
        parameters (dict): A dictionary containing the parameters to be added or updated in Firestore.

    Returns:
        MessageResponse: A response message indicating whether the parameters were successfully added.
    """
    key_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\api-fourdrinoy-7d86e5dd6927.json"

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        parameters_ref = db.collection('parameters').document('parameters')

        parameters_ref.set(parameters, merge=True)
        return MessageResponse(message="Parameters added successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding parameters: {e}")


@router.put("/update_parameters", name="Update parameters in Firestore", response_model=MessageResponse)
def update_parameters(updates: dict) -> MessageResponse:
    """Update parameters in Firestore.

    This endpoint updates the existing parameters stored in the Firestore 'parameters' document
    with the provided updates. If the parameters document does not exist, a 404 error is raised.

    Args:
        updates (dict): A dictionary containing the updates to be applied to the Firestore parameters document.

    Returns:
        MessageResponse: A response message indicating whether the parameters were successfully updated.
    """
    key_path = r"C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\api-fourdrinoy-7d86e5dd6927.json"

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        parameters_ref = db.collection('parameters').document('parameters')

        if not parameters_ref.get().exists:
            raise HTTPException(status_code=404, detail="Parameters document does not exist.")

        parameters_ref.update(updates)
        return MessageResponse(message="Parameters updated successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameters: {e}")
