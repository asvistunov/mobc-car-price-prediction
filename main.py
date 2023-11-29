import pickle
import re
import pandas as pd

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# Load pipeline parameters from a pickle file
with open('pipeline_params.pkl', 'rb') as f:
    pipeline_params = pickle.load(f)
    pipeline, train_medians = pipeline_params['modeling_pipeline'], pd.Series(pipeline_params['train_median_weights'])

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]



def process_data(df):
    """
    Process the input DataFrame by cleaning and preparing it for modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing vehicle data.

    Returns
    -------
    None
        The function modifies the DataFrame in-place.
    """

    df.drop(['torque', 'name'], axis=1, inplace=True)

    # Extracting numerical part from strings for specific columns
    df[['mileage', 'engine', 'max_power']] = (
        df[['mileage', 'engine', 'max_power']]
        .map(str)
        .map(lambda el: match.group(0) if (match := re.match('[0-9.]+', el)) else None)
        .astype(float)
    )

    # Fill missing values with training medians
    df[train_medians.index] = df[train_medians.index].fillna(train_medians)

    # Typecasting specific columns to integer
    int_types = {'seats': int, 'engine': int}
    df = df.astype(int_types)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Predict the selling price of a single vehicle item.

    Parameters
    ----------
    item : Item
        An instance of Item containing vehicle data.

    Returns
    -------
    float
        The predicted selling price.
    """
    sample = pd.DataFrame.from_dict([item.model_dump()])
    process_data(sample)
    return pipeline.predict(sample)[0]

@app.post("/predict_items_from_file", response_class=StreamingResponse)
def predict_items_from_file(file: UploadFile = File()):
    """
    Predict selling prices for multiple vehicle items from a CSV file.

    Parameters
    ----------
    file : UploadFile
        A CSV file containing multiple vehicle data entries.

    Returns
    -------
    StreamingResponse
        A CSV file containing the original data along with the predicted selling prices.
    """
    X = pd.read_csv(file.file)
    X_copy = X.copy()
    process_data(X)
    y = pd.Series(pipeline.predict(X), name='selling_price')
    X_y = pd.concat([X_copy, y], axis=1)
    response = StreamingResponse(iter([X_y.to_csv(index=False)]), media_type='text/csv')
    return response

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    """
    Predict the selling prices for a list of vehicle items.

    Parameters
    ----------
    items : List[Item]
        A list of Item instances containing vehicle data.

    Returns
    -------
    List[float]
        A list of predicted selling prices.
    """
    X = pd.DataFrame.from_records([item.model_dump() for item in items])
    process_data(X)

    return pd.Series(pipeline.predict(X), name='selling_price').tolist()
