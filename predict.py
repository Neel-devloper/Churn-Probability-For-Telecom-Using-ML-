import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# Define preprocessor (same as in notebook)
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Define categories for OneHotEncoder
categories = [
    ['Female', 'Male'],  # gender
    [0, 1],  # SeniorCitizen
    ['No', 'Yes'],  # Partner
    ['No', 'Yes'],  # Dependents
    ['No', 'Yes'],  # PhoneService
    ['No', 'No phone service', 'Yes'],  # MultipleLines
    ['DSL', 'Fiber optic', 'No'],  # InternetService
    ['No', 'No internet service', 'Yes'],  # OnlineSecurity
    ['No', 'No internet service', 'Yes'],  # OnlineBackup
    ['No', 'No internet service', 'Yes'],  # DeviceProtection
    ['No', 'No internet service', 'Yes'],  # TechSupport
    ['No', 'No internet service', 'Yes'],  # StreamingTV
    ['No', 'No internet service', 'Yes'],  # StreamingMovies
    ['Month-to-month', 'One year', 'Two year'],  # Contract
    ['No', 'Yes'],  # PaperlessBilling
    ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']  # PaymentMethod
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Fit preprocessor on dummy data to set the scalers
dummy_data = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'tenure': [1, 72],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic'],
    'OnlineSecurity': ['No', 'Yes'],
    'OnlineBackup': ['No', 'Yes'],
    'DeviceProtection': ['No', 'Yes'],
    'TechSupport': ['No', 'Yes'],
    'StreamingTV': ['No', 'Yes'],
    'StreamingMovies': ['No', 'Yes'],
    'Contract': ['Month-to-month', 'One year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check'],
    'MonthlyCharges': [18.25, 118.75],
    'TotalCharges': [18.8, 8684.8]
}

import pandas as pd
dummy_df = pd.DataFrame(dummy_data)
preprocessor.fit(dummy_df)

# model = torch.load('churn_model_entire.pth', weights_only=False)

# model.eval()
# with torch.inference_mode():
#     def predict_churn(gender: int, SeniorCitizen: int, Partner: int, Dependents: int, tenure: int, PhoneService: int,
#                       MultipleLines: int, InternetService: int, OnlineSecurity: int, OnlineBackup: int,
#                       DeviceProtection: int, TechSupport: int, StreamingTV: int, StreamingMovies: int,
#                       Contract: int, PaperlessBilling: int, PaymentMethod: int, MonthlyCharges: float,
#                       TotalCharges: float) -> str:
#         # Preprocess input data
#         input_data = torch.tensor([[
#             gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
#             MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
#             DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
#             Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
#             TotalCharges
#         ]], dtype=torch.float32)
#         # Make prediction
#         prediction = model(input_data)
#         return 'Churn' if torch.round(prediction).item() == 1 else 'No Churn'
    



# predict_churn(1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 29.85, 29.85)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(46, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drp1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drp2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drp3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does not apply sigmoid
        Returns raw logits
        """
        x = self.relu(self.drp1(self.bn1(self.fc1(x))))
        x = self.relu(self.drp2(self.bn2(self.fc2(x))))
        x = self.relu(self.drp3(self.bn3(self.fc3(x))))
        x = self.fc4(x)

        return x


state_dict = torch.load('churn_model_parameters.pth')
model = Model()
model.load_state_dict(state_dict)

model.eval()
with torch.inference_mode():
    def predict(gender: int, SeniorCitizen: int, Partner: int, Dependents: int, tenure: int, PhoneService: int,
                MultipleLines: int, InternetService: int, OnlineSecurity: int, OnlineBackup: int,
                DeviceProtection: int, TechSupport: int, StreamingTV: int, StreamingMovies: int,
                Contract: int, PaperlessBilling: int, PaymentMethod: int, MonthlyCharges: float,
                TotalCharges: float, return_prob: bool = True) -> str:

        # Map integers to categorical strings
        gender_map = {0: 'Female', 1: 'Male'}
        binary_map = {0: 'No', 1: 'Yes'}
        multiple_lines_map = {0: 'No', 1: 'Yes', 2: 'No phone service'}
        internet_service_map = {0: 'DSL', 1: 'Fiber optic', 2: 'No'}
        internet_options_map = {0: 'No', 1: 'Yes', 2: 'No internet service'}
        contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
        payment_method_map = {0: 'Bank transfer (automatic)', 1: 'Credit card (automatic)', 2: 'Electronic check', 3: 'Mailed check'}

        input_dict = {
            'gender': gender_map[gender],
            'SeniorCitizen': SeniorCitizen,  # already int
            'Partner': binary_map[Partner],
            'Dependents': binary_map[Dependents],
            'tenure': tenure,
            'PhoneService': binary_map[PhoneService],
            'MultipleLines': multiple_lines_map[MultipleLines],
            'InternetService': internet_service_map[InternetService],
            'OnlineSecurity': internet_options_map[OnlineSecurity],
            'OnlineBackup': internet_options_map[OnlineBackup],
            'DeviceProtection': internet_options_map[DeviceProtection],
            'TechSupport': internet_options_map[TechSupport],
            'StreamingTV': internet_options_map[StreamingTV],
            'StreamingMovies': internet_options_map[StreamingMovies],
            'Contract': contract_map[Contract],
            'PaperlessBilling': binary_map[PaperlessBilling],
            'PaymentMethod': payment_method_map[PaymentMethod],
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        input_df = pd.DataFrame([input_dict])
        input_processed = preprocessor.transform(input_df)
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)

        prediction = model.forward(input_tensor)

        if return_prob:
            return torch.sigmoid(prediction).item()
        return 'Churn' if torch.round(torch.sigmoid(prediction)).item() == 1 else 'No Churn'


result =predict(1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 29.85, 29.85)
print(result)