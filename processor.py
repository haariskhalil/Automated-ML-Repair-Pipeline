import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MachineLearningRepairKit:
    def __init__(self):
        """
        Initializes the repair kit with advanced ML models.
        """
        # predicts missing values based on other columns
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        
        # identifies and flags data outliers
        self.outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        
        # To scale data before outlier detection (crucial for distance/variance based models)
        self.scaler = StandardScaler()
        
        # Memory to store what we removed (for the report)
        self.report = {
            "missing_fixed": 0,
            "outliers_detected": 0,
            "original_shape": (0,0),
            "final_shape": (0,0)
        }

    def _force_numeric(self, df):
        """
        Private method: Coerces data to numeric. 
        Strings like '1,000' or 'Error' become NaN so they can be imputed.
        """
        df_numeric = df.copy()
        for col in df_numeric.columns:
            # force_numeric will turn "abc" into NaN
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        return df_numeric

    def repair(self, df):
        """
        The main pipeline execution.
        1. Coerce to Numeric
        2. Impute Missing Values (Iterative)
        3. Detect Outliers (Isolation Forest)
        """
        # 0. Stats before
        self.report["original_shape"] = df.shape
        self.report["missing_fixed"] = df.isnull().sum().sum()

        # 1. Force Numeric (Preprocessing)
        # We only work on the numeric interpretation of the data
        df_clean = self._force_numeric(df)

        # 2. Advanced Imputation (The 'Repair' Step)
        # Learns relationships between columns to fill NaNs
        imputed_matrix = self.imputer.fit_transform(df_clean)
        df_imputed = pd.DataFrame(imputed_matrix, columns=df_clean.columns)

        # 3. Outlier Detection (The 'Filter' Step)
        # We scale first because IsolationForest is sensitive to magnitude
        scaled_data = self.scaler.fit_transform(df_imputed)
        
        # Predict: -1 is outlier, 1 is inlier
        # FIX: Use fit_predict, not fit_transform
        outlier_labels = self.outlier_detector.fit_predict(scaled_data)
        
        # Count outliers
        self.report["outliers_detected"] = np.sum(outlier_labels == -1)

        # 4. Filter the dataset
        # We keep only the rows that are NOT outliers (label == 1)
        df_final = df_imputed[outlier_labels == 1]
        
        self.report["final_shape"] = df_final.shape

        return df_final, self.report