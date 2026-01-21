import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class MachineLearningRepairKit:
    def __init__(self):
        """
        Initializes the repair kit with advanced ML models.
        """
        # UPGRADE: Using RandomForestRegressor
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=-1),
            max_iter=10,
            random_state=42
        )
        
        self.outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        
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
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        return df_numeric

    def repair(self, df):
        """
        The main pipeline execution.
        """
        # 0. Stats before
        self.report["original_shape"] = df.shape
        self.report["missing_fixed"] = df.isnull().sum().sum()

        # 1. Force Numeric (Preprocessing)
        df_clean = self._force_numeric(df)

        # 2. Advanced Imputation
        imputed_matrix = self.imputer.fit_transform(df_clean)
        df_imputed = pd.DataFrame(imputed_matrix, columns=df_clean.columns)

        # 3. Outlier Detection
        scaled_data = self.scaler.fit_transform(df_imputed)
        
        # fit_predict matches the Estimator API
        outlier_labels = self.outlier_detector.fit_predict(scaled_data)
        
        self.report["outliers_detected"] = np.sum(outlier_labels == -1)

        # 4. Filter the dataset
        df_final = df_imputed[outlier_labels == 1]
        
        # Round to 1 decimal place for cleaner UI
        df_final = df_final.round(1)
        
        self.report["final_shape"] = df_final.shape

        return df_final, self.report