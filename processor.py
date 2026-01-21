import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

class MachineLearningRepairKit:
    def __init__(self):
        """
        Initializes the repair kit with advanced ML models.
        """
        # Predicts missing values based on other columns
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=-1),
            max_iter=10,
            random_state=42
        )
        
        # Identifies and flags data outliers
        self.outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        
        # To scale data before outlier detection
        self.scaler = StandardScaler()
        
        # Memory to store what was removed
        self.report = {
            "missing_fixed": 0,
            "outliers_detected": 0,
            "outliers": pd.DataFrame(), # Initialize empty storage
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
        
        # Predict: -1 is outlier, 1 is inlier
        outlier_labels = self.outlier_detector.fit_predict(scaled_data)
        
        self.report["outliers_detected"] = np.sum(outlier_labels == -1)

        # Capture the specific rows flagged as outliers for review
        self.report["outliers"] = df_imputed[outlier_labels == -1]

        # 4. Filter the dataset
        df_final = df_imputed[outlier_labels == 1]
        
        # Rounding
        df_final = df_final.round(1)
        self.report["outliers"] = self.report["outliers"].round(1)
        
        self.report["final_shape"] = df_final.shape

        return df_final, self.report

    def evaluate_model(self, df, target_col):
        """
        Trains two models to predict the 'target_col':
        1. Baseline: Simple Mean Imputation + Linear Regression
        2. Advanced: Advanced Cleaned Data + Random Forest
        """
        # Separate Features (X) and Target (y)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # force the TARGET to be numeric. "error" becomes NaN.
        y = pd.to_numeric(y, errors='coerce')
        
        # Force numeric on X for the baseline to work
        X = self._force_numeric(X)
        
        # For X, we fill missing with MEAN.
        clean_idx = ~y.isna()
        X_base = X[clean_idx]
        y_base = y[clean_idx]
        
        # Check if enough data left after dropping NaNs
        if len(y_base) < 2:
            return {"error": "Not enough valid data in target column to train baseline."}

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
        
        # Simple Pipeline
        baseline_model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        baseline_model.fit(X_train, y_train)
        y_pred_base = baseline_model.predict(X_test)
        
        # Run the internal repair to get a clean dataset first
        df_clean, _ = self.repair(df)
        
        if target_col not in df_clean.columns:
            return {"error": "Target column was removed as an outlier!"}
            
        X_adv = df_clean.drop(columns=[target_col])
        y_adv = df_clean[target_col]
        
        # Split
        X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)
        
        # Advanced Model
        model_adv = RandomForestRegressor(random_state=42)
        model_adv.fit(X_train_adv, y_train_adv)
        y_pred_adv = model_adv.predict(X_test_adv)
        
        # 3. COMPARE METRICS
        return {
            "baseline_mae": mean_absolute_error(y_test, y_pred_base),
            "advanced_mae": mean_absolute_error(y_test_adv, y_pred_adv),
            "baseline_r2": r2_score(y_test, y_pred_base),
            "advanced_r2": r2_score(y_test_adv, y_pred_adv)
        }