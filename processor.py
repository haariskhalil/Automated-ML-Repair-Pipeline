import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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
        
        # Memory to store encoders for each column so we can decode later
        self.encoders = {}
        
        # Memory to store what was removed
        self.report = {
            "missing_fixed": 0,
            "outliers_detected": 0,
            "outliers": pd.DataFrame(),
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
            # Only force numeric if the column is NOT categorical (object type)
            if df_numeric[col].dtype != 'object':
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        return df_numeric

    def _encode_categories(self, df):
        """
        Converts text columns (e.g., 'Sales', 'HR') into numbers (0, 1).
        """
        df_encoded = df.copy()
        object_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            
            # Fit on valid data
            non_nulls = df_encoded[col].dropna().values.reshape(-1, 1)
            
            if len(non_nulls) > 0:
                encoder.fit(non_nulls)
                self.encoders[col] = encoder
                
                # Transform
                existing_values = df_encoded[col].values.reshape(-1, 1)
                # Use .ravel() to flatten 2D array to 1D
                df_encoded[col] = encoder.transform(existing_values).ravel()
                
        return df_encoded

    def _decode_categories(self, df):
        """
        Converts numbers (0, 1) back into text (e.g., 'Sales', 'HR').
        """
        df_decoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df_decoded.columns:
                # 1. Clip & Round
                max_category_id = len(encoder.categories_[0]) - 1
                df_decoded[col] = df_decoded[col].clip(0, max_category_id).round()
                
                # 2. Inverse Transform
                # Use .ravel() to flatten 2D array to 1D
                df_decoded[col] = encoder.inverse_transform(df_decoded[[col]]).ravel()
                
        return df_decoded

    def repair(self, df):
        """
        The main pipeline execution.
        """
        # 0. Stats before
        self.report["original_shape"] = df.shape
        self.report["missing_fixed"] = df.isnull().sum().sum()

        # 1. Force Numeric (Only for non-categorical columns)
        df_clean = self._force_numeric(df)
        
        # 2. Encode Text -> Numbers
        df_encoded = self._encode_categories(df_clean)

        # 3. Advanced Imputation
        imputed_matrix = self.imputer.fit_transform(df_encoded)
        df_imputed = pd.DataFrame(imputed_matrix, columns=df_encoded.columns)

        # 4. Outlier Detection
        scaled_data = self.scaler.fit_transform(df_imputed)
        outlier_labels = self.outlier_detector.fit_predict(scaled_data)
        
        # Extract the Anomaly Score (Lower = More Abnormal)
        anomaly_scores = self.outlier_detector.decision_function(scaled_data)
        
        self.report["outliers_detected"] = np.sum(outlier_labels == -1)

        # Capture Outliers with Reasoning
        # 1. Get the outlier rows (Encoded version)
        outliers_encoded = df_imputed[outlier_labels == -1].copy()
        
        # 2. Add the Score
        outliers_encoded['Anomaly_Score'] = anomaly_scores[outlier_labels == -1]
        
        # 3. Decode them back to text for readability
        outliers_decoded = self._decode_categories(outliers_encoded)
        
        # 4. Simple Heuristic: Which column contributed most?
        # We look at the Z-Score (standard deviation) of the scaled data for these rows
        # If a column has Z > 3 or Z < -3, it's likely the reason.
        outlier_indices = np.where(outlier_labels == -1)[0]
        outlier_z_scores = scaled_data[outlier_indices]
        
        reasons = []
        feature_names = df_imputed.columns
        
        for row_z in outlier_z_scores:
            # Find column with max absolute Z-score
            abs_z = np.abs(row_z)
            max_z_idx = np.argmax(abs_z)
            max_z_val = row_z[max_z_idx]
            feature = feature_names[max_z_idx]
            
            # 1. CATEGORICAL CHECK: Is it a rare category?
            if feature in self.encoders:
                reasons.append(f"Rare value in '{feature}'")
            
            # 2. NUMERIC CHECK: Is it a single extreme value?
            elif max_z_val > 3:
                reasons.append(f"{feature} is extremely high")
            elif max_z_val < -3:
                reasons.append(f"{feature} is extremely low")
            elif max_z_val > 2:
                reasons.append(f"{feature} is high")
            elif max_z_val < -2:
                reasons.append(f"{feature} is low")
                
            # 3. CONTEXTUAL CHECK: It's a mismatch combination
            else:
                # Get indices of the top 2 deviations
                # sort the indices by absolute Z-score (descending)
                top_2_indices = np.argsort(abs_z)[-2:][::-1]
                f1 = feature_names[top_2_indices[0]]
                f2 = feature_names[top_2_indices[1]]
                
                reasons.append(f"Unusual combination of {f1} and {f2}")
        
        outliers_decoded['Likely_Reason'] = reasons
        
        # Move 'Likely_Reason' and 'Score' to the front for visibility
        cols = ['Likely_Reason', 'Anomaly_Score'] + [c for c in outliers_decoded.columns if c not in ['Likely_Reason', 'Anomaly_Score']]
        self.report["outliers"] = outliers_decoded[cols]

        # 5. Filter the dataset (Keep Inliers)
        df_final_encoded = df_imputed[outlier_labels == 1]
        
        # 6. Decode Clean Data
        df_final = self._decode_categories(df_final_encoded)
        
        # Rounding
        numeric_cols = df_final.select_dtypes(include=['number']).columns
        df_final[numeric_cols] = df_final[numeric_cols].round(1)
        self.report["outliers"][numeric_cols] = self.report["outliers"][numeric_cols].round(3) # Keep more precision for outliers
        
        self.report["final_shape"] = df_final.shape

        return df_final, self.report

    def evaluate_model(self, df, target_col):
        """
        Trains two models to predict the 'target_col'.
        Handles categorical encoding automatically.
        """
        # Separate Features (X) and Target (y)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Force numeric on Target (y) ONLY if it's meant to be numeric (like Salary)
        # If the user tries to predict "Department", we need a Classifier, not Regressor.
        # let's assume y is numeric.
        y = pd.to_numeric(y, errors='coerce')
        
        # 1. Clean X for Baseline
        # need to encode X so Linear Regression can handle text
        X_encoded = self._force_numeric(X)
        X_encoded = self._encode_categories(X_encoded)
        
        # For X, we fill missing with MEAN (works because we encoded text to numbers)
        clean_idx = ~y.isna()
        X_base = X_encoded[clean_idx]
        y_base = y[clean_idx]
        
        if len(y_base) < 2:
            return {"error": "Not enough valid data in target column to train baseline."}

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
        
        # Simple Pipeline
        baseline_model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        baseline_model.fit(X_train, y_train)
        y_pred_base = baseline_model.predict(X_test)
        
        # 2. Advanced Pipeline
        # run the internal repair to get a clean dataset first
        df_clean, _ = self.repair(df)
        
        if target_col not in df_clean.columns:
            return {"error": "Target column was removed as an outlier!"}
            
        X_adv = df_clean.drop(columns=[target_col])
        y_adv = df_clean[target_col]
        
        # need to Ensure X_adv is encoded for the Evaluation Model (RandomForest) to work
        # (The repair function returns Decoded text)
        # must Re-Encode X_adv
        X_adv_encoded = self._encode_categories(X_adv)
        
        # Split
        X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_adv_encoded, y_adv, test_size=0.2, random_state=42)
        
        # Advanced Model
        model_adv = RandomForestRegressor(random_state=42)
        model_adv.fit(X_train_adv, y_train_adv)
        y_pred_adv = model_adv.predict(X_test_adv)
        
        return {
            "baseline_mae": mean_absolute_error(y_test, y_pred_base),
            "advanced_mae": mean_absolute_error(y_test_adv, y_pred_adv),
            "baseline_r2": r2_score(y_test, y_pred_base),
            "advanced_r2": r2_score(y_test_adv, y_pred_adv)
        }