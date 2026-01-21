import pandas as pd
import numpy as np
import random

# Configuration
NUM_ROWS = 300
np.random.seed(42)

def generate_messy_data():
    print("Generating synthetic messy data...")
    
    # 1. Base Data Generation (Correlated)
    # Experience (0 to 40 years)
    experience = np.random.normal(loc=10, scale=8, size=NUM_ROWS)
    experience = np.abs(experience)  # No negative years
    
    # Age (correlated with experience + randomness)
    age = 22 + experience + np.random.normal(0, 3, NUM_ROWS)
    
    # Salary (Strongly correlated with experience)
    # Base: 40k, + 3k per year of experience, + random noise
    salary = 40000 + (experience * 3000) + np.random.normal(0, 5000, NUM_ROWS)
    
    # Credit Score (Weakly correlated with Age)
    credit_score = 600 + (age * 2) + np.random.normal(0, 40, NUM_ROWS)
    credit_score = np.clip(credit_score, 300, 850) # Clip to realistic range

    # Create DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Experience": experience,
        "Salary": salary,
        "Credit_Score": credit_score
    })

    # Rounding for realism
    df = df.round(1)

    # 2. Injecting The "Mess"
    
    # A. Inject Missing Values (NaN)
    # Randomly remove 15% of Age and 10% of Salary
    df.loc[df.sample(frac=0.15).index, "Age"] = np.nan
    df.loc[df.sample(frac=0.10).index, "Salary"] = np.nan

    # B. Inject Text Errors
    # Add garbage strings to numeric columns
    error_indices = df.sample(n=5).index
    df.loc[error_indices, "Salary"] = "Pending"
    
    error_indices_2 = df.sample(n=3).index
    df.loc[error_indices_2, "Credit_Score"] = "Error_404"

    # C. Inject Extreme Outliers (The "IsolationForest" Test)
    df.loc[NUM_ROWS-1, "Salary"] = 500000000 
    df.loc[NUM_ROWS-1, "Age"] = 150
    

    df.loc[NUM_ROWS-2, "Age"] = 5
    df.loc[NUM_ROWS-2, "Salary"] = 200000
    

    df.loc[NUM_ROWS-3, "Credit_Score"] = 5000

    print("Data generated successfully.")
    print(f"Shape: {df.shape}")
    print("Saving to 'large_messy_data.csv'...")
    
    df.to_csv("large_messy_data.csv", index=False)
    print("Done! 'large_messy_data.csv' created")

if __name__ == "__main__":
    generate_messy_data()