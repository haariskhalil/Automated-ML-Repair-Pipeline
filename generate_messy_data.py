import pandas as pd
import numpy as np
import random

# Configuration
NUM_ROWS = 300
np.random.seed(42)

def generate_messy_data():
    print("Generating synthetic messy data (Allowed Start Age: 18)...")
    
    # 1. Base Data Generation
    # Experience (0 to 40 years)
    experience = np.random.normal(loc=10, scale=8, size=NUM_ROWS)
    experience = np.abs(experience)
    
    # Age = 18 + Experience
    # add random noise (-2 to +5) to account for gap years, career switches, etc.
    # ensure Age is never LESS than (18 + Experience)
    noise = np.random.uniform(low=0, high=10, size=NUM_ROWS) 
    age = 18 + experience + noise
    
    # CATEGORY 1: Department
    depts = np.random.choice(["Support", "Engineering", "Management"], NUM_ROWS, p=[0.4, 0.4, 0.2])
    
    # CATEGORY 2: Gender
    genders = np.random.choice(["Male", "Female", "Non-Binary"], NUM_ROWS, p=[0.48, 0.48, 0.04])
    
    # Salary logic
    salary = []
    for i in range(NUM_ROWS):
        base = 30000
        if depts[i] == "Engineering": base = 60000
        if depts[i] == "Management": base = 90000
        
        sal = base + (experience[i] * 2000) + np.random.normal(0, 5000)
        salary.append(sal)
    
    salary = np.array(salary)

    df = pd.DataFrame({
        "Age": age,
        "Experience": experience,
        "Department": depts, 
        "Gender": genders,
        "Salary": salary
    })

    df = df.round(1)

    # 2. Injecting The "Mess"
    
    # A. Missing Values
    df.loc[df.sample(frac=0.15).index, "Age"] = np.nan
    df.loc[df.sample(frac=0.10).index, "Salary"] = np.nan
    df.loc[df.sample(frac=0.10).index, "Department"] = np.nan
    df.loc[df.sample(frac=0.10).index, "Gender"] = np.nan

    # B.Contextual Outlier
    # create someone who claims to be 20 but has 15 years experience
    df.loc[10, "Age"] = 20
    df.loc[10, "Experience"] = 15 
    
    # C. Extreme Outliers
    df.loc[NUM_ROWS-1, "Salary"] = 500000000
    df.loc[NUM_ROWS-1, "Age"] = 150
    
    print("Data generated successfully.")
    df.to_csv("large_messy_data_v3.csv", index=False)

if __name__ == "__main__":
    generate_messy_data()