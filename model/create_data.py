import os
import numpy as np
import pandas as pd

np.random.seed(42)

n = 500

area = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 5, n)

locations = np.random.choice(
    ["Mumbai", "Pune", "Delhi", "Bangalore"],
    n
)

location_prices = {
    "Mumbai": 5000,
    "Pune": 3000,
    "Delhi": 4000,
    "Bangalore": 3500
}

price = (
    area * 3000
    + bedrooms * 500000
    + bathrooms * 300000
    + np.array([location_prices[x] for x in locations]) * area
    + np.random.normal(0, 500000, n)
)

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "location": locations,
    "price": price
})

os.makedirs("data", exist_ok=True)

df.to_csv("data/house_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Dataset shape:", df.shape)