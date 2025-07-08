import random
import string
import pandas as pd

# Define the base aircraft data
aircraft_data = [
    {"model_name": "F-35 Lightning", "number_engines": 1, "manufacturer": "Lockheed", "carrier_compatible": "Yes", "country": "USA", "dataset_comprisal": 0.30},
    {"model_name": "F-16 Falcon", "number_engines": 1, "manufacturer": "General Dynamics", "carrier_compatible": "No", "country": "USA", "dataset_comprisal": 0.25},
    {"model_name": "F/A-18 Hornet", "number_engines": 2, "manufacturer": "Boeing", "carrier_compatible": "Yes", "country": "USA", "dataset_comprisal": 0.10},
    {"model_name": "B-2 Spirit", "number_engines": 4, "manufacturer": "Northrup Grumman", "carrier_compatible": "No", "country": "USA", "dataset_comprisal": 0.002},
    {"model_name": "B-52 Stratofortress", "number_engines": 8, "manufacturer": "Boeing", "carrier_compatible": "No", "country": "USA", "dataset_comprisal": 0.003},
    {"model_name": "Su-57 Felon", "number_engines": 2, "manufacturer": "Sukhoi", "carrier_compatible": "No", "country": "Russia", "dataset_comprisal": 0.25},
    {"model_name": "MiG-15 Fagot", "number_engines": 1, "manufacturer": "Mikoyan", "carrier_compatible": "No", "country": "Russia", "dataset_comprisal": 0.0955}
]

# Parameters
total_rows = 1000
sighting_countries = ["USA", "Ukraine", "Russia", "China", "Taiwan", "Denmark", "Greenland"]

# Helper function to generate a random 4-letter serial number
def generate_serial():
    return random.randint(1111, 9999)

# Generate dataset
dataset = []
for aircraft in aircraft_data:
    count = int(aircraft["dataset_comprisal"] * total_rows)
    for _ in range(count):
        # row = aircraft.copy()
        row = {k: v for k, v in aircraft.items() if k != "dataset_comprisal"}  # Exclude dataset_comprisal
        row["serial_number"] = generate_serial()
        row["country_sighted"] = random.choice(sighting_countries)
        dataset.append(row)

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Optional: shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv("aircraft_dataset.csv", index=False)

print("Dataset generated with", len(df), "rows and saved as 'aircraft_dataset.csv'")
