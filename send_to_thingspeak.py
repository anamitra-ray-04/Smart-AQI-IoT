import requests
import time
import re
import csv
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access them
API_KEY = os.getenv("WRITE_API_KEY")

URL     = f"https://api.thingspeak.com/update?api_key={API_KEY}&field1=0"

with open("serial_output.txt", "r") as f:
    content = f.read()

# Your exact labels to look for
LABELS = ["HAZARDOUS", "V.Unhealthy", "Unhealthy", "Moderate", "Good"]

# Build one big string with no extra spaces
content = re.sub(r'\s+', ' ', content).strip()

# Find every reading using regex
# Pattern: number,number,number,LABEL
pattern = r'(-?[\d.]+),(-?[\d.]+),([\d.]+),(HAZARDOUS|V\.Unhealthy|Unhealthy|Moderate|Good)'
matches = re.findall(pattern, content)

print(f"Found {len(matches)} readings\n")

# Filter out bad/unrealistic values
valid = []
for temp_s, hum_s, ppm_s, label in matches:
    temp = float(temp_s)
    hum  = float(hum_s)
    ppm  = float(ppm_s)

    if temp < 0 or temp > 60:
        continue
    if hum < 0 or hum > 100:
        continue
    if ppm > 9999:
        continue

    valid.append((temp, hum, ppm, label))

# Remove consecutive duplicate readings
# (same values repeated = Wokwi printing same reading multiple times)
deduped = []
prev = None
for row in valid:
    if row != prev:
        deduped.append(row)
        prev = row

print(f"After filtering and deduplication: {len(deduped)} readings\n")

# Save as CSV for ML training
with open("aqi_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["temperature", "humidity", "gas_ppm", "aqi_label"])
    for t, h, p, l in deduped:
        writer.writerow([t, h, p, l])
print(f"Saved aqi_dataset.csv with {len(deduped)} rows\n")

