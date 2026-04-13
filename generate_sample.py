import pandas as pd
import numpy as np

data = {
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Temperature': np.random.normal(25, 5, 100),
    'Humidity': np.random.normal(60, 10, 100),
    'Rainfall': np.random.exponential(2, 100),
    'Pressure': np.random.normal(1013, 5, 100),
    'WindSpeed': np.random.normal(15, 5, 100),
    'WeatherType': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 100)
}
df = pd.DataFrame(data)
df.to_csv('c:\\Users\\LENOVO\\Documents\\DAVL_exam\\weather_sample.csv', index=False)
print("Created weather_sample.csv")
