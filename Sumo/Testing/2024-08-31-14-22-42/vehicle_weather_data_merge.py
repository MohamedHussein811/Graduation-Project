import pandas as pd
from scipy.spatial import KDTree
import time

vehicle_data = pd.read_csv('SumoData/vehicle_data_astar_distance_weather_traffic.csv')
weather_data = pd.read_csv('WeatherData/weather_data.csv')

vehicle_data['gpscoord'] = vehicle_data['gpscoord'].apply(lambda x: eval(x))

weather_coords = weather_data[['Latitude', 'Longitude']].to_numpy()

weather_tree = KDTree(weather_coords)

start_time = time.time()

def find_nearest_weather_kdtree(veh_coords, weather_data, weather_tree):
    vehicle_gps = (veh_coords[1], veh_coords[0])
    
    distance, idx = weather_tree.query([vehicle_gps[0], vehicle_gps[1]])
    
    nearest_weather = weather_data.iloc[idx]
    
    return nearest_weather['temperature']

total_vehicles = len(vehicle_data)

def process_vehicle_data(vehicle_data, total_vehicles, weather_data, weather_tree):
    for i, row in vehicle_data.iterrows():
        vehicle_data.at[i, 'temperature'] = find_nearest_weather_kdtree(row['gpscoord'], weather_data, weather_tree)
        
        if (i + 1) % 10000 == 0 or i == total_vehicles - 1:
            print(f"Processed {i + 1}/{total_vehicles} vehicles. Vehicles left: {total_vehicles - (i + 1)}")

print("Starting to merge vehicle data with weather data...")
process_vehicle_data(vehicle_data, total_vehicles, weather_data, weather_tree)

vehicle_data.to_csv('ML_Datasets/vehicle_data_with_weather.csv', index=False)

print(f"Merge completed. Merged data saved to 'ML_Datasets/vehicle_data_with_weather.csv'.")
print(f"Total time taken: {time.time() - start_time:.2f} seconds")
