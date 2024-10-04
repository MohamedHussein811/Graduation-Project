import csv

# Bounding box coordinates
top_left_lat = 52.511387
top_left_lon = 13.346622
bottom_right_lat = 52.42349
bottom_right_lon = 13.434519

# Number of grid points
num_points = 50

# Calculate increments
lat_increment = (top_left_lat - bottom_right_lat) / (num_points - 1)
lon_increment = (bottom_right_lon - top_left_lon) / (num_points - 1)

# Generate coordinates
coordinates = []

for i in range(num_points):
    for j in range(num_points):
        lat = top_left_lat - i * lat_increment
        lon = top_left_lon + j * lon_increment
        coordinates.append((lat, lon))

# Save coordinates to CSV file
with open('coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Latitude', 'Longitude'])  # Write header
    writer.writerows(coordinates)  # Write data rows

print("Coordinates have been saved to 'coordinates.csv'.")
