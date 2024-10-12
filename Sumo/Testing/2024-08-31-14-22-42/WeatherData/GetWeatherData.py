import requests
import csv
import time

coords = []
output_csv = "WeatherData/weather_data.csv"
sub_key = "S9ofosom0MBJA2SEL_eeEDFxrT5FkiCJD-9dGKk5eGI"
def extractData(data):
    try:
        result = data["results"][0]
        weather_data = {
            "Latitude": lat,
            "Longitude": lon,
            "dateTime": result.get("dateTime", ""),
            "phrase": result.get("phrase", ""),
            "iconCode": result.get("iconCode", ""),
            "temperature": result.get("temperature", {}).get("value", ""),
            "realFeelTemperature": result.get("realFeelTemperature", {}).get("value", ""),
            "realFeelTemperatureShade": result.get("realFeelTemperatureShade", {}).get("value", ""),
            "relativeHumidity": result.get("relativeHumidity", ""),
            "dewPoint": result.get("dewPoint", {}).get("value", ""),
            "windDirection": result.get("wind", {}).get("direction", {}).get("degrees", ""),
            "windSpeed": result.get("wind", {}).get("speed", {}).get("value", ""),
            "windGust": result.get("windGust", {}).get("speed", {}).get("value", ""),
            "uvIndex": result.get("uvIndex", ""),
            "uvIndexPhrase": result.get("uvIndexPhrase", ""),
            "visibility": result.get("visibility", {}).get("value", ""),
            "cloudCover": result.get("cloudCover", ""),
            "ceiling": result.get("ceiling", {}).get("value", ""),
            "pressure": result.get("pressure", {}).get("value", ""),
            "pressureTendency": result.get("pressureTendency", {}).get("localizedDescription", ""),
            "past24HourTemperatureDeparture": result.get("past24HourTemperatureDeparture", {}).get("value", ""),
            "apparentTemperature": result.get("apparentTemperature", {}).get("value", ""),
            "windChillTemperature": result.get("windChillTemperature", {}).get("value", ""),
            "wetBulbTemperature": result.get("wetBulbTemperature", {}).get("value", ""),
            "pastHourPrecipitation": result.get("precipitationSummary", {}).get("pastHour", {}).get("value", ""),
            "past3HoursPrecipitation": result.get("precipitationSummary", {}).get("past3Hours", {}).get("value", ""),
            "past6HoursPrecipitation": result.get("precipitationSummary", {}).get("past6Hours", {}).get("value", ""),
            "past9HoursPrecipitation": result.get("precipitationSummary", {}).get("past9Hours", {}).get("value", ""),
            "past12HoursPrecipitation": result.get("precipitationSummary", {}).get("past12Hours", {}).get("value", ""),
            "past18HoursPrecipitation": result.get("precipitationSummary", {}).get("past18Hours", {}).get("value", ""),
            "past24HoursPrecipitation": result.get("precipitationSummary", {}).get("past24Hours", {}).get("value", ""),
            "past6HoursTemperatureMin": result.get("temperatureSummary", {}).get("past6Hours", {}).get("minimum", {}).get("value", ""),
            "past6HoursTemperatureMax": result.get("temperatureSummary", {}).get("past6Hours", {}).get("maximum", {}).get("value", ""),
            "past12HoursTemperatureMin": result.get("temperatureSummary", {}).get("past12Hours", {}).get("minimum", {}).get("value", ""),
            "past12HoursTemperatureMax": result.get("temperatureSummary", {}).get("past12Hours", {}).get("maximum", {}).get("value", ""),
            "past24HoursTemperatureMin": result.get("temperatureSummary", {}).get("past24Hours", {}).get("minimum", {}).get("value", ""),
            "past24HoursTemperatureMax": result.get("temperatureSummary", {}).get("past24Hours", {}).get("maximum", {}).get("value", "")
        }
        writer.writerow(weather_data)   
    except KeyError:
        print(f"Error extracting data for {lat}, {lon}: {data}")

with open("WeatherData/coordinates.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lat = row['Latitude']
        lon = row['Longitude']
        coords.append((lat, lon))

# Prepare CSV file for writing
with open(output_csv, "w", newline='') as csvfile:
    fieldnames = [
        "Latitude", "Longitude", "dateTime", "phrase", "iconCode", "temperature",
        "realFeelTemperature", "realFeelTemperatureShade", "relativeHumidity", "dewPoint",
        "windDirection", "windSpeed", "windGust", "uvIndex", "uvIndexPhrase",
        "visibility", "cloudCover", "ceiling", "pressure", "pressureTendency",
        "past24HourTemperatureDeparture", "apparentTemperature",
        "windChillTemperature", "wetBulbTemperature", "pastHourPrecipitation",
        "past3HoursPrecipitation", "past6HoursPrecipitation",
        "past9HoursPrecipitation", "past12HoursPrecipitation",
        "past18HoursPrecipitation", "past24HoursPrecipitation",
        "past6HoursTemperatureMin", "past6HoursTemperatureMax",
        "past12HoursTemperatureMin", "past12HoursTemperatureMax",
        "past24HoursTemperatureMin", "past24HoursTemperatureMax"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for lat, lon in coords:
        url = f"https://atlas.microsoft.com/weather/currentConditions/json?api-version=1.1&query={lat},{lon}&subscription-key={sub_key}"
        print(f"Processing data for {lat}, {lon}...")
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                extractData(data)
                print(f"Data for {lat}, {lon} processed successfully.")
            except ValueError:
                print(f"Error decoding JSON for {lat}, {lon}: {response.text}")
        else:
            print(f"Error fetching data for {lat}, {lon}: HTTP {response.status_code}")
        time.sleep(1)
