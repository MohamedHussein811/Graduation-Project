import traci
import pytz
import datetime
import pandas as pd
import sumolib
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.models import load_model

MODEL_NAME = 'gru'
MODEL_FILE = f'Models/exit_edge_{MODEL_NAME}.h5'
SIMULATION_OUTPUT_FILE = f'ML_Datasets/Output/simulation_output_{MODEL_NAME}.csv'
weatherDataFile = "WeatherData/weather_data.csv"
weather_df = pd.read_csv(weatherDataFile)

model = load_model(MODEL_FILE)

def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if isinstance(element, list):
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def get_weather_penalty(node_coord, weather_df):
    lat, lon = node_coord[1], node_coord[0]
    weather_df['distance'] = ((weather_df['Latitude'] - lat)**2 + (weather_df['Longitude'] - lon)**2)**0.5
    nearest_weather = weather_df.loc[weather_df['distance'].idxmin()]
    temperature = nearest_weather['Temperature']
    if temperature > 30: 
        return (temperature - 30) * 0.1 
    elif temperature < 0:
        return abs(temperature) * 0.1
    else:
        return 0

def get_traffic_penalty(edge_id):
    try:
        traffic_density = traci.edge.getLastStepVehicleNumber(edge_id)
        if traffic_density > 100: 
            return (traffic_density - 100) * 0.1 
        else:
            return 0
    except Exception as e:
        print(f"Error getting traffic data for edge {edge_id}: {e}")
        return 0

def predict_path_features(current_edge, goal_edge, weather_df):
    features = {
        "current_edge": current_edge.getID(),
        "goal_edge": goal_edge.getID(),
        "weather": get_weather_penalty(current_edge.getCoord(), weather_df),
        "traffic": get_traffic_penalty(current_edge.getID()),
    }
    
    feature_vector = np.array([
        features['current_edge'],
        features['goal_edge'],
        features['weather'],
        features['traffic'],
    ]).reshape(1, -1)

    # Predict using the model
    predicted_path = model.predict(feature_vector)
    return predicted_path

def get_exit_direction(angle):
    if 45 <= angle < 135:
        return "Right"
    elif 135 <= angle < 225:
        return "U-turn"
    elif 225 <= angle < 315:
        return "Left"
    else:
        return "Straight"

vehicle_stops = {}
try:
    tree = ET.parse('Sumo_Config/osm.passenger.trips.xml')
    root = tree.getroot()
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        stops = [stop.get('busStop') for stop in vehicle.findall('stop')]
        vehicle_stops[vid] = stops
except FileNotFoundError as e:
    print(f"Error loading trips file: {e}")

bus_stops = {}
try:
    tree = ET.parse('Sumo_Config/osm.poly.xml')
    root = tree.getroot()
    for bus_stop in root.findall('busStop'):
        bs_id = bus_stop.get('id')
        lane = bus_stop.get('lane')
        bus_stops[bs_id] = lane
except FileNotFoundError as e:
    print(f"Error loading bus stops file: {e}")

net = sumolib.net.readNet('Sumo_Config/osm.net.xml')

sumoCmd = ["sumo", "-c", "Sumo_Config/osm.sumocfg", "--routing-algorithm", "astar", "--no-warnings", "true"]
traci.start(sumoCmd)

packBigData = []
junction_edges = set()
edge_junction_map = {}
packPersonData = []
personState = {}
bus_junction_track = {}

for junction_id in traci.junction.getIDList():
    for incoming_edge in traci.junction.getIncomingEdges(junction_id):
        junction_edges.add(incoming_edge)
        edge_junction_map[incoming_edge] = junction_id

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    trafficlights = traci.trafficlight.getIDList()

    for vehid in vehicles:
        try:
            x, y = traci.vehicle.getPosition(vehid)
            coord = [x, y]
            lon, lat = traci.simulation.convertGeo(x, y)
            gpscoord = [lon, lat]
            spd = round(traci.vehicle.getSpeed(vehid) * 3.6, 2)
            edge = traci.vehicle.getRoadID(vehid)
            lane = traci.vehicle.getLaneID(vehid)
            displacement = round(traci.vehicle.getDistance(vehid), 2)
            turnAngle = round(traci.vehicle.getAngle(vehid), 2)
            nextTLS = traci.vehicle.getNextTLS(vehid)
            
            # Assign goals and set path for vehicles
            if vehid in vehicle_stops:
                stops = vehicle_stops[vehid]
                if stops:
                    current_stop = stops.pop(0)
                    next_stop = stops[0] if stops else None
                    if next_stop:
                        current_lane = bus_stops[current_stop]
                        goal_lane = bus_stops[next_stop]
                        
                        current_edge_id = current_lane
                        goal_edge_id = goal_lane
                        
                        if current_edge_id and goal_edge_id:
                            current_edge = net.getEdge(current_edge_id)
                            goal_edge = net.getEdge(goal_edge_id)

                            # Use the model to predict the path
                            predicted_path = predict_path_features(current_edge, goal_edge, weather_df)

                            # Check if the predicted path is valid
                            valid_path = all(net.getEdge(edge_id) for edge_id in predicted_path)

                            if valid_path:
                                traci.vehicle.setRoute(vehid, predicted_path)
                                traci.vehicle.resume(vehid)
                            else:
                                print(f"Invalid path for vehicle {vehid}: {predicted_path}")
                        else:
                            print(f"Invalid lanes for vehicle {vehid}: {current_lane}, {goal_lane}")
                        
            junction_id = ""
            exit_edge = ""
            exit_angle = ""
            exit_direction = ""

            if edge in junction_edges:
                junction_id = edge_junction_map[edge]
                if vehid in bus_junction_track:
                    last_junction, last_edge = bus_junction_track[vehid]
                    if last_junction == junction_id and last_edge != edge:
                        exit_edge = edge
                        exit_angle = turnAngle
                        exit_direction = get_exit_direction(exit_angle)
                        print(f"Vehicle {vehid} exited junction {junction_id} to edge {exit_edge} with angle {exit_angle} degrees, direction: {exit_direction}")
                bus_junction_track[vehid] = (junction_id, edge)
            
            vehList = [
                getdatetime(), vehid, coord, gpscoord, spd, edge, lane,
                displacement, turnAngle, nextTLS, junction_id, exit_edge, exit_angle, exit_direction
            ]

            tlsList = []
            for trafficlight_id in trafficlights:
                if lane in traci.trafficlight.getControlledLanes(trafficlight_id):
                    tflight = trafficlight_id
                    tl_state = traci.trafficlight.getRedYellowGreenState(trafficlight_id)
                    tl_phase_duration = traci.trafficlight.getPhaseDuration(trafficlight_id)
                    tl_lanes_controlled = traci.trafficlight.getControlledLanes(trafficlight_id)
                    tl_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight_id)
                    tl_next_switch = traci.trafficlight.getNextSwitch(trafficlight_id)

                    tlsList = [tflight, tl_state, tl_phase_duration, tl_lanes_controlled, tl_program, tl_next_switch]

            bigList = flatten_list([vehList, tlsList])
            packBigData.append(bigList)
        except Exception as e:
            continue

    persons = traci.person.getIDList()
    for person_id in persons:
        person_pos = traci.person.getPosition(person_id)
        person_edge = traci.person.getRoadID(person_id)
        person_speed = traci.person.getSpeed(person_id)
        person_vehicle = traci.person.getVehicle(person_id)

        # Check if the person is boarding a bus
        if person_id not in personState:
            personState[person_id] = {"status": "walking", "vehicle": "", "last_edge": person_edge}

        current_status = personState[person_id]["status"]
        last_vehicle = personState[person_id]["vehicle"]

        # Boarding a bus
        if person_vehicle and current_status == "walking":
            personState[person_id]["status"] = "boarding"
            personState[person_id]["vehicle"] = person_vehicle
            print(f"Person {person_id} boarding vehicle {person_vehicle}")

        # Exiting the bus
        if last_vehicle and person_vehicle is None:
            personState[person_id]["status"] = "walking"
            personState[person_id]["vehicle"] = ""
            print(f"Person {person_id} exited vehicle {last_vehicle}")

# Save the collected data
data_df = pd.DataFrame(packBigData, columns=['dateandtime', 'vehid', 'coord', 'gpscoord', 'spd', 'edge', 'lane',
    'displacement', 'turnAngle', 'nextTLS', 'junction_id', 'exit_edge', 'exit_angle', 'exit_direction',
    'tflight', 'tl_state', 'tl_phase_duration', 'tl_lanes_controlled',
    'tl_program', 'tl_next_switch'])


data_df.to_csv(SIMULATION_OUTPUT_FILE, index=False)

traci.close()
