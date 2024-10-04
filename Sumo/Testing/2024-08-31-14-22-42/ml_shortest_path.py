import traci
import pytz
import datetime
import pandas as pd
import numpy as np
import sumolib
import xml.etree.ElementTree as ET
from keras.models import load_model

# Load the pre-trained model
model = load_model('ML_Datasets/vehicle_rerouting_model.h5')

weatherDataFile = "WeatherData/weather_data.csv"
weather_df = pd.read_csv(weatherDataFile)

def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    return currentDT.strftime("%Y-%m-%d %H:%M:%S")

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list

def validate_lane_and_edge(lane, net):
    try:
        if lane and net.getLane(lane):
            return net.getLane(lane).getEdge().getID()
        else:
            print(f"Invalid lane: {lane}")
            return None
    except Exception as e:
        print(f"Validation error for lane {lane}: {e}")
        return None

vehicle_stops = {}
try:
    tree = ET.parse('osm.passenger.trips.xml')
    root = tree.getroot()
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id')
        stops = [stop.get('busStop') for stop in vehicle.findall('stop')]
        vehicle_stops[vid] = stops
except FileNotFoundError as e:
    print(f"Error loading trips file: {e}")

bus_stops = {}
try:
    tree = ET.parse('osm.poly.xml')
    root = tree.getroot()
    for bus_stop in root.findall('busStop'):
        bs_id = bus_stop.get('id')
        lane = bus_stop.get('lane')
        bus_stops[bs_id] = lane
except FileNotFoundError as e:
    print(f"Error loading bus stops file: {e}")

net = sumolib.net.readNet('osm.net.xml')

sumoCmd = ["sumo", "-c", "osm.sumocfg", "--no-warnings", "true"]
traci.start(sumoCmd)

packBigData = []
packPersonData = []
personState = {}

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    trafficlights = traci.trafficlight.getIDList()

    for vehid in vehicles:
        try:
            x, y = traci.vehicle.getPosition(vehid)
            coord = [x, y]
            spd = round(traci.vehicle.getSpeed(vehid) * 3.6, 2)
            edge = traci.vehicle.getRoadID(vehid)
            lane = traci.vehicle.getLaneID(vehid)
            displacement = round(traci.vehicle.getDistance(vehid), 2)
            turnAngle = round(traci.vehicle.getAngle(vehid), 2)

            # Assign goals and set path for vehicles
            if vehid in vehicle_stops:
                stops = vehicle_stops[vehid]
                if stops:
                    current_stop = stops.pop(0)
                    next_stop = stops[0] if stops else None
                    if next_stop:
                        current_lane = bus_stops[current_stop]
                        goal_lane = bus_stops[next_stop]

                        current_edge_id = validate_lane_and_edge(current_lane, net)
                        goal_edge_id = validate_lane_and_edge(goal_lane, net)

                        if current_edge_id and goal_edge_id:
                            # Prepare the input for the model
                            features = np.array([[spd, displacement, turnAngle]])  # Customize based on your features
                            predicted_edge_id = model.predict(features)  # Use the model to predict next edge
                            predicted_edge_id = np.argmax(predicted_edge_id)  # Assuming model outputs probabilities
                            predicted_edge = net.getEdge(predicted_edge_id)  # Get the predicted edge

                            if predicted_edge:
                                traci.vehicle.setRoute(vehid, [predicted_edge.getID()])  # Set the new route
                                traci.vehicle.resume(vehid)
                            else:
                                print(f"Invalid edge predicted for vehicle {vehid}: {predicted_edge_id}")

            vehList = [
                getdatetime(), vehid, coord, spd, edge, lane, displacement, turnAngle
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
            print(f"Error processing vehicle {vehid}: {e}")
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
            personState[person_id] = {"status": "on_veh", "vehicle": person_vehicle, "last_edge": person_edge}
            packPersonData.append([getdatetime(), person_id, person_pos, person_edge, person_speed, person_vehicle, "Boarded Bus"])
            print(f"Person {person_id} boarded {person_vehicle} at {getdatetime()} on edge {person_edge}.")

        # Alighting a bus
        elif not person_vehicle and current_status == "on_veh":
            personState[person_id] = {"status": "walking", "vehicle": "", "last_edge": person_edge}
            packPersonData.append([getdatetime(), person_id, person_pos, person_edge, person_speed, last_vehicle, "Alighted Bus"])
            print(f"Person {person_id} alighted {last_vehicle} at {getdatetime()} on edge {person_edge}.")

        # Update status based on speed
        if person_speed == 0:
            if current_status == "walking":
                current_status = "waiting"
            else:
                if current_status == "waiting":
                    current_status = "walking"

        packPersonData.append([getdatetime(), person_id, person_pos, person_edge, person_speed, person_vehicle, current_status])

traci.close()

vehicle_columns = [
    'dateandtime', 'vehid', 'coord', 'spd', 'edge', 'lane',
    'displacement', 'turnAngle', 'tflight', 'tl_state', 'tl_phase_duration',
    'tl_lanes_controlled', 'tl_program', 'tl_next_switch'
]
person_columns = ['dateandtime', 'person_id', 'position', 'current_edge', 'speed', 'vehicle', 'activity']

dataset = pd.DataFrame(packBigData, columns=vehicle_columns)
dataset.to_csv('SumoData/vehicle_data_deep_learning.csv', index=False)

person_dataset = pd.DataFrame(packPersonData, index=None, columns=person_columns)
person_dataset.to_csv('SumoData/person_data_deep_learning.csv', index=False)
