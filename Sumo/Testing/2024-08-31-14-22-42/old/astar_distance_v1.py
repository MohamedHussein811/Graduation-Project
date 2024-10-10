import traci
import pytz
import datetime
import pandas as pd
from heapq import heappop, heappush
import sumolib
import xml.etree.ElementTree as ET

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

def heuristic(node, goal):
    try:
        return traci.simulation.getDistance2D(node.getCoord()[0], node.getCoord()[1], goal.getCoord()[0], goal.getCoord()[1])
    except Exception as e:
        print(f"Error calculating heuristic: {e}")
        return float('inf')

def astar_search(start, goal, net):
    frontier = []
    heappush(frontier, (0, start.getID()))
    came_from = {}
    cost_so_far = {}
    came_from[start.getID()] = None
    cost_so_far[start.getID()] = 0

    while frontier:
        current = net.getEdge(heappop(frontier)[1])

        if current == goal:
            break

        for next_edge in current.getOutgoing():
            next_node = next_edge.getToNode()
            new_cost = cost_so_far[current.getID()] + current.getLength()
            if next_node.getID() not in cost_so_far or new_cost < cost_so_far[next_node.getID()]:
                cost_so_far[next_node.getID()] = new_cost
                priority = new_cost + heuristic(next_node, goal.getToNode())
                heappush(frontier, (priority, next_edge.getID()))
                came_from[next_edge.getID()] = current.getID()

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            break
    path.append(start)
    path.reverse()
    return path

def get_exit_direction(angle):
    if 45 <= angle < 135:
        return "Right"
    elif 135 <= angle < 225:
        return "U-turn"
    elif 225 <= angle < 315:
        return "Left"
    else:
        return "Straight"

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

sumoCmd = ["sumo", "-c", "Sumo_Config/osm.sumocfg", "--no-warnings", "true"]
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
                        
                        # Validate and get edge IDs
                        current_edge_id = validate_lane_and_edge(current_lane, net)
                        goal_edge_id = validate_lane_and_edge(goal_lane, net)
                        
                        if current_edge_id and goal_edge_id:
                            current_edge = net.getEdge(current_edge_id)
                            goal_edge = net.getEdge(goal_edge_id)
                            came_from, cost_so_far = astar_search(current_edge, goal_edge, net)
                            path = reconstruct_path(came_from, current_edge.getID(), goal_edge.getID())

                            # Check entire path is valid before setting route
                            valid_path = all(net.getEdge(edge_id) for edge_id in path)

                            if valid_path:
                                traci.vehicle.setRoute(vehid, path)
                                traci.vehicle.resume(vehid)
                            else:
                                print(f"Invalid path for vehicle {vehid}: {path}")
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
            personState[person_id] = {"status": "on_veh", "vehicle": person_vehicle, "last_edge": person_edge}
            packPersonData.append([
                getdatetime(), person_id, person_pos, person_edge, person_speed, person_vehicle, "Boarded Bus"
            ])
            print(f"Person {person_id} boarded {person_vehicle} at {getdatetime()} on edge {person_edge}.")

        # Alighting a bus
        elif not person_vehicle and current_status == "on_veh":
            personState[person_id] = {"status": "walking", "vehicle": "", "last_edge": person_edge}
            packPersonData.append([
                getdatetime(), person_id, person_pos, person_edge, person_speed, last_vehicle, "Alighted Bus"
            ])
            print(f"Person {person_id} alighted {last_vehicle} at {getdatetime()} on edge {person_edge}.")

        # Update status based on speed
        if person_speed == 0:
            if current_status == "walking":
                current_status = "waiting"
            else:
                if current_status == "waiting":
                    current_status = "walking"

        packPersonData.append([
            getdatetime(), person_id, person_pos, person_edge, person_speed, person_vehicle, current_status
        ])

traci.close()

vehicle_columns = [
    'dateandtime', 'vehid', 'coord', 'gpscoord', 'spd', 'edge', 'lane',
    'displacement', 'turnAngle', 'nextTLS', 'junction_id', 'exit_edge', 'exit_angle', 'exit_direction',
    'tflight', 'tl_state', 'tl_phase_duration', 'tl_lanes_controlled',
    'tl_program', 'tl_next_switch'
]
person_columns = ['dateandtime', 'person_id', 'position', 'current_edge', 'speed', 'vehicle', 'activity']

dataset = pd.DataFrame(packBigData, columns=vehicle_columns)
dataset.to_csv('SumoData/vehicle_data_astar_distance.csv', index=False)

person_dataset = pd.DataFrame(packPersonData, index=None, columns=person_columns)
person_dataset.to_csv('SumoData/person_data_astar_distance.csv', index=False)