import xml.etree.ElementTree as ET
import random

RoutesList = {
    "busRoute1": {"stops": ["bs_1", "bs_2"], "edges": "-16374393#0 -1199889721#0 -505526660#0 -356359316#3"},
    "busRoute2": {"stops": ["bs_2"], "edges": "37289167#4 37289167#6 37289167#7 36259516#0 36259516#1 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 -505526660#0 -356359316#3"},
    "busRoute3": {"stops": ["bs_2"], "edges": "-1132166349#0 -65040958#2 -16374393#0 -1199889721#0 -505526660#0 -356359316#3 -356359316#2"},
    "busRoute4": {"stops": ["bs_2"], "edges": "1147754937#2 563341816#1 563341816#2 1147755019#2 1147755019#3 4396526#1 4396526#2 4396526#3 4396526#4 328064927#0 466891838#1 466891838#3 1147755017#0 1147755017#1 1147755014#2 1147755014#3 1147755013#1 328064919#1 328064919#2 328064919#3 1147755012 24214089#1 24214089#2 24214089#3 24214089#4 24214089#5 24214089#6 24214089#7 24214089#8 11845212 328064918#0 328064918#2 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 707100745#1 1199889721#1 -16374393#0 -1199889721#0 -505526660#0 -356359316#3 -356359316#2 -356359316#1"},
    "busRoute5": {"stops": ["bs_3", "bs_4", "bs_0", "bs_1", "bs_2"], "edges": "328064927#0 1057138146#1 1147413257#0 1147413257#1 1147413256#1 1147413255#1 1147413255#2 1147413249#0 1147413246 -1147413269#1 -1147413268 -39839435#1 -1147413271#1 -28402260#3 -28402260#2 1152723799#1 4611774#1 4611774#2 4611774#3 4611774#4 1152723801  -1152876521#0 -1152876518#0 -67099797#1 -435164424#0 199847244#1 199847244#2 199847244#3 81113584#0 -327462543#1 -1132164863 -16374393#0 -1199889721#0 -505526660#0 -356359316#3"},
    "busRoute6": {"stops": ["bs_5","bs_6","bs_7","bs_8", "bs_2"], "edges": "37289167#3 37289167#4 37289167#6 37289167#7 36259516#0 36259516#1 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 -505526660#0 -356359316#3"},
    "busRoute7": {"stops": ["bs_9" ,"bs_0", "bs_1", "bs_2"], "edges": "1132165078#1  1132165078#2 1132164893#1 -16374393#0 -1199889721#0 -505526660#0 -356359316#3"},
    "busRoute8": {"stops": ["bs_10", "bs_9" ,"bs_0", "bs_1", "bs_2"], "edges": "665241888#1 1136657266#1 1136657266#2 4588224#0 4588224#1 1132165078#1  1132165078#2 1132164893#1 -16374393#0 -1199889721#0 -505526660#0 -356359316#3"},
    "busRoute9": {"stops":["bs_11","bs_12", "bs_7","bs_8", "bs_2"], "edges":"-23453218  -67557767 -1128232941#1 1174253786#3 1100552769#2 25121299 -1174265474#0 -110450332#1 -110450332#0 1209494754 1209494764#2 -1152906480 -1152906481 -28497262#2 28497260#0 -1152906482#1 -32936807#5 -32936807#4 -32936807#3 -32936807#2 -32936807#0 -1152906483#2 -104652096#3 -80366517#11 -80366517#8 -80366517#7 -80366517#6 -80366517#5 -80366517#4 -80366517#2 -490361982#3 -490361981#2 -490361981#0 -25121300#2 -25121300#0 1152890201 1152890200 25121294#1 25121294#2 37294341#0 4395698 16943611#0 16943612 763211372#3 1125773127#0 1125773127#1 1125773128#0 1125773128#1 25041996#2 1124624271 404598741#0 664586644#0 37289167#3 37289167#4 37289167#6 37289167#7 36259516#0 36259516#1 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 -505526660#0 -356359316#3"},
    "busRoute10": {"stops":["bs_7","bs_8", "bs_2"], "edges":"1209494754 1209494764#2 -1152906480 -1152906481 -28497262#2 28497260#0 -1152906482#1 -32936807#5 -32936807#4 -32936807#3 -32936807#2 -32936807#0 -1152906483#2 -104652096#3 -80366517#11 -80366517#8 -80366517#7 -80366517#6 -80366517#5 -80366517#4 -80366517#2 -490361982#3 -490361981#2 -490361981#0 -25121300#2 -25121300#0 1152890201 1152890200 25121294#1 25121294#2 37294341#0 4395698 16943611#0 16943612 763211372#3 1125773127#0 1125773127#1 1125773128#0 1125773128#1 25041996#2 1124624271 404598741#0 664586644#0 37289167#3 37289167#4 37289167#6 37289167#7 36259516#0 36259516#1 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 -505526660#0 -356359316#3"},
    "busRoute11": {"stops":["bs_12", "bs_7","bs_8", "bs_2"], "edges":"-110450332#0 1209494754 1209494764#2 -1152906480 -1152906481 -28497262#2 28497260#0 -1152906482#1 -32936807#5 -32936807#4 -32936807#3 -32936807#2 -32936807#0 -1152906483#2 -104652096#3 -80366517#11 -80366517#8 -80366517#7 -80366517#6 -80366517#5 -80366517#4 -80366517#2 -490361982#3 -490361981#2 -490361981#0 -25121300#2 -25121300#0 1152890201 1152890200 25121294#1 25121294#2 37294341#0 4395698 16943611#0 16943612 763211372#3 1125773127#0 1125773127#1 1125773128#0 1125773128#1 25041996#2 1124624271 404598741#0 664586644#0 37289167#3 37289167#4 37289167#6 37289167#7 36259516#0 36259516#1 159081772#1 159081772#2 4396109 453379904 356359316#1 356359316#2 356359316#3 1199404285 -505526660#0 -356359316#3"},
}

VehRoutes = {
    "vehRoute1": {"stops": [], "edges": "4531624#5 4531624#6 4531624#10 -1128209803#3 -1128209803#2 -1128209803#0 -1214394197#0 -4531625#1 -4531625#0 152953803 1174257961#1 4618893#1 1174253788#1 1174253779#1 67707649#1 508614986 1144048223#1 1144047048#0 1144047048#3 593851959#0 593851959#1 593851959#2 593851959#4 593851959#5 593851959#6 593851959#7 1165173551#0 1165173551#4 593851958#0 593851958#1 593851958#2 1185347574 110450330#1 110450330#2 397052779#0 397052779#1 1165448858#0 "},
    "vehRoute2": {"stops":[], "edges":"-593851965#1 155396257#1 155396257#2 -4603390#2 -4603390#1 13863115 187939148#0 187939148#1 593851965#1 1254450668#1 155396256#1"},
    "vehRoute3": {"stops":["st_1","st_2"], "edges":"-181407580#13 -181407580#12 -181407580#11 -181407580#9 -181407580#7 -181407580#6 -181407580#5 -181407580#4 -181407580#3 -181407580#2 -181407580#1 -181407580#0 -1133304144#3 -790444789#1 -4579851#3 -4579851#2 666317537#0 666317537#2 1079316194 1079316193#0 1079316193#1 1133296854 155073131#0 155073131#1 155073131#3 155073131#4 318668065#0 10996272#3 391493957#0 4526202#2 1142744721#1 690575144#0 690575145 539116809 1191505117#0 1191505455#1 35373236#2 539116564#0 1057192512#2 1057192512#3 155071250#2 155071250#3 155071250#4 1133296876#1 1133296876#2 1133296876#5 1133296874#1 1133296873#1 52158841#2 489480666#1"},
    "vehRoute4": {"stops": ["st_2"], "edges":"35373236#2 539116564#0 1057192512#2 1057192512#3 155071250#2 155071250#3 155071250#4 1133296876#1 1133296876#2 1133296876#5 1133296874#1 1133296873#1 52158841#2 489480666#1"},
    "vehRoute5": {"stops": ["st_3","st_4"], "edges":"1057528473#1 1057528473#2 1147755002 1147755001#0 1147755001#1 1147755001#2 -563341816#2 -563341816#1 -563341816#0 -1147754937#1 -1147754937#0 -1147754939#0 -1147754940#0 -1147754941 1147754944 1147754949#1 1147754946#1 42998616#2 42998616#3 1186574031 1126171520#1 1126171520#2 1046077064#1 1046077064#3 1147768014#0 1147768014#1 1147768018#0 1147768018#1 1147768018#2 1147768018#3 4611862#2 4611862#5 1147768017#2 466891839#0"},
} 

personsList = [
    # Students
    {"initStop": {"edge": "1217880400#1", "duration": "10"}, "walks": {"busStop": "bs_1", "duration": "20"}, "rides": {"busStop": "bs_1"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1190798655#2", "duration": "10"}, "walks": {"busStop": "bs_3", "duration": "20"}, "rides": {"busStop": "bs_3"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1149847405#0", "duration": "10"}, "walks": {"busStop": "bs_4", "duration": "20"}, "rides": {"busStop": "bs_4"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "462764504#3", "duration": "10"}, "walks": {"busStop": "bs_5", "duration": "20"}, "rides": {"busStop": "bs_5"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1124627672#0", "duration": "10"}, "walks": {"busStop": "bs_6", "duration": "20"}, "rides": {"busStop": "bs_6"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1228446772#3", "duration": "10"}, "walks": {"busStop": "bs_9", "duration": "20"}, "rides": {"busStop": "bs_9"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1228446777#3", "duration": "10"}, "walks": {"busStop": "bs_10", "duration": "20"}, "rides": {"busStop": "bs_10"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1106891943#1", "duration": "10"}, "walks": {"busStop": "bs_11", "duration": "20"}, "rides": {"busStop": "bs_11"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1106891943#1", "duration": "10"}, "walks": {"busStop": "bs_12", "duration": "20"}, "rides": {"busStop": "bs_12"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "241574704#1", "duration": "10"}, "walks": {"busStop": "bs_7", "duration": "20"}, "rides": {"busStop": "bs_7"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},
    {"initStop": {"edge": "1149851903#4", "duration": "10"}, "walks": {"busStop": "bs_8", "duration": "20"}, "rides": {"busStop": "bs_8"}, "finalEdge": "-356359316#3", "finalStation": "bs_2"},

    # Normal passengers
    {"initStop": {"edge": "1117897479#1", "duration": "20"}, "walks": {"busStop": "st_1", "duration": "20"}, "rides": {"busStop": "st_1"}, "finalEdge": "1191505455#1", "finalStation": "st_2"},
    {"initStop": {"edge": "465683476#2", "duration": "20"}, "walks": {"busStop": "st_2", "duration": "20"}, "rides": {"busStop": "st_2"}, "finalEdge": "","finalStation": ""},
    {"initStop": {"edge": "1186574016#1", "duration": "20"}, "walks": {"busStop": "st_3", "duration": "20"}, "rides": {"busStop": "st_3"}, "finalEdge":"466891839#0", "finalStation": "st_4"},
]

def create_element(tag, attributes, parent):
    """Create an XML element with given attributes and append it to the parent."""
    element = ET.SubElement(parent, tag)
    for key, value in attributes.items():
        element.set(key, value)
    return element

def create_routes(routes_dict, root):
    for route_id, route_data in routes_dict.items():
        create_element("route", {"id": route_id, "edges": route_data["edges"]}, root)

def generate_random_vehs(num_vehs, route_ids):
    """Generate a list of buses with random routes."""
    vehs = []
    for i in range(num_vehs):
        veh_id = str(i + 1)
        route = random.choice(route_ids)
        bus = {"id": veh_id, "route": route, "capacity": random.randint(30, 60), "speed": 60}
        vehs.append(bus)
    return vehs

def generate_random_persons(num_persons, base_persons_list):
    """Generate a list of persons with unique IDs based on the base persons list."""
    persons = []
    for i in range(num_persons):
        base_person = random.choice(base_persons_list)
        person_id = str(i + 1)
        person = {
            "id": person_id,
            "initStop": base_person["initStop"],
            "walks": base_person["walks"],
            "rides": base_person["rides"],
            "finalEdge": base_person.get("finalEdge", ""),
            "finalStation": base_person.get("finalStation", ""),
        }
        persons.append(person)
    return persons

# =======================================================================================
root = ET.Element("routes")

create_element("vType", {"id": "bus","vClass":"bus", "accel": "1.0", "decel": "1.0", "sigma": "0.5", "length": "12", "minGap": "2.5", "maxSpeed": "30"}, root)

create_element("vType", {"id": "car", "accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "4.5", "minGap": "2.5", "maxSpeed": "80", "color": "blue"}, root)

create_element("vType", {"id": "pedestrian", "vClass": "pedestrian","width":"0.70", "length":"0.60"}, root)

create_routes(RoutesList, root)
create_routes(VehRoutes, root)


# =======================================================================================

num_buses = 50
route_ids = list(RoutesList.keys())
generated_buses = generate_random_vehs(num_buses, route_ids)

# Create vehicle elements for generated buses
for bus in generated_buses:
    route_data = RoutesList[bus["route"]]
    vehicle = create_element("vehicle", {"id": bus["id"], "type": "bus", "route": bus["route"], "depart": "2"}, root)
    for stop_id in route_data["stops"]:
        create_element("stop", {"busStop": stop_id, "duration": "50"}, vehicle)

num_persons = 300
generated_persons = generate_random_persons(num_persons, personsList)

# Create person elements for generated persons
for person in generated_persons:
    person_element = create_element("person", {"id": person["id"], "type": "pedestrian", "depart": "2"}, root)
    create_element("stop", person["initStop"], person_element)
    create_element("walk", person["walks"], person_element)
    if person["finalEdge"]:
        create_element("ride", {"busStop": person["rides"]["busStop"], "to": person["finalEdge"]}, person_element)
    else:
        create_element("ride", {"busStop": person["rides"]["busStop"]}, person_element)
    create_element("stop", {"busStop": person["finalStation"], "duration": "100"}, person_element)

# Generate 300 vehicles with random routes
num_vehicles = 300
veh_route_ids = list(VehRoutes.keys())
generated_vehicles = generate_random_vehs(num_vehicles, veh_route_ids)

# Create vehicle elements for generated vehicles
for vehicle in generated_vehicles:
    route_data = VehRoutes[vehicle["route"]]
    vehicle_element = create_element("vehicle", {"id": "veh" + vehicle["id"], "type": "car", "route": vehicle["route"], "depart": "2"}, root)
    for stop_id in route_data["stops"]:
        create_element("stop", {"busStop": stop_id, "duration": "50"}, vehicle_element)

# =======================================================================================

tree = ET.ElementTree(root)
tree.write("osm.passenger.trips.xml")
ET.dump(root)

print("Data written to osm.passenger.trips.xml")
