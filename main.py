# ================== IMPORTS ==================

import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ml_model import train_model, predict_with_confidence, save_new_data

# ================== VALID FACILITIES ==================

VALID_FACILITIES = [
    "cardiac", "emergency", "neurology", "orthopedic",
    "pediatric", "ent", "ophthalmology", "oncology", "general"
]

# ================== DATA ==================

hospitals = [
    {"name": "City Hospital", "facility": "general", "pos": (1,11)},
    {"name": "Heart Care Center", "facility": "cardiac", "pos": (6,11)},
    {"name": "Trauma Center", "facility": "emergency", "pos": (11,11)},
    {"name": "Neuro Hospital", "facility": "neurology", "pos": (1,5)},
    {"name": "Ortho Care", "facility": "orthopedic", "pos": (6,5)},
    {"name": "Children Hospital", "facility": "pediatric", "pos": (11,5)},
    {"name": "Cancer Institute", "facility": "oncology", "pos": (1,-1)},
    {"name": "ENT Hospital", "facility": "ent", "pos": (6,-1)},
    {"name": "Eye Care Center", "facility": "ophthalmology", "pos": (11,-1)}
]

houses = {
    1:(1,8), 2:(4,8), 3:(7,8), 4:(10,8), 5:(13,8),
    6:(1,2), 7:(4,2), 8:(7,2), 9:(10,2), 10:(13,2)
}

ambulances = [
    {"id": "Amb1", "pos": (0,10)},
    {"id": "Amb2", "pos": (15,10)},
    {"id": "Amb3", "pos": (0,0)},
    {"id": "Amb4", "pos": (15,0)}
]

# ================== ROAD GRID ==================

roads = set()

for x in range(0,16):
    for y in range(-2,14):
        roads.add((x,y))

def block_area(x, y, w, h):
    return {(i, j) for i in range(x, x+w) for j in range(y, y+h)}

for h in hospitals:
    roads -= block_area(h["pos"][0], h["pos"][1], 4, 2)

for pos in houses.values():
    roads -= block_area(pos[0], pos[1], 2, 2)

# ================== FUNCTIONS ==================

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def nearest_road(point):
    return min(roads, key=lambda r: manhattan(r, point))

def find_best_hospital(facility, house_pos):
    match = [h for h in hospitals if h["facility"] == facility]
    if not match:
        match = hospitals
    return min(match, key=lambda h: manhattan(house_pos, h["pos"]))

def find_nearest_ambulance(house_pos):
    return min(ambulances, key=lambda a: manhattan(a["pos"], house_pos))

# ================== A* ==================

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if neighbor not in roads:
                continue

            temp = g[current] + 1

            if neighbor not in g or temp < g[neighbor]:
                g[neighbor] = temp
                f = temp + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    return None

# ================== DRAW ==================

def draw_map(path1, path2, hospital, ambulance):
    fig, ax = plt.subplots()

    def draw_rect(x, y, w, h, color):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color))

    def draw_static():
        ax.clear()

        draw_rect(0, -2, 16, 16, "black")

        for h in hospitals:
            x, y = h["pos"]
            color = "yellow" if h == hospital else "red"
            draw_rect(x, y, 4, 2, color)
            ax.text(x+2, y+1, h["name"], color='white',
                    ha='center', va='center', fontsize=7)

        for num, (x, y) in houses.items():
            draw_rect(x, y, 2, 2, "lime")
            ax.text(x+1, y+1, f"H{num}", color='black',
                    ha='center', va='center', fontsize=8)

        for a in ambulances:
            x, y = a["pos"]
            ax.plot(x+0.5, y+0.5, marker='o', color='cyan', markersize=6)
            ax.text(x+0.5, y+0.8, a["id"], color='cyan',
                    ha='center', fontsize=7)

        ax.set_xlim(0,16)
        ax.set_ylim(-2,14)
        ax.set_aspect('equal')
        ax.axis('off')

    for (x,y) in path1:
        draw_static()
        for (px,py) in path1:
            ax.plot(px+0.5, py+0.5, marker='s', color='blue', markersize=4)
        ax.plot(x+0.5, y+0.5, marker='o', color='white', markersize=10)
        plt.pause(0.1)

    for (x,y) in path2:
        draw_static()
        for (px,py) in path1:
            ax.plot(px+0.5, py+0.5, marker='s', color='blue', markersize=4)
        for (px,py) in path2:
            ax.plot(px+0.5, py+0.5, marker='s', color='yellow', markersize=4)
        ax.plot(x+0.5, y+0.5, marker='o', color='white', markersize=10)
        plt.pause(0.1)

    plt.show()

# ================== MAIN ==================

print("\n--- SMART EMERGENCY SYSTEM (ML + CONFIDENCE) ---")

print("Training model...")
model, vectorizer = train_model()

symptom = input("Enter symptom: ")
house_no = int(input("Enter house number (1-10): "))
show_graph = input("Do you want to see the graph visualization? (yes/no): ").lower()

if house_no not in houses:
    print("Invalid house number")
    exit()

house_pos = houses[house_no]

facility, confidence = predict_with_confidence(symptom, model, vectorizer)

print("\nPrediction:", facility)
print("Confidence:", round(confidence, 2))

THRESHOLD = 0.6

# LOW CONFIDENCE → ASK USER
if confidence < THRESHOLD:
    print("\n⚠️ Low confidence prediction")

    print("Choose correct facility:")
    for f in VALID_FACILITIES:
        print("-", f)

    correct = input("Enter correct facility: ").lower()

    if correct in VALID_FACILITIES:
        save_new_data(symptom, correct)
        facility = correct
        print("Learning saved!")
    else:
        print("Invalid input, using predicted result.")

hospital = find_best_hospital(facility, house_pos)
ambulance = find_nearest_ambulance(house_pos)

start = nearest_road(ambulance["pos"])
house_r = nearest_road(house_pos)
hospital_r = nearest_road(hospital["pos"])

path1 = astar(start, house_r)
path2 = astar(house_r, hospital_r)

if path1 is None or path2 is None:
    print("No path found!")
    exit()

print("\nHospital:", hospital["name"])
print("Ambulance:", ambulance["id"])

if show_graph == "yes":
    draw_map(path1, path2, hospital, ambulance)
else:
    print("\nGraph visualization skipped.")