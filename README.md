# Smart Emergency Response System

## Project Description

This project is a Smart Emergency Response System that recommends the most suitable hospital based on user symptoms and finds the shortest path for ambulance routing. It combines Machine Learning for symptom classification and the A* algorithm for optimal pathfinding.

The system takes user input from the command line and optionally displays a graphical visualization of the ambulance route using matplotlib.

---

## Features

* Predicts required medical facility based on symptoms using Machine Learning
* Displays prediction confidence score
* Self-learning system that improves with user feedback
* Finds nearest ambulance and optimal hospital
* Uses A* algorithm for shortest path calculation
* Optional graphical visualization of ambulance movement
* Grid-based simulation of a city map

---

## Technologies Used

* Python
* scikit-learn (TF-IDF, Naive Bayes)
* matplotlib (visualization)
* Heap Queue (priority queue for A*)

---

## How It Works

1. User enters symptom and house number
2. ML model predicts the required medical facility
3. If confidence is low, user can correct the prediction
4. System selects:

   * Nearest suitable hospital
   * Nearest ambulance
5. A* algorithm calculates shortest path
6. System optionally displays graphical route

---

## Project Structure

```
├── main.py                # Main application (CLI + visualization)
├── ml_model.py            # Machine learning model
├── training_data.txt      # Dataset for training
└── README.md              # Project documentation
```

---

## Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/tfkhtr/aiml-project.git
cd aiml-project
```

### 2. Install Dependencies

```
pip install matplotlib scikit-learn
```

### 3. Run the Project

```
python main.py
```

---

## Usage

* Enter a symptom (e.g., chest pain, headache, fracture)
* Enter house number (1–10)
* View predicted hospital and ambulance details
* Choose whether to display graphical visualization

---

## Limitations

* Uses a simulated grid instead of real maps
* Model retrains every time (no saved model)
* Dataset may contain duplicates or noise
* Not connected to real hospital data

---

## Future Improvements

* Convert to GUI or web application
* Integrate real-time maps (Google Maps API)
* Improve dataset quality
* Save trained model for faster execution
* Add real-time ambulance tracking

---

## Conclusion

This project demonstrates the integration of Machine Learning and graph-based algorithms to solve a real-world problem in emergency response systems. It highlights how intelligent systems can assist in faster decision-making and efficient resource allocation.

---
