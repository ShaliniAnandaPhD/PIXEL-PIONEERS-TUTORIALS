# Python Growing Food on Mars: Early Warning System for Plant Diseases

# Welcome to the tenth class in the "Growing Food on Mars" series!
# In this class, we will learn how to implement a Python-based early warning system for plant diseases in a Martian greenhouse.

# First, let's define a dictionary of common plant diseases and their symptoms:
plant_diseases = {
   "Powdery Mildew": ["white powdery spots", "yellow leaves", "stunted growth"],
   "Fusarium Wilt": ["yellow leaves", "wilting", "brown stem discoloration"],
   "Bacterial Blight": ["water-soaked spots", "yellowing leaves", "leaf drop"]
}

# Let's create a function to check for disease symptoms:
def check_for_disease(symptoms):
   for disease, disease_symptoms in plant_diseases.items():
       if set(symptoms).intersection(disease_symptoms):
           print(f"Warning: Plant may have {disease}. Symptoms observed: {', '.join(symptoms)}")
       else:
           print(f"No symptoms of {disease} observed.")

# Now, let's define a function to monitor plants for disease symptoms:
def monitor_plants(plant_data):
   for plant_id, symptoms in plant_data.items():
       print(f"\nMonitoring Plant ID: {plant_id}")
       check_for_disease(symptoms)

# Let's create a sample dataset of plant observations:
plant_observations = {
   1: ["yellow leaves", "stunted growth"],
   2: ["wilting", "brown stem discoloration"],
   3: ["water-soaked spots", "leaf drop"],
   4: ["healthy"],
   5: ["white powdery spots", "yellow leaves"]
}

# Let's start monitoring the plants:
print("Starting plant health monitoring...\n")
monitor_plants(plant_observations)

# Let's define a function to send alerts when a disease is detected:
def send_alert(plant_id, disease):
   print(f"Alert: Plant ID {plant_id} may have {disease}. Take immediate action!")

# Now, let's modify the check_for_disease function to send alerts:
def check_for_disease(plant_id, symptoms):
   for disease, disease_symptoms in plant_diseases.items():
       if set(symptoms).intersection(disease_symptoms):
           print(f"Warning: Plant ID {plant_id} may have {disease}. Symptoms observed: {', '.join(symptoms)}")
           send_alert(plant_id, disease)
       else:
           print(f"No symptoms of {disease} observed in Plant ID {plant_id}.")

# Let's update the monitor_plants function to include plant IDs:
def monitor_plants(plant_data):
   for plant_id, symptoms in plant_data.items():
       print(f"\nMonitoring Plant ID: {plant_id}")
       check_for_disease(plant_id, symptoms)

# Let's start monitoring the plants again with the updated system:
print("\nRestarting plant health monitoring with alert system...\n")
monitor_plants(plant_observations)

# Follow along and rewrite this code in your own Python file.
# Feel free to experiment with different disease symptoms and alerting mechanisms to further improve the early warning system.

# In the next class, we will explore how to simulate the effects of Martian gravity on plant growth using Python.
# Get ready to dive deeper into the world of plant adaptation on Mars!
