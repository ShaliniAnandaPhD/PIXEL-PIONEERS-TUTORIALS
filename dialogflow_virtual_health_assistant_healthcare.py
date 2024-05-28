# Use case: Healthcare - AI-Powered Virtual Health Assistant

import dialogflow_v2 as dialogflow
import pandas as pd
import numpy as np
import random

# Set up Dialogflow credentials
project_id = "your-project-id"
session_id = "your-session-id"
language_code = "en-US"
credentials_file = "path/to/your/credentials.json"

# Create a Dialogflow session client
session_client = dialogflow.SessionsClient.from_service_account_file(credentials_file)
session = session_client.session_path(project_id, session_id)

# Load sample health data
health_data = pd.DataFrame({
    "Symptom": ["Headache", "Cough", "Fever", "Fatigue", "Nausea"],
    "Severity": ["Mild", "Moderate", "High", "Mild", "Moderate"],
    "Duration (days)": [2, 5, 3, 7, 1]
})

# Function to get user input
def get_user_input():
    user_input = input("User: ")
    return user_input

# Function to get Dialogflow response
def get_dialogflow_response(user_input):
    # Prepare the text input for Dialogflow
    text_input = dialogflow.types.TextInput(text=user_input, language_code=language_code)
    query_input = dialogflow.types.QueryInput(text=text_input)
    
    # Send the text input to Dialogflow and get the response
    response = session_client.detect_intent(session=session, query_input=query_input)
    return response.query_result.fulfillment_text

# Function to analyze health data and provide recommendations
def analyze_health_data(symptom, severity, duration):
    # Perform analysis based on symptom, severity, and duration
    if symptom == "Fever" and severity == "High" and duration >= 3:
        recommendation = "It seems like you have a severe fever. Please consult a doctor immediately."
    elif symptom == "Cough" and duration >= 5:
        recommendation = "Your cough has persisted for several days. Consider seeing a doctor if it doesn't improve."
    else:
        recommendation = "Based on your symptoms, it doesn't seem to be a serious concern. Rest and monitor your condition."
    return recommendation

# Additional function to simulate health data based on user input
def simulate_health_data(user_input):
    # Extract symptoms mentioned in user input
    symptoms = [symptom for symptom in health_data["Symptom"] if symptom.lower() in user_input.lower()]
    
    if symptoms:
        symptom = random.choice(symptoms)
        severity = health_data[health_data["Symptom"] == symptom]["Severity"].values[0]
        duration = health_data[health_data["Symptom"] == symptom]["Duration (days)"].values[0]
    else:
        # Default values if no specific symptom is mentioned
        symptom = random.choice(health_data["Symptom"].values)
        severity = "Mild"
        duration = random.randint(1, 7)
        
    return symptom, severity, duration

# Main interaction loop
print("Virtual Health Assistant: How can I assist you today?")

while True:
    # Get user input
    user_input = get_user_input()
    
    # Exit the loop if the user says 'bye'
    if user_input.lower() == "bye":
        print("Virtual Health Assistant: Thank you for using the Virtual Health Assistant. Take care!")
        break
    
    # Get the Dialogflow response for the user input
    response = get_dialogflow_response(user_input)
    print("Virtual Health Assistant:", response)
    
    # If the user input contains a symptom, simulate health data and provide a recommendation
    if any(symptom.lower() in user_input.lower() for symptom in health_data["Symptom"].values):
        symptom, severity, duration = simulate_health_data(user_input)
        recommendation = analyze_health_data(symptom, severity, duration)
        print("Virtual Health Assistant:", recommendation)

# Additional functionalities to make the assistant more sophisticated

# Function to log the conversation history
def log_conversation(user_input, assistant_response, recommendation=None):
    with open("conversation_log.txt", "a") as log_file:
        log_file.write(f"User: {user_input}\n")
        log_file.write(f"Assistant: {assistant_response}\n")
        if recommendation:
            log_file.write(f"Recommendation: {recommendation}\n")
        log_file.write("\n")

# Function to load and display health tips
def load_health_tips():
    # Load health tips from a file or API
    health_tips = [
        "Stay hydrated by drinking at least 8 cups of water a day.",
        "Get at least 7-8 hours of sleep each night.",
        "Incorporate fruits and vegetables into your diet.",
        "Exercise regularly to maintain a healthy body and mind.",
        "Wash your hands frequently to prevent the spread of germs."
    ]
    return health_tips

# Display health tips periodically
health_tips = load_health_tips()
tip_index = 0

while True:
    user_input = get_user_input()
    
    if user_input.lower() == "bye":
        print("Virtual Health Assistant: Thank you for using the Virtual Health Assistant. Take care!")
        break
    
    response = get_dialogflow_response(user_input)
    print("Virtual Health Assistant:", response)
    
    if any(symptom.lower() in user_input.lower() for symptom in health_data["Symptom"].values):
        symptom, severity, duration = simulate_health_data(user_input)
        recommendation = analyze_health_data(symptom, severity, duration)
        print("Virtual Health Assistant:", recommendation)
        log_conversation(user_input, response, recommendation)
    else:
        log_conversation(user_input, response)
    
    # Display a health tip every 5 interactions
    if tip_index % 5 == 0:
        tip = health_tips[tip_index % len(health_tips)]
        print("Health Tip:", tip)
    tip_index += 1

# Function to handle user feedback
def handle_user_feedback():
    feedback = input("Virtual Health Assistant: Do you have any feedback for me? (yes/no): ")
    if feedback.lower() == "yes":
        user_feedback = input("Virtual Health Assistant: Please provide your feedback: ")
        with open("user_feedback.txt", "a") as feedback_file:
            feedback_file.write(f"Feedback: {user_feedback}\n")
        print("Virtual Health Assistant: Thank you for your feedback!")
    else:
        print("Virtual Health Assistant: Thank you for using the Virtual Health Assistant.")

# Call the feedback function at the end of the session
handle_user_feedback()

#this code demonstrates how to build an AI-powered virtual health assistant using Dialogflow. The assistant interacts with the user, processes their input using Dialogflow's natural language understanding capabilities, and provides relevant responses. It also includes a sample health data analysis component that provides recommendations based on the user's symptoms.
#To run this code, you need to set up a Dialogflow project, obtain the necessary credentials, and replace the placeholders (your-project-id, your-session-id, path/to/your/credentials.json) with your actual project details.
#Remember to have the required libraries installed (Dialogflow, Pandas, NumPy) before running the code. You can install the Dialogflow library using pip install dialogflow
