import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('Training.csv')


X = data.drop(columns='prognosis')  
y = data['prognosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------- Random Forest Model ----------------------------
best_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


best_rf_model.fit(X_train, y_train)

# ---------------------------- Naive Bayes Model ----------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


def predict_disease(symptoms_input, model_choice='random_forest'):
    
    symptoms_array = np.zeros(X.shape[1])  

    for symptom in symptoms_input:
        if symptom in symptom_map:
            index = symptom_map[symptom] - 1  
            symptoms_array[index] = 1.0 

    if model_choice == 'random_forest':
        model = best_rf_model
    elif model_choice == 'naive_bayes':
        model = nb_model
    else:
        raise ValueError("Invalid model choice. Please choose 'random_forest' or 'naive_bayes'.")

    
    predicted_disease = model.predict(symptoms_array.reshape(1, -1))
    return predicted_disease[0]

symptom_map = {
    "itching": 1,
    "skin_rash": 2,
    "nodal_skin_eruptions": 3,
    "continuous_sneezing": 4,
    "shivering": 5,
    "chills": 6,
    "joint_pain": 7,
    "stomach_pain": 8,
    "acidity": 9,
    "ulcers_on_tongue": 10,
    "muscle_wasting": 11,
    "vomiting": 12,
    "burning_micturition": 13,
    "spotting_urination": 14,
    "fatigue": 15,
    "weight_gain": 16,
    "anxiety": 17,
    "cold_hands_and_feets": 18,
    "mood_swings": 19,
    "weight_loss": 20,
    "restlessness": 21,
    "lethargy": 22,
    "patches_in_throat": 23,
    "irregular_sugar_level": 24,
    "cough": 25,
    "high_fever": 26,
    "sunken_eyes": 27,
    "breathlessness": 28,
    "sweating": 29,
    "dehydration": 30,
    "indigestion": 31,
    "headache": 32,
    "yellowish_skin": 33,
    "dark_urine": 34,
    "nausea": 35,
    "loss_of_appetite": 36,
    "pain_behind_the_eyes": 37,
    "back_pain": 38,
    "constipation": 39,
    "abdominal_pain": 40,
    "diarrhoea": 41,
    "mild_fever": 42,
    "yellow_urine": 43,
    "yellowing_of_eyes": 44,

}

name = input("Enter the name of the person: ")
symptoms_input = []

print("Available symptoms:", list(symptom_map.keys()))
print("Enter symptoms from the list above (up to a maximum of 5):")

for i in range(5): 
    symptom_name = input(f"Symptom {i + 1}: ").lower()
    
    if symptom_name in symptom_map:
        symptoms_input.append(symptom_name)
        
        if len(symptoms_input) == 5:
            break
    else:
        print("Invalid symptom! Please choose from the list.")
        break

if len(symptoms_input) == 5:
    
    model_choice = input("Enter the model you want to use (random_forest/naive_bayes): ").lower()
    
    predicted_disease = predict_disease(symptoms_input, model_choice)
    print("\n")
    
    print(f"\n{name}, based on the symptoms you provided, the predicted disease is: {predicted_disease}")
else:
    print("Error: Please provide exactly 5 valid symptoms.")
