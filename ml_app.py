import streamlit as st 
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

attrib_info = """
#### Información de las variables:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle stiffness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.

"""
# Diccionarios para recodificar las variables
label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
       'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
       'itching', 'irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'alopecia', 'obesity', 'class']

# Funciones prar recodificar las variables
def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 



# Carga de modelo ML

# Memoria cache: solo ejecuta la función si algo cambia
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache_resource
def load_model(model_file):
	try:
		loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
		return loaded_model
	except Exception as e:
		error_message = str(e)
		st.error(f"Error loading model {model_file}: {error_message}")
		
		if "incompatible dtype" in error_message and "missing_go_to_left" in error_message and "decision_tree" in model_file:
			st.warning("""
			Este error se debe a una incompatibilidad de versiones de scikit-learn. 
			Se creará un nuevo modelo de árbol de decisión con la versión actual.
			""")
			return create_decision_tree_model()
		else:
			st.info("Asegúrate de que scikit-learn está instalado correctamente. Ejecuta: pip install scikit-learn")
		
		return None

@st.cache_resource
def create_decision_tree_model():
	"""Create a new decision tree model using the current scikit-learn version"""
	try:
		# Load the data
		st.info("Creando un nuevo modelo de árbol de decisión compatible...")
		df = pd.read_csv("data/diabetes_data_upload_clean.csv")
		
		# Encode categorical variables
		le = LabelEncoder()
		for column in df.columns:
			if df[column].dtype == 'object':
				df[column] = le.fit_transform(df[column])
		
		# Split the data
		X = df.drop('class', axis=1)
		y = df['class']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Create and train model
		model = DecisionTreeClassifier(random_state=42)
		model.fit(X_train, y_train)
		
		# Calculate accuracy
		accuracy = model.score(X_test, y_test)
		st.success(f"Nuevo modelo creado con una precisión de {accuracy:.2f}")
		
		return model
	except Exception as e:
		st.error(f"Error al crear el nuevo modelo: {str(e)}")
		return None

def run_ml_app():
	st.subheader("Sección de Machine Learning")
	loaded_model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")

	with st.expander("Informacion sobre las variables"):
		st.markdown(attrib_info,unsafe_allow_html=True)

	# Layout
	col1,col2 = st.columns(2)

	with col1:
		age = st.number_input("Age",10,100)
		gender = st.radio("Gender",("Female","Male"))
		polyuria = st.radio("Polyuria",["No","Yes"])
		polydipsia = st.radio("Polydipsia",["No","Yes"]) 
		sudden_weight_loss = st.selectbox("Sudden_weight_loss",["No","Yes"])
		weakness = st.radio("weakness",["No","Yes"]) 
		polyphagia = st.radio("polyphagia",["No","Yes"]) 
		genital_thrush = st.selectbox("Genital_thrush",["No","Yes"]) 
		
	
	with col2:
		visual_blurring = st.selectbox("Visual_blurring",["No","Yes"])
		itching = st.radio("itching",["No","Yes"]) 
		irritability = st.radio("irritability",["No","Yes"]) 
		delayed_healing = st.radio("delayed_healing",["No","Yes"]) 
		partial_paresis = st.selectbox("Partial_paresis",["No","Yes"])
		muscle_stiffness = st.radio("muscle_stiffness",["No","Yes"]) 
		alopecia = st.radio("alopecia",["No","Yes"]) 
		obesity = st.select_slider("obesity",["No","Yes"]) 

	with st.expander("Valores introducidos"):
		result = {'age':age,
		'gender':gender,
		'polyuria':polyuria,
		'polydipsia':polydipsia,
		'sudden_weight_loss':sudden_weight_loss,
		'weakness':weakness,
		'polyphagia':polyphagia,
		'genital_thrush':genital_thrush,
		'visual_blurring':visual_blurring,
		'itching':itching,
		'irritability':irritability,
		'delayed_healing':delayed_healing,
		'partial_paresis':partial_paresis,
		'muscle_stiffness':muscle_stiffness,
		'alopecia':alopecia,
		'obesity':obesity}
		st.write(result)
		encoded_result = []
		for i in result.values():
			if type(i) == int:
				encoded_result.append(i)
			elif i in ["Female","Male"]:
				res = get_value(i,gender_map)
				encoded_result.append(res)
			else:
				encoded_result.append(get_fvalue(i))


		st.write(encoded_result)
	with st.expander("Resultados de la Predicción"):
		single_sample = np.array(encoded_result).reshape(1,-1)

		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		st.write(prediction)
		# Para ver más detalles y hacerlo más completo: 
		if prediction == 1:
			st.warning("Riesgo Positivo-{}".format(prediction[0]))
			pred_probability_score = {"Negativo en DM":pred_prob[0][0]*100,"Positivo en DM":pred_prob[0][1]*100}
			st.subheader("Valores de Predicción en probabilidades")
			st.json(pred_probability_score)
		else:
			st.success("Riesgo Negativo-{}".format(prediction[0]))
			pred_probability_score = {"Negativo en DM":pred_prob[0][0]*100,"Positivo en DM":pred_prob[0][1]*100}
			st.subheader("Valores de Predicción en probabilidades")
			st.json(pred_probability_score)

