#  PROYECTO CONSOLIDACION STREAMLIT: FUNCION app.py

# Importaciones
import streamlit as st
import streamlit.components.v1 as stc
from eda_app import run_eda_app
from ml_app import run_ml_app


# Función main()
def main():
	st.sidebar.title("Menu")
	section = st.sidebar.selectbox("Go to", ["Home", "EDA", "ML", "Info"])

	if section == "Home":
		st.title("App para la deteccion temprana de DM")
		st.subheader("(Diabetes Mellitus)")
		st.write("Dataset que contiene señales y síntomas que pueden indicar diabetes o posibilidad de diabetes.")
		
		st.subheader("Fuente de datos")
		st.write("- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset)")

		st.subheader("Contenidos de la App")
		st.write("- EDA Section: Análisis exploratorio de los datos")
		st.write("- ML Section: Predicción de Diabetes basada en ML (Machine Learning)")

	elif section == "EDA":
		run_eda_app()
	elif section == "ML":
		run_ml_app()
	elif section == "Info":
		st.title("Info")
		st.subheader("MBIT, Proyecto de consolidación, librería Streamlit.")
		st.write("* * *")
		
		st.write("## MBIT Data School")
		
		# No funciona el iframe, MBIT lo bloquea
		st.markdown("""
		<div style="text-align: center; margin: 20px 0;">
			<p>Para visitar el sitio web oficial de MBIT School, haga clic en el botón a continuación:</p>
			<a href="https://www.mbitschool.com/" target="_blank" style="
				display: inline-block;
				background-color: #FF4B4B;
				color: white;
				padding: 12px 24px;
				text-align: center;
				text-decoration: none;
				font-size: 16px;
				border-radius: 4px;
				margin: 20px 0;
			">Visitar MBIT School</a>
		</div>
		""", unsafe_allow_html=True)
		
		st.write("### Sobre MBIT")
		st.write("MBIT Data School es una institución especializada en formación para data science, inteligencia artificial y big data.")
		st.info("Este proyecto es parte del curso de Data Science de MBIT School, utilizando la librería Streamlit para crear aplicaciones web interactivas para análisis de datos.")
		
		st.write("### Acerca de este proyecto")
		st.write("Desarrollado como parte del proyecto de consolidación de conocimientos en Streamlit y análisis de datos.")


if __name__ == '__main__':
	main()