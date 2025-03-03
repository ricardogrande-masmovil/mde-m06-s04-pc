# PROYECTO DE CONSOLIDACION STREAMLIT: FUNCION eda_app.py

# Importaciones: streamlit, pandas, matplotlib, seaborn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

@st.cache_data
def load_age_data():
	return pd.read_csv("data/freqdist_of_age_data.csv")

# Función para la versión mejorada con Plotly
def run_plotly_eda():
	st.title("Sección EDA con Plotly")
	st.write("Versión mejorada con gráficos interactivos de Plotly")
	
	df = pd.read_csv("data/diabetes_data_upload.csv")
	
	st.sidebar.subheader("SubMenu")
	submenu = st.sidebar.selectbox("Tipo de Gráfico", ["Gráfico", "Correlación", "Distribución"])
	
	if submenu == "Gráfico":
		st.subheader("Visualización gráfica con Plotly")

		with st.expander("Gráfico de distribución por género (Gender)"):
			gender_counts = df['Gender'].value_counts()
			fig = px.pie(
				values=gender_counts.values, 
				names=gender_counts.index, 
				title="Distribución por Género",
				color_discrete_sequence=px.colors.qualitative.Pastel
			)
			fig.update_traces(textposition='inside', textinfo='percent+label')
			st.plotly_chart(fig, use_container_width=True)
			
			st.write("Distribución por Gender")
			st.table(df['Gender'].value_counts())
			
		with st.expander("Gráfico de distribución por clase (Class)"):
			class_counts = df['class'].value_counts()
			fig = px.pie(
				values=class_counts.values, 
				names=class_counts.index, 
				title="Distribución por Clase",
				color_discrete_sequence=px.colors.qualitative.Bold
			)
			fig.update_traces(textposition='inside', textinfo='percent+label')
			st.plotly_chart(fig, use_container_width=True)
			
			st.write("Distribución por Class")
			st.table(df['class'].value_counts())
		
		with st.expander("Distribución por edades (Age)"):
			age_data = load_age_data()
			st.write(age_data.head())
			
			fig = px.bar(
				age_data, 
				x='Age', 
				y='count', 
				title="Conteo de frecuencia por Edad",
				color_discrete_sequence=['#4C78A8'],
				text='count'
			)
			fig.update_layout(
				xaxis_title="Rango de Edad",
				yaxis_title="Frecuencia",
				xaxis={'categoryorder':'total ascending'}
			)
			st.plotly_chart(fig, use_container_width=True)
		
		with st.expander("Detección de Outliers"):
			fig = px.box(
				df, 
				x='Gender', 
				y='Age', 
				color='Gender',
				title="Boxplot de las edades, por géneros",
				color_discrete_map={'Male': '#4C72B0', 'Female': '#DD8452'}
			)
			fig.update_layout(
				xaxis_title="Género",
				yaxis_title="Edad"
			)
			st.plotly_chart(fig, use_container_width=True)
	
	elif submenu == "Correlación":
		st.subheader("Análisis de correlación con Plotly")
		
		diabetes_df = pd.read_csv("data/diabetes_data_upload_clean.csv")
		
		numeric_df = diabetes_df.select_dtypes(include=['number'])
		corr = numeric_df.corr()
		
		fig = px.imshow(
			corr, 
			text_auto='.2f',
			color_continuous_scale='RdBu_r',
			zmin=-0.4,
			zmax=1.0
		)
		fig.update_layout(
			title="Matriz de Correlación - Dataset de Diabetes",
			width=900,
			height=700
		)
		st.plotly_chart(fig)
	
	elif submenu == "Distribución":
		st.subheader("Distribución de variables")
		
		numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
		
		for column in numeric_cols:
			st.write(f"Distribución de {column}")
			fig = px.histogram(
				df, 
				x=column,
				title=f"Histograma de {column}",
				color_discrete_sequence=['#636EFA']
			)
			fig.update_layout(bargap=0.1)
			st.plotly_chart(fig, use_container_width=True)

# Función principal que emplearemos en la APP
def run_eda_app():
	st.title("Sección EDA")
	
	visualization_type = st.radio(
		"Seleccione el tipo de visualización:",
		("Visualización Estándar (Matplotlib/Seaborn)", "Visualización Interactiva (Plotly)")
	)
	
	if visualization_type == "Visualización Interactiva (Plotly)":
		run_plotly_eda()
		return
	
	df = pd.read_csv("data/diabetes_data_upload.csv")

	submenu = st.sidebar.selectbox("SubMenu", ["Descriptivo", "Gráfico"])

	if submenu == "Descriptivo":
		st.subheader("Análisis descriptivo")
		
		st.dataframe(df)

		with st.expander("Tipos de datos"):
			st.write(df.dtypes)

		with st.expander("Resumen descriptivo"):
			df_clean = pd.read_csv("data/diabetes_data_upload_clean.csv")
			st.write(df_clean.describe())

		with st.expander("Distribución por género (Gender)"):
			st.write(df['Gender'].value_counts())

		with st.expander("Distribución por clase/label (Class)"):
			st.write(df['class'].value_counts())

	elif submenu == "Gráfico":
		st.subheader("Análisis gráfico")

		with st.expander("Gráfico de distribución por género (Gender)"):
			fig, ax = plt.subplots()
			sns.countplot(x='Gender', data=df, ax=ax)
			st.pyplot(fig)
			st.write("Distribución por Gender")
			st.table(df['Gender'].value_counts())

		with st.expander("Gráfico de distribución por clase (Class)"):
			fig, ax = plt.subplots()
			sns.countplot(x='class', data=df, ax=ax)
			st.pyplot(fig)
			st.write("Distribución por Class")
			st.table(df['class'].value_counts())

		with st.expander("Distribución por edades (Age)"):
			age_data = load_age_data()
			st.write(age_data.head())

			fig, ax = plt.subplots(figsize=(10, 6))
			sns.barplot(x='Age', y='count', data=age_data, ax=ax)
			ax.set_title("Conteo de frecuencia por Edad")
			
			plt.xticks(rotation=45, ha='right')
			
			plt.tight_layout()
			st.pyplot(fig)

		with st.expander("Detección de Outliers"):
			fig, ax = plt.subplots(figsize=(10, 6))
			sns.boxplot(x='Gender', y='Age', data=df, ax=ax, hue='Gender', palette=['#4C72B0', '#DD8452'], legend=False)
			ax.set_title("Boxplot de las edades, por géneros")
			ax.set_ylabel("Age")
			ax.set_xlabel("Gender")
			
			plt.tight_layout()
			st.pyplot(fig)

		with st.expander("Gráfico de Correlación"):
			diabetes_df = pd.read_csv("data/diabetes_data_upload_clean.csv")
			
			corr = diabetes_df.select_dtypes(include=['number']).corr()
			
			plt.figure(figsize=(14, 12))
			
			sns.heatmap(corr, 
						annot=True,
						fmt='.2f',
						cmap='RdBu_r',
						linewidths=0.5,
						vmin=-0.4,
						vmax=1,
						annot_kws={"size": 8})
			
			plt.tight_layout()
			st.pyplot(plt)

	elif submenu == "Correlación":
		st.subheader("Análisis de correlación")
		corr = df.corr()
		st.write(corr)
		fig, ax = plt.subplots()
		sns.heatmap(corr, ax=ax, annot=True)
		st.pyplot(fig)

	elif submenu == "Distribución":
		st.subheader("Distribución de variables")
		for column in df.select_dtypes(include=['int64', 'float64']).columns:
			st.write(f"Distribución de {column}")
			fig, ax = plt.subplots()
			sns.histplot(df[column], ax=ax)
			st.pyplot(fig)

# Fin de la FUNCION







