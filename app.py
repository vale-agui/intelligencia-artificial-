import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


st.set_page_config(page_title="Laboratorio estadístico", page_icon="📊", layout="wide")


@st.cache_data
def generar_datos(n: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    ciudades = np.array(["Medellín", "Bogotá", "Cali", "Barranquilla", "Pereira"])
    categorias = np.array(["Tecnología", "Servicios", "Comercio", "Educación"])
    niveles = np.array(["Bajo", "Medio", "Alto"])

    edad = np.clip(np.rint(rng.normal(36, 11, n)), 18, 70).astype(int)
    experiencia = np.maximum(0, edad - rng.integers(18, 29, n))
    ingresos = np.maximum(1_300_000, rng.lognormal(np.log(3_200_000), 0.42, n))
    satisfaccion = np.clip(
        np.rint(5.8 + experiencia * 0.04 + rng.normal(0, 1.5, n)), 1, 10
    ).astype(int)
    fecha_inicio = pd.Timestamp("2024-01-01")

    return pd.DataFrame(
        {
            "ID": [f"REG-{i:04d}" for i in range(1, n + 1)],
            "Fecha": fecha_inicio + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
            "Edad": edad,
            "Experiencia_anios": experiencia,
            "Ingresos_COP": np.round(ingresos, -3).astype(int),
            "Satisfaccion": satisfaccion,
            "Ciudad": rng.choice(ciudades, n, p=[0.30, 0.30, 0.18, 0.12, 0.10]),
            "Sector": rng.choice(categorias, n, p=[0.28, 0.32, 0.24, 0.16]),
            "Nivel": rng.choice(niveles, n, p=[0.22, 0.53, 0.25]),
            "Recomienda": rng.choice(["Sí", "No"], n, p=[0.76, 0.24]),
        }
    )


def cargar_archivo(archivo) -> pd.DataFrame:
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        contenido = archivo.getvalue()
        try:
            return pd.read_csv(io.BytesIO(contenido), sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", encoding="latin-1")
    return pd.read_excel(archivo)


def tabla_cuantitativa(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    filas = []
    for col in columnas:
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.empty:
            continue
        moda = serie.mode()
        filas.append(
            {
                "Variable": col,
                "N": serie.size,
                "Faltantes": int(df[col].isna().sum()),
                "Media": serie.mean(),
                "Mediana": serie.median(),
                "Moda": moda.iloc[0] if not moda.empty else np.nan,
                "Desv. estándar": serie.std(),
                "Varianza": serie.var(),
                "Mínimo": serie.min(),
                "Q1": serie.quantile(0.25),
                "Q3": serie.quantile(0.75),
                "Máximo": serie.max(),
                "Asimetría": stats.skew(serie, bias=False) if serie.size > 2 else np.nan,
                "Curtosis": stats.kurtosis(serie, bias=False) if serie.size > 3 else np.nan,
            }
        )
    return pd.DataFrame(filas).set_index("Variable") if filas else pd.DataFrame()


def descargar_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("📊 Laboratorio de datos y estadística")
st.caption("Genera datos sintéticos o analiza un archivo propio con estadísticas y gráficas interactivas.")

with st.sidebar:
    st.header("Fuente de datos")
    fuente = st.radio("Selecciona una opción", ["Datos sintéticos", "Cargar archivo"])
    if fuente == "Datos sintéticos":
        cantidad = st.slider("Número de registros", 50, 10_000, 500, 50)
        semilla = st.number_input("Semilla aleatoria", min_value=0, value=42, step=1)
        df = generar_datos(cantidad, int(semilla))
    else:
        archivo = st.file_uploader("Archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if archivo is None:
            st.info("Carga un archivo para comenzar.")
            st.stop()
        try:
            df = cargar_archivo(archivo)
        except Exception as exc:
            st.error(f"No fue posible leer el archivo: {exc}")
            st.stop()

if df.empty:
    st.warning("El conjunto de datos no contiene registros.")
    st.stop()

numericas = df.select_dtypes(include=np.number).columns.tolist()
categoricas = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
fechas = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(df):,}")
c2.metric("Variables", len(df.columns))
c3.metric("Numéricas", len(numericas))
c4.metric("Categóricas", len(categoricas))

tab_datos, tab_cuant, tab_cuali, tab_graficos = st.tabs(
    ["Datos", "Estadística cuantitativa", "Estadística cualitativa", "Gráficas interactivas"]
)

with tab_datos:
    st.subheader("Vista previa")
    st.dataframe(df, use_container_width=True, hide_index=True)
    faltantes = pd.DataFrame({"Tipo": df.dtypes.astype(str), "Faltantes": df.isna().sum()})
    st.subheader("Calidad y tipos de datos")
    st.dataframe(faltantes, use_container_width=True)
    st.download_button(
        "Descargar datos en CSV", descargar_csv(df), "datos_analizados.csv", "text/csv"
    )

with tab_cuant:
    if not numericas:
        st.info("No se detectaron variables numéricas.")
    else:
        seleccion_num = st.multiselect(
            "Variables numéricas", numericas, default=numericas[: min(5, len(numericas))]
        )
        resumen = tabla_cuantitativa(df, seleccion_num)
        if resumen.empty:
            st.info("Selecciona al menos una variable.")
        else:
            st.dataframe(resumen.style.format(precision=2), use_container_width=True)
            if len(seleccion_num) >= 2:
                st.subheader("Matriz de correlación de Pearson")
                corr = df[seleccion_num].corr(numeric_only=True)
                fig_corr = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
                )
                fig_corr.update_layout(height=520)
                st.plotly_chart(fig_corr, use_container_width=True)

with tab_cuali:
    if not categoricas:
        st.info("No se detectaron variables categóricas.")
    else:
        col_cat = st.selectbox("Variable cualitativa", categoricas)
        frecuencias = (
            df[col_cat]
            .fillna("(Sin dato)")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("Categoría")
            .reset_index(name="Frecuencia")
        )
        frecuencias["Porcentaje"] = frecuencias["Frecuencia"] / len(df) * 100
        moda = frecuencias.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Categorías", len(frecuencias))
        b.metric("Moda", str(moda["Categoría"]))
        c.metric("Frecuencia modal", int(moda["Frecuencia"]))
        st.dataframe(
            frecuencias.style.format({"Porcentaje": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        fig_cat = px.bar(
            frecuencias,
            x="Categoría",
            y="Frecuencia",
            color="Frecuencia",
            text_auto=True,
            title=f"Distribución de {col_cat}",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

with tab_graficos:
    tipos = ["Histograma", "Caja y bigotes", "Dispersión", "Barras", "Circular"]
    if fechas and numericas:
        tipos.append("Serie temporal")
    tipo = st.selectbox("Tipo de gráfica", tipos)

    if tipo == "Histograma":
        if not numericas:
            st.info("Se requiere una variable numérica.")
        else:
            x = st.selectbox("Variable", numericas, key="hist_x")
            color = st.selectbox("Agrupar por", ["Ninguna"] + categoricas, key="hist_color")
            bins = st.slider("Número de intervalos", 5, 100, 30)
            fig = px.histogram(df, x=x, color=None if color == "Ninguna" else color, nbins=bins, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Caja y bigotes":
        if not numericas:
            st.info("Se requiere una variable numérica.")
        else:
            y = st.selectbox("Variable numérica", numericas, key="box_y")
            x = st.selectbox("Agrupar por", ["Ninguna"] + categoricas, key="box_x")
            fig = px.box(df, x=None if x == "Ninguna" else x, y=y, color=None if x == "Ninguna" else x, points="outliers")
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Dispersión":
        if len(numericas) < 2:
            st.info("Se requieren al menos dos variables numéricas.")
        else:
            x = st.selectbox("Eje X", numericas, key="scatter_x")
            y = st.selectbox("Eje Y", numericas, index=1, key="scatter_y")
            color = st.selectbox("Color", ["Ninguna"] + categoricas, key="scatter_color")
            tendencia = st.checkbox("Mostrar línea de tendencia")
            fig = px.scatter(
                df, x=x, y=y, color=None if color == "Ninguna" else color,
                trendline="ols" if tendencia else None, hover_data=df.columns[: min(5, len(df.columns))]
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tipo in ["Barras", "Circular"]:
        if not categoricas:
            st.info("Se requiere una variable categórica.")
        else:
            x = st.selectbox("Categoría", categoricas, key="cat_graph")
            conteo = df[x].fillna("(Sin dato)").astype(str).value_counts().reset_index()
            conteo.columns = [x, "Frecuencia"]
            if tipo == "Barras":
                fig = px.bar(conteo, x=x, y="Frecuencia", color="Frecuencia", text_auto=True)
            else:
                fig = px.pie(conteo, names=x, values="Frecuencia", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Serie temporal":
        fecha = st.selectbox("Fecha", fechas)
        valor = st.selectbox("Variable numérica", numericas, key="time_y")
        temporal = df[[fecha, valor]].dropna().sort_values(fecha)
        fig = px.line(temporal, x=fecha, y=valor, markers=True)
        st.plotly_chart(fig, use_container_width=True)

st.caption("Las gráficas permiten zoom, selección, desplazamiento y descarga desde su barra de herramientas.")
