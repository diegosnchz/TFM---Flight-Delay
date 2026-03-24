# CLAUDE.md - TFM: Análisis Predictivo de Retrasos Aéreos y Arbitraje EU261

## Contexto del Proyecto

Este es un Trabajo de Fin de Máster (TFM) para un Máster en Big Data e Inteligencia Artificial. El alumno es Diego, un junior data professional con experiencia en Power BI, DAX, Python y SQL. El TFM debe demostrar competencia técnica real, no código copiado sin entender.

**Título del TFM:**
"Análisis predictivo de retrasos aéreos en aerolíneas low-cost europeas: Evaluación de oportunidades de arbitraje regulatorio bajo la normativa EU261/2004"

**Hipótesis central:**
Es posible construir un modelo predictivo que identifique vuelos con alta probabilidad de retraso compensable (>3 horas) bajo la normativa EU261, generando un valor esperado positivo al comprar billetes baratos de forma estratégica.

---

## Estructura del Proyecto

```
tfm-eu261/
├── CLAUDE.md                    # Este archivo
├── README.md                    # Descripción del proyecto para GitHub
├── requirements.txt             # Dependencias Python
├── data/
│   ├── raw/                     # Datos crudos de ADRR/Kaggle (NO subir a git)
│   ├── processed/               # Datos limpios
│   └── external/                # Datos auxiliares (aeropuertos, distancias, precios)
├── notebooks/                   # Notebooks exploratorios (solo para exploración rápida)
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuración global, rutas, constantes
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py            # Carga y validación de datos crudos
│   │   ├── clean.py             # Limpieza y transformaciones
│   │   └── features.py          # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py             # Entrenamiento de modelos
│   │   ├── evaluate.py          # Métricas y evaluación
│   │   └── predict.py           # Predicción y escenarios de arbitraje
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py         # Gráficos del EDA
│   │   ├── model_plots.py       # Curvas ROC, matrices de confusión, SHAP
│   │   └── arbitrage_plots.py   # Visualizaciones del modelo de arbitraje
│   └── utils/
│       ├── __init__.py
│       ├── eu261.py             # Lógica de compensaciones EU261
│       └── geo.py               # Cálculo de distancias entre aeropuertos (Haversine)
├── outputs/
│   ├── figures/                 # Gráficos exportados para el TFM (PNG alta resolución)
│   ├── tables/                  # Tablas exportadas (CSV/LaTeX)
│   └── models/                  # Modelos serializados (joblib/pickle)
├── docs/
│   └── tfm/                     # Documento TFM
│       ├── tfm_eu261.docx       # Documento Word final
│       └── assets/              # Imágenes y recursos del documento
├── tests/                       # Tests unitarios básicos
│   └── test_eu261.py
└── Makefile                     # Comandos: make data, make train, make figures, make tfm
```

---

## Fuente de Datos Principal

### Eurocontrol ADRR (Aviation Data Repository for Research)
- URL: https://www.eurocontrol.int/dashboard/aviation-data-research
- Acceso: Registro gratuito en OneSky Online + solicitud de acceso al dataset
- Cobertura: Vuelos comerciales europeos, datos a nivel de vuelo individual
- Incluye: hora programada, hora real, aeropuerto origen/destino, aerolínea, tipo de aeronave

### Datos Auxiliares Necesarios
1. **Coordenadas de aeropuertos**: Para calcular distancias (necesario para bandas EU261)
   - Fuente: OpenFlights airports.dat (https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat)
   - Formato: CSV con IATA code, latitud, longitud
2. **Precios de billetes**: Promedios por ruta low-cost
   - Fuente: Estimaciones razonables basadas en datos públicos (Google Flights, Kiwi)
   - Rango típico: 15-80 EUR para low-cost intraeuropeas
3. **Datos meteorológicos** (opcional, enriquecimiento):
   - Fuente: Open-Meteo API (gratuita, histórica)
   - Variables: temperatura, precipitación, velocidad del viento por aeropuerto/fecha

### Fallback: Si ADRR no está disponible
Usar el dataset de Kaggle "European Flights Dataset" (https://www.kaggle.com/datasets/umerhaddii/european-flights-dataset) o similar. En ese caso, adaptar el script de ingesta pero mantener la misma estructura de features.

---

## Fases de Ejecución (Ejecutar en Orden)

### FASE 0: Setup del Entorno
```bash
make setup
```
- Crear virtualenv Python 3.10+
- Instalar dependencias de requirements.txt
- Crear estructura de directorios
- Verificar que los datos crudos existen en data/raw/

**requirements.txt debe incluir:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
scipy>=1.11
imbalanced-learn>=0.11
joblib>=1.3
geopy>=2.4
```

### FASE 1: Ingesta y Limpieza (`src/data/ingest.py` + `src/data/clean.py`)

**ingest.py:**
- Cargar datos crudos del ADRR
- Validar schema: columnas esperadas, tipos de datos
- Filtrar SOLO las 4 aerolíneas low-cost: Ryanair (FR), easyJet (U2), Wizz Air (W6), Vueling (VY)
- Log de registros totales vs filtrados
- Guardar en data/processed/flights_raw_filtered.parquet

**clean.py:**
- Calcular delay_minutes = (hora_real_llegada - hora_programada_llegada) en minutos
- Crear variable binaria: is_eu261_eligible = 1 si delay_minutes >= 180
- Eliminar vuelos cancelados (son otro tipo de compensación)
- Eliminar duplicados
- Tratar missing values (documentar decisión para cada columna)
- Crear features temporales: hour, day_of_week, month, is_weekend, is_summer (jun-sep)
- Merge con coordenadas de aeropuertos (OpenFlights)
- Calcular distance_km entre origen y destino (Haversine)
- Asignar banda EU261: <1500km = 250 EUR, 1500-3500km = 400 EUR, >3500km = 600 EUR
- Guardar en data/processed/flights_clean.parquet

### FASE 2: Feature Engineering (`src/data/features.py`)

Features a crear:
- **route**: concatenación origen_destino (ej: "CIY-HAJ")
- **airline_code**: código IATA de la aerolínea
- **aircraft_type**: tipo de avión (A320, B737, etc.)
- **aircraft_age**: edad del avión en años (si disponible en ADRR)
- **departure_hour_bin**: bloques horarios (madrugada 0-6, mañana 6-12, tarde 12-18, noche 18-24)
- **origin_delay_rate**: tasa histórica de retrasos EU261 del aeropuerto de origen (calculada sobre training set para evitar data leakage)
- **dest_delay_rate**: tasa histórica de retrasos EU261 del aeropuerto de destino
- **airline_delay_rate**: tasa histórica de retrasos EU261 de la aerolínea
- **route_delay_rate**: tasa histórica de retrasos EU261 de la ruta
- **day_of_year**: día del año (1-365) para capturar estacionalidad
- **is_holiday_period**: periodos vacacionales europeos (Navidad, Semana Santa, verano)

**IMPORTANTE sobre data leakage:**
- Las tasas históricas (origin_delay_rate, etc.) deben calcularse SOLO con datos del training set
- Usar encoders fitted en train y aplicados en test
- Documentar explícitamente este punto en el TFM (los tribunales lo valoran)

Guardar en data/processed/flights_features.parquet

### FASE 3: Análisis Exploratorio de Datos (EDA) (`src/visualization/eda_plots.py`)

Generar y exportar a outputs/figures/ los siguientes gráficos (alta resolución, 300 DPI, formato PNG):

1. **fig_01_delay_distribution.png**: Distribución de retrasos en minutos (histograma + KDE), con línea vertical en 180 min
2. **fig_02_eu261_rate_by_airline.png**: Tasa de retrasos EU261 por aerolínea (barplot horizontal)
3. **fig_03_eu261_rate_by_hour.png**: Tasa de retrasos EU261 por hora del día (lineplot)
4. **fig_04_eu261_rate_by_month.png**: Tasa de retrasos EU261 por mes (barplot)
5. **fig_05_top20_origin_airports.png**: Top 20 aeropuertos de origen con mayor tasa de retraso EU261
6. **fig_06_top20_dest_airports.png**: Top 20 aeropuertos de destino con mayor tasa de retraso EU261
7. **fig_07_aircraft_age_paradox.png**: Tasa de retraso vs edad del avión (scatterplot con tendencia)
8. **fig_08_heatmap_day_hour.png**: Heatmap día de la semana x hora con tasa de retraso
9. **fig_09_correlation_matrix.png**: Matriz de correlación de features numéricas
10. **fig_10_class_imbalance.png**: Gráfico de desbalance de clases (pie chart o barplot)

**Estilo de los gráficos:**
- Usar palette profesional consistente (sugiero: "muted" de seaborn o una palette personalizada)
- Títulos en español
- Labels claros, tamaño de fuente legible (12pt mínimo)
- Fondo blanco, sin grid excesivo
- Incluir anotaciones donde aporten valor (ej: porcentaje exacto en barplots)

### FASE 4: Modelado (`src/models/train.py`)

**Modelos a entrenar y comparar:**

1. **Regresión Logística** (baseline + explicabilidad)
   - Con regularización L2
   - StandardScaler en features numéricas
   - OneHotEncoder en features categóricas (con handle_unknown='ignore')

2. **Random Forest**
   - Para comparación y feature importance basada en impureza

3. **XGBoost** (modelo principal)
   - Tuning con GridSearchCV o RandomizedSearchCV
   - scale_pos_weight para manejar desbalance de clases
   - Early stopping con eval set

4. **LightGBM** (alternativa a XGBoost)
   - is_unbalance=True para desbalance

**Manejo del desbalance de clases (CRITICO - solo 0.62% positivos):**
- Probar: SMOTE, class_weight='balanced', scale_pos_weight, undersampling
- Comparar rendimiento con y sin cada técnica
- Documentar cuál funciona mejor y por qué

**Split de datos:**
- Train/Validation/Test: 70/15/15
- Stratified split para mantener proporción de clases
- Split temporal si los datos tienen suficiente rango de fechas (más realista)

**Pipeline de scikit-learn:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Definir preprocessor y modelo en un Pipeline limpio
# Esto asegura que no hay data leakage en el preprocesamiento
```

### FASE 5: Evaluación (`src/models/evaluate.py`)

**Métricas a calcular:**
- AUC-ROC (métrica principal para comparación)
- AUC-PR (Precision-Recall, más informativa con clases desbalanceadas)
- F1-Score (con threshold optimizado)
- Precision y Recall por separado
- Log Loss
- Matriz de confusión

**Gráficos de evaluación** (exportar a outputs/figures/):
- **fig_11_roc_curves.png**: Curvas ROC de todos los modelos superpuestas
- **fig_12_pr_curves.png**: Curvas Precision-Recall de todos los modelos
- **fig_13_confusion_matrix.png**: Matriz de confusión del mejor modelo (con threshold óptimo)
- **fig_14_shap_summary.png**: SHAP summary plot del mejor modelo
- **fig_15_shap_waterfall.png**: SHAP waterfall plot para el "escenario óptimo"
- **fig_16_feature_importance.png**: Feature importance comparativa (modelo vs SHAP)
- **fig_17_threshold_analysis.png**: Precision/Recall/F1 vs threshold (para elegir punto de corte)

**Análisis de la Paradoja de Simpson:**
- Comparar tasas brutas de retraso por aerolínea vs coeficientes del modelo controlando por ruta
- Generar tabla que demuestre que la aerolínea con mayor tasa bruta no necesariamente tiene mayor probabilidad intrínseca
- **fig_18_simpson_paradox.png**: Visualización de la paradoja

### FASE 6: Modelo de Arbitraje (`src/models/predict.py` + `src/visualization/arbitrage_plots.py`)

**Fórmula de Expected Value (EV) extendida:**
```
EV = P(retraso>=3h) * Compensacion_EU261 - Precio_Billete - Coste_Transporte_Aeropuerto - (Valor_Hora * Horas_Invertidas) - P(reclamacion_denegada) * Compensacion_EU261
```

**Variables del modelo de arbitraje:**
- `P(retraso>=3h)`: Output del modelo predictivo
- `Compensacion_EU261`: 250/400/600 EUR según distancia
- `Precio_Billete`: Estimado por ruta (usar promedio low-cost)
- `Coste_Transporte`: Estimación fija por aeropuerto (ej: 15-30 EUR transporte al aeropuerto)
- `Valor_Hora`: Configurable (usar salario mínimo España como baseline: ~8.87 EUR/h en 2025)
- `Horas_Invertidas`: Estimación por viaje (ej: 6-10 horas entre ida al aeropuerto, espera, vuelo, vuelta)
- `P(reclamacion_denegada)`: Tasa de rechazo de reclamaciones (estimar 20-40% por circunstancias extraordinarias)

**Escenarios a simular:**
1. **Escenario Óptimo**: Ruta + aerolínea + horario con mayor P(retraso), billete más barato
2. **Top 10 rutas rentables**: Ranking por EV positivo
3. **Análisis de sensibilidad**: Cómo cambia el EV al variar cada parámetro
4. **Break-even analysis**: P(retraso) mínima para EV > 0 dado un precio de billete

**Gráficos de arbitraje:**
- **fig_19_top_routes_ev.png**: Top rutas por Expected Value
- **fig_20_sensitivity_analysis.png**: Análisis de sensibilidad (spider chart o heatmap)
- **fig_21_breakeven.png**: Curva de break-even P(retraso) vs precio billete
- **fig_22_ev_simulation.png**: Distribución de EV con simulación Monte Carlo (10,000 iteraciones)

### FASE 7: Generación del Documento TFM (`docs/tfm/`)

**Generar un documento Word (.docx) completo con la siguiente estructura:**

#### Portada
- Título del TFM
- Autor: Diego Sánchez (ajustar apellido si es necesario)
- Máster en Big Data e Inteligencia Artificial - Euroformac
- Fecha: Junio 2026
- Tutor: [PENDIENTE DE RELLENAR]

#### Índice de contenidos (autogenerado)

#### Abstract (español + inglés)
- 250-300 palabras cada uno
- Resumen del problema, metodología, hallazgos principales y conclusiones
- Palabras clave: machine learning, retrasos aéreos, EU261, arbitraje regulatorio, regresión logística, XGBoost

#### 1. Introducción
- 1.1 Motivación y contexto
- 1.2 Objetivos del trabajo
- 1.3 Alcance y limitaciones
- 1.4 Estructura del documento

#### 2. Estado del Arte
- 2.1 Machine Learning aplicado a la predicción de retrasos aéreos
- 2.2 Normativa EU261/2004: marco legal y jurisprudencia
- 2.3 Comportamiento operativo de aerolíneas low-cost europeas
- 2.4 Teoría de juegos y arbitraje de riesgos regulatorios
- 2.5 Empresas de reclamación (Claim Agencies): FlightRight, AirHelp, etc.
- **NOTA**: Este capítulo necesita bibliografía real. Insertar placeholders [REF_XX] donde Diego debe buscar y añadir citas de papers reales. Sugerir papers concretos a buscar en Google Scholar.

#### 3. Metodología
- 3.1 Fuentes de datos y adquisición
- 3.2 Preprocesamiento y limpieza
- 3.3 Feature engineering
- 3.4 Selección de modelos y justificación
- 3.5 Estrategia de validación y manejo de desbalance
- 3.6 Métricas de evaluación
- 3.7 Modelo de Expected Value para arbitraje

#### 4. Análisis Exploratorio de Datos
- Insertar los gráficos generados en FASE 3
- Narrar hallazgos clave: tasa base de 0.62%, diferencias por aerolínea, paradoja de edad de aviones, aeropuertos problemáticos
- Cada gráfico debe tener: título, descripción del hallazgo, implicación para el modelo

#### 5. Resultados del Modelado
- 5.1 Comparativa de modelos (tabla con métricas)
- 5.2 Análisis del mejor modelo
- 5.3 Interpretabilidad: SHAP values y feature importance
- 5.4 Paradoja de Simpson: análisis por aerolínea
- Insertar gráficos de FASE 5

#### 6. Modelo de Arbitraje y Simulación
- 6.1 Definición del modelo de Expected Value extendido
- 6.2 Escenario óptimo
- 6.3 Top rutas rentables
- 6.4 Análisis de sensibilidad
- 6.5 Simulación Monte Carlo
- Insertar gráficos de FASE 6

#### 7. Discusión
- 7.1 Viabilidad práctica del modelo de arbitraje
- 7.2 Limitaciones del estudio (sección CRITICA, ver abajo)
- 7.3 Implicaciones éticas y legales
- 7.4 Posibles respuestas de las aerolíneas

#### 8. Conclusiones y Trabajo Futuro
- 8.1 Conclusiones principales
- 8.2 Contribuciones del trabajo
- 8.3 Líneas de trabajo futuro

#### Bibliografía
- Formato APA 7ª edición
- Incluir placeholders con las referencias sugeridas

#### Anexos
- A. Código fuente principal (fragmentos relevantes, no todo el repo)
- B. Tablas completas de resultados
- C. Glosario de términos

### Limitaciones del Estudio (OBLIGATORIAS en el TFM)

Estas limitaciones son CRITICAS y un tribunal las va a preguntar. Incluir TODAS:

1. **Circunstancias extraordinarias**: Las aerolíneas pueden (y de hecho lo hacen) alegar causas de fuerza mayor (meteorología, huelgas ATC, emergencias sanitarias) para denegar la compensación. El modelo no puede predecir esto.

2. **Tasa de rechazo de reclamaciones**: En la vida real, entre el 20-40% de reclamaciones son rechazadas inicialmente. El EV real es menor que el teórico.

3. **Coste de oportunidad temporal**: El modelo asume que el "arbitrajista" tiene tiempo ilimitado. En realidad, ir al aeropuerto, esperar, volar y volver consume un día entero mínimo.

4. **Costes de reclamación**: Tiempo en gestionar la reclamación (cartas, formularios, posible mediación/juicio). Las claim agencies cobran 25-35% de la compensación.

5. **Cancelaciones previas al vuelo**: Si la aerolínea cancela el vuelo con >14 días de antelación, no hay compensación. El modelo no puede predecir cancelaciones anticipadas.

6. **Adaptación de las aerolíneas**: Si este tipo de arbitraje se popularizase, las aerolíneas podrían ajustar horarios, subir precios en rutas problemáticas o mejorar puntualidad en esas rutas (teoría de juegos).

7. **Calidad de los datos**: Los datos de ADRR pueden no coincidir exactamente con lo que el pasajero experimenta (diferencias entre hora de llegada a parking vs hora de apertura de puertas).

8. **Sesgo de supervivencia**: Solo se analizan vuelos operados, no los cancelados antes de operar.

9. **Periodo de datos limitado**: El modelo se entrena con datos históricos que pueden no reflejar el comportamiento futuro (COVID, nuevas regulaciones, cambios de flota).

10. **No se consideran vuelos con escala**: El EU261 tiene reglas diferentes para vuelos con conexión.

---

## Reglas de Estilo y Formato

### Código Python
- Docstrings en todas las funciones (estilo Google)
- Type hints en parámetros y returns
- Logging con módulo `logging` (no prints)
- Constantes en MAYÚSCULAS en config.py
- Paths con pathlib, nunca strings hardcodeados
- Seed fijo (42) para reproducibilidad en TODOS los modelos y splits

### Documento TFM
- Tono académico formal, tercera persona ("se ha analizado", "se observa")
- Sin emojis, sin lenguaje coloquial
- Cada gráfico referenciado en el texto con "como se muestra en la Figura X"
- Tablas numeradas y referenciadas
- NO usar guiones largos (em dashes --) en ningún texto generado. Usar comas, puntos o puntos y comas en su lugar.

### Gráficos
- Resolución: 300 DPI mínimo
- Formato: PNG para figuras del TFM
- Tamaño: figsize=(12, 8) por defecto, ajustar según contenido
- Guardar con bbox_inches='tight'
- Paleta consistente en todo el proyecto

---

## Makefile

```makefile
.PHONY: setup data features eda train evaluate arbitrage figures tfm all clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	mkdir -p data/{raw,processed,external} outputs/{figures,tables,models} docs/tfm/assets

data:
	python -m src.data.ingest
	python -m src.data.clean

features:
	python -m src.data.features

eda:
	python -m src.visualization.eda_plots

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

arbitrage:
	python -m src.models.predict

figures: eda evaluate arbitrage

tfm:
	python docs/tfm/generate_tfm.py

all: data features eda train evaluate arbitrage figures tfm

clean:
	rm -rf data/processed/* outputs/* .venv
```

---

## Orden de Ejecución para Claude Code

1. Leer este CLAUDE.md completo
2. Crear la estructura de directorios
3. Crear requirements.txt y Makefile
4. Crear src/config.py con constantes
5. Crear src/utils/eu261.py y src/utils/geo.py
6. **PREGUNTAR A DIEGO**: "¿Ya tienes los datos de ADRR descargados? ¿En qué formato están y dónde los has puesto?" -- NO avanzar sin datos reales
7. Una vez confirmados los datos: ejecutar FASE 1 a FASE 6 en orden
8. FASE 7: Generar documento TFM
9. Revisar outputs y ajustar

---

## Notas Importantes

- **NO inventar datos**: Si los datos de ADRR no están disponibles, crear un script de ingesta con schema esperado y placeholder, pero NO generar datos sintéticos sin avisar.
- **Explicar cada decisión**: Diego necesita entender lo que se hace. Añadir comentarios explicativos en el código, no solo código funcional.
- **Reproducibilidad**: Todo debe poder ejecutarse de cero con `make all` una vez los datos estén en data/raw/.
- **Git-ready**: Incluir .gitignore que excluya data/raw/, data/processed/, outputs/models/, .venv/
