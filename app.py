import os
import time
import math
import tempfile
import threading
import datetime

import av
import cv2
import pandas as pd
import mediapipe as mp
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF

from streamlit_webrtc import webrtc_streamer


# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

st.set_page_config(
    page_title="zenithz",
    page_icon="🦘",
    layout="wide"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker_lite.task"
)


# =====================================================
# 2. ESTADOS DE LA APLICACION
# =====================================================

if "ficha_completa" not in st.session_state:
    st.session_state.ficha_completa = False

if "perfil_evaluacion" not in st.session_state:
    st.session_state.perfil_evaluacion = {}

if "historial_saltos" not in st.session_state:
    st.session_state.historial_saltos = pd.DataFrame(columns=["Intento", "Altura (cm)", "Potencia (W)"])

if "resultado_video" not in st.session_state:
    st.session_state.resultado_video = None

if "identidad_video" not in st.session_state:
    st.session_state.identidad_video = None

if "registro_camara" not in st.session_state:
    st.session_state.registro_camara = {
        "lock": threading.Lock(),
        "muestras": [],
        "inicio": None,
        "ultima_muestra": -1.0,
        "grabando": False
    }

registro = st.session_state.registro_camara


# =====================================================
# 3. FICHA DE CARACTERIZACION DEPORTIVA
# =====================================================

def mostrar_ficha():
    st.markdown("## A. Datos Personales")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: nombre = st.text_input("Nombre completo", placeholder="Ej: Juan Pérez")
    with col2: edad = st.number_input("Edad", min_value=10, max_value=100, value=21, step=1)
    with col3: estatura = st.number_input("Estatura (cm)", min_value=100, max_value=250, value=161, step=1)
    with col4: peso = st.number_input("Peso (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.1)
    with col5: fecha_nacimiento = st.date_input("Nacimiento", value=datetime.date(2005, 9, 5), min_value=datetime.date(1950, 1, 1))

    st.markdown("## B. Experiencia deportiva general")
    experiencia_general = st.text_area("Describe tu experiencia previa y actual", placeholder="Ej: Básquetbol competitivo...")

    st.markdown("## C. Competencia y objetivos deportivos")
    competencias = st.text_area("¿Participas habitualmente en competencias deportivas?", placeholder="Describe tu experiencia...")
    temporada = st.text_area("¿En qué periodo del año tienes mayor carga deportiva?", placeholder="Escribe tu respuesta...")
    objetivo_deportivo = st.text_area("¿Cuál es tu principal objetivo deportivo actualmente?", placeholder="Describe tus objetivos...")

    st.markdown("## D. Características del entrenamiento")
    duracion_sesion = st.text_area("¿Cuánto dura una sesión de entrenamiento?", placeholder="Escribe tu respuesta...")
    grupos_musculares = st.text_area("¿Qué grupos musculares trabajas habitualmente?", placeholder="Ej: cuádriceps, glúteos...")
    ejercicios_fuerza = st.text_area("¿Qué ejercicios utilizas para fuerza de piernas?", placeholder="Ej: sentadillas...")
    cambios_direccion = st.text_area("¿Realizas ejercicios de velocidad/cambios de dirección?", placeholder="Describe...")

    st.markdown("## E. Experiencia pliométrica")
    tipos_saltos = st.text_area("¿Qué tipos de saltos realizas habitualmente?", placeholder="Describe tus ejercicios...")
    tipo_apoyo = st.text_area("¿Realizas saltos con apoyo de una pierna o ambas?", placeholder="Escribe tu respuesta...")
    dificultad_saltos = st.text_area("¿Qué dificultades experimentas al saltar?", placeholder="Describe tu experiencia...")

    st.markdown("## F. Percepción de la técnica del CMJ")
    conocimiento_cmj = st.text_area("¿Cómo describirías la técnica correcta del CMJ?", placeholder="Escribe lo que conoces...")
    profundidad = st.text_area("¿Cómo describirías tu profundidad en el descenso?", placeholder="Describe tu movimiento...")
    aterrizaje = st.text_area("¿Cómo percibes tu técnica de aterrizaje?", placeholder="Describe...")
    control_corporal = st.text_area("¿Mantienes control de tu cuerpo en el salto?", placeholder="Escribe tu respuesta...")
    dificultad_tecnica = st.text_area("¿Qué parte de la ejecución te resulta más difícil?", placeholder="Escribe...")

    st.markdown("## G. Antecedentes de evaluación física")
    evaluaciones = st.text_area("¿Has realizado evaluaciones de salto?", placeholder="Describe tu experiencia...")
    instrumentos = st.text_area("¿Has utilizado plataformas de fuerza u otros?", placeholder="Escribe tu respuesta...")
    resultados_previos = st.text_area("¿Conoces algún resultado anterior de salto?", placeholder="Puedes escribir el resultado...")
    seguimiento = st.text_area("¿Te gustaría comparar resultados con futuras evaluaciones?", placeholder="Escribe...")

    st.markdown("## H. Configuración del análisis")
    perspectiva = st.radio("Indica la perspectiva desde la que grabarás el salto", ["Lateral", "Frontal", "Otra perspectiva"], index=0)

    if st.button("Guardar ficha e iniciar evaluación CMJ", type="primary", use_container_width=True):
        st.session_state.perfil_evaluacion = {
            "Nombre": nombre, "Edad": edad, 
            "Estatura (cm)": estatura, "Peso (kg)": peso,
            "Fecha de nacimiento": fecha_nacimiento.strftime("%d/%m/%Y") if fecha_nacimiento else "",
            "Experiencia general": experiencia_general, "Competencias": competencias,
            "Temporada": temporada, "Objetivo": objetivo_deportivo,
            "Duración sesión": duracion_sesion, "Grupos musculares": grupos_musculares,
            "Fuerza extremidades": ejercicios_fuerza, "Velocidad y dirección": cambios_direccion,
            "Tipos de saltos": tipos_saltos, "Tipo de apoyo": tipo_apoyo,
            "Dificultades de salto": dificultad_saltos, "Conocimiento CMJ": conocimiento_cmj,
            "Profundidad": profundidad, "Aterrizaje": aterrizaje,
            "Control corporal": control_corporal, "Dificultad técnica": dificultad_tecnica,
            "Evaluaciones anteriores": evaluaciones, "Instrumentos": instrumentos,
            "Resultados previos": resultados_previos, "Seguimiento": seguimiento,
            "Perspectiva de cámara": perspectiva
        }
        st.session_state.ficha_completa = True
        st.rerun()


# =====================================================
# 4. CONEXIONES DEL ESQUELETO
# =====================================================

CONEXIONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32)
]


# =====================================================
# 5. MODELO DE MEDIAPIPE
# =====================================================

def crear_opciones():
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )

@st.cache_resource
def cargar_modelo(ruta):
    return mp.tasks.vision.PoseLandmarker.create_from_options(crear_opciones())


# =====================================================
# 6. FUNCIONES BIOMECANICAS Y MATEMATICAS
# =====================================================

def punto_visible(punto):
    visibilidad = getattr(punto, "visibility", None)
    if visibilidad is None: return True
    return float(visibilidad) >= 0.4

def calcular_kasr(puntos):
    rod_izq, rod_der = puntos[25], puntos[26]
    tob_izq, tob_der = puntos[27], puntos[28]
    if not all(punto_visible(p) for p in [rod_izq, rod_der, tob_izq, tob_der]): return None
    sep_rodillas = abs(rod_izq.x - rod_der.x)
    sep_tobillos = abs(tob_izq.x - tob_der.x)
    if sep_tobillos < 0.01: return None
    return sep_rodillas / sep_tobillos

def calcular_angulo(p1, p2, p3):
    angulo = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x))
    angulo = abs(angulo)
    if angulo > 180.0: angulo = 360.0 - angulo
    return angulo

def extraer_metricas(puntos, vista_frontal):
    hombro_izq, hombro_der = puntos[11], puntos[12]
    cadera_izq, cadera_der = puntos[23], puntos[24]
    rodilla_izq, rodilla_der = puntos[25], puntos[26]
    tobillo_izq, tobillo_der = puntos[27], puntos[28]
    punta_izq, punta_der = puntos[31], puntos[32]
    cabeza = puntos[0]
    
    if not all(punto_visible(p) for p in [hombro_izq, hombro_der, cadera_izq, cadera_der, rodilla_izq, rodilla_der, tobillo_izq, tobillo_der]):
        return None
        
    kasr = calcular_kasr(puntos) if vista_frontal else None
    ang_rodilla_medio = (calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq) + calcular_angulo(cadera_der, rodilla_der, tobillo_der)) / 2.0
    ang_cadera_medio = (calcular_angulo(hombro_izq, cadera_izq, rodilla_izq) + calcular_angulo(hombro_der, cadera_der, rodilla_der)) / 2.0
    
    x_cadera = (cadera_izq.x + cadera_der.x) / 2.0
    y_cadera = (cadera_izq.y + cadera_der.y) / 2.0
    
    y_pie_mas_bajo = max(tobillo_izq.y, tobillo_der.y, punta_izq.y, punta_der.y)
    
    estatura_norm = abs(y_pie_mas_bajo - cabeza.y)
    if estatura_norm < 0.1: estatura_norm = 0.1
        
    return {
        "kasr": kasr,
        "angulo_rodilla": ang_rodilla_medio,
        "angulo_cadera": ang_cadera_medio,
        "x_cadera": x_cadera,
        "y_cadera": y_cadera,
        "y_pie": y_pie_mas_bajo,
        "estatura_norm": estatura_norm
    }

def dibujar_esqueleto(imagen, puntos, estatura_cm, estatura_norm, y_suelo_px, x_centro_px, trayectoria=None):
    alto, ancho = imagen.shape[:2]
    
    # 1. SISTEMA DE COORDENADAS FIJO
    if estatura_cm and estatura_norm and estatura_norm > 0 and y_suelo_px is not None and x_centro_px is not None:
        pixeles_por_cm = (estatura_norm * alto) / estatura_cm
        
        cv2.line(imagen, (0, y_suelo_px), (ancho, y_suelo_px), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(imagen, (x_centro_px, 0), (x_centro_px, alto), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(imagen, "(0,0)", (x_centro_px + 5, y_suelo_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for cm in range(10, 150, 10):
            y_linea = int(y_suelo_px - (cm * pixeles_por_cm))
            if 0 <= y_linea <= alto:
                cv2.line(imagen, (0, y_linea), (ancho, y_linea), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.putText(imagen, f"{cm}y", (5, y_linea - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            x_linea_der = int(x_centro_px + (cm * pixeles_por_cm))
            x_linea_izq = int(x_centro_px - (cm * pixeles_por_cm))
            
            if x_linea_der <= ancho:
                cv2.line(imagen, (x_linea_der, 0), (x_linea_der, alto), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            if x_linea_izq >= 0:
                cv2.line(imagen, (x_linea_izq, 0), (x_linea_izq, alto), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
        x_cadera_px = int((puntos[23].x + puntos[24].x) / 2 * ancho)
        y_cadera_px = int((puntos[23].y + puntos[24].y) / 2 * alto)
        coord_x_cm = (x_cadera_px - x_centro_px) / pixeles_por_cm
        coord_y_cm = (y_suelo_px - y_cadera_px) / pixeles_por_cm
        texto_coord = f"CM({coord_x_cm:.1f}, {coord_y_cm:.1f})"
        cv2.putText(imagen, texto_coord, (x_cadera_px + 15, y_cadera_px), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 2. TRAYECTORIA Y ESQUELETO
    if trayectoria is not None and len(trayectoria) > 1:
        for i in range(1, len(trayectoria)):
            p1 = (int(trayectoria[i-1][0] * ancho), int(trayectoria[i-1][1] * alto))
            p2 = (int(trayectoria[i][0] * ancho), int(trayectoria[i][1] * alto))
            cv2.line(imagen, p1, p2, (255, 0, 255), 4)
            
    for inicio, final in CONEXIONES:
        p1, p2 = puntos[inicio], puntos[final]
        if punto_visible(p1) and punto_visible(p2):
            cv2.line(imagen, (int(p1.x * ancho), int(p1.y * alto)), (int(p2.x * ancho), int(p2.y * alto)), (0, 255, 0), 3)
            
    for punto in puntos:
        if punto_visible(punto):
            x, y = int(punto.x * ancho), int(punto.y * alto)
            if 0 <= x < ancho and 0 <= y < alto:
                cv2.circle(imagen, (x, y), 5, (0, 0, 255), -1)
                
    return imagen


# =====================================================
# 7. GENERADOR DE REPORTES (PDF)
# =====================================================

def generar_reporte_pdf(perfil, altura, tiempo, potencia, potencia_rel, rodilla, cadera, diagnostico):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado Oficial
    pdf.set_font("Arial", "B", 18)
    pdf.cell(200, 10, txt="Reporte Biomecanico CMJ - zenithz", ln=True, align="C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(200, 8, txt="Universidad Santo Tomas - Ciencias del Deporte", ln=True, align="C")
    pdf.ln(10)
    
    # Datos del Paciente
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, txt=" 1. Antecedentes del Deportista", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    
    nombre_paciente = perfil.get('Nombre', 'Sin nombre')
    pdf.cell(0, 8, txt=f"Nombre: {nombre_paciente}", ln=True)
    pdf.cell(0, 8, txt=f"Edad: {perfil.get('Edad', 'N/A')} anos | Estatura: {perfil.get('Estatura (cm)', 'N/A')} cm | Peso: {perfil.get('Peso (kg)', 'N/A')} kg", ln=True)
    pdf.ln(5)
    
    # Resultados Cinemáticos
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt=" 2. Resultados de Rendimiento (Potencia y Vuelo)", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, txt=f"Altura Maxima Estimada: {altura:.1f} cm", ln=True)
    pdf.cell(0, 8, txt=f"Tiempo en el Aire: {tiempo:.3f} segundos", ln=True)
    pdf.cell(0, 8, txt=f"Potencia Pico (Sayers): {potencia:.0f} W", ln=True)
    pdf.cell(0, 8, txt=f"Potencia Relativa: {potencia_rel:.1f} W/kg", ln=True)
    pdf.ln(5)

    # Perfil Articular
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt=" 3. Perfil Articular y Riesgo de Lesion", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, txt=f"Amplitud Maxima Flexion de Rodilla: {rodilla:.1f} grados", ln=True)
    pdf.cell(0, 8, txt=f"Amplitud Maxima Flexion de Cadera: {cadera:.1f} grados", ln=True)
    
    diag_limpio = diagnostico.replace("🔴", "").replace("🟡", "").replace("🟢", "")
    pdf.cell(0, 8, txt=f"Control Motor Frontal (KASR): {diag_limpio}", ln=True)
    pdf.ln(15)

    # Firma
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt=f"Evaluacion generada el {datetime.date.today().strftime('%d/%m/%Y')} mediante zenithz AI.", ln=True, align="C")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp.name)
    
    return pdf_bytes, nombre_paciente


# =====================================================
# 8. PROCESAMIENTO EN DIRECTO Y VIDEO
# =====================================================

def crear_procesador(modelo, registro, vista_frontal, estatura_cm):
    ultimo_timestamp = [0]
    trayectoria_en_vivo = []
    ref_fija = {"y_suelo": None, "x_centro": None, "estatura_norm": None}

    def procesar_frame(frame):
        imagen = frame.to_ndarray(format="bgr24")
        alto, ancho = imagen.shape[:2]
        if ancho > 640:
            nuevo_alto = int(alto * 640 / ancho)
            imagen = cv2.resize(imagen, (640, nuevo_alto))

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(time.monotonic_ns() // 1_000_000, ultimo_timestamp[0] + 1)
        ultimo_timestamp[0] = timestamp
        resultado = modelo.detect_for_video(mp_imagen, timestamp)

        if resultado.pose_landmarks:
            puntos = resultado.pose_landmarks[0]
            metricas = extraer_metricas(puntos, vista_frontal)
            
            if metricas is not None:
                trayectoria_en_vivo.append((metricas["x_cadera"], metricas["y_cadera"]))
                if len(trayectoria_en_vivo) > 45: trayectoria_en_vivo.pop(0)
                
                curr_y_suelo = int(max(puntos[27].y, puntos[28].y, puntos[31].y, puntos[32].y) * alto)
                curr_x_centro = int(((puntos[31].x + puntos[32].x) / 2) * ancho)
                
                with registro["lock"]:
                    if not registro["grabando"]:
                        ref_fija["y_suelo"] = curr_y_suelo
                        ref_fija["x_centro"] = curr_x_centro
                        ref_fija["estatura_norm"] = metricas["estatura_norm"]

                    if registro["grabando"]:
                        ahora = time.monotonic()
                        if registro["inicio"] is None: registro["inicio"] = ahora
                        tiempo = ahora - registro["inicio"]
                        if (tiempo - registro["ultima_muestra"]) >= 0.1:
                            registro["muestras"].append({
                                "Tiempo (s)": round(tiempo, 3),
                                "KASR": round(metricas["kasr"], 4) if metricas["kasr"] else None,
                                "Angulo Rodilla": round(metricas["angulo_rodilla"], 2),
                                "Angulo Cadera": round(metricas["angulo_cadera"], 2),
                                "Y Cadera": round(metricas["y_cadera"], 4),
                                "Y Pie": round(metricas["y_pie"], 4),
                                "Estatura Norm": round(metricas["estatura_norm"], 4)
                            })
                            registro["ultima_muestra"] = tiempo
                            if len(registro["muestras"]) > 3000: registro["muestras"].pop(0)
                            
            imagen = dibujar_esqueleto(
                imagen, puntos, estatura_cm, 
                ref_fija["estatura_norm"], ref_fija["y_suelo"], ref_fija["x_centro"], 
                trayectoria_en_vivo
            )
            cv2.putText(imagen, "SISTEMA ANCLADO (0,0)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            trayectoria_en_vivo.clear()
            cv2.putText(imagen, "BUSCANDO CUERPO...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return av.VideoFrame.from_ndarray(imagen, format="bgr24")
    return procesar_frame


def analizar_video(archivo, vista_frontal, estatura_cm):
    extension = os.path.splitext(archivo.name)[1] or ".mp4"
    ruta_temporal = None
    captura = None
    progreso = st.progress(0)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temporal:
            temporal.write(archivo.getvalue())
            ruta_temporal = temporal.name

        captura = cv2.VideoCapture(ruta_temporal)
        if not captura.isOpened(): raise ValueError("No se pudo abrir el video.")

        fps = captura.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps < 1: fps = 30.0
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        
        resultados = []
        trayectoria_historica = []
        frame_pico_salto = None
        min_y_cadera_pico = 999.0
        
        indice = 0
        ultimo_timestamp = -1
        ultima_muestra = -1.0
        
        ref_y_suelo = None
        ref_x_centro = None
        ref_estatura_norm = None
        frames_calibracion = 0

        with mp.tasks.vision.PoseLandmarker.create_from_options(crear_opciones()) as detector:
            while True:
                correcto, frame = captura.read()
                if not correcto: break

                alto, ancho = frame.shape[:2]
                if ancho > 640:
                    nuevo_alto = int(alto * 640 / ancho)
                    frame = cv2.resize(frame, (640, nuevo_alto))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int(indice * 1000 / fps), ultimo_timestamp + 1)
                ultimo_timestamp = timestamp
                deteccion = detector.detect_for_video(mp_imagen, timestamp)

                if deteccion.pose_landmarks:
                    puntos = deteccion.pose_landmarks[0]
                    metricas = extraer_metricas(puntos, vista_frontal)
                    
                    if metricas is not None:
                        curr_y_suelo = int(max(puntos[27].y, puntos[28].y, puntos[31].y, puntos[32].y) * alto)
                        curr_x_centro = int(((puntos[31].x + puntos[32].x) / 2) * ancho)
                        
                        if frames_calibracion < 10:
                            if ref_y_suelo is None or curr_y_suelo > ref_y_suelo:
                                ref_y_suelo = curr_y_suelo
                                ref_x_centro = curr_x_centro
                                ref_estatura_norm = metricas["estatura_norm"]
                            frames_calibracion += 1

                        trayectoria_historica.append((metricas["x_cadera"], metricas["y_cadera"]))
                        
                        tiempo = indice / fps
                        if tiempo - ultima_muestra >= 0.05:
                            resultados.append({
                                "Tiempo (s)": round(tiempo, 3),
                                "KASR": round(metricas["kasr"], 4) if metricas["kasr"] else None,
                                "Angulo Rodilla": round(metricas["angulo_rodilla"], 2),
                                "Angulo Cadera": round(metricas["angulo_cadera"], 2),
                                "Y Cadera": round(metricas["y_cadera"], 4),
                                "Y Pie": round(metricas["y_pie"], 4),
                                "Estatura Norm": round(metricas["estatura_norm"], 4)
                            })
                            ultima_muestra = tiempo

                    frame_dibujado = dibujar_esqueleto(frame.copy(), puntos, estatura_cm, ref_estatura_norm, ref_y_suelo, ref_x_centro, trayectoria_historica)
                    
                    if metricas is not None and metricas["y_cadera"] < min_y_cadera_pico:
                        min_y_cadera_pico = metricas["y_cadera"]
                        frame_pico_salto = frame_dibujado.copy()

                indice += 1
                if total > 0 and indice % 10 == 0: progreso.progress(min(indice / total, 1.0))

        progreso.progress(1.0)
        return pd.DataFrame(resultados), frame_pico_salto, indice, fps

    finally:
        if captura is not None: captura.release()
        if ruta_temporal and os.path.exists(ruta_temporal): os.remove(ruta_temporal)
        progreso.empty()


# =====================================================
# 9. MOSTRAR RESULTADOS FINALES Y DASHBOARD
# =====================================================

def mostrar_resultados(df, nombre):
    if df.empty or 'Angulo Rodilla' not in df.columns:
        st.warning("Datos insuficientes para el análisis.")
        return

    df_clean = df.dropna(subset=['Y Cadera', 'Angulo Rodilla', 'Angulo Cadera']).reset_index(drop=True)
    if df_clean.empty:
        st.warning("Datos inconsistentes.")
        return

    estatura_cm = st.session_state.perfil_evaluacion.get("Estatura (cm)", 161)
    peso_kg = st.session_state.perfil_evaluacion.get("Peso (kg)", 60.0)

    # METODO: RASTREO ESPACIAL DEL CENTRO DE MASA
    y_pico_salto = df_clean['Y Cadera'].min() 
    estatura_norm_calibrada = df_clean['Estatura Norm'].max()
    factor_conversion = estatura_cm / estatura_norm_calibrada if estatura_norm_calibrada > 0 else 0
    y_cadera_reposo = df_clean['Y Cadera'].iloc[:5].mean()
    
    altura_final = max(0, (y_cadera_reposo - y_pico_salto) * factor_conversion)
    altura_metros = altura_final / 100.0
    tiempo_vuelo_real = math.sqrt((8 * altura_metros) / 9.81) if altura_metros > 0 else 0

    potencia_watts = (60.7 * altura_final) + (45.3 * peso_kg) - 2055
    if potencia_watts < 0: potencia_watts = 0
    potencia_relativa = potencia_watts / peso_kg if peso_kg > 0 else 0

    idx_profundo = df_clean['Angulo Rodilla'].idxmin()
    flex_max_rodilla = df_clean.loc[idx_profundo, 'Angulo Rodilla']
    flex_max_cadera = df_clean['Angulo Cadera'].min() 

    kasr_profundo = df_clean.loc[idx_profundo, 'KASR'] if 'KASR' in df_clean.columns else None
    if pd.isna(kasr_profundo): diagnostico = "No evaluable"
    elif kasr_profundo < 0.85: diagnostico = "Tendencia a VALGO 🔴"
    elif kasr_profundo > 1.15: diagnostico = "Tendencia a VARO 🟡"
    else: diagnostico = "Alineación NEUTRA 🟢"

    # HISTORIAL
    st.markdown("### 💾 Registro de Saltos")
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("➕ Guardar salto en el Historial", type="primary", use_container_width=True):
            nuevo_salto = pd.DataFrame({
                "Intento": [f"Salto {len(st.session_state.historial_saltos) + 1}"],
                "Altura (cm)": [round(altura_final, 2)],
                "Potencia (W)": [round(potencia_watts, 0)]
            })
            st.session_state.historial_saltos = pd.concat([st.session_state.historial_saltos, nuevo_salto], ignore_index=True)
            st.success("¡Salto registrado!")

    if not st.session_state.historial_saltos.empty:
        st.markdown("#### 📊 Comparativa de Saltos")
        fig = go.Figure(data=[
            go.Bar(
                x=st.session_state.historial_saltos["Intento"],
                y=st.session_state.historial_saltos["Altura (cm)"],
                text=st.session_state.historial_saltos["Altura (cm)"],
                textposition='inside',
                marker_color='#5C8CE6',
                textfont=dict(color='#FFD700', size=24, family="Arial Black")
            )
        ])
        fig.update_layout(
            plot_bgcolor="#E5E5E5", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Altura (cm)", gridcolor="white", range=[0, max(st.session_state.historial_saltos["Altura (cm)"])*1.3]),
            margin=dict(l=20, r=20, t=30, b=20), height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("🗑️ Borrar Historial"):
            st.session_state.historial_saltos = pd.DataFrame(columns=["Intento", "Altura (cm)", "Potencia (W)"])
            st.rerun()

    st.divider()

    # METRICAS
    st.markdown("#### 🚀 Métricas de Rendimiento")
    c1, c2, c3 = st.columns(3)
    c1.metric("Altura Máxima", f"{altura_final:.1f} cm", "Tracking de Centro de Masa")
    c2.metric("Tiempo de Vuelo", f"{tiempo_vuelo_real:.3f} s", "Física Cinemática")
    c3.metric("Potencia Pico", f"{potencia_watts:.0f} W", f"Relativa: {potencia_relativa:.1f} W/kg")

    st.markdown("#### 📐 Métricas Articulares")
    c4, c5, c6 = st.columns(3)
    c4.metric("Alineación Rodillas", diagnostico, "Índice KASR")
    c5.metric("Flexión Máx. Rodilla", f"{flex_max_rodilla:.1f}°", "Amplitud articular")
    c6.metric("Flexión Máx. Cadera", f"{flex_max_cadera:.1f}°", "Cierre de tronco")
    
    st.divider()
    
    # FEEDBACK IA
    st.markdown("### 🧠 Análisis Clínico Automático")
    col_fb1, col_fb2, col_fb3 = st.columns(3)
    
    with col_fb1:
        if flex_max_rodilla < 85:
            st.error("**Técnica Excéntrica:**\n\n⚠️ Flexión excesiva. Bajar a menos de 85° disipa la energía elástica.")
        elif flex_max_rodilla > 115:
            st.warning("**Técnica Excéntrica:**\n\n⚠️ Flexión insuficiente. Al bajar tan poco (>115°), no reclutas fibras rápidas.")
        else:
            st.success("**Técnica Excéntrica:**\n\n✅ Excelente profundidad de flexión. Estás en el rango óptimo (90°-110°).")

    with col_fb2:
        if diagnostico == "Tendencia a VALGO 🔴":
            st.error("**Riesgo Articular:**\n\n⚠️ Peligro de Valgo. Se sugiere trabajo compensatorio de Glúteo Medio para proteger ligamentos.")
        elif diagnostico == "Tendencia a VARO 🟡":
            st.warning("**Riesgo Articular:**\n\n🟡 Tendencia a Varo. Las rodillas apuntan hacia afuera en exceso.")
        elif diagnostico == "Alineación NEUTRA 🟢":
            st.success("**Riesgo Articular:**\n\n✅ Alineación perfecta. Excelente control motor frontal.")
        else:
            st.info("**Riesgo Articular:**\n\nNo se detectó el Valgo/Varo (Solo disponible en vista frontal).")

    with col_fb3:
        if potencia_relativa < 40:
            st.warning("**Perfil de Fuerza:**\n\n🟡 Potencia baja (< 40 W/kg). Se recomienda pliometría básica y fuerza máxima.")
        elif potencia_relativa >= 55:
            st.success("**Perfil de Fuerza:**\n\n🔥 Nivel Atleta de Élite (> 55 W/kg). Excelente desarrollo explosivo.")
        else:
            st.success("**Perfil de Fuerza:**\n\n✅ Buena producción de potencia. Parámetros estables (40 - 55 W/kg).")

    st.divider()

    # DESCARGAS
    st.markdown("### 📑 Exportar Informe del Paciente")
    pdf_bytes, nom_paciente = generar_reporte_pdf(
        st.session_state.perfil_evaluacion, 
        altura_final, tiempo_vuelo_real, potencia_watts, potencia_relativa, 
        flex_max_rodilla, flex_max_cadera, diagnostico
    )
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📄 Descargar Reporte Clínico (PDF)",
            data=pdf_bytes,
            file_name=f"Reporte_CMJ_{nom_paciente.replace(' ', '_')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_d2:
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Descargar Base de Datos (CSV)",
            data=csv,
            file_name=f"Datos_Tracking_{nom_paciente.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.divider()

    st.subheader("Análisis Temporal")
    tab1, tab2, tab3 = st.tabs(["📐 Ángulos Articulares", "🚀 Tracking Vertical", "⚖️ Índice KASR"])
    
    with tab1:
        st.line_chart(df_clean, x="Tiempo (s)", y=["Angulo Rodilla", "Angulo Cadera"], height=350)
    
    with tab2:
        df_plot = df_clean.copy()
        df_plot["Elevación de Cadera"] = -1 * df_plot["Y Cadera"]
        st.line_chart(df_plot, x="Tiempo (s)", y="Elevación de Cadera", height=300)

    with tab3:
        if not df_clean['KASR'].isna().all():
            st.line_chart(df_clean, x="Tiempo (s)", y="KASR", height=300)
        else:
            st.info("Gráfico KASR no disponible (Se requiere cámara Frontal).")


# =====================================================
# 10. PANTALLA INICIAL Y PANEL PRINCIPAL
# =====================================================

if not st.session_state.ficha_completa:
    mostrar_ficha()
    st.stop()

st.title("🦘 zenithz")
st.subheader("Evaluación biomecánica del salto vertical CMJ")
st.success("Ficha deportiva completada.")

with st.expander("Ver antecedentes del paciente"):
    st.json(st.session_state.perfil_evaluacion)

if st.button("Volver a la ficha inicial"):
    st.session_state.ficha_completa = False
    st.rerun()

st.divider()

st.header("1. Seleccionar método de evaluación")
modo = st.radio("¿Cómo deseas analizar tu salto?", ["Cámara en directo", "Subir video grabado"], horizontal=True)
vista_frontal = (st.session_state.perfil_evaluacion.get("Perspectiva de cámara") == "Frontal")
estatura_cm_actual = st.session_state.perfil_evaluacion.get("Estatura (cm)", 161)

# =====================================================
# 11. EVALUACION CON CAMARA Y WEBRTC SEGURO (STUN/TURN)
# =====================================================

# =====================================================
# 11. EVALUACION CON CAMARA (OPTIMIZADA PARA NUBE)
# =====================================================

if modo == "Cámara en directo":
    st.subheader("📱 Evaluación mediante cámara en tiempo real")
    st.info("Ubícate frente a la cámara. Para asegurar fluidez, el análisis se calculará tras la grabación.")

    if not vista_frontal:
        st.warning("Vista Lateral: Se medirán ángulos y altura, pero el Valgo/Varo (KASR) estará desactivado.")

    if os.path.exists(MODEL_PATH):
        try:
            # 1. Iniciamos el Streamer básico y liviano (sin callbacks pesados en la nube)
            webrtc_ctx = webrtc_streamer(
                key="zenithz-camera-light",
                mode=webrtc_streamer.Mode.SENDRECV,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                },
                media_stream_constraints={"video": True, "audio": False},
                video_html_attrs={
                    "style": {
                        "width": "100%", "max-width": "600px", "margin": "0 auto",
                        "border-radius": "12px", "border": "3px solid #5C8CE6",
                        "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.2)",
                        "display": "block"
                    },
                    "playsinline": True,
                    "autoplay": True
                }
            )

            # 2. Controladores de estado
            st.markdown("### 🎛️ Control de Evaluación")
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if st.button("🔴 Iniciar Grabación de Salto", use_container_width=True, disabled=not webrtc_ctx.state.playing):
                    st.session_state.grabando_ahora = True
                    st.success("¡Grabando! Realiza tu salto...")
            
            with col_rec2:
                if st.button("⏹️ Finalizar y Analizar", use_container_width=True):
                    st.session_state.grabando_ahora = False
                    st.info("Función de análisis en nube en desarrollo para esta versión ligera.")
                    # Aquí irá la lógica futura para mandar los frames recopilados a analizar.

            if not webrtc_ctx.state.playing:
                st.warning("👈 Presiona 'START' para encender la cámara antes de grabar.")

        except Exception as error:
            st.error(f"Error al iniciar el módulo webRTC: {error}")
    else:
        st.error("Falta el modelo pose_landmarker_lite.task en la carpeta.")
# =====================================================
# 12. EVALUACION CON VIDEO
# =====================================================

elif modo == "Subir video grabado":
    st.subheader("Evaluación mediante video")
    st.write("Sube un video de salto CMJ. El sistema anclará el origen (0,0) al inicio del video.")

    if not vista_frontal:
        st.warning("Tu ficha indica vista no frontal. El tracking articular funcionará excelente, pero el índice KASR (Valgo) se deshabilitará.")

    archivo = st.file_uploader("Sube tu archivo (mp4, mov)", type=["mp4", "mov", "avi"], key="video_cmj")

    if archivo is not None:
        identidad = (archivo.name, archivo.size)
        if identidad != st.session_state.identidad_video:
            st.session_state.identidad_video = identidad
            st.session_state.resultado_video = None

        st.video(archivo)

        if st.button("Analizar Video con Ejes Fijos", use_container_width=True):
            try:
                with st.spinner("Ejecutando tracking articular y anclaje de coordenadas..."):
                    df, frame, total, fps = analizar_video(archivo, vista_frontal, estatura_cm_actual)
                    st.session_state.resultado_video = {"df": df, "frame": frame, "total": total, "fps": fps}
            except Exception as error:
                st.error(f"Error al procesar: {error}")

        resultado = st.session_state.resultado_video
        if resultado is not None:
            st.success("Tracking cinemático finalizado.")
            if resultado["frame"] is not None:
                st.image(resultado["frame"], channels="BGR", caption="Sistema Cartesiano Anclado al Suelo")
            
            mostrar_resultados(resultado["df"], "video")

st.divider()
st.header("2. Consideraciones técnicas")
st.info(
    "El sistema cartesiano ancla su origen (0,0) en base a los primeros frames válidos registrados. "
    "Mide la altura aislando matemáticamente el desplazamiento puro del centro de masa, "
    "proyectando una grilla espacial en pantalla calibrada a la estatura del sujeto."
)
st.caption("zenithz | Herramienta de Ciencias del Deporte") 
