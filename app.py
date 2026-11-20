# =====================================================
# Zenithz - Evaluación Biomecánica de Salto Vertical CMJ
# Versión Fusionada: Ficha Completa + Cámara en Vivo (WebRTC)
# + Video Grabado con Dempster 2D, H.264, Visor de Cuadros,
# KASR, Valgo/Varo, Potencia Sayers, Historial y Reporte PDF
# =====================================================

import os
import time
import math
import shutil
import tempfile
import threading
import datetime
import subprocess

import av
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
from streamlit_webrtc import webrtc_streamer


# =====================================================
# 1. CONFIGURACION GENERAL Y CONSTANTES
# =====================================================

st.set_page_config(
    page_title="Zenithz | Análisis Biomecánico CMJ",
    page_icon="🦘",
    layout="wide",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pose_landmarker_lite.task"
)

MAX_ANCHO_VIDEO = 960    # Ancho máximo para video grabado
MAX_ANCHO_VIVO = 640     # Ancho máximo para cámara en vivo (fluidez en nube)
N_CALIB = 15             # Cuadros iniciales para calibración automática
FRACCION_NARIZ = 0.92    # La nariz está a ~92% de la estatura total
VIS_MIN = 0.45           # Visibilidad mínima de un landmark (0 a 1)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Índices de MediaPipe Pose (33 puntos)
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

CONEXIONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32)
]

# Estilos visuales
st.markdown(
    """
    <style>
      .block-container {
        max-width: 1600px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
      }
      .zenithz-subtitle {
        color: #7b8494;
        margin-top: -0.7rem;
        margin-bottom: 1.2rem;
      }
      div[data-testid="stVideo"] {
        width: 100%;
      }
      div[data-testid="stVideo"] video {
        width: 100%;
        max-height: 75vh;
        object-fit: contain;
        background: #111;
      }
      .step-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: rgba(92, 140, 230, 0.16);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 2. ESTADOS DE LA APLICACION Y SERVIDORES WEBRTC
# =====================================================

if "pagina" not in st.session_state:
    st.session_state.pagina = "ficha"

if "perfil_evaluacion" not in st.session_state:
    st.session_state.perfil_evaluacion = {}

if "historial_saltos" not in st.session_state:
    st.session_state.historial_saltos = pd.DataFrame(
        columns=["Intento", "Método", "Altura (cm)", "Potencia (W)"]
    )

if "registro_camara" not in st.session_state:
    st.session_state.registro_camara = {
        "lock": threading.Lock(),
        "muestras": [],
        "inicio": None,
        "ultima_muestra": -1.0,
        "grabando": False,
        "trayectoria": [],
        "cal": None,
        "mostrar_grilla": True,
    }

registro = st.session_state.registro_camara


def obtener_ice_servers():
    """Carga credenciales TURN desde st.secrets con respaldo seguro STUN."""
    ice_base = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
    ]
    try:
        username = st.secrets["TURN_USERNAME"]
        credential = st.secrets["TURN_CREDENTIAL"]
        ice_base.append({
            "urls": [
                "turn:sa.relay.metered.ca:80",
                "turn:sa.relay.metered.ca:80?transport=tcp",
                "turn:sa.relay.metered.ca:443",
                "turns:sa.relay.metered.ca:443?transport=tcp",
            ],
            "username": username,
            "credential": credential,
        })
        return ice_base, True
    except Exception:
        return ice_base, False


# =====================================================
# 3. MODELO DE MEDIAPIPE TASKS
# =====================================================

def crear_opciones(modo_ejecucion=mp.tasks.vision.RunningMode.VIDEO):
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=modo_ejecucion,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )


@st.cache_resource
def cargar_modelo_vivo(ruta):
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        crear_opciones(mp.tasks.vision.RunningMode.VIDEO)
    )


# =====================================================
# 4. GEOMETRIA, CALIBRACION Y DEMPSTER 2D
# =====================================================

def punto_visible(lm_punto, umbral=VIS_MIN):
    vis = getattr(lm_punto, "visibility", None)
    if vis is None:
        return True
    return float(vis) >= umbral


def punto_px(lm, idx, ancho, alto):
    return (lm[idx].x * ancho, lm[idx].y * alto)


def calcular_angulo(a, b, c):
    """Ángulo en el vértice B formado por A-B-C (180° = pierna extendida)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    coseno = np.clip(coseno, -1.0, 1.0)
    return float(np.degrees(np.arccos(coseno)))


def desviacion_valgo(cadera, rodilla, tobillo, x_centro):
    """Positivo = valgo (rodilla hacia adentro). Negativo = varo (hacia afuera)."""
    desvio = 180.0 - calcular_angulo(cadera, rodilla, tobillo)
    t = (rodilla[1] - cadera[1]) / (tobillo[1] - cadera[1] + 1e-9)
    x_linea = cadera[0] + t * (tobillo[0] - cadera[0])
    hacia_adentro = (x_centro - cadera[0]) * (rodilla[0] - x_linea) > 0
    return desvio if hacia_adentro else -desvio


def calcular_kasr(lm):
    """Índice KASR = Separación horizontal Rodillas / Separación horizontal Tobillos."""
    if not all(punto_visible(lm[i]) for i in (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)):
        return None
    sep_rodillas = abs(lm[LEFT_KNEE].x - lm[RIGHT_KNEE].x)
    sep_tobillos = abs(lm[LEFT_ANKLE].x - lm[RIGHT_ANKLE].x)
    if sep_tobillos < 0.01:
        return None
    return float(sep_rodillas / sep_tobillos)


def piernas_visibles(lm, vista, lado):
    if vista == "Lateral":
        indices = (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE) if lado == "Derecho" else (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    else:
        indices = (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    return all(punto_visible(lm[i]) for i in indices)


def centro_de_masa_dempster(lm, ancho, alto):
    """Centro de masa 2D basado en el modelo antropométrico de segmentos de Dempster."""
    def pt(idx):
        return np.array(punto_px(lm, idx, ancho, alto), dtype=float)

    def medio(a, b):
        return (pt(a) + pt(b)) / 2.0

    def en_segmento(a, b, fraccion):
        return pt(a) + fraccion * (pt(b) - pt(a))

    hombros = medio(LEFT_SHOULDER, RIGHT_SHOULDER)
    caderas = medio(LEFT_HIP, RIGHT_HIP)

    partes = [
        (medio(LEFT_EAR, RIGHT_EAR), 0.081),
        (hombros + 0.5 * (caderas - hombros), 0.497),
        # Lado izquierdo
        (en_segmento(LEFT_SHOULDER, LEFT_ELBOW, 0.436), 0.028),
        (en_segmento(LEFT_ELBOW, LEFT_WRIST, 0.430), 0.016),
        (en_segmento(LEFT_WRIST, LEFT_INDEX, 0.506), 0.006),
        (en_segmento(LEFT_HIP, LEFT_KNEE, 0.433), 0.100),
        (en_segmento(LEFT_KNEE, LEFT_ANKLE, 0.433), 0.0465),
        (en_segmento(LEFT_HEEL, LEFT_FOOT_INDEX, 0.5), 0.0145),
        # Lado derecho
        (en_segmento(RIGHT_SHOULDER, RIGHT_ELBOW, 0.436), 0.028),
        (en_segmento(RIGHT_ELBOW, RIGHT_WRIST, 0.430), 0.016),
        (en_segmento(RIGHT_WRIST, RIGHT_INDEX, 0.506), 0.006),
        (en_segmento(RIGHT_HIP, RIGHT_KNEE, 0.433), 0.100),
        (en_segmento(RIGHT_KNEE, RIGHT_ANKLE, 0.433), 0.0465),
        (en_segmento(RIGHT_HEEL, RIGHT_FOOT_INDEX, 0.5), 0.0145),
    ]
    masa_total = sum(m for _, m in partes)
    cm = sum(pos * m for pos, m in partes) / masa_total
    return (float(cm[0]), float(cm[1]))


def calibracion_desde_medidas(talon_x, talon_y, altura_px, estatura_cm):
    estatura_px = altura_px / FRACCION_NARIZ
    return {
        "ox": float(talon_x),
        "oy": float(talon_y),
        "k": float(estatura_cm / estatura_px),
        "unidad": "cm",
    }


def calibracion_desde_landmarks(lm, ancho, alto, estatura_cm):
    ti = punto_px(lm, LEFT_HEEL, ancho, alto)
    td = punto_px(lm, RIGHT_HEEL, ancho, alto)
    nariz = punto_px(lm, NOSE, ancho, alto)
    tx = (ti[0] + td[0]) / 2.0
    ty = (ti[1] + td[1]) / 2.0
    altura_px = ty - nariz[1]
    if altura_px > 20:
        return calibracion_desde_medidas(tx, ty, altura_px, estatura_cm)
    return {"ox": ancho / 2.0, "oy": alto * 0.92, "k": 1.0, "unidad": "px"}


def a_coordenadas(pt, cal):
    """Pasa de píxeles a (x, y) con Y positivo hacia arriba."""
    return ((pt[0] - cal["ox"]) * cal["k"], (cal["oy"] - pt[1]) * cal["k"])


def agregar_coords(fila, nombre, pt, cal):
    x, y = a_coordenadas(pt, cal)
    u = cal["unidad"]
    fila[f"{nombre} x ({u})"] = round(x, 2)
    fila[f"{nombre} y ({u})"] = round(y, 2)


def medir_cuadro(lm, ancho, alto, vista, lado, cal):
    fila = {}
    if vista == "Lateral":
        if lado == "Derecho":
            hombro_i, cad_i, rod_i, tob_i = RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        else:
            hombro_i, cad_i, rod_i, tob_i = LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

        hombro = punto_px(lm, hombro_i, ancho, alto)
        cadera = punto_px(lm, cad_i, ancho, alto)
        rodilla = punto_px(lm, rod_i, ancho, alto)
        tobillo = punto_px(lm, tob_i, ancho, alto)

        fila["Angulo Rodilla"] = round(calcular_angulo(cadera, rodilla, tobillo), 2)
        fila["Angulo Cadera"] = round(calcular_angulo(hombro, cadera, rodilla), 2)
        fila["KASR"] = None
        fila["Valgo izquierda (°)"] = None
        fila["Valgo derecha (°)"] = None

        agregar_coords(fila, "Cadera", cadera, cal)
        agregar_coords(fila, "Rodilla", rodilla, cal)
        agregar_coords(fila, "Tobillo", tobillo, cal)
        return fila

    # Vista Frontal (o perspectiva frontal/mixta)
    hom_i, cad_i, rod_i, tob_i = [punto_px(lm, i, ancho, alto) for i in (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)]
    hom_d, cad_d, rod_d, tob_d = [punto_px(lm, i, ancho, alto) for i in (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]
    x_centro = (cad_i[0] + cad_d[0]) / 2.0

    ang_rod = (calcular_angulo(cad_i, rod_i, tob_i) + calcular_angulo(cad_d, rod_d, tob_d)) / 2.0
    ang_cad = (calcular_angulo(hom_i, cad_i, rod_i) + calcular_angulo(hom_d, cad_d, rod_d)) / 2.0
    kasr = calcular_kasr(lm)

    fila["Angulo Rodilla"] = round(ang_rod, 2)
    fila["Angulo Cadera"] = round(ang_cad, 2)
    fila["KASR"] = round(kasr, 4) if kasr is not None else None
    fila["Valgo izquierda (°)"] = round(desviacion_valgo(cad_i, rod_i, tob_i, x_centro), 2)
    fila["Valgo derecha (°)"] = round(desviacion_valgo(cad_d, rod_d, tob_d, x_centro), 2)

    for nombre, pt in [
        ("Cadera izq", cad_i), ("Rodilla izq", rod_i), ("Tobillo izq", tob_i),
        ("Cadera der", cad_d), ("Rodilla der", rod_d), ("Tobillo der", tob_d),
    ]:
        agregar_coords(fila, nombre, pt, cal)
    return fila


# =====================================================
# 5. DIBUJO SOBRE VIDEO (CUADRICULA, EJES, ESQUELETO, CM)
# =====================================================

def crear_cuadricula(ancho, alto, cal):
    capa = np.zeros((alto, ancho, 3), np.uint8)
    paso = max(20.0 / cal["k"] if cal["unidad"] == "cm" else 100.0, 12.0)
    x = cal["ox"] % paso
    while x < ancho:
        cv2.line(capa, (int(x), 0), (int(x), alto), (255, 255, 255), 1)
        x += paso
    y = cal["oy"] % paso
    while y < alto:
        cv2.line(capa, (0, int(y)), (ancho, int(y)), (255, 255, 255), 1)
        y += paso
    return capa


def dibujar_ejes(cuadro, cal):
    ox, oy = int(cal["ox"]), int(cal["oy"])
    largo = int(100.0 / cal["k"]) if cal["unidad"] == "cm" else 150
    cv2.arrowedLine(cuadro, (ox, oy), (ox + largo, oy), (0, 0, 255), 2, tipLength=0.05)
    cv2.arrowedLine(cuadro, (ox, oy), (ox, oy - largo), (0, 255, 0), 2, tipLength=0.05)
    cv2.putText(cuadro, "X", (ox + largo + 8, oy + 8), FUENTE, 0.7, (0, 0, 255), 2)
    cv2.putText(cuadro, "Y", (ox + 8, oy - largo - 8), FUENTE, 0.7, (0, 255, 0), 2)
    cv2.circle(cuadro, (ox, oy), 5, (255, 255, 255), -1)
    if cal["unidad"] == "cm":
        for cm in range(20, 100, 20):
            d = int(cm / cal["k"])
            cv2.line(cuadro, (ox + d, oy - 5), (ox + d, oy + 5), (0, 0, 255), 2)
            cv2.line(cuadro, (ox - 5, oy - d), (ox + 5, oy - d), (0, 255, 0), 2)


def dibujar_esqueleto(cuadro, lm):
    alto, ancho = cuadro.shape[:2]
    for i, j in CONEXIONES:
        if punto_visible(lm[i]) and punto_visible(lm[j]):
            p1 = (int(lm[i].x * ancho), int(lm[i].y * alto))
            p2 = (int(lm[j].x * ancho), int(lm[j].y * alto))
            cv2.line(cuadro, p1, p2, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    for pt in lm:
        if punto_visible(pt):
            x, y = int(pt.x * ancho), int(pt.y * alto)
            if 0 <= x < ancho and 0 <= y < alto:
                cv2.circle(cuadro, (x, y), 4, (0, 0, 255), -1)


def escribir_cm(cuadro, cm_px, x_cm, y_cm, unidad):
    texto = f"CM ({x_cm:.1f}, {y_cm:.1f}) {unidad}"
    centro = (int(cm_px[0]), int(cm_px[1]))
    pos = (centro[0] + 12, centro[1] - 12)
    cv2.putText(cuadro, texto, pos, FUENTE, 0.65, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.putText(cuadro, texto, pos, FUENTE, 0.65, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(cuadro, centro, 7, (0, 0, 255), -1)


def escribir_metricas_overlay(cuadro, fila, vista):
    if not fila:
        return
    if vista == "Lateral" and fila.get("Angulo Rodilla") is not None:
        texto = f"Rodilla: {fila['Angulo Rodilla']:.0f} deg | Cadera: {fila['Angulo Cadera']:.0f} deg"
    elif fila.get("Valgo izquierda (°)") is not None:
        kasr_txt = f" | KASR: {fila['KASR']:.2f}" if fila.get("KASR") is not None else ""
        texto = f"Valgo I:{fila['Valgo izquierda (°)']:.0f} D:{fila['Valgo derecha (°)']:.0f} deg{kasr_txt}"
    else:
        return
    cv2.putText(cuadro, texto, (18, 36), FUENTE, 0.72, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.putText(cuadro, texto, (18, 36), FUENTE, 0.72, (0, 255, 255), 2, lineType=cv2.LINE_AA)


# =====================================================
# 6. VIDEO: LECTURA, CALIBRACION, PROCESAMIENTO Y H.264
# =====================================================

def leer_cuadro(cap, max_ancho=MAX_ANCHO_VIDEO):
    ok, cuadro = cap.read()
    if not ok:
        return None
    alto, ancho = cuadro.shape[:2]
    if ancho > max_ancho:
        escala = max_ancho / ancho
        cuadro = cv2.resize(cuadro, (max_ancho, int(alto * escala)))
    return cuadro


def calibrar_video(ruta, estatura_cm):
    cap = cv2.VideoCapture(ruta)
    xs, ys, alturas = [], [], []
    ancho = alto = None
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        crear_opciones(mp.tasks.vision.RunningMode.IMAGE)
    ) as detector:
        for _ in range(N_CALIB):
            cuadro = leer_cuadro(cap, MAX_ANCHO_VIDEO)
            if cuadro is None:
                break
            alto, ancho = cuadro.shape[:2]
            rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                ti = punto_px(lm, LEFT_HEEL, ancho, alto)
                td = punto_px(lm, RIGHT_HEEL, ancho, alto)
                nariz = punto_px(lm, NOSE, ancho, alto)
                xs.append((ti[0] + td[0]) / 2.0)
                ys.append((ti[1] + td[1]) / 2.0)
                alturas.append(((ti[1] + td[1]) / 2.0) - nariz[1])
    cap.release()
    if alto is None:
        return None
    if len(alturas) == 0 or np.median(alturas) <= 20:
        return {"ox": ancho / 2.0, "oy": alto * 0.95, "k": 1.0, "unidad": "px"}
    return calibracion_desde_medidas(np.median(xs), np.median(ys), np.median(alturas), estatura_cm)


def procesar_video_completo(ruta, vista, lado, cal, mostrar_grilla, barra):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps < 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ruta_salida = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    escritor = None
    filas = []
    trayectoria = []
    capa_grilla = None
    n = 0
    ultimo_ts = -1

    with mp.tasks.vision.PoseLandmarker.create_from_options(
        crear_opciones(mp.tasks.vision.RunningMode.VIDEO)
    ) as detector:
        while True:
            cuadro = leer_cuadro(cap, MAX_ANCHO_VIDEO)
            if cuadro is None:
                break
            alto, ancho = cuadro.shape[:2]
            if escritor is None:
                escritor = cv2.VideoWriter(
                    ruta_salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
                )

            rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
            ts = max(int(n * 1000 / fps), ultimo_ts + 1)
            ultimo_ts = ts
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )

            if mostrar_grilla:
                if capa_grilla is None:
                    capa_grilla = crear_cuadricula(ancho, alto, cal)
                cuadro = cv2.addWeighted(cuadro, 1.0, capa_grilla, 0.35, 0)
                cv2.line(cuadro, (int(cal["ox"]), 0), (int(cal["ox"]), alto), (0, 255, 255), 2)
            dibujar_ejes(cuadro, cal)

            fila = {}
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                dibujar_esqueleto(cuadro, lm)

                if piernas_visibles(lm, vista, lado):
                    fila = medir_cuadro(lm, ancho, alto, vista, lado, cal)
                    escribir_metricas_overlay(cuadro, fila, vista)

                cm_px = centro_de_masa_dempster(lm, ancho, alto)
                agregar_coords(fila, "CM", cm_px, cal)
                trayectoria.append((int(cm_px[0]), int(cm_px[1])))
                if len(trayectoria) > 1:
                    cv2.polylines(cuadro, [np.array(trayectoria, np.int32)], False, (255, 0, 255), 3)
                x_cm, y_cm = a_coordenadas(cm_px, cal)
                escribir_cm(cuadro, cm_px, x_cm, y_cm, cal["unidad"])

            filas.append(fila)
            escritor.write(cuadro)
            n += 1
            if total > 0:
                barra.progress(min(n / total, 1.0), text=f"Procesando cuadro {n} de {total}...")

    cap.release()
    if escritor is not None:
        escritor.release()
    return pd.DataFrame(filas), fps, ruta_salida


def a_h264(ruta_entrada):
    """Convierte video MP4 a H.264 compatible con navegadores web."""
    ruta_salida = os.path.splitext(ruta_entrada)[0] + "_web.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", ruta_entrada,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags", "+faststart", "-an", ruta_salida
            ],
            check=True,
            capture_output=True,
        )
        return ruta_salida, None
    except subprocess.CalledProcessError as e:
        return ruta_entrada, e.stderr.decode(errors="replace")[-300:]
    except Exception as e:
        return ruta_entrada, str(e)


def cuadro_de_salida(ruta, indice):
    cap = cv2.VideoCapture(ruta)
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, cuadro = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)


# =====================================================
# 7. PROCESADOR EN DIRECTO (WEBRTC)
# =====================================================

def crear_procesador_vivo(modelo, reg, vista, lado, estatura_cm):
    ultimo_ts = [0]
    capa_grilla_cache = {"shape": None, "cal": None, "capa": None}

    def procesar_frame(frame):
        imagen = frame.to_ndarray(format="bgr24")
        alto, ancho = imagen.shape[:2]
        if ancho > MAX_ANCHO_VIVO:
            nuevo_alto = int(alto * MAX_ANCHO_VIVO / ancho)
            imagen = cv2.resize(imagen, (MAX_ANCHO_VIVO, nuevo_alto))
            alto, ancho = imagen.shape[:2]

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        ts = max(time.monotonic_ns() // 1_000_000, ultimo_ts[0] + 1)
        ultimo_ts[0] = ts
        res = modelo.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        with reg["lock"]:
            grabando = reg["grabando"]
            cal_actual = reg["cal"]
            ver_grilla = reg["mostrar_grilla"]

        if res.pose_landmarks:
            lm = res.pose_landmarks[0]

            if not grabando or cal_actual is None:
                cal_actual = calibracion_desde_landmarks(lm, ancho, alto, estatura_cm)
                with reg["lock"]:
                    reg["cal"] = cal_actual

            if ver_grilla and cal_actual is not None:
                if (
                    capa_grilla_cache["shape"] != (alto, ancho)
                    or capa_grilla_cache["cal"] != cal_actual
                ):
                    capa_grilla_cache["shape"] = (alto, ancho)
                    capa_grilla_cache["cal"] = dict(cal_actual)
                    capa_grilla_cache["capa"] = crear_cuadricula(ancho, alto, cal_actual)
                imagen = cv2.addWeighted(imagen, 1.0, capa_grilla_cache["capa"], 0.30, 0)
                cv2.line(imagen, (int(cal_actual["ox"]), 0), (int(cal_actual["ox"]), alto), (0, 255, 255), 2)

            dibujar_ejes(imagen, cal_actual)
            dibujar_esqueleto(imagen, lm)

            fila = {}
            if piernas_visibles(lm, vista, lado):
                fila = medir_cuadro(lm, ancho, alto, vista, lado, cal_actual)
                escribir_metricas_overlay(imagen, fila, vista)

            cm_px = centro_de_masa_dempster(lm, ancho, alto)
            agregar_coords(fila, "CM", cm_px, cal_actual)

            with reg["lock"]:
                tray = reg["trayectoria"]
                tray.append((int(cm_px[0]), int(cm_px[1])))
                if len(tray) > 60:
                    tray.pop(0)
                tray_copia = list(tray)

                if reg["grabando"]:
                    ahora = time.monotonic()
                    if reg["inicio"] is None:
                        reg["inicio"] = ahora
                    tiempo = ahora - reg["inicio"]
                    if (tiempo - reg["ultima_muestra"]) >= 0.05:
                        fila["Tiempo (s)"] = round(tiempo, 3)
                        reg["muestras"].append(fila)
                        reg["ultima_muestra"] = tiempo
                        if len(reg["muestras"]) > 3000:
                            reg["muestras"].pop(0)

            if len(tray_copia) > 1:
                cv2.polylines(imagen, [np.array(tray_copia, np.int32)], False, (255, 0, 255), 3)
            x_cm, y_cm = a_coordenadas(cm_px, cal_actual)
            escribir_cm(imagen, cm_px, x_cm, y_cm, cal_actual["unidad"])

            estado_txt = "GRABANDO SALTO (EJES FIJOS)" if grabando else "LISTO - CALIBRANDO (0,0)"
            color_est = (0, 0, 255) if grabando else (0, 255, 255)
            cv2.putText(imagen, estado_txt, (18, alto - 18), FUENTE, 0.6, color_est, 2)
        else:
            with reg["lock"]:
                reg["trayectoria"].clear()
            cv2.putText(imagen, "BUSCANDO CUERPO COMPLETO...", (18, 36), FUENTE, 0.7, (0, 165, 255), 2)

        return av.VideoFrame.from_ndarray(imagen, format="bgr24")

    return procesar_frame


# =====================================================
# 8. GENERADOR DE REPORTES (PDF)
# =====================================================

def limpiar_latin1(texto):
    limpio = str(texto).replace("🔴", "").replace("🟡", "").replace("🟢", "").replace("⚠️", "").replace("✅", "")
    return limpio.encode("latin-1", "replace").decode("latin-1")


def generar_reporte_pdf(
    perfil, altura, desp_total, tiempo, potencia, potencia_rel,
    rodilla, cadera, diagnostico, valgo_izq=None, valgo_der=None
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt=limpiar_latin1("Reporte Biomecánico CMJ - Zenithz"), ln=True, align="C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 8, txt=limpiar_latin1("Universidad Santo Tomás - Ciencias del Deporte"), ln=True, align="C")
    pdf.ln(8)

    # 1. Antecedentes
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(230, 235, 245)
    pdf.cell(0, 9, txt=limpiar_latin1(" 1. Antecedentes del Deportista"), ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    nombre_paciente = perfil.get("Nombre") or "Sin nombre"
    pdf.cell(0, 7, txt=limpiar_latin1(f"Nombre: {nombre_paciente} | Sexo: {perfil.get('Sexo', 'N/A')}"), ln=True)
    pdf.cell(
        0, 7,
        txt=limpiar_latin1(
            f"Edad: {perfil.get('Edad', 'N/A')} años | "
            f"Estatura: {perfil.get('Estatura (cm)', 'N/A')} cm | "
            f"Peso: {perfil.get('Peso (kg)', 'N/A')} kg"
        ),
        ln=True,
    )
    if perfil.get("Deporte y nivel"):
        pdf.cell(0, 7, txt=limpiar_latin1(f"Deporte y nivel: {perfil.get('Deporte y nivel')}"), ln=True)
    if perfil.get("Lesiones previas"):
        pdf.multi_cell(0, 7, txt=limpiar_latin1(f"Lesiones/molestias: {perfil.get('Lesiones previas')}"))
    pdf.ln(4)

    # 2. Rendimiento
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, txt=limpiar_latin1(" 2. Rendimiento CMJ (Centro de Masa Dempster 2D)"), ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, txt=limpiar_latin1(f"Altura Máxima de Salto Estimada: {altura:.1f} cm"), ln=True)
    pdf.cell(0, 7, txt=limpiar_latin1(f"Desplazamiento Vertical Total del CM (descenso a pico): {desp_total:.1f} cm"), ln=True)
    pdf.cell(0, 7, txt=limpiar_latin1(f"Tiempo de Vuelo Estimado: {tiempo:.3f} s"), ln=True)
    pdf.cell(0, 7, txt=limpiar_latin1(f"Potencia Pico (Sayers): {potencia:.0f} W | Potencia Relativa: {potencia_rel:.1f} W/kg"), ln=True)
    pdf.ln(4)

    # 3. Perfil articular
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 9, txt=limpiar_latin1(" 3. Cinemática Articular y Control Frontal"), ln=True, fill=True)
    pdf.set_font("Arial", "", 11)
    if rodilla is not None:
        pdf.cell(0, 7, txt=limpiar_latin1(f"Ángulo Mínimo de Rodilla: {rodilla:.1f} grados (Flexión máx.: {180 - rodilla:.1f} grados)"), ln=True)
    if cadera is not None:
        pdf.cell(0, 7, txt=limpiar_latin1(f"Ángulo Mínimo de Cadera: {cadera:.1f} grados"), ln=True)
    pdf.cell(0, 7, txt=limpiar_latin1(f"Control Motor Frontal (KASR): {diagnostico}"), ln=True)
    if valgo_izq is not None and valgo_der is not None:
        pdf.cell(0, 7, txt=limpiar_latin1(f"Valgo Máximo Estimado -> Izquierda: {valgo_izq:.1f} deg | Derecha: {valgo_der:.1f} deg"), ln=True)

    pdf.ln(12)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        txt=limpiar_latin1(f"Evaluación generada el {datetime.date.today().strftime('%d/%m/%Y')} mediante Zenithz AI."),
        ln=True, align="C"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp.name)
    return pdf_bytes, nombre_paciente


# =====================================================
# 9. DASHBOARD DE RESULTADOS (COMPARTIDO VIDEO Y VIVO)
# =====================================================

def mostrar_dashboard_resultados(df, cal, vista, origen_label):
    u = cal["unidad"]
    col_cm_y = f"CM y ({u})"

    if df.empty or col_cm_y not in df.columns:
        st.warning("No se registraron suficientes datos del centro de masa para el análisis.")
        return

    datos = df.dropna(subset=[col_cm_y]).reset_index(drop=True)
    if datos.empty:
        st.warning("Datos inconsistentes para calcular métricas.")
        return

    perfil = st.session_state.perfil_evaluacion
    peso_kg = float(perfil.get("Peso (kg)", 60.0))

    # Altura CMJ (desde posición erguida inicial al pico) y desplazamiento total
    n_base = min(8, len(datos))
    y_reposo = float(datos[col_cm_y].iloc[:n_base].median())
    y_pico = float(datos[col_cm_y].max())
    y_min = float(datos[col_cm_y].min())
    desplazamiento_total = max(0.0, y_pico - y_min)

    if u == "cm":
        altura_final = max(0.0, y_pico - y_reposo)
        altura_m = altura_final / 100.0
        tiempo_vuelo = math.sqrt((8.0 * altura_m) / 9.81) if altura_m > 0 else 0.0
        potencia_w = max(0.0, (60.7 * altura_final) + (45.3 * peso_kg) - 2055.0)
        potencia_rel = potencia_w / peso_kg if peso_kg > 0 else 0.0
    else:
        st.warning("No se pudo calibrar la escala en cm (persona de pie al inicio no detectada). Las coordenadas se muestran en píxeles.")
        altura_final = tiempo_vuelo = potencia_w = potencia_rel = 0.0

    # Ángulos articulares
    tiene_rodilla = "Angulo Rodilla" in datos.columns and not datos["Angulo Rodilla"].isna().all()
    tiene_cadera = "Angulo Cadera" in datos.columns and not datos["Angulo Cadera"].isna().all()

    if tiene_rodilla:
        idx_profundo = datos["Angulo Rodilla"].idxmin()
        ang_min_rodilla = float(datos.loc[idx_profundo, "Angulo Rodilla"])
        flex_max_rodilla = 180.0 - ang_min_rodilla
    else:
        idx_profundo = datos[col_cm_y].idxmin()
        ang_min_rodilla = None
        flex_max_rodilla = None

    ang_min_cadera = float(datos["Angulo Cadera"].min()) if tiene_cadera else None

    # KASR y Valgo/Varo (Frontal)
    kasr_profundo = None
    if vista == "Frontal" and "KASR" in datos.columns and not datos["KASR"].isna().all():
        val_kasr = datos.loc[idx_profundo, "KASR"]
        kasr_profundo = float(val_kasr) if pd.notna(val_kasr) else float(datos["KASR"].dropna().median())

    if kasr_profundo is None:
        diagnostico = "No evaluable (requiere vista frontal)"
    elif kasr_profundo < 0.85:
        diagnostico = "Tendencia a VALGO 🔴"
    elif kasr_profundo > 1.15:
        diagnostico = "Tendencia a VARO 🟡"
    else:
        diagnostico = "Alineación NEUTRA 🟢"

    valgo_izq_max = valgo_der_max = valgo_izq_fondo = valgo_der_fondo = None
    cols_valgo = ["Valgo izquierda (°)", "Valgo derecha (°)"]
    tiene_valgo = vista == "Frontal" and all(c in datos.columns and not datos[c].isna().all() for c in cols_valgo)
    if tiene_valgo:
        fondo_cm = datos[col_cm_y].idxmin()
        valgo_izq_max = float(datos[cols_valgo[0]].max())
        valgo_der_max = float(datos[cols_valgo[1]].max())
        valgo_izq_fondo = float(datos.loc[fondo_cm, cols_valgo[0]]) if pd.notna(datos.loc[fondo_cm, cols_valgo[0]]) else 0.0
        valgo_der_fondo = float(datos.loc[fondo_cm, cols_valgo[1]]) if pd.notna(datos.loc[fondo_cm, cols_valgo[1]]) else 0.0

    # --- HISTORIAL DE SALTOS ---
    st.markdown("### 💾 Registro de Saltos")
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("➕ Guardar salto en el Historial", type="primary", use_container_width=True, key=f"btn_hist_{origen_label}"):
            nuevo = pd.DataFrame({
                "Intento": [f"Salto {len(st.session_state.historial_saltos) + 1}"],
                "Método": [origen_label],
                "Altura (cm)": [round(altura_final, 2)],
                "Potencia (W)": [round(potencia_w, 0)],
            })
            st.session_state.historial_saltos = pd.concat(
                [st.session_state.historial_saltos, nuevo], ignore_index=True
            )
            st.success("¡Salto registrado en el historial!")

    if not st.session_state.historial_saltos.empty:
        st.markdown("#### 📊 Comparativa de Saltos")
        fig = go.Figure(data=[
            go.Bar(
                x=st.session_state.historial_saltos["Intento"],
                y=st.session_state.historial_saltos["Altura (cm)"],
                text=st.session_state.historial_saltos["Altura (cm)"],
                textposition="inside",
                marker_color="#5C8CE6",
                textfont=dict(color="#FFD700", size=22, family="Arial Black"),
            )
        ])
        max_h = max(st.session_state.historial_saltos["Altura (cm)"].max(), 10.0)
        fig.update_layout(
            plot_bgcolor="#E5E5E5",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Altura (cm)", gridcolor="white", range=[0, max_h * 1.3]),
            margin=dict(l=20, r=20, t=30, b=20),
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("🗑️ Borrar Historial", key=f"btn_del_{origen_label}"):
            st.session_state.historial_saltos = pd.DataFrame(
                columns=["Intento", "Método", "Altura (cm)", "Potencia (W)"]
            )
            st.rerun()

    st.divider()

    # --- METRICAS DE RENDIMIENTO ---
    st.markdown("#### 🚀 Métricas de Rendimiento (Modelo Dempster 2D)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Altura Máxima CMJ", f"{altura_final:.1f} {u}", "Reposo a pico CM")
    c2.metric("Desplazamiento Total CM", f"{desplazamiento_total:.1f} {u}", "Fondo a pico CM")
    c3.metric("Tiempo de Vuelo", f"{tiempo_vuelo:.3f} s", "Cinemática vertical")
    c4.metric("Potencia Pico (Sayers)", f"{potencia_w:.0f} W", f"Relativa: {potencia_rel:.1f} W/kg")

    # --- METRICAS ARTICULARES ---
    st.markdown("#### 📐 Métricas Articulares y Control Frontal")
    c5, c6, c7 = st.columns(3)
    c5.metric(
        "Ángulo Mínimo Rodilla",
        f"{ang_min_rodilla:.1f}°" if ang_min_rodilla is not None else "N/A",
        f"Flexión máx: {flex_max_rodilla:.1f}°" if flex_max_rodilla is not None else "Sin detección clara",
    )
    c6.metric(
        "Ángulo Mínimo Cadera",
        f"{ang_min_cadera:.1f}°" if ang_min_cadera is not None else "N/A",
        "Cierre de tronco",
    )
    c7.metric("Alineación Rodillas (KASR)", diagnostico, f"Índice: {kasr_profundo:.2f}" if kasr_profundo else "Solo vista frontal")

    if tiene_valgo:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Valgo máx. Izquierda", f"{valgo_izq_max:.1f}°")
        v2.metric("Valgo máx. Derecha", f"{valgo_der_max:.1f}°")
        v3.metric("Valgo Izq. en punto más bajo", f"{valgo_izq_fondo:.1f}°")
        v4.metric("Valgo Der. en punto más bajo", f"{valgo_der_fondo:.1f}°")
        st.caption("Positivo (+) = valgo (rodilla hacia adentro); negativo (-) = varo (hacia afuera).")

    st.divider()

    # --- ANALISIS CLINICO AUTOMATICO ---
    st.markdown("### 🧠 Análisis Clínico Automático")
    fb1, fb2, fb3 = st.columns(3)

    with fb1:
        if ang_min_rodilla is None:
            st.info("**Técnica Excéntrica:**\n\nNo se detectó el ángulo de rodilla con claridad.")
        elif ang_min_rodilla < 85:
            st.error("**Técnica Excéntrica:**\n\n⚠️ Flexión excesiva (<85° internos). Puede disipar energía elástica.")
        elif ang_min_rodilla > 115:
            st.warning("**Técnica Excéntrica:**\n\n⚠️ Flexión insuficiente (>115° internos). Menor recorrido de impulso.")
        else:
            st.success("**Técnica Excéntrica:**\n\n✅ Profundidad óptima de flexión (rango 85°–115° internos).")

    with fb2:
        if diagnostico == "Tendencia a VALGO 🔴":
            st.error("**Riesgo Articular:**\n\n⚠️ Tendencia a Valgo dinámico. Se sugiere fortalecer glúteo medio y control lumbopélvico.")
        elif diagnostico == "Tendencia a VARO 🟡":
            st.warning("**Riesgo Articular:**\n\n🟡 Tendencia a Varo. Rodillas desplazadas hacia afuera en exceso.")
        elif diagnostico == "Alineación NEUTRA 🟢":
            st.success("**Riesgo Articular:**\n\n✅ Alineación neutra. Buen control motor en el plano frontal.")
        else:
            st.info("**Riesgo Articular:**\n\nEl análisis de valgo/varo y KASR está disponible en vista Frontal.")

    with fb3:
        if u != "cm":
            st.info("**Perfil de Fuerza:**\n\nRequiere calibración en cm para estimar potencia.")
        elif potencia_rel < 40:
            st.warning("**Perfil de Fuerza:**\n\n🟡 Potencia relativa baja (< 40 W/kg). Priorizar fuerza máxima y pliometría básica.")
        elif potencia_rel >= 55:
            st.success("**Perfil de Fuerza:**\n\n🔥 Nivel explosivo alto (> 55 W/kg).")
        else:
            st.success("**Perfil de Fuerza:**\n\n✅ Producción de potencia estable (40–55 W/kg).")

    st.divider()

    # --- EXPORTAR PDF Y CSV ---
    st.markdown("### 📑 Exportar Informe y Datos")
    pdf_bytes, nom_pac = generar_reporte_pdf(
        perfil, altura_final, desplazamiento_total, tiempo_vuelo,
        potencia_w, potencia_rel, ang_min_rodilla, ang_min_cadera,
        diagnostico, valgo_izq_max, valgo_der_max
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📄 Descargar Reporte Clínico (PDF)",
            data=pdf_bytes,
            file_name=f"Reporte_CMJ_{nom_pac.replace(' ', '_')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key=f"dl_pdf_{origen_label}",
        )
    with d2:
        st.download_button(
            "📊 Descargar Coordenadas y Métricas (CSV)",
            data=datos.to_csv(index=False).encode("utf-8"),
            file_name=f"Zenithz_Datos_{nom_pac.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_csv_{origen_label}",
        )

    st.divider()

    # --- GRAFICOS TEMPORALES ---
    st.subheader("📈 Análisis Temporal")
    t1, t2, t3 = st.tabs(["🚀 Altura Centro de Masa", "📐 Ángulos Articulares", "⚖️ Control Frontal (Valgo / KASR)"])

    with t1:
        st.line_chart(datos, x="Tiempo (s)", y=col_cm_y, height=320)

    with t2:
        cols_ang = [c for c in ["Angulo Rodilla", "Angulo Cadera"] if c in datos.columns and not datos[c].isna().all()]
        if cols_ang:
            st.line_chart(datos, x="Tiempo (s)", y=cols_ang, height=340)
            st.caption("180° representa extensión completa; valores menores indican mayor flexión.")
        else:
            st.info("No hubo suficientes detecciones claras de rodilla/cadera.")

    with t3:
        if tiene_valgo:
            st.markdown("**Desviación angular de rodillas (Valgo + / Varo -)**")
            st.line_chart(datos, x="Tiempo (s)", y=cols_valgo, height=280)
        if "KASR" in datos.columns and not datos["KASR"].isna().all():
            st.markdown("**Evolución Índice KASR (neutro ≈ 1.0)**")
            st.line_chart(datos, x="Tiempo (s)", y="KASR", height=260)
        if not tiene_valgo and ("KASR" not in datos.columns or datos["KASR"].isna().all()):
            st.info("Gráficos frontales disponibles únicamente cuando la vista de cámara es Frontal.")

    with st.expander("Ver tabla completa de coordenadas y mediciones"):
        st.write(f"Coordenadas (x, y) en **{u}**, origen en el suelo entre talones, eje Y positivo hacia arriba.")
        st.dataframe(datos, use_container_width=True)


# =====================================================
# 10. INTERFAZ PRINCIPAL DE DOS PASOS
# =====================================================

st.title("🦘 Zenithz")
st.markdown(
    '<div class="zenithz-subtitle">Evaluación biomecánica del salto vertical CMJ · Ciencias del Deporte</div>',
    unsafe_allow_html=True,
)

paso_actual = 1 if st.session_state.pagina == "ficha" else 2
st.progress(paso_actual / 2, text=f"Paso {paso_actual} de 2")


# -----------------------------------------------------
# PANTALLA 1: FICHA DEL DEPORTISTA (COMPLETA)
# -----------------------------------------------------
if st.session_state.pagina == "ficha":
    st.markdown('<span class="step-pill">PASO 1 · FICHA DE CARACTERIZACIÓN DEPORTIVA</span>', unsafe_allow_html=True)
    st.write("Completa los antecedentes del deportista antes de iniciar la captura o subir el video.")

    p = st.session_state.perfil_evaluacion

    with st.form("form_ficha_completa"):
        st.markdown("### A. Datos Personales")
        c1, c2, c3 = st.columns(3)
        with c1:
            nombre = st.text_input("Nombre completo", value=p.get("Nombre", ""), placeholder="Ej: Michelle Castro R.")
            sexo = st.selectbox(
                "Sexo", ["Femenino", "Masculino", "Otro"],
                index=["Femenino", "Masculino", "Otro"].index(p.get("Sexo", "Femenino"))
            )
        with c2:
            edad = st.number_input("Edad (años)", min_value=10, max_value=100, value=int(p.get("Edad", 21)), step=1)
            estatura = st.number_input("Estatura (cm)", min_value=100, max_value=240, value=int(p.get("Estatura (cm)", 161)), step=1)
        with c3:
            peso = st.number_input("Peso / Masa corporal (kg)", min_value=30.0, max_value=200.0, value=float(p.get("Peso (kg)", 60.0)), step=0.1)
            fecha_nac = st.date_input("Fecha de nacimiento", value=datetime.date(2005, 9, 5), min_value=datetime.date(1950, 1, 1))

        deporte = st.text_input("Deporte y nivel", value=p.get("Deporte y nivel", ""), placeholder="Ej: Atletismo / Vóleibol universitario")
        lesiones = st.text_area("Lesiones previas o molestias actuales", value=p.get("Lesiones previas", ""), height=80)

        with st.expander("B – G. Antecedentes Deportivos, Entrenamiento y Técnica CMJ (Opcional)", expanded=False):
            st.markdown("#### B. Experiencia deportiva general")
            experiencia_general = st.text_area("Describe tu experiencia previa y actual", value=p.get("Experiencia general", ""))

            st.markdown("#### C. Competencia y objetivos")
            competencias = st.text_area("¿Participas habitualmente en competencias?", value=p.get("Competencias", ""))
            temporada = st.text_area("¿En qué periodo del año tienes mayor carga?", value=p.get("Temporada", ""))
            objetivo = st.text_area("Principal objetivo deportivo actual", value=p.get("Objetivo", ""))

            st.markdown("#### D. Características del entrenamiento")
            duracion_sesion = st.text_area("Duración habitual de la sesión", value=p.get("Duración sesión", ""))
            grupos_musculares = st.text_area("Grupos musculares trabajados habitualmente", value=p.get("Grupos musculares", ""))
            ejercicios_fuerza = st.text_area("Ejercicios de fuerza de piernas", value=p.get("Fuerza extremidades", ""))
            cambios_direccion = st.text_area("Ejercicios de velocidad / cambios de dirección", value=p.get("Velocidad y dirección", ""))

            st.markdown("#### E. Experiencia pliométrica")
            tipos_saltos = st.text_area("Tipos de saltos habituales", value=p.get("Tipos de saltos", ""))
            tipo_apoyo = st.text_area("Saltos unipodales o bipodales", value=p.get("Tipo de apoyo", ""))
            dificultad_saltos = st.text_area("Dificultades al saltar", value=p.get("Dificultades de salto", ""))

            st.markdown("#### F. Percepción de la técnica CMJ")
            conocimiento_cmj = st.text_area("Descripción propia de la técnica CMJ", value=p.get("Conocimiento CMJ", ""))
            profundidad = st.text_area("Percepción de profundidad en el descenso", value=p.get("Profundidad", ""))
            aterrizaje = st.text_area("Percepción de la técnica de aterrizaje", value=p.get("Aterrizaje", ""))
            control_corporal = st.text_area("Control corporal durante el vuelo", value=p.get("Control corporal", ""))
            dificultad_tecnica = st.text_area("Parte más difícil de la ejecución", value=p.get("Dificultad técnica", ""))

            st.markdown("#### G. Antecedentes de evaluación física")
            evaluaciones = st.text_area("Evaluaciones de salto anteriores", value=p.get("Evaluaciones anteriores", ""))
            instrumentos = st.text_area("Instrumentos usados (plataforma, video, etc.)", value=p.get("Instrumentos", ""))
            resultados_previos = st.text_area("Resultados previos conocidos", value=p.get("Resultados previos", ""))
            seguimiento = st.text_area("Interés en seguimiento comparativo", value=p.get("Seguimiento", ""))

        guardar = st.form_submit_button(
            "Guardar ficha y continuar a Evaluación →", type="primary", use_container_width=True
        )

    if guardar:
        st.session_state.perfil_evaluacion = {
            "Nombre": nombre,
            "Sexo": sexo,
            "Edad": edad,
            "Estatura (cm)": estatura,
            "Peso (kg)": peso,
            "Fecha de nacimiento": fecha_nac.strftime("%d/%m/%Y") if fecha_nac else "",
            "Deporte y nivel": deporte,
            "Lesiones previas": lesiones,
            "Experiencia general": experiencia_general,
            "Competencias": competencias,
            "Temporada": temporada,
            "Objetivo": objetivo,
            "Duración sesión": duracion_sesion,
            "Grupos musculares": grupos_musculares,
            "Fuerza extremidades": ejercicios_fuerza,
            "Velocidad y dirección": cambios_direccion,
            "Tipos de saltos": tipos_saltos,
            "Tipo de apoyo": tipo_apoyo,
            "Dificultades de salto": dificultad_saltos,
            "Conocimiento CMJ": conocimiento_cmj,
            "Profundidad": profundidad,
            "Aterrizaje": aterrizaje,
            "Control corporal": control_corporal,
            "Dificultad técnica": dificultad_tecnica,
            "Evaluaciones anteriores": evaluaciones,
            "Instrumentos": instrumentos,
            "Resultados previos": resultados_previos,
            "Seguimiento": seguimiento,
        }
        st.session_state.pagina = "evaluacion"
        st.rerun()


# -----------------------------------------------------
# PANTALLA 2: EVALUACION (VIDEO GRABADO O CAMARA EN VIVO)
# -----------------------------------------------------
elif st.session_state.pagina == "evaluacion":
    if not st.session_state.perfil_evaluacion:
        st.session_state.pagina = "ficha"
        st.rerun()

    perfil = st.session_state.perfil_evaluacion
    estatura_cm = float(perfil.get("Estatura (cm)", 161))

    st.markdown('<span class="step-pill">PASO 2 · CAPTURA Y ANÁLISIS BIOMECÁNICO</span>', unsafe_allow_html=True)

    top1, top2 = st.columns([1, 4])
    with top1:
        if st.button("← Editar ficha", use_container_width=True):
            st.session_state.pagina = "ficha"
            st.rerun()
    with top2:
        with st.expander(f"Ver ficha de {perfil.get('Nombre') or 'Deportista'} ({estatura_cm:.0f} cm · {perfil.get('Peso (kg)', 60.0)} kg)"):
            st.json(perfil)

    st.divider()

    # Configuración común de cámara y perspectiva
    st.subheader("1. Configuración de la toma")
    cfg1, cfg2 = st.columns([1, 1])
    with cfg1:
        modo = st.radio(
            "Método de evaluación",
            ["Subir video grabado", "Cámara en directo"],
            horizontal=True,
        )
        vista = st.radio(
            "Perspectiva de la cámara",
            ["Lateral", "Frontal"],
            horizontal=True,
        )
        if vista == "Lateral":
            lado = st.selectbox("¿Qué lado del cuerpo ve la cámara?", ["Derecho", "Izquierdo"])
        else:
            lado = None
    with cfg2:
        st.info(
            "**Recomendaciones para máxima precisión:**\n"
            "- Mantener la cámara fija y estable.\n"
            "- Dejar visible el cuerpo completo (cabeza a pies).\n"
            "- Comenzar de pie y quieto ~1 segundo para calibrar el origen (0,0).\n"
            "- En video grabado, idealmente usar 60 fps o más."
        )

    if not os.path.exists(MODEL_PATH):
        st.error("Falta el archivo `pose_landmarker_lite.task` en la carpeta del proyecto.")
        st.stop()

    st.divider()

    # -------------------------------------------------
    # MODO A: SUBIR VIDEO GRABADO (CON H.264 Y VISOR)
    # -------------------------------------------------
    if modo == "Subir video grabado":
        st.subheader("2. Sube el video del salto")
        video_subido = st.file_uploader(
            "Selecciona un archivo de video (MP4, MOV, AVI)",
            type=["mp4", "mov", "avi"],
            key="cargador_video",
        )

        if video_subido is not None:
            bytes_nuevos = video_subido.getvalue()
            firma_nueva = (video_subido.name, len(bytes_nuevos))
            if st.session_state.get("firma_video") != firma_nueva:
                st.session_state["firma_video"] = firma_nueva
                st.session_state["video_bytes"] = bytes_nuevos
                st.session_state["video_nombre"] = video_subido.name
                st.session_state.pop("video_original_path", None)
                st.session_state.pop("video_preview_bytes", None)
                st.session_state.pop("resultado_video", None)
                st.session_state.pop("visor_cuadro", None)

        if "video_bytes" not in st.session_state:
            st.warning("Selecciona un video para habilitar la vista previa y el análisis.")
            st.stop()

        if "video_original_path" not in st.session_state:
            ext = os.path.splitext(st.session_state["video_nombre"])[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(st.session_state["video_bytes"])
                st.session_state["video_original_path"] = tmp.name

        ruta_original = st.session_state["video_original_path"]

        if "video_preview_bytes" not in st.session_state:
            with st.spinner("Preparando vista previa compatible con navegador..."):
                ruta_prev, err_prev = a_h264(ruta_original)
                try:
                    with open(ruta_prev, "rb") as f:
                        st.session_state["video_preview_bytes"] = f.read()
                    st.session_state["error_preview"] = err_prev
                except Exception as e:
                    st.session_state["video_preview_bytes"] = st.session_state["video_bytes"]
                    st.session_state["error_preview"] = str(e)

        st.markdown("#### Vista previa")
        st.video(st.session_state["video_preview_bytes"])

        st.subheader("3. Opciones de análisis")
        opt1, opt2 = st.columns(2)
        with opt1:
            mostrar_grilla = st.checkbox("Mostrar cuadrícula de 20 cm sobre el video", value=True)
        with opt2:
            ventana = st.slider(
                "Suavizado de señales (cuadros)",
                min_value=1, max_value=9, value=3, step=2,
                help="1 = sin suavizar. Valores mayores reducen ruido pero suavizan picos."
            )

        if st.button("🚀 Analizar video completo", type="primary", use_container_width=True):
            st.session_state.pop("resultado_video", None)
            st.session_state.pop("visor_cuadro", None)

            with st.spinner("Calibrando ejes (0,0) y escala corporal..."):
                cal = calibrar_video(ruta_original, estatura_cm)
            if cal is None:
                st.error("No se pudo leer el video. Revisa que el archivo no esté dañado.")
                st.stop()

            barra = st.progress(0.0, text="Iniciando tracking cinemático...")
            tabla, fps, ruta_crudo = procesar_video_completo(
                ruta_original, vista, lado, cal, mostrar_grilla, barra
            )
            barra.empty()

            u = cal["unidad"]
            col_cm_y = f"CM y ({u})"
            if tabla.empty or col_cm_y not in tabla.columns or tabla[col_cm_y].isna().all():
                st.error("No se detectó a la persona en el video. Verifica encuadre e iluminación.")
                st.stop()

            # Interpolación y suavizado de columnas numéricas
            cols_num = tabla.select_dtypes(include=[np.number]).columns
            tabla[cols_num] = (
                tabla[cols_num]
                .interpolate(limit_direction="both")
                .rolling(ventana, center=True, min_periods=1)
                .mean()
            )
            tabla.insert(0, "Tiempo (s)", np.round(np.arange(len(tabla)) / fps, 3))

            with st.spinner("Codificando video con tracking para el navegador (H.264)..."):
                ruta_web, err_web = a_h264(ruta_crudo)
                with open(ruta_web, "rb") as f:
                    bytes_web = f.read()

            st.session_state["resultado_video"] = {
                "tabla": tabla,
                "fps": fps,
                "vista": vista,
                "cal": cal,
                "video_bytes": bytes_web,
                "error_video": err_web,
                "video_crudo": ruta_crudo,
            }
            st.rerun()

        if "resultado_video" in st.session_state:
            r = st.session_state["resultado_video"]
            u = r["cal"]["unidad"]
            col_cm_y = f"CM y ({u})"
            datos = r["tabla"]

            st.divider()
            st.header("Resultados del análisis de video")
            st.caption(f"Frecuencia detectada: {r['fps']:.0f} fps · Escala: 1 px = {r['cal']['k']:.3f} {u}")
            if r["fps"] < 60:
                st.warning("El video tiene menos de 60 fps; las mediciones temporales pueden tener mayor margen de error.")

            st.subheader("🎥 Video con tracking cinemático")
            st.video(r["video_bytes"])
            st.download_button(
                "⬇️ Descargar video con tracking (MP4)",
                data=r["video_bytes"],
                file_name="zenithz_markerless.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            st.subheader("🎞️ Visor cuadro por cuadro")
            if len(datos) > 1:
                idx_pico_default = int(datos[col_cm_y].idxmax())
                indice = st.slider(
                    "Desplaza para revisar cada cuadro (inicia en el pico del salto)",
                    0, len(datos) - 1, idx_pico_default, key="visor_cuadro"
                )
                img = cuadro_de_salida(r["video_crudo"], indice)
                if img is not None:
                    st.image(
                        img,
                        caption=f"Cuadro {indice} · Tiempo: {indice / r['fps']:.3f} s",
                        use_container_width=True,
                    )

            st.divider()
            mostrar_dashboard_resultados(datos, r["cal"], r["vista"], "Video grabado")

    # -------------------------------------------------
    # MODO B: CAMARA EN DIRECTO (WEBRTC SEGURO)
    # -------------------------------------------------
    else:
        st.subheader("2. Evaluación mediante cámara en directo")
        mostrar_grilla_vivo = st.checkbox("Mostrar cuadrícula de 20 cm en vivo", value=True)
        with registro["lock"]:
            registro["mostrar_grilla"] = mostrar_grilla_vivo

        try:
            modelo_vivo = cargar_modelo_vivo(MODEL_PATH)
            procesador = crear_procesador_vivo(modelo_vivo, registro, vista, lado, estatura_cm)
            ice_servers, tiene_turn = obtener_ice_servers()

            col_esp1, col_cam, col_esp2 = st.columns([1, 2, 1])
            with col_cam:
                ctx = webrtc_streamer(
                    key="zenithz-camera-fused",
                    video_frame_callback=procesador,
                    media_stream_constraints={
                        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                        "audio": False,
                    },
                    async_processing=True,
                    rtc_configuration={"iceServers": ice_servers},
                    video_html_attrs={
                        "style": {
                            "width": "100%",
                            "border-radius": "12px",
                            "border": "3px solid #5C8CE6",
                            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.2)",
                        },
                        "playsinline": True,
                        "autoplay": True,
                    },
                )
                if ctx.state.signalling and not ctx.state.playing:
                    st.caption("Conectando con la cámara... Si tarda más de 15 s, revisa permisos o usa 'Subir video grabado'.")
                if not tiene_turn:
                    st.caption("ℹ️ Ejecutando solo con STUN (sin secretos TURN configurados en este entorno).")

            st.markdown("### 🎛️ Control del registro en vivo")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔴 Iniciar registro (Anclar ejes 0,0)", use_container_width=True):
                    with registro["lock"]:
                        registro["muestras"].clear()
                        registro["trayectoria"].clear()
                        registro["inicio"] = None
                        registro["ultima_muestra"] = -1.0
                        registro["grabando"] = True
                    st.success("¡Coordenadas ancladas! Realiza tu salto CMJ.")
            with b2:
                if st.button("⏹️ Detener registro", use_container_width=True):
                    with registro["lock"]:
                        registro["grabando"] = False
                    st.success("Registro detenido.")

            with registro["lock"]:
                grabando = registro["grabando"]
                datos_vivo = list(registro["muestras"])
                cal_vivo = registro["cal"]

            if grabando:
                st.info("Recopilando métricas cinemáticas en tiempo real...")
            st.metric("Muestras registradas", len(datos_vivo))

            b3, b4 = st.columns(2)
            with b3:
                ver_res = st.button("📊 Procesar y Ver Resultados", type="primary", use_container_width=True)
            with b4:
                limpiar = st.button("🗑️ Limpiar registro", use_container_width=True)

            if limpiar:
                with registro["lock"]:
                    registro["muestras"].clear()
                    registro["trayectoria"].clear()
                    registro["inicio"] = None
                    registro["ultima_muestra"] = -1.0
                    registro["grabando"] = False
                st.rerun()

            if ver_res:
                if not datos_vivo or cal_vivo is None:
                    st.warning("Primero inicia la cámara, presiona 'Iniciar registro' y ejecuta un salto.")
                else:
                    df_vivo = pd.DataFrame(datos_vivo)
                    mostrar_dashboard_resultados(df_vivo, cal_vivo, vista, "Cámara en vivo")

        except Exception as error:
            st.error(f"Error al iniciar la cámara: {error}")

st.divider()
st.caption(
    "Zenithz | Herramienta de Ciencias del Deporte · Centro de masa calculado con modelo de segmentos de Dempster (2D) "
    "y ejes anclados al suelo entre los talones."
)
