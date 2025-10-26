import streamlit as st

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Contador de curls")

st.title("🏋️ Contador de Curls en Tiempo Real")
st.write("Permite usar la cámara para detectar movimientos de bíceps y contar repeticiones automáticamente.")

# Inicializar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# Variables del contador
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "stage" not in st.session_state:
    st.session_state.stage = None

# Cámara en tiempo real
frame_window = st.image([])

run = st.toggle("Activar cámara")

# Configuración de Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se detecta cámara o está siendo usada por otra aplicación.")
            break

        # Procesar imagen
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            # Coordenadas relevantes
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calcular ángulos
            left_angle_bc = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle_bc = calculate_angle(right_shoulder, right_elbow, right_wrist)

            left_angle_st = calculate_angle(left_hip, left_shoulder, left_elbow)
            right_angle_st = calculate_angle(right_hip, right_shoulder, right_elbow)

            # Condiciones
            down_cond = left_angle_bc > 160 and right_angle_bc > 160
            contract_cond_left = left_angle_bc < 30 and left_angle_st < 20
            contract_cond_right = right_angle_bc < 30 and right_angle_st < 20

            if down_cond:
                st.session_state.stage = "down"
            if (contract_cond_left and contract_cond_right and
                    st.session_state.stage == "down"):
                st.session_state.stage = "up"
                st.session_state.counter += 1

            # Dibujar pose
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            pass

        # Mostrar datos
        cv2.putText(image, f'Reps: {st.session_state.counter}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Stage: {st.session_state.stage}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        frame_window.image(image, channels="RGB")

    cap.release()
else:
    st.info("Activa la cámara para comenzar el conteo de repeticiones.")

pose.close()
