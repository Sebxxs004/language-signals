import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Menu, messagebox
from PIL import Image, ImageTk
import pyttsx3
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

numero_letras = 3 #AGREGAR CANTIDAD DE LETRAS
cant_imagenes = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
engine = pyttsx3.init()

letras_reconocidas = []
letra_anterior = None
frames_repetidos = 0 

def obtener_palabras(letras):
    if len(letras) < 2:
        return []
    
    letras_combinadas = "".join(letras[-2:])
    url = f"https://api.datamuse.com/words?sp=*{letras_combinadas}*&v=es&max=5"
    try:
        response = requests.get(url)
        palabras = [word['word'].upper() for word in response.json()]
        return palabras
    except requests.RequestException as e:
        print("Error al conectar con Datamuse:", e)
        return ["Error al obtener palabras"]

def recolectar_imagenes():
    cap = cv2.VideoCapture(0)
    for j in range(numero_letras):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        contador = 0
        while contador < cant_imagenes:
            ret, frame = cap.read()
            cv2.putText(frame, f'Recolectando frames para la letra {j}.', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Presiona "k" para continuar.', 
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('k'):
                cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{contador}.jpg'), frame)
                contador += 1
            

    cap.release()
    cv2.destroyAllWindows()
    entrenar_modelo()

def entrenar_modelo():
    data, labels = [], []
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux, x_, y_ = [], [], []
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                data.append(data_aux)
                labels.append(dir_)

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    entrenar_clasificador()

def entrenar_clasificador():
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    data, labels = np.asarray(data_dict['data']), np.asarray(data_dict['labels'])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f'{score * 100:.2f}% de precisión en la clasificación.')

    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)

def probar_lectura(texto_sugerencias):
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    labels_dict = {0: 'A', 1: 'B', 2: 'L'} #AGREGAR LAS LETRAS

    cap = cv2.VideoCapture(0)
    global letra_anterior, frames_repetidos

    while True:
        data_aux, x_, y_ = [], [], []
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for i in range(len(hand_landmarks.landmark)):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character == letra_anterior:
                frames_repetidos += 1
            else:
                letra_anterior = predicted_character
                frames_repetidos = 0

            if frames_repetidos < 15:
                letras_reconocidas.append(predicted_character)

            engine.say(predicted_character)
            engine.runAndWait()

            if len(letras_reconocidas) >= 2:
                palabras_sugeridas = obtener_palabras(letras_reconocidas)
                mensaje_sugerencias = "Sugerencias: " + ", ".join(palabras_sugeridas)
                texto_sugerencias.delete(1.0, tk.END)
                texto_sugerencias.insert(tk.END, mensaje_sugerencias)

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_imagen.config(image=img_tk)
        label_imagen.image = img_tk

        ventana_lectura.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def abrir_ventana_lectura():

    global ventana_lectura
    ventana_lectura = tk.Toplevel(ventana)
    ventana_lectura.title("Lectura de Señas en Vivo")
    ventana_lectura.geometry("800x460")
    
    frame_lectura = tk.Frame(ventana_lectura, width=800, height=350, bg="lightgray")
    frame_lectura.pack(padx=10, pady=10)
    
    texto_sugerencias = tk.Text(frame_lectura, height=6, width=30, font=("Arial", 14))
    texto_sugerencias.grid(row=0, column=1, padx=10, pady=10)
    
    global label_imagen
    label_imagen = tk.Label(frame_lectura)
    label_imagen.grid(row=0, column=0, padx=10, pady=10)

    probar_lectura(texto_sugerencias)

ventana = tk.Tk()
ventana.title("Programa de Lectura de Señas")
ventana.geometry("800x470")
ventana.iconbitmap("icon.ico")

try:
    imagen_fondo = Image.open("img-src.png")
    imagen_fondo = imagen_fondo.resize((800, 500), Image.LANCZOS)
    imagen_fondo = ImageTk.PhotoImage(imagen_fondo)
    label_fondo = tk.Label(ventana, image=imagen_fondo, bg="white")
    label_fondo.place(x=0, y=0)
except Exception as e:
    print("Error al cargar la imagen:", e)

titulo = tk.Label(ventana, text="PROGRAMA DE LECTURA\nDE SEÑAS", font=("Arial", 36, "bold"), bg="white", fg="black")
titulo.place(x=80, y=80)

boton_lectura = tk.Button(ventana, text="INICIAR LECTURA DE SEÑAS", font=("Arial", 16), command=abrir_ventana_lectura)
boton_lectura.place(x=260, y=250)

def mostrar_menu(event):
    menu = Menu(ventana, tearoff=0)
    menu.add_command(label="Información del programa", command=lambda: messagebox.showinfo("Información", "Este programa ayuda a interpretar señas en tiempo real.\n EL recolecta imagenes las cuales\n son procesadas y guardadas en una red la cual\n tras la lectura de la mano, detecta una letra\n y luego sugiere palabras que podrian contener dichas letras."))
    menu.add_command(label="Recolectar imágenes", command=recolectar_imagenes)
    menu.post(event.x_root, event.y_root)

boton_ayuda = tk.Button(ventana, text="?", font=("Arial", 24, "bold"), fg="red", bg="white", borderwidth=0)
boton_ayuda.bind("<Button-1>", mostrar_menu)
boton_ayuda.place(x=10, y=400)

ventana.mainloop()
