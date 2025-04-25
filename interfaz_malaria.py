import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
import sys
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB0

class MalariaClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NhAI-Malaria Classifier")
        # self.root.geometry("900x700")
        self.root.geometry("1000x2000")  # Tamaño más grande
        self.root.configure(bg="#f0f0f0")  # Fondo claro
        
        # Configuración de rutas
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Configuración de modelos
        self.models = {
            "ResNet50": os.path.join(self.BASE_DIR, "modelos", "resnet50_malaria.h5"),
            "InceptionV3": os.path.join(self.BASE_DIR, "modelos", "inceptionv3_malaria.h5"),
            "EfficientNetB0": os.path.join(self.BASE_DIR, "modelos", "efficientnetb0_malaria.h5")
        }
        self.loaded_models = {}
        
        # Variables de control
        self.image_path = tk.StringVar()
        self.selected_model = tk.StringVar(value="ResNet50")
        self.prediction_result = tk.StringVar(value="Seleccione una imagen y modelo")
        self.probability = tk.StringVar(value="")
        
        # Interfaz gráfica
        self.create_widgets()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de imagen
        img_frame = ttk.LabelFrame(main_frame, text="Imagen", padding="10")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.image_label = ttk.Label(img_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Botones de carga
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Cargar Imagen", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Imagen Aleatoria", command=self.load_random_image).pack(fill=tk.X, pady=2)
        
       
        # Selección de modelo
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(model_frame, text="Modelo:").pack()
        for model_name in self.models.keys():
            ttk.Radiobutton(model_frame, text=model_name, variable=self.selected_model, 
                           value=model_name).pack(anchor=tk.W)
        
        # Botón de predicción
        ttk.Button(control_frame, text="Clasificar", command=self.predict_image).pack(side=tk.RIGHT, padx=5)
        
        # Panel de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultado", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(result_frame, textvariable=self.prediction_result, font=('Arial', 12, 'bold')).pack()
        ttk.Label(result_frame, textvariable=self.probability, font=('Arial', 10)).pack()
        
        # Créditos
        ttk.Label(main_frame, text="Malaria Classifier – © 2025 by Researcher Ñahui-Vargas Luis-Edison", foreground="gray").pack(side=tk.BOTTOM)
   

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            self.image_path.set(file_path)
            self.display_image(file_path)
    
    def load_random_image(self):
        try:
            dataset_path = os.path.join(self.BASE_DIR, "cell_images")
            classes = ["Parasitized", "Uninfected"]
            selected_class = random.choice(classes)
            class_path = os.path.join(dataset_path, selected_class)
            
            if os.path.exists(class_path):
                images = [img for img in os.listdir(class_path) 
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    random_image = random.choice(images)
                    image_path = os.path.join(class_path, random_image)
                    self.image_path.set(image_path)
                    self.display_image(image_path)
                else:
                    self.prediction_result.set("No hay imágenes en el directorio")
            else:
                self.prediction_result.set(f"Directorio no encontrado: {class_path}")
        except Exception as e:
            self.prediction_result.set(f"Error: {str(e)}")
    
    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((500, 500))
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            self.prediction_result.set("Imagen cargada correctamente")
            self.probability.set("")
        except Exception as e:
            self.prediction_result.set(f"Error al cargar imagen: {str(e)}")
    
    def load_model(self, model_name):
        if model_name not in self.loaded_models:
            try:
                model_path = self.models[model_name]
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Archivo no encontrado: {model_path}")
                
                self.loaded_models[model_name] = load_model(model_path, compile=False)
                print(f"Modelo {model_name} cargado exitosamente")
            except Exception as e:
                print(f"Error cargando modelo {model_name}: {str(e)}")
                return None
        return self.loaded_models[model_name]



    # def load_model(self, model_name):
    #     if model_name not in self.loaded_models:
    #         try:
    #             # Reconstruye la arquitectura base
    #             if model_name == "ResNet50":
    #                 base_model = ResNet50(weights=None, include_top=False)
    #             elif model_name == "InceptionV3":
    #                 base_model = InceptionV3(weights=None, include_top=False)
    #             elif model_name == "EfficientNetB0":
    #                 base_model = EfficientNetB0(weights=None, include_top=False)
                    
    #             # Carga solo los pesos
    #             model_path = self.models[model_name]
    #             base_model.load_weights(model_path)
                
    #             self.loaded_models[model_name] = base_model
    #             print(f"Modelo {model_name} cargado exitosamente")
    #             return base_model
                
    #         except Exception as e:
    #             print(f"Error cargando modelo {model_name}: {str(e)}")
    #             return None
    #     return self.loaded_models[model_name]
    
    def preprocess_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = img.resize((224, 224))  # Tamaño esperado por los modelos
            img_array = np.array(img) / 255.0  # Normalización
            img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
            return img_array
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            return None
    
    def predict_image(self):
        if not self.image_path.get():
            self.prediction_result.set("Primero carga una imagen")
            return
            
        model_name = self.selected_model.get()
        model = self.load_model(model_name)
        
        if model is None:
            self.prediction_result.set(f"Error al cargar modelo {model_name}")
            return
        
        try:
            img_array = self.preprocess_image(self.image_path.get())
            if img_array is None:
                self.prediction_result.set("Error al procesar imagen")
                return
                
            prediction = model.predict(img_array)
            probability = float(prediction[0][0])
            
            if probability > 0.5:
                result = "No Parasitado (Sano)"
                prob_percent = probability * 100
            else:
                result = "Parasitado (Infectado)"
                prob_percent = (1 - probability) * 100
                
            self.prediction_result.set(f"Resultado ({model_name}): {result}")
            self.probability.set(f"Probabilidad: {prob_percent:.2f}%")
            
        except Exception as e:
            self.prediction_result.set(f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MalariaClassifierApp(root)
    root.mainloop()