# NhAI-Malaria-Classifier
Sistema de escritorio realizado en python para deteccion de celulas de malaria

NhAI-Malaria-Classifier is a desktop application developed in Python for the automated detection of Plasmodium-infected blood cells, the parasite responsible for malaria, using convolutional neural network (CNN) models. This system was developed as part of the research "Detection of Malaria Infections Using Convolutional Neural Networks, 2025", which evaluated the performance of three pre-trained architectures: EfficientNetB0, InceptionV3, and ResNet50.

The system aims to provide an accessible and efficient tool for malaria diagnosis, particularly useful in clinical and educational contexts with limited resources. It features an intuitive graphical user interface (GUI) that allows users to upload blood smear images and visualize model predictions, combining the power of deep learning with user-friendly interaction.

If you're interested in artificial intelligence applied to healthcare, we invite you to read the full article, which details the methodology, results, comparative analysis, and the feasibility of these solutions for infectious disease diagnosis.

## 🔧 Guía de Instalación y Ejecución

### 1. Descargar el Repositorio
* Descarga el repositorio completo desde:

https://github.com/lu-mala/NhAI-Malaria-Classifier

---

### 2. Preparar las Subcarpetas
Descomprime los siguientes archivos antes de ejecutar el sistema:

- Dentro de la carpeta `cell_images/`:
  - `Parasitized/`
  - `Uninfected/`

- Dentro de la carpeta `modelos/`:
  - `InceptionV3/`
  - `ResNet50/`

---

### 3. Crear y Activar un Entorno Virtual

Ejecuta los siguientes comandos desde la raíz del proyecto:

- python -m venv venv
- .\venv\Scripts\activate

💡 En Linux/macOS usa: 
source venv/bin/activate

---
### 4. Instalar las Dependencias
Instala las bibliotecas necesarias (versiones compatibles):

- pip install tensorflow==2.17.0 pillow==10.1.0 numpy==1.24.4 opencv-python==4.8.1.78

---
### 5. Verificar TensorFlow y GPU (opcional)
Asegúrate de que TensorFlow se instaló correctamente:
- python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print('GPU disponible:', bool(tf.config.list_physical_devices('GPU')))"

✅ Salida esperada:

###### TensorFlow 2.17.0

###### GPU disponible: True  o False si no se detecta GPU

---
### 6. Ejecutar la Interfaz Gráfica
Lanza la interfaz con:
- python interfaz_malaria.py
