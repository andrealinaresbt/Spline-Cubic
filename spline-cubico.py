import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline



#se cargan las imagenes y se convierten a escala de grises
def loadImages(path):
    image = cv2.imread(path)
    #Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image

#Se detectan los bordes de la imagen basado en el algoritmo de Canny
def edgeDetection(gray_image):
    #Se desenfoca la imagen para reducir el ruido usando el filtro Gaussiano, en este caso usamos una matriz de 7x7 y 0 de desviación estándar
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    #Detectar bordes usando el detector de Canny. 10 se usa como umbral inferior y 80 como superior
    edges = cv2.Canny(blurred, 10, 80)
    return edges

def countourExtraction(edges, image):
    # Definir un kernel de tamaño 5x5 para operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Aplicar operación de cierre morfológico (dilatación seguida de erosión)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Aplicar dilatación para expandir áreas blancas en la imagen
    dilated = cv2.dilate(cleaned, kernel, iterations=3)

    # Aplicar erosión para reducir áreas dilatadas a su tamaño original
    closed = cv2.erode(dilated, kernel, iterations=3)

    # Encontrar todos los contornos en la imagen procesada
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print("Contornos encontrados: ", len(contours))

    if not contours:
        raise ValueError("No se encontraron contornos")


    # Seleccionar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    #agarramos todos los puntos del contorno mayor y todas sus dimensiones (x,y)
    points = largest_contour[:, 0, :]
    #altura del contorno
    height, _ = edges.shape

    # Seleccionar puntos que están en la parte superior de la imagen (por encima del 90% de la altura)
    upper_points = points[points[:, 1] < height * 0.9]

    if len(upper_points) < 2:
        raise ValueError("No se detectaron suficientes puntos superiores")

    # Crear un diccionario para eliminar puntos duplicados en el eje x
    unique_points = {}
    for point in upper_points:
        x, y = point
        if x not in unique_points or y < unique_points[x]:
            unique_points[x] = y

    # Convertir el diccionario en arrays numpy ordenados por x
    unique_x = np.array(sorted(unique_points.keys()))
    unique_y = np.array([unique_points[x] for x in unique_x])
    #Crea un array de puntos X, Y 
    unique_points = np.column_stack((unique_x, unique_y))

    return unique_points



def CalcCubicSpline(points):
    x = points[:, 0]
    y = points[:, 1]

    #Se ordenan los puntos
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]


    #Se eliminan los puntos duplicados
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]

    if len(unique_x) < 2:
        raise ValueError("Not enough unique points for spline interpolation")

    cs = CubicSpline(unique_x, unique_y)

    return cs

def Plotting(original_image, points, spline):
    plt.figure(figsize=(10, 6))
    #se muestra la imagen original con sus colores originales
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #se muestran los puntos del contorno superior en rojo
    plt.scatter(points[:, 0], points[:, 1], color='red', s=5, label='Puntos encontrados del Contorno')

    #se crea un array de 1000 puntos entre el punto mínimo y máximo de los puntos del contorno
    x_new = np.linspace(points[:, 0].min(), points[:, 0].max(), 1000)
    #se interpola la función en los puntos obtenidos de la linea anterior
    y_new = spline(x_new)
    plt.plot(x_new, y_new, color='blue', label='Interpolación Spline Cúbico')

    plt.legend()
    plt.title('Contorno Superior y Spline Cúbico')
    plt.show()

def main(image_path):
    #cargamos imagen y la convertimos en gris
    grayImage = loadImages(image_path)
    #detectamos los bordes de la imagen
    edges = edgeDetection(grayImage)
    #cargamos la imagen original
    originalImage = cv2.imread(image_path)
    #extraemos los puntos del contorno
    points = countourExtraction(edges, originalImage)

    #aplicamos spline cubico
    cs = CalcCubicSpline(points)

    #Graficamos en la imagen
    Plotting(originalImage, points, cs)

path = 'imagen.jpg'
main(path)