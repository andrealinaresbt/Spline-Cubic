import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Carga la imagen y la convierte a escala de grises
def loadImages(path):
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Detecta los bordes de la imagen usando el algoritmo de Canny
def edgeDetection(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(blurred, 10, 80)
    return edges

def countourExtraction(edges, image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(cleaned, kernel, iterations=3)
    closed = cv2.erode(dilated, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print("Contornos encontrados: ", len(contours))

    if not contours:
        raise ValueError("No se encontraron contornos")

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour[:, 0, :]
    height, _ = edges.shape
    upper_points = points[points[:, 1] < height * 0.9]

    if len(upper_points) < 2:
        raise ValueError("No se detectaron suficientes puntos superiores")

    unique_points = {}
    for point in upper_points:
        x, y = point
        if x not in unique_points or y < unique_points[x]:
            unique_points[x] = y

    unique_x = np.array(sorted(unique_points.keys()))
    unique_y = np.array([unique_points[x] for x in unique_x])
    unique_points = np.column_stack((unique_x, unique_y))

    return unique_points

def CalcAndPlotCubicSpline(points, original_image):
    x = points[:, 0]
    y = points[:, 1]

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]

    if len(unique_x) < 2:
        raise ValueError("No hay suficientes puntos únicos para la interpolación")

    cs = CubicSpline(unique_x, unique_y)

    # Graficar la imagen con los puntos y el spline
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:, 0], points[:, 1], color='green', s=20, label='Puntos del Contorno')
    x_new = np.linspace(points[:, 0].min(), points[:, 0].max(), 1000)
    y_new = cs(x_new)
    plt.plot(x_new, y_new, color='blue', label='Spline Cúbico')
    
    # Mostrar solo una ecuación del spline cúbico
    i = 0  # Seleccionar el primer intervalo como ejemplo
    coef = cs.c[:, i]
    x_interval = np.linspace(unique_x[i], unique_x[i+1], 100)
    y_interval = cs(x_interval)
    plt.plot(x_interval, y_interval, color='blue')
    
    equation = f"S(x) = {coef[3]:.4f}(x - {unique_x[i]:.2f})^3 + {coef[2]:.4f}(x - {unique_x[i]:.2f})^2 + {coef[1]:.4f}(x - {unique_x[i]:.2f}) + {coef[0]:.4f}"
    plt.text((unique_x[i] + unique_x[i+1]) / 2, np.mean(y_interval), equation, fontsize=8, color='red')
    print('Ecuacion: '+ " "+ equation)
    plt.legend()
    plt.title('Contorno Superior y Spline Cúbico')
    plt.show()

def main(image_path):
    grayImage = loadImages(image_path)
    edges = edgeDetection(grayImage)
    originalImage = cv2.imread(image_path)
    points = countourExtraction(edges, originalImage)
    CalcAndPlotCubicSpline(points, originalImage)

# Ruta de la imagen
path = 'Spline-Cubic/imagen.jpg'
main(path)
