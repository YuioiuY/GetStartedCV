import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread("five_pr\\img\\input_image.jpg", cv2.IMREAD_GRAYSCALE)

# Применение фильтра Гаусса для сглаживания изображения
gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)

# Применение фильтров Собеля для выделения краев
sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)

# Комбинирование результатов Собеля для получения общего изображения градиентов
sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

# Отображение и сохранение результатов
cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blur", gaussian_blur)
cv2.imshow("Sobel Edge Detection", sobel_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
