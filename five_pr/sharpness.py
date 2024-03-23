import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread("five_pr\\img\\input_image.jpg")

# Применение фильтра увеличения резкости
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

# Отображение и сохранение результатов
cv2.imshow("Original Image", image)
cv2.imshow("Sharpened Image", sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
