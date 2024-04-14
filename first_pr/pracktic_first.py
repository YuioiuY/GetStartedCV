#region libs
import cv2
import numpy as np 
from PIL import Image as im 
from PIL import ImageFilter
import sys
#endregion
#shift+alt+a

#region PILLOW

#region start
""" image = im.open('img/street.jpg')
image.show() """
#endregion

#region crop

""" image = im.open('img/street.jpg')
cropped = image.crop((0, 80, 200, 400))
cropped.save('new_street.png')
 """
#endregion

#region rotate

""" try:
    image = im.open("img/street.jpg")
except IOError:
    print("Unable to load image")
    sys.exit(1)
    
rotated = image.rotate(180)
rotated.save('new_street.jpg')
 """
#endregion

#region filter 

""" image = im.open('img/street.jpg')
blurred_jelly = image.filter(ImageFilter.BLUR) #ImageFilter.SHARPEN
blurred_jelly.save('new_street.png') """

#endregion

#region GrayScale
""" try:
    image = im.open("img/street.jpg")
except IOError:
    print("Unable to load image")
    sys.exit(1)
    
grayscale = image.convert('L')
grayscale.show() """
#endregion

#region resize

""" image = im.open("img/street.jpg")
image = image.resize((100, 100), im.ANTIALIAS)
image.show() """
#endregion

#region resize
""" image = im.open("img/street.jpg")
 
width, height = image.size
new_width  = 680 # ширина
new_height = int(new_width * height / width)
 
image = image.resize((new_width, new_height), im.ANTIALIAS)
image.show() """
#endregion

#region anim
""" square_animation = []
for offset in range(0, 100, 2):
    red = np.zeros((600, 600))
    green = np.zeros((600, 600))
    blue = np.zeros((600, 600))
    red[101 + offset : 301 + offset, 101 + offset : 301 + offset] = 255
    green[200:400, 200:400] = 255
    blue[299 - offset : 499 - offset, 299 - offset : 499 - offset] = 255
    red_img = im.fromarray(red).convert("L")
    green_img = im.fromarray(green).convert("L")
    blue_img = im.fromarray((blue)).convert("L")
    square_animation.append(
        im.merge(
            "RGB",
            (red_img, green_img, blue_img)
        )
    )

square_animation[0].save(
    "animation.gif", save_all=True, append_images=square_animation[1:]
) """
#endregion

#endregion

# region OPENCV

#region first try

photo_file = cv2.imread('img/street.jpg') # write image

print(photo_file)   # read image
print(type(photo_file)) # <class 'numpy.ndarray'>

cv2.imshow('photo_file',photo_file) # show image

cv2.waitKey(0) # wait for quit

#endregion

#region chage something

""" photo_file_to_chage = cv2.imread('img/street.jpg') # write image

photo_file_to_chage = cv2.rotate(photo_file_to_chage, cv2.ROTATE_90_CLOCKWISE) # rotate

photo_file_to_chage = cv2.resize(photo_file_to_chage,(480, 640)) # resize

cv2.imshow('photo_file',photo_file_to_chage) # show image

cv2.waitKey(0) # wait for quit """

#endregion

#region noise

# Load the image

'''В следующей программе мы использовали функцию cv2.randn() для генерации гауссовского шума, 
   распределение которого определяется средним значением и значением стандартного отклонения. 
   Затем мы добавляем этот шум к изображению и сохраняем зашумленное изображение.'''


""" 
img = cv2.imread('img/street.jpg')

# Generate random Gaussian noise
mean = 0
stddev = 180
noise = np.zeros(img.shape, np.uint8)
cv2.randn(noise, mean, stddev)

noisy_img = cv2.add(img, noise)# Add noise to image

cv2.imshow('photo_file',noisy_img) # show image

cv2.waitKey(0) # wait for quit 
"""


#endregion

#region hsv
'''
Очень часто с изображениями работают как с массивом пикселей в формате RGB. 
Хотя такое представление оказывается относительно интуитивным, его нельзя назвать оптимальным для задачи, которую нам предстоит решить. 
В RGB цвет пикселя определяется насыщенностью красным, зеленым и голубым. 
Таким образом, выбор диапазона оттенков одного и того же цвета становится не самой простой задачей.

С форматом HSV дела обстоят иначе. Эта цветовая схема определяется тремя компонентами:

Hue - цветовой тон;
Saturation - насыщенность;
Value - яркость.
'''

""" 
photo_file = cv2.imread('jupiter/img/street.jpg') # write image

# Take each frame
frame = photo_file
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110,50,50]) # lower blue color
upper_blue = np.array([130,255,255]) # upper blue color

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask= mask)

#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
cv2.imshow('res',res)
k = cv2.waitKey(0)  
"""
    

#endregion

#region contours

""" # параметры цветового фильтра
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

img = cv2.imread('img/street.png')

hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV 
thresh = cv2.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
# ищем контуры и складируем их в переменную contours
contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# отображаем контуры поверх изображения
cv2.drawContours( img, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
cv2.imshow('contours', img) # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows() """

#endregion

#region contours2

""" # параметры цветового фильтра
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

img = cv2.imread('img/street.png')

hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV 
thresh = cv2.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
# ищем контуры и складируем их в переменную contours
contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
index = 0
layer = 0

def update():
   vis = img.copy()
   cv2.drawContours( vis, contours, index, (255,0,0), 2, cv2.LINE_AA, hierarchy, layer )
   cv2.imshow('contours', vis)

def update_index(v):
   global index
   index = v-1
   update()

def update_layer(v):
   global layer
   layer = v
   update()

update_index(0)
update_layer(0)
cv2.createTrackbar( "contour", "contours", 0, 7, update_index )
cv2.createTrackbar( "layers", "contours", 0, 7, update_layer )

cv2.waitKey()
cv2.destroyAllWindows() """
#endregion

#region contours3
""" def nothing(*arg):
        pass



cv2.namedWindow( "result" ) # создаем главное окно
cv2.namedWindow( "settings" ) # создаем окно настроек

cap = cv2.VideoCapture(0)
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
crange = [0,0,0, 0,0,0]

while True:
   flag, img = cap.read()
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

   # считываем значения бегунков
   h1 = cv2.getTrackbarPos('h1', 'settings')
   s1 = cv2.getTrackbarPos('s1', 'settings')
   v1 = cv2.getTrackbarPos('v1', 'settings')
   h2 = cv2.getTrackbarPos('h2', 'settings')
   s2 = cv2.getTrackbarPos('s2', 'settings')
   v2 = cv2.getTrackbarPos('v2', 'settings')

   # формируем начальный и конечный цвет фильтра
   h_min = np.array((h1, s1, v1), np.uint8)
   h_max = np.array((h2, s2, v2), np.uint8)

   # накладываем фильтр на кадр в модели HSV
   thresh = cv2.inRange(hsv, h_min, h_max)

   contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


   cv2.drawContours( img, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1 )
   cv2.imshow('result', img) 
   cv2.imshow('thresh', thresh) 
 
   ch = cv2.waitKey(5)
   if ch == 27:
      break

cap.release()
cv2.destroyAllWindows() """

#endregion

#endregion

#region FACE DETECTION

#region static face detection
""" # Загрузка предварительно обученного каскадного классификатора для детекции лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения
image = cv2.imread('img/people.jpg')
image = cv2.resize(image,(480, 640))
 
# Преобразование изображения в оттенки серого (для улучшения производительности)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Детекция лиц на изображении
# Детекция лиц на изображении
    
#detectMultiScale(image, scaleFactor, minNeighbors, minSize, flags)

#image: Это входное изображение, на котором будет производиться поиск объектов. Оно должно быть представлено в виде массива NumPy или матрицы изображения.

#scaleFactor: Этот параметр определяет насколько изображение уменьшается на каждом масштабе обнаружения. 
#Он используется для создания масштабной пирамиды изображения для обнаружения объектов в различных масштабах.
#Например, если scaleFactor = 1.1, то изображение будет уменьшаться на 10% на каждом уровне масштабирования. По умолчанию его значение равно 1.1.

#minNeighbors: Этот параметр определяет, сколько соседей должно иметь каждый прямоугольник-кандидат, чтобы быть признанным как объект. 
#Это помогает устранить ложные положительные результаты. 
#Чем выше значение, тем более консервативным будет алгоритм. Значение по умолчанию обычно равно 3 или 5.

#minSize: Этот параметр определяет минимальный размер объекта. 
#Лица меньше этого размера будут проигнорированы. Он представлен в виде кортежа (width, height).

#flags: Этот параметр используется для управления режимом работы алгоритма. 
#Он может принимать различные значения, такие как cv2.CASCADE_SCALE_IMAGE, cv2.CASCADE_FIND_BIGGEST_OBJECT, cv2.CASCADE_DO_ROUGH_SEARCH и т. д. Значение по умолчанию - 0.

   
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Отрисовка прямоугольников вокруг обнаруженных лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Вывод изображения с обозначенными лицами
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows() """
#endregion

#region dinamic faces detection
""" лол всё не настолько так просто :D """
#endregion

#endregion