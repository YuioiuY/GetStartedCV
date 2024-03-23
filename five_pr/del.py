from PIL import Image, ImageFilter

# Открываем изображение
image = Image.open("five_pr\\img\\input_image.jpg")

# Применяем медианный фильтр для удаления шума
filtered_image = image.filter(ImageFilter.MedianFilter(size=3))  # Размер окна фильтра (3х3)

# Сохраняем результат
filtered_image.save("five_pr\\output\\output_image.jpg")

# Показываем результат на экране (необязательно)
filtered_image.show()
