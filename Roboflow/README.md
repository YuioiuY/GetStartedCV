# (ಥ_ಥ)━☆ﾟ.*･｡ﾟ GetStartedCV 

# Roboflow

Данная тема является ознакомительной, так как для работы с такими вещами требуется очень много времени и конкретная цель. 

Но для дополнительного развития очень вам советую ознакомиться с лекционным материалом, где на слайдах я постарался подробно показать как 
работать с инструментом для составления датасета:  

https://github.com/YuioiuY/GetStartedCV/tree/main/0lecture/ 📖

Вам нужна 7-ая лекция под названием Roboflow. 💡


# Ссылки для изучения и работы с Robodlow

Ссылка на сам Roboflow: 

- > https://roboflow.com/

Здесь можно ознакомиться с инструкцией для обучения и самостоятельно опробовать работоспособность: 

- > https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb#scrollTo=GD9gUQpaBxNa 🤓

Ссылка на репозиторий с исходным кодом: 

- > https://github.com/ultralytics/yolov5 🤓

Доп инфа: 

- > https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data 🤓



# Основные команды для тестирования и обучения  

python train.py --img 416 --batch 16 --epochs 100 --data YOUR_DATASET/data.yaml --cfg models/yolov5s.yaml  --name yolov5s_results  --cache

python detect.py --weights runs/train/yolov5s_results15/weights/last.pt --img 416 --conf 0.4 --source YOUR_DATASET/test/

**ВНИМНИЕ!!!**  
Читайте консоль после завершения обучения и поиска объектов! 
В консоли пишется куда именно сохранены файлы.

Так же прошу обратить внимание что у меня в проекте датасет находится в корневой папке репозитория который я скачал. 
