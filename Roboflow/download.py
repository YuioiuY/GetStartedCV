"""                  NOT WORKING                 """

""" EXAMPLE OF DOWNLOADING DATASET FROM ROBOFLOW """

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_SANBOX").project("YOUR_NAME_PROJECT")
version = project.version(1)
dataset = version.download("yolov5") 