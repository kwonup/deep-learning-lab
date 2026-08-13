
from ultralytics import YOLO

def yolo_train():
    # yaml_path =r'./yolo_setting.yaml'

    #YOLO 훈련
    result = YOLO('yolov8n.pt').train(
        data = yaml_path,
        epochs=50,
        imgsz=640,
        batch = 16,
        save =True,
        device = 0,
        plots=True,
        name='peach_train01'
    )

    print('훈련 완료')

def yolo_predict(source,weight_path):

    #YOLO 평가
    weight_path = r'runs\detect\peach_train01\weights\best.pt'
    source = r'fruit4.jpg'
    model =YOLO(weight_path)

    model.predict(source=source,
                    device=0,
                    save=True)