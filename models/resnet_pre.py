from torchvision.models import resnet34,ResNet34_Weights

def get_resnet_model():
    model = resnet34(weights=ResNet34_Weights.DEFAULT)

    #모델의 모든 가중치를 동결합니다.
    for p in model.parameters():
        p.requires_grad = False #“이 가중치는 학습 중 수정하지 말라”는 뜻

    #모델의 마지막 FC 파라미터만 '훈련 가능'으로 동결 해제
    for p in model.fc.parameters():
        p.requires_grad = True
    
    return model