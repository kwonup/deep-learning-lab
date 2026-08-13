#모델 Zoo : https://docs.pytorch.org/serve/model_zoo.html
import torch
import torch.nn as nn
import torch.optim as optim

from utils.DataLoader import get_dataloader
from models.vgg import get_vgg_model
from models.resnet_pre import get_resnet_model
from train import train_one_epoch
from eval import evaluate
from utils.graph import draw_plot
from tqdm import tqdm

def run_epoch(model,history,EPOCHS):
    #꼭 해야 할 것 -> GPU를 쓰기 위해
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    #에포크마다 train_one_epoch랑 eval 함수를 돌리면 됨!
    #오차함수 ,최적화함수,히스토리 딕셔너리 추가,에포크 정해주기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0

    epoch_bar = tqdm(range(EPOCHS), desc='Training', unit='epoch')
    for e in epoch_bar:
        #평균 loss, 맞춘 비율 %
        #model,loader, criterion,optimizer,device
        train_loss,train_acc = train_one_epoch(model,
                                                train_loader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                device=device)
        valid_loss,valid_acc = evaluate(model,
                                                valid_loader,
                                                criterion=criterion,
                                                device=device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        epoch_bar.set_postfix(
            train_loss=f'{train_loss:.4f}',
            train_acc=f'{train_acc:.2f}%',
            valid_loss=f'{valid_loss:.4f}',
            valid_acc=f'{valid_acc:.2f}%'
        )

        if valid_acc > best_acc :
            best_acc = valid_acc
            save_path = f'./CIFAR10_VGG11_{e+1}epoch_{valid_acc:.2f}.pth'

            torch.save(model.state_dict(),save_path)
            print(f'최고기록! {e+1}회 에포크 --> {valid_acc:.2f}')
    return history 

if __name__ == '__main__':
    #get_dataloader = 데이터를 로딩하는 함수
    train_loader,valid_loader = get_dataloader()

    #vgg11모델과 가중치를 로딩하는 함수
    model = get_resnet_model()
    history = {'train_loss':[], 'train_acc':[], 'valid_loss':[], 'valid_acc':[]}
     #model_history 훈련이 끝났을 때의 총 history를 반환
    model_history = run_epoch(model, history, EPOCHS=30)

    draw_plot(model_history,save_path=r'./result.jpg')
