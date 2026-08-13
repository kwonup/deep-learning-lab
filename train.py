import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """학습 데이터를 한 번 순회하고 평균 loss와 accuracy를 반환한다."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total * 100
