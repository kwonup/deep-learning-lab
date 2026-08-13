import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """검증 데이터를 평가하고 평균 loss와 accuracy를 반환한다."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total * 100
