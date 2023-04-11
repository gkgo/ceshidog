import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from torchvision.datasets import ImageFolder
from tqdm import tqdm  # 添加进度条库
import matplotlib.pyplot as plt
from dg import resnet50
# from zhong import resnet50

torch.manual_seed(2)


GPU = torch.cuda.is_available()

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

def train(epoch, model, device, train_loader, optimizer):
    model.train()
    size = len(train_loader.dataset)
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Training epoch {epoch}", unit="batch", leave=True)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += output.argmax(dim=1).eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / size
    print(f"Train Epoch: {epoch}\tLoss: {train_loss:.4f}\tAccuracy: {train_acc:.2f}%")



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing", unit="batch", leave=True)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            test_correct += output.argmax(dim=1).eq(target).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}\tAccuracy: {test_acc:.2f}%")

    return test_acc


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小到224x224像素
        transforms.ToTensor(),  # 转换为Tensor类型
        transforms.Normalize(  # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomCrop((224, 224))  # 随机裁剪
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data

    train_data = ImageFolder('./cropped/train', transform=train_transform)
    val_data = ImageFolder('./cropped/test', transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

    # model = resnet18gai().to(device)
    model = resnet50(num_classes=120).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)
    best_acc = 0.0

    for epoch in range(100):
        start_time = time.time()
        train(epoch, model, device, train_loader, optimizer)
        test_acc = test(model, device, val_loader)
        lr_scheduler.step()
        epoch_time = time.time() - start_time
        if test_acc > best_acc:
            best_acc = test_acc
        print(f'[ log ] roughly {(100 - epoch) / 3600. * epoch_time:.2f} h left')
        print(f"Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()

    # Plot loss and accuracy curves
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(train_loss_list, label='Train Loss')
    ax[0].plot(test_loss_list, label='Test Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(train_acc_list, label='Train Accuracy')
    ax[1].plot(test_acc_list, label='Test Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.show()
