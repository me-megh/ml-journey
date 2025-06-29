from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
