import torch
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0], [1])
     ])
data = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
num_batches = len(data_loader)


class Encoder(torch.nn.Module):
    def __init__(self, encoded_space_size):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.lin1 = torch.nn.Linear(in_features=169, out_features=encoded_space_size)

    def forward(self, x):
        a = x.shape[0]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(a, 1, -1)
        x = torch.sigmoid(self.lin1(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, encoded_space_size):
        super(Decoder, self).__init__()

        self.lin1 = torch.nn.Linear(in_features=encoded_space_size, out_features=80)
        self.lin2 = torch.nn.Linear(in_features=80, out_features=200)
        self.lin3 = torch.nn.Linear(in_features=200, out_features=784)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoded_space_size):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(encoded_space_size)
        self.decoder = Decoder(encoded_space_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


num_epochs = 10000

encoded_space_size = 20

autoencoder = AutoEncoder(encoded_space_size)
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-5)


for epoch in range(num_epochs):
    error = None
    for image_batch, _ in data_loader:
        out = autoencoder(image_batch)
        out = vectors_to_images(out)
        error = loss(out, image_batch)
        error.backward()
        optimizer.step()
    print(error.detach().item())
    plt.imshow(
        vectors_to_images(
            autoencoder(
                data_loader.dataset[0][0].reshape(1, 1, 28, 28))
        ).detach().numpy()[0, 0], cmap='gray')
    plt.show()


