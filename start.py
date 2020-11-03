from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ]
)

data = datasets.MNIST(root='./dataset',
                      train=True,
                      download=True,
                      transform=transform)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=32,
                                          shuffle=True)


class Encoder(torch.nn.Module):
    def __init__(self, space_size):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
        self.lin1 = torch.nn.Linear(in_features=676, out_features=space_size)

    def forward(self, x):
        a = x.shape[0]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(a, 1, -1)
        x = torch.relu(self.lin1(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, space_size):
        super(Decoder, self).__init__()
        self.lin1 = torch.nn.Linear(in_features=space_size, out_features=90)
        self.lin2 = torch.nn.Linear(in_features=90, out_features=100)
        self.lin3 = torch.nn.Linear(in_features=100, out_features=784)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, space_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(space_size)
        self.decoder = Decoder(space_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


SPACE_SIZE = 15
autoencoder = Autoencoder(space_size=SPACE_SIZE)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1.5e-4)


def to_img(vec):
    return vec.reshape(vec.shape[0], 1, 28, 28)


for epoch in range(100):
    error_sum = 0
    a = 0
    for image_batch, _ in data_loader:
        a += 1
        out = autoencoder(image_batch)
        out = to_img(out)
        optimizer.zero_grad()
        error = loss(out, image_batch)
        error.backward()
        optimizer.step()
        error_sum += error.detach().item()
    print(error_sum / a)
    plt.imshow(
        to_img(
            autoencoder(
                data_loader.dataset[0][0].reshape(1, 1, 28, 28)
            ).detach().numpy()
        )[0, 0], cmap='gray'
    )
    plt.show()
