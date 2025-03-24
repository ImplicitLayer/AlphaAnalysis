import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Определение энкодера
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Среднее значение
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Стандартное отклонение

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var


# Определение декодера
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h2 = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h2))  # Используем сигмоиду для восстановления данных


# Класс VAE
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        # Считаем MSE для восстановления
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL divergence между полученным распределением и нормальным распределением
        KL = torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1.0)

        return MSE + KL

    def train_vae(self, train_data, epochs=10, batch_size=128, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        data_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.float()
                optimizer.zero_grad()
                recon_batch, mu, log_var = self(data)
                loss = self.loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {train_loss / len(data_loader.dataset)}')

    def generate(self, num_samples, latent_dim):
        self.eval()
        z = torch.randn(num_samples, latent_dim)
        with torch.no_grad():
            generated_data = self.decoder(z)
        return generated_data


# Пример использования

if __name__ == "__main__":
    # Параметры для модели
    input_dim = 30  # Размерность входных данных
    hidden_dim = 64
    latent_dim = 10

    # Создание объектов Encoder и Decoder
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)

    # Создание модели VAE
    vae_model = VAE(encoder, decoder)

    # Пример случайных данных для обучения
    train_data = torch.randn(1000, input_dim)  # 1000 примеров, каждый размерностью 30

    # Обучение модели
    vae_model.train_vae(train_data, epochs=50)

    # Генерация новых данных
    generated_data = vae_model.generate(5, latent_dim)
    print(generated_data)
