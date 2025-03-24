import torch
import torch.nn as nn
import torch.optim as optim


class GAN:
    def __init__(self, generator, discriminator, latent_dim=100, lr=0.0002, beta1=0.5, device=None):
        """
        Инициализация GAN.

        :param generator: torch.nn.Module — модель генератора
        :param discriminator: torch.nn.Module — модель дискриминатора
        :param latent_dim: int — размерность шума, передаваемого в генератор
        :param lr: float — скорость обучения
        :param beta1: float — параметр beta1 для Adam-оптимизатора
        :param device: str — устройство ('cuda' или 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        # Передача моделей на нужное устройство
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Функция потерь
        self.criterion = nn.BCELoss()

        # Оптимизаторы
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    def train(self, dataloader, epochs=100, print_every=10):
        """
        Обучение GAN.

        :param dataloader: DataLoader — обучающий датасет
        :param epochs: int — количество эпох
        :param print_every: int — как часто печатать результаты
        """
        for epoch in range(epochs):
            for i, (real_images, _) in enumerate(dataloader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)

                # Метки
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # ----- Обучение дискриминатора -----
                self.optimizer_D.zero_grad()

                # Оценка реальных изображений
                real_outputs = self.discriminator(real_images)
                loss_real = self.criterion(real_outputs, real_labels)

                # Генерация фейковых изображений
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise)

                # Оценка фейковых изображений
                fake_outputs = self.discriminator(fake_images.detach())
                loss_fake = self.criterion(fake_outputs, fake_labels)

                # Итоговый лосс дискриминатора
                loss_D = loss_real + loss_fake
                loss_D.backward()
                self.optimizer_D.step()

                # ----- Обучение генератора -----
                self.optimizer_G.zero_grad()

                fake_outputs = self.discriminator(fake_images)
                loss_G = self.criterion(fake_outputs, real_labels)
                loss_G.backward()
                self.optimizer_G.step()

            # Вывод результатов
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    def generate(self, num_samples=1):
        """
        Генерация новых данных с помощью обученного генератора.

        :param num_samples: int — количество сэмплов для генерации
        :return: torch.Tensor — сгенерированные данные
        """
        self.generator.eval()  # Устанавливаем режим eval
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            generated_data = self.generator(noise)
        self.generator.train()  # Возвращаем в режим train
        return generated_data


# ----- Пример использования -----
if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    class SimpleGenerator(nn.Module):
        def __init__(self, latent_dim, output_dim=28 * 28):
            super(SimpleGenerator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.Tanh()
            )

        def forward(self, z):
            return self.model(z)

    class SimpleDiscriminator(nn.Module):
        def __init__(self, input_dim=28 * 28):
            super(SimpleDiscriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # Параметры
    latent_dim = 10
    sequence_length = 30  # Длина временного ряда
    feature_dim = 3  # Например, [цена, объем, волатильность]

    # Загружаем финансовые данные (заглушка)
    np.random.seed(42)
    real_data = np.random.randn(1000, sequence_length * feature_dim)  # 1000 примеров, по 30 точек в 3 измерениях

    # Преобразуем данные в PyTorch тензоры
    dataset = TensorDataset(torch.tensor(real_data, dtype=torch.float32), torch.zeros(1000))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Создаем модели
    generator = SimpleGenerator(latent_dim, output_dim=sequence_length * feature_dim)
    discriminator = SimpleDiscriminator(input_dim=sequence_length * feature_dim)

    # Обучаем GAN
    gan = GAN(generator, discriminator, latent_dim=latent_dim)
    gan.train(dataloader, epochs=200, print_every=20)

    # Генерируем синтетические финансовые временные ряды
    synthetic_data = gan.generate(num_samples=5)
    print(synthetic_data.shape)  # Ожидаем (5, 90) → 5 рядов по 30 точек в 3 измерениях
