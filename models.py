import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AutoencoderMCDSVDD(nn.Module):
    """Pytorch implementation of an Autoencoder with Multiclass Deep SVDD (MCDSVDD)."""

    def __init__(self, in_dim, z_dim, num_classes):
        super(AutoencoderMCDSVDD, self).__init__()

        self.num_classes = num_classes

        # Encoder Architecture
        self.enc1 = nn.Linear(in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, 64)
        self.encbn4 = nn.BatchNorm1d(64)
        self.enc5 = nn.Linear(64, z_dim, bias=False)

        # Decoder Architecture
        self.dec1 = nn.Linear(z_dim, 64)
        self.decbn1 = nn.BatchNorm1d(64)
        self.dec2 = nn.Linear(64, 128)
        self.decbn2 = nn.BatchNorm1d(128)
        self.dec3 = nn.Linear(128, 256)
        self.decbn3 = nn.BatchNorm1d(256)
        self.dec4 = nn.Linear(256, 512)
        self.decbn4 = nn.BatchNorm1d(512)
        self.dec5 = nn.Linear(512, in_dim, bias=False)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        h = F.leaky_relu(self.encbn4(self.enc4(h)))
        return self.enc5(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        h = F.leaky_relu(self.decbn4(self.dec4(h)))
        return torch.tanh(self.dec5(h))

    def forward(self, x):
        """Forward pass over the network architecture"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def compute_loss(self, x):
        """Compute MSE Loss for autoencoder training."""
        _, x_hat = self.forward(x)
        return F.mse_loss(x_hat, x, reduction='mean')


    def set_centers(self, dataloader, eps=0.01):
        """Initialize the centers for the hyperspheres."""
        latents, labels = self.get_latent_space(dataloader)
        c = []
        for i in range(self.num_classes):
            ixs = np.where(labels == i)[0]
            c.append(torch.mean(latents[ixs], dim=0))

        # Add small perturbations to avoid zero centers
        c = torch.stack(c)
        for i in range(len(c)):
            c[i][(abs(c[i]) < eps) & (c[i] < 0)] = -eps
            c[i][(abs(c[i]) < eps) & (c[i] > 0)] = eps

        self.c = c

    def get_latent_space(self, dataloader, dataloader_c=None):
        """Extract latent representations from the data and their labels."""
        latents = []
        labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.float(), y.long()
                z, _ = self.forward(x)
                latents.append(z.detach().cpu())
                labels.append(y)
        return torch.cat(latents), torch.cat(labels)


    def train_autoencoder(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Train the Autoencoder with MSE loss (reconstruction)."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loss_list, val_loss_list = [], []

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader): # Unpack data and target
                optimizer.zero_grad()
                loss = self.compute_loss(data) # Pass data (features) to compute_loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            total_train_loss /= len(train_loader)
            train_loss_list.append(total_train_loss)

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    loss = self.compute_loss(data)
                    total_val_loss += loss.item()

            total_val_loss /= len(val_loader)
            val_loss_list.append(total_val_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}')


        return train_loss_list, val_loss_list

    def train_mcdsvdd(self, train_loader, val_loader, epochs=100, lr=0.001, lambda_reg=0.5e-6):
        """Train the model using MCDSVDD loss, adjusting centers of hyperspheres, with validation."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                x, labels = data
                x = x.float()
                labels = labels.long()

                # Forward pass to get latent space
                z, _ = self.forward(x)

                # Compute MCDSVDD loss (distance from centers)
                loss = 0
                for i in range(self.num_classes):
                    mask = (labels == i).float()
                    distance = torch.norm(z - self.c[i], p=2, dim=1)
                    loss += mask * distance

                # Add regularization term for weight decay
                l2_reg = sum(torch.norm(param, p=2) for param in self.parameters())
                loss = loss.mean() + lambda_reg * l2_reg

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            total_train_loss /= len(train_loader)
            train_loss_list.append(total_train_loss)

            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    x, labels = data
                    x = x.float()
                    labels = labels.long()

                    # Forward pass to get latent space
                    z, _ = self.forward(x)

                    # Compute MCDSVDD loss for validation
                    val_loss = 0
                    for i in range(self.num_classes):
                        mask = (labels == i).float()
                        distance = torch.norm(z - self.c[i], p=2, dim=1)
                        val_loss += mask * distance

                    l2_reg = sum(torch.norm(param, p=2) for param in self.parameters())
                    val_loss = val_loss.mean() + lambda_reg * l2_reg
                    total_val_loss += val_loss.item()

            total_val_loss /= len(val_loader)
            val_loss_list.append(total_val_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Val Loss: {total_val_loss:.4f}')

        return train_loss_list, val_loss_list


class AutoencoderGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=2, n_gmm=5, num_epochs=100,
                 lambda_energy=0.1, lambda_cov_diag=0.005):
        super(AutoencoderGMM, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, input_dim)
        )

        # GMM estimation network
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim, 10),  # Encoded dim + cosine and euclidean distances
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        )

        # GMM parameters
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim ))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim + 1, latent_dim))

        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)



    def forward(self, x):
        # Encoding step
        enc = self.encoder(x)

        # Decoding step
        dec = self.decoder(enc)

        # Reconstruction errors (cosine similarity and euclidean distance)
        rec_euclidean = torch.norm(x - dec, dim=1) / torch.norm(x, dim=1)


        z = enc
        # GMM estimation
        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = (sum_gamma / N)
        self.phi = phi.data
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, epsilon=1e-12):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov

        k, D, _ = cov.size()
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # Agregar epsilon a la diagonal de cada matriz de covarianza
        cov_adjusted = cov + epsilon * torch.eye(D).to(cov.device).unsqueeze(0)

        # Calcular la inversa de cada covarianza ajustada
        cov_inverse = [torch.inverse(cov_adjusted[i]) for i in range(k)]
        cov_inverse = torch.stack(cov_inverse)

        # Cálculo del término exponencial
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)

        max_val = torch.max(exp_term_tmp, dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term, dim=1))

        # Extraer la diagonal de las matrices de covarianza ajustadas
        cov_diag = torch.stack([torch.diag(cov_adjusted[i]) for i in range(k)])

        return sample_energy, cov_diag


    def loss_gmm(self, x, x_hat, z, gamma):
        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + self.lambda_energy * sample_energy.mean() + self.lambda_cov_diag * torch.mean(cov_diag)
        return loss, sample_energy.mean(), recon_error, cov_diag

    def dagmm_step(self, input_data):
        self.train()
        enc, dec, z, gamma = self(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.loss_gmm(input_data, dec, z, gamma)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag

    def train_model(self, data_loader):
        self.train()
        iters_per_epoch = len(data_loader)

        loss_history = []
        reconstruction_history = []
        energy_history = []

        for epoch in range(self.num_epochs):
            total_loss_list = []
            recon_error_list = []
            energy_list = []


            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.float()
                target = target.long()
                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(data)

                # Almacenar las pérdidas para estadísticas
                total_loss_list.append(total_loss.item())
                recon_error_list.append(recon_error.item())
                energy_list.append(sample_energy.item())

            # Imprimir estadísticas por época
            avg_loss = np.mean(total_loss_list)
            avg_recon_error = np.mean(recon_error_list)
            avg_energy = np.mean(energy_list)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Recon Error: {avg_recon_error:.4f}, Energy: {avg_energy:.4f}')

            # Almacenar la pérdida promedio por época
            loss_history.append(avg_loss)
            reconstruction_history.append(avg_recon_error)
            energy_history.append(avg_energy)

        self.loss_history = loss_history
        self.reconstruction_history = reconstruction_history
        self.energy_history = energy_history

    def get_latent_space(self, dataloader):
        """Extract latent representations from the data and their labels."""
        latents = []
        labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.float(), y.long()
                _,_,z,_ = self.forward(x)
                latents.append(z.detach().cpu())
                labels.append(y)
        return torch.cat(latents), torch.cat(labels)

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.num_epochs + 1), self.loss_history)
        plt.title('Loss as a function of the number of epochs')
        plt.xlabel('Number of epochs')
        plt.ylabel('Average loss')
        plt.show()

    def plot_reconstruction_error(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.num_epochs + 1), self.reconstruction_history)
        plt.title('Reconstruction error as a function of the number of epochs')
        plt.xlabel('Number of epochs')
        plt.ylabel('Average reconstruction error')
        plt.show()

    def plot_energy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.num_epochs + 1), self.energy_history)
        plt.title('Energy as a function of the number of epochs')
        plt.xlabel('Number of epochs')
        plt.ylabel('Average energy')
        plt.show()


