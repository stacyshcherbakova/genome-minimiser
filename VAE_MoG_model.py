import torch 
import torch.nn as nn
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAEWithMoGPrior(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, num_components):
        super(VAEWithMoGPrior, self).__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU()
        )
        
        # latent mean and variance
        self.mean_layer = nn.Linear(hidden_dim*2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim*2, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

        # Mixture of Gaussians parameters
        self.mog_means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.mog_logvars = nn.Parameter(torch.randn(num_components, latent_dim))

     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar


    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # Compute the standard deviation
        epsilon = torch.randn_like(std).to(device)  # Generate epsilon
        z = mean + std * epsilon
        return z


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar, z
    

    def compute_mog_prior(self, z):
        log_probs = []
        for i in range(self.num_components):
            mog_mean = self.mog_means[i]
            mog_logvar = self.mog_logvars[i]
            log_prob = -0.5 * (torch.sum(mog_logvar) + torch.sum((z - mog_mean) ** 2 / torch.exp(mog_logvar), dim=1))
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)
        log_probs = nn.functional.log_softmax(log_probs, dim=1)
        return torch.logsumexp(log_probs, dim=1)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    