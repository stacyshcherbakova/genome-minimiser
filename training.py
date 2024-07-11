# Used libraries
import torch.nn as nn
import torch 
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l1_regularization(model, lambda_l1):
    l1_penalty = 0.0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty

def cosine_annealing_schedule(t, T, min_beta, max_beta):
    return min_beta + (max_beta - min_beta) / 2 * (1 + np.cos(np.pi * (t % T) / T))

def exponential_decay_schedule(t, initial_beta, decay_rate):
    return initial_beta * np.exp(-decay_rate * t)

# Function used to train the model with KL annealing two additional additional loss terms which penalises model for generating larger genomes 
def train_KL_annealing_additional_loss_2(model, optimizer, scheduler, n_epochs, train_loader, val_loader, beta_start, beta_end, gamma_start, gamma_end, max_norm, lambda_l1):
    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4
    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)


            gene_abundance = recon_x.sum(axis=0)
            gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            genome_size = recon_x.sum(axis=1)
            genome_size_loss = torch.sum(torch.abs(genome_size)) 

            #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + (gamma * genome_size_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                gene_abundance = recon_x.sum(axis=0)
                gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

                genome_size = recon_x.sum(axis=1)
                genome_size_loss = torch.sum(torch.abs(genome_size)) 

                #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + (gamma * genome_size_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals, epoch + 1

# FUnction cycllic annealing _= genomes size
def train_cyclic_KL_annealing_additional_loss_2(model, optimizer, scheduler, n_epochs, train_loader, val_loader, min_beta, max_beta, gamma_start, gamma_end, max_norm, lambda_l1):
    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    T = 50
    counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=20
    min_delta=1e-4

    train_recon_loss_vals = []
    train_kl_loss_vals = []
    train_gene_abundance_loss_vals = []
    train_genome_size_loss_vals = []
    train_l1_loss_vals = []

    val_recon_loss_vals = []
    val_kl_loss_vals = []
    val_gene_abundance_loss_vals = []
    val_genome_size_loss_vals = []


    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        # beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0
        epoch_train_recon_loss = 0.0
        epoch_train_kl_loss = 0.0
        epoch_train_gene_abundance_loss = 0.0
        epoch_train_genome_size_loss = 0.0
        epoch_train_l1_loss = 0.0

        for batch in train_loader:
            t = epoch * 32 + counter
            beta = cosine_annealing_schedule(t, T, min_beta, max_beta)
            # gamma = exponential_decay_schedule(t, gamma_start, decay_rate)
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)


            gene_abundance = recon_x.sum(axis=0)
            gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            genome_size = recon_x.sum(axis=1)
            genome_size_loss = torch.sum(torch.abs(genome_size)) 

            #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + (gamma * genome_size_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            counter += 1

            #######
            epoch_train_loss += loss.item()
            epoch_train_recon_loss += reconstruction_loss.item()
            epoch_train_kl_loss += kl_divergence_loss.item()
            epoch_train_gene_abundance_loss += gene_abundance_loss.item()
            epoch_train_genome_size_loss += genome_size_loss.item()
            epoch_train_l1_loss += l1_penalty.item()

        train_recon_loss_vals.append(epoch_train_recon_loss / len(train_loader.dataset))
        train_kl_loss_vals.append(epoch_train_kl_loss / len(train_loader.dataset))
        train_gene_abundance_loss_vals.append(epoch_train_gene_abundance_loss / len(train_loader.dataset))
        train_genome_size_loss_vals.append(epoch_train_genome_size_loss / len(train_loader.dataset))
        train_l1_loss_vals.append(epoch_train_l1_loss / len(train_loader.dataset))
        #######

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_recon_loss = 0.0
        epoch_val_kl_loss = 0.0
        epoch_val_gene_abundance_loss = 0.0
        epoch_val_genome_size_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                gene_abundance = recon_x.sum(axis=0)
                gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

                genome_size = recon_x.sum(axis=1)
                genome_size_loss = torch.sum(torch.abs(genome_size)) 

                #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + (gamma * genome_size_loss)

                epoch_val_loss += loss.item()
                epoch_val_recon_loss += reconstruction_loss.item()
                epoch_val_kl_loss += kl_divergence_loss.item()
                epoch_val_gene_abundance_loss += gene_abundance_loss.item()
                epoch_val_genome_size_loss += genome_size_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        val_recon_loss_vals.append(epoch_val_recon_loss / len(val_loader.dataset))
        val_kl_loss_vals.append(epoch_val_kl_loss / len(val_loader.dataset))
        val_gene_abundance_loss_vals.append(epoch_val_gene_abundance_loss / len(val_loader.dataset))
        val_genome_size_loss_vals.append(epoch_val_genome_size_loss / len(val_loader.dataset))

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    # Plotting training loss components
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss_vals, label='Train Loss')
    plt.plot(train_recon_loss_vals, label='Train Reconstruction Loss')
    plt.plot(train_kl_loss_vals, label='Train KL Loss')
    plt.plot(train_gene_abundance_loss_vals, label='Train Gene Abundance Loss')
    plt.plot(train_genome_size_loss_vals, label='Train Genome Size Loss')
    plt.plot(train_l1_loss_vals, label='Train L1 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Components over Epochs')
    plt.savefig("train_losses.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plotting validation loss components
    plt.figure(figsize=(12, 8))
    plt.plot(val_loss_vals, label='Validation Loss')
    plt.plot(val_recon_loss_vals, label='Val Reconstruction Loss')
    plt.plot(val_kl_loss_vals, label='Val KL Loss')
    plt.plot(val_gene_abundance_loss_vals, label='Val Gene Abundance Loss')
    plt.plot(val_genome_size_loss_vals, label='Val Genome Size Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss Components over Epochs')
    plt.savefig("validation_losses.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return train_loss_vals, val_loss_vals, epoch + 1



# Function used to train the model with KL annealing and additional loss term which penalises model for generating larger genomes 
def train_cyclic_KL_annealing_additional_loss(model, optimizer, scheduler, n_epochs, train_loader, val_loader, min_beta, max_beta, gamma_start, gamma_end, max_norm, lambda_l1):
    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    T = 10
    counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4

    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        # beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            t = epoch * 32 + counter
            beta = cosine_annealing_schedule(t, T, min_beta, max_beta)
            # gamma = exponential_decay_schedule(t, gamma_start, decay_rate)
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)


            gene_abundance = recon_x.sum(axis=0)
            gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            # genome_size = recon_x.sum(axis=1)
            # genome_size_loss = torch.sum(torch.abs(genome_size)) 

            #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            counter += 1

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                gene_abundance = recon_x.sum(axis=0)
                gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

                # genome_size = recon_x.sum(axis=1)
                # genome_size_loss = torch.sum(torch.abs(genome_size)) 

                #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals, epoch + 1


# Function used to train the model with KL annealing and additional loss term which penalises model for generating larger genomes 
def train_KL_annealing_additional_loss(model, optimizer, scheduler, n_epochs, train_loader, val_loader, beta_start, beta_end, gamma_start, gamma_end, max_norm, lambda_l1):
    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4
    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)


            gene_abundance = recon_x.sum(axis=0)
            gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            # genome_size = recon_x.sum(axis=1)
            # genome_size_loss = torch.sum(torch.abs(genome_size)) 

            #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                gene_abundance = recon_x.sum(axis=0)
                gene_abundance_loss = torch.sum(torch.abs(gene_abundance)) 

                # genome_size = recon_x.sum(axis=1)
                # genome_size_loss = torch.sum(torch.abs(genome_size)) 

                #scaled_gene_abundance_loss = scaling_factor * gene_abundance_loss
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * gene_abundance_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals, epoch + 1



# Function used to train the model with no KL annealing
def train_with_KL_annelaing(model, optimizer, scheduler, n_epochs, train_loader, val_loader, beta_start, beta_end, max_norm):
    # global train_loss_vals 
    # train_loss_vals = []
    # global train_loss_vals2 
    train_loss_vals2 = []
    # global val_loss_vals
    val_loss_vals = []
    # train_loss = 0.0
    train_loss2 = 0.0
    val_loss = 0.0
    # best_val_loss = float('inf')
    # early_stopping_patience = 5
    # early_stopping_counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        model.train()

        # epoch_train_loss = 0.0
        epoch_train_loss2 = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)
            # print('reco_x:', recon_x[:1, :5])
            # print('data:', data[:1, :5])

            # print(recon_x.shape)
            # print(data.shape) 

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # loss = reconstruction_loss + kl_divergence_loss
            loss2 = reconstruction_loss + (beta * kl_divergence_loss)

            loss2.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            # epoch_train_loss += loss.item()
            epoch_train_loss2 += loss2.item()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} gradient: {param.grad.abs().mean().item()}') 

        # avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_loss2 = epoch_train_loss2 / len(train_loader.dataset)
        # train_loss_vals.append(avg_train_loss)
        train_loss_vals2.append(avg_train_loss2)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss2 = reconstruction_loss + (beta * kl_divergence_loss)

                epoch_val_loss += loss2.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                #   f" Train Loss (method 1): {avg_train_loss}\n"
                  f" Train Loss (method 2): {avg_train_loss2}\n"
                  f" Validation Loss: {avg_val_loss}")

        # train_loss += avg_train_loss
        train_loss2 += avg_train_loss2
        val_loss += avg_val_loss

        # # Check for early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter >= early_stopping_patience:
        #     print("Early stopping triggered")
        #     break


        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # final_avg_train_loss = train_loss / n_epochs
    final_avg_train_loss2 = train_loss2 / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    # print(f"\nFinal Average Training Loss (method 1): {final_avg_train_loss}")
    print(f"Final Average Training Loss (method 2): {final_avg_train_loss2}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals2, val_loss_vals, epoch + 1


# Function used to train the model with KL annealing
def train_no_KL_annelaing(model, optimizer, scheduler, n_epochs, train_loader, val_loader, max_norm):
    # global train_loss_vals 
    train_loss_vals = []
    # global train_loss_vals2 
    # train_loss_vals2 = []
    # global val_loss_vals
    val_loss_vals = []
    train_loss = 0.0
    # train_loss2 = 0.0
    val_loss = 0.0
    # best_val_loss = float('inf')
    # early_stopping_patience = 5
    # early_stopping_counter = 0

    for epoch in range(n_epochs):
        model.train()

        epoch_train_loss = 0.0
        # epoch_train_loss2 = 0.0

        for batch in train_loader:
            if batch[0].size(0) == 1:
                continue 
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)
            # print('reco_x:', recon_x[:1, :5])
            # print('data:', data[:1, :5])

            # print(recon_x.shape)
            # print(data.shape) 

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence_loss
            # loss2 = reconstruction_loss + (beta * kl_divergence_loss)

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            # epoch_train_loss2 += loss2.item()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} gradient: {param.grad.abs().mean().item()}') 

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        # avg_train_loss2 = epoch_train_loss2 / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)
        # train_loss_vals2.append(avg_train_loss2)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch[0].size(0) == 1:
                    continue 
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_divergence_loss

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss (method 1): {avg_train_loss}\n"
                #   f" Train Loss (method 2): {avg_train_loss2}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        # train_loss2 += avg_train_loss2
        val_loss += avg_val_loss

        # # Check for early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter >= early_stopping_patience:
        #     print("Early stopping triggered")
        #     break

    final_avg_train_loss = train_loss / n_epochs
    # final_avg_train_loss2 = train_loss2 / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"\nFinal Average Training Loss (method 1): {final_avg_train_loss}")
    # print(f"Final Average Training Loss (method 2): {final_avg_train_loss2}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals