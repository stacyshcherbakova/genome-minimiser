# Used libraries
import torch.nn as nn
import torch 

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

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        model.train()

        # epoch_train_loss = 0.0
        epoch_train_loss2 = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float)
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
                data = batch[0]
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

    # final_avg_train_loss = train_loss / n_epochs
    final_avg_train_loss2 = train_loss2 / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    # print(f"\nFinal Average Training Loss (method 1): {final_avg_train_loss}")
    print(f"Final Average Training Loss (method 2): {final_avg_train_loss2}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals2, val_loss_vals


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
            data = batch[0].to(torch.float)
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
                data = batch[0]
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