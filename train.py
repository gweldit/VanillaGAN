import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import functional as F
from tqdm import tqdm


def train_gcn_model(
    model, train_loader, vocab_size=342, epochs=20, device="cpu", lr=1e-3
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.5, 0.99)
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    # val_losses = []
    train_f1_scores = []

    for epoch in range(epochs):
        loss_ = 0.0
        f1_score_ = 0.0
        for data in tqdm(train_loader):
            # print(data)
            x = F.one_hot(data.x[0], num_classes=vocab_size).float()
            x = x.to(device)
            edge_index = data.edge_index.to(device)
            edge_weight = data.weight.float().to(device)
            # y = torch.LongTensor(data.y).to(device)
            y = torch.LongTensor(data.y).to(device)
            batch = data.batch.to(device)

            optimizer.zero_grad()

            pred = model(x, edge_index, edge_weight=edge_weight, batch=batch)

            # print("pred", pred.shape)
            loss = criterion(pred, y)

            # print("loss", loss.item())
            loss.backward()
            optimizer.step()
            loss_ += loss.item()

            # Convert the tensors to numpy arrays
            outputs = torch.argmax(torch.clone(pred).detach().cpu(), dim=1).numpy()
            targets = torch.clone(y).detach().cpu().numpy()
            # Compute the F1 score
            f1 = f1_score(targets, outputs, zero_division=0.0)
            f1_score_ += f1

        # Compute the average loss and F1 score
        n_batches = len(train_loader)
        loss_ = loss_ / n_batches
        f1_score_ = f1_score_ / n_batches
        train_losses.append(loss_)
        train_f1_scores.append(f1_score_)

        print(
            f"Epoch {epoch + 1} / {epochs}, Loss: {loss_:.4f}, F1 Score: { f1_score_:.4f}"
        )

    return train_losses, train_f1_scores


def train_gan_model(
    generator,
    discriminator,
    gen_optimizer,
    disc_optimizer,
    train_loader,
    epochs,
    vocab_size,
    device="cpu",
    tau=0.1,
    hard=False,
    warm_up_epochs=10,
):
    generator_losses = []
    discriminator_losses = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} / {epochs}")

        gloss = 0.0
        dloss = 0.0
        number_of_batches = len(train_loader)

        for data, labels in tqdm(train_loader):
            # print(labels)

            # encode data to one-hot encoding
            data = F.one_hot(data.long(), vocab_size).float().to(device)

            labels = labels.long().to(device)

            # Train Discriminator
            disc_optimizer.zero_grad()
            # generate random noise
            latent_dim = generator.latent_dim
            z = torch.randn(data.shape[0], latent_dim).to(device)
            # generate fake data
            gen_data = generator(z, labels)

            # print("gen data shape", gen_data.shape)

            # perform discrete categorical sampling
            # gen_data = F.gumbel_softmax(gen_data, tau=temperature, hard=True)

            # feed real data to discriminator
            disc_real = discriminator(data, labels)

            # feed fake data to discriminator
            disc_fake = discriminator(gen_data.detach(), labels)

            # compute discriminator loss
            disc_loss = -torch.mean(torch.log(disc_real) + torch.log(1 - disc_fake))

            dloss += disc_loss.item()

            # backward pass
            disc_loss.backward()

            # update discriminator weights
            disc_optimizer.step()

            # Train Generator
            if epoch >= warm_up_epochs:
                gen_optimizer.zero_grad()
                z = torch.randn(data.shape[0], latent_dim).to(device)

                # generate fake data
                gen_data = generator(z, labels, tau=tau, hard=hard)

                # perform discrete categorical sampling
                # print("gen data shape", gen_data.shape)
                # gen_data = F.gumbel_softmax(gen_data, tau=tau, hard=False, dim=-1)

                # feed fake data to discriminator
                disc_fake = discriminator(gen_data, labels)

                # compute generator loss: minmax GAN's generator loss
                gen_loss = torch.mean(torch.log(1 - disc_fake))

                # for non-saturated minmax GAN's generator loss :==> generator: maximize log(D(G(z)))
                # gen_loss = -torch.mean(torch.log(disc_fake))

                gloss += gen_loss.item()

                # backward pass
                gen_loss.backward()

                # update generator weights
                gen_optimizer.step()

        dloss = torch.round(torch.tensor(dloss / number_of_batches), decimals=4)
        gloss = torch.round(torch.tensor(gloss / number_of_batches), decimals=4)

        discriminator_losses.append(dloss)
        generator_losses.append(gloss)

        # print(f"D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")
        print(f"D Loss: {dloss.item():.4f}, G Loss: {gloss.item():.4f}")

    return generator_losses, discriminator_losses
