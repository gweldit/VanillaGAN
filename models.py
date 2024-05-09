import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        seq_len,
        hidden_dim,
        output_dim,
        embed_dim,
        dropout,
        conditional_info=False,
        num_classes=2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.input_dim = self.latent_dim
        self.conditional_info = conditional_info
        if self.conditional_info:
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided if conditional_info is True"
                )

            self.input_dim = latent_dim + embed_dim
            self.embedding = nn.Embedding(num_classes, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * output_dim),
            nn.ReLU(),  # added ReLU here to avoid invalid logits to the gumbel softmax
        )

    def forward(self, z, labels=None):

        if self.conditional_info:
            if labels is None:
                raise ValueError("labels must be provided if conditional_info is True")
            self.embedding.to(z.device)
            z = torch.cat((z, self.embedding(labels)), dim=1)

        out = self.model(z)
        # reshape output to [batch_size, seq_len, output_dim]
        #  i.e., [batch_size, seq_len, vocab_size]
        return out.view(out.shape[0], self.seq_len, self.output_dim)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        hidden_dim,
        embed_dim,
        dropout,
        conditional_info=False,
        num_classes=None,
    ):
        super().__init__()

        self.conditional_info = conditional_info
        self.input_dim = input_dim * seq_len
        if self.conditional_info:
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided if conditional_info is True"
                )
            self.embedding = nn.Embedding(num_classes, embed_dim)
            self.input_dim = self.input_dim + embed_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels=None):
        # reshape x from [batch_size, seq_len, vocab_size] to [batch_size, seq_len * vocab_size]
        x = x.reshape(x.shape[0], -1)
        if self.conditional_info:
            if labels is None:
                raise ValueError("labels must be provided if conditional_info is True")
            self.embedding.to(x.device)
            # concatenate labels with input
            c = self.embedding(labels)
            # print(x.shape, "|", c.shape)
            x = torch.cat((x, c), dim=-1)
        out = self.model(x)
        return out
