import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self,
                 zdim=64,
                 image_size=28,
                 bidirect=True,
                 channels=(64, 32, 16, 1),
                 kernel_sizes=(4,4,4,4),
                 pads=(0,0,1,1),
                 strides=(1,1,2,2),
                 encoder_fc_out_size=64,
                 decoder_fc_out_size=64,
                 **kwargs):

        super().__init__()
        self.zdim = zdim
        self.image_size = image_size
        self.kernel_sizes = kernel_sizes
        self.pads = pads
        self.strides = strides
        self.n_decoder_layers = len(channels)
        self.fc_in = 2*zdim if bidirect else zdim
        self.encoder_fc_out_size = encoder_fc_out_size
        self.decoder_fc_out_size = decoder_fc_out_size
        self.bidirect = bidirect

        self.encoder_lstm = nn.LSTM(image_size, zdim, bidirectional=bidirect, batch_first=True)
        self.encoder_mu = nn.Linear(self.fc_in, encoder_fc_out_size)
        self.encoder_logvar = nn.Linear(self.fc_in, encoder_fc_out_size)

        self.decoder_fc = nn.Linear(encoder_fc_out_size, decoder_fc_out_size)
        self.decoder_list = nn.ModuleList([])
        for i, kernel in enumerate(kernel_sizes):
            in_ch = channels[i-1] if i>0 else self.decoder_fc_out_size
            out_ch = channels[i]
            self.decoder_list.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel, strides[i], pads[i])
            )

        self.check_out_dim()

    def encode(self, x):
        _, (lstm_out, _) = self.encoder_lstm(x)
        if self.bidirect:
            lstm_out_dir0 = lstm_out[0].view(-1, self.zdim)
            lstm_out_dir1 = lstm_out[1].view(-1, self.zdim)
            lstm_out = torch.cat([lstm_out_dir0, lstm_out_dir1], axis=1)
        else:
            lstm_out = lstm_out.view(-1, self.fc_in)
        mu = self.encoder_mu(lstm_out)
        logvar = self.encoder_logvar(lstm_out)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        z = torch.randn_like(mu)
        return mu + z*sigma

    def decode(self, x):
        x = self.decoder_fc(x)
        x = F.relu(x)
        x = x.view(-1, self.decoder_fc_out_size, 1 , 1)

        for i, layer in enumerate(self.decoder_list):
            x = layer(x)
            if i+1 < self.n_decoder_layers:
                x = F.relu(x)
            else:
                x = torch.sigmoid(x)

        return x


    def forward(self, x):
        mu, logvar = self.encode(x)
        sample = self.reparametrize(mu, logvar)
        decoded = self.decode(sample)
        return decoded, mu, logvar

    def check_out_dim(self):
        out = 1
        for kernel_size, pad, stride in zip(self.kernel_sizes, self.pads, self.strides):
            out = stride*(out-1) + kernel_size - 2*pad #+ (out+2*pad-kernel_size)%stride

        print(f"Output Dim: {out}" )
        assert out == self.image_size
