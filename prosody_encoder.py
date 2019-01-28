# outer packages
import librosa
import numpy as np
import random
import os
import torch
from torch.utils import data
from torch.utils.data import DataLoader

# inner packages
import stft
import layers
from hparams import create_hparams

sr = 22050
n_class = 4

hparams = create_hparams('')
max_wav_value=hparams.max_wav_value
stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def get_mel(audio):
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, meta_path, prefix):
        'Initialization'
        f = open(meta_path, encoding='utf-8')
        meta_list = f.readlines()
        f.close()
        self.labels_and_wavs = [m.strip().split('|') for m in meta_list]
        self.prefix = prefix
        random.seed(1234)
        random.shuffle(self.labels_and_wavs)
        self.segment_length = 16380*5

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels_and_wavs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = os.path.join(self.prefix, self.labels_and_wavs[index][0])
        audio, sampling_rate = load_wav_to_torch(path)

        label_index = int(self.labels_and_wavs[index][1])
        label = np.zeros(n_class)
        label[label_index] = 1
        label = torch.from_numpy(label).float()

        # 최대 길이에 맞춘 padding
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = get_mel(audio)
        mel = mel.unsqueeze(0)
        assert list(mel.size())[2] % 2 == 0

        return mel, label

class prosody_encoder(torch.nn.Module):
    '''
    Args:
      inputs: A 3d tensor with shape of (N, n_mels, Ty), with dtype of float32.
                Melspectrogram of reference audio.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      Prosody vectors. Has the shape of (N, 128).
    '''
    def __init__(self):
        super(prosody_encoder, self).__init__()

        self.kernel = 3
        self.stride = 2
        self.pad = max(self.kernel - self.stride, 0)

        # 6-Layer Strided Conv2D -> (N, 128, ceil(n_mels/64), ceil(T/64))
        self.conv2d_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_1_bn = torch.nn.BatchNorm2d(num_features=32)
        self.conv2d_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_2_bn = torch.nn.BatchNorm2d(num_features=32)
        self.conv2d_3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_3_bn = torch.nn.BatchNorm2d(num_features=64)
        self.conv2d_4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_4_bn = torch.nn.BatchNorm2d(num_features=64)
        self.conv2d_5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_5_bn = torch.nn.BatchNorm2d(num_features=128)
        self.conv2d_6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, stride=self.stride, padding=self.pad, dilation=1, groups=1, bias=True)
        self.conv2d_6_bn = torch.nn.BatchNorm2d(num_features=128)

        # GRU in [N, ceil(T/64), 128*ceil(n_mel/64)], out [N, ceil(T/64), 128]
        self.gru = torch.nn.GRU(input_size=128*2, hidden_size=128, num_layers=1)

        # FC in [N, 128], out [N, 128]
        self.fc = torch.nn.Linear(in_features=128, out_features=128)
        self.tanh = torch.nn.Tanh()

        # for test classification
        self.fc2 = torch.nn.Linear(in_features=128, out_features=n_class)
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        """
        in [N, 1, 80, Ty], out [N, 128]
        out -> [N, n_class] for test
        """
        # check if Ty % stride == 0
        x_size = list(x.size())
        assert x_size[3] % self.stride == 0

        # 2c CNN in [N, 1, 80, Ty] out [N, 128, ceil(80/64), ceil(Ty/64)]
        c1 = self.conv2d_1(x)
        cb1 = self.conv2d_1_bn(c1)
        c2 = self.conv2d_2(cb1)
        cb2 = self.conv2d_2_bn(c2)
        c3 = self.conv2d_3(cb2)
        cb3 = self.conv2d_3_bn(c3)
        c4 = self.conv2d_4(cb3)
        cb4 = self.conv2d_4_bn(c4)
        c5 = self.conv2d_5(cb4)
        cb5 = self.conv2d_5_bn(c5)
        c6 = self.conv2d_6(cb5)
        c2d_output = self.conv2d_6_bn(c6)

        # unrolling in [N, 128, ceil(n_mel / 64), ceil(T / 64)] out [N, ceil(T/64), 128*ceil(n_mel/64)]
        N, C, ceil_nmel_64, ceil_T_64 = list(c2d_output.size())
        c2d_output_permute= c2d_output.permute(0, 3, 1, 2) # [N, 128, ceil(n_mel / 64), ceil(T / 64)] to [N, ceil(n_mel / 64), 128, ceil(T / 64)]
        unrooling_output = c2d_output_permute.view(N, ceil_T_64, C*ceil_nmel_64)

        # GRU in [N, ceil(T/64), 128*ceil(n_mel/64)], out [N, 128]
        gru_output, _ = self.gru(unrooling_output)
        gru_output = gru_output[:, -1, :] # take last value

        # FC [N, 128]
        fc_output = self.tanh(self.fc(gru_output))

        # Fc [N, n_class]
        last_fc_output = self.softmax(self.fc2(fc_output))

        return last_fc_output

class loss_fn(torch.nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()
        self.mseloss = torch.nn.MSELoss()

    def forward(self, output, target):
        mel_loss = self.mseloss(output, target)
        return mel_loss


def train():
    meta_path = 'prosody_embedding_test/metadata.txt'
    perfix = 'prosody_embedding_test'
    batch_size = 10

    model = prosody_encoder()
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = loss_fn()

    iteration = 0
    epoch_offset = 0

    trainset = Dataset(meta_path, perfix)
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    model.train().cuda()
    epoch_offset = max(0, int(iteration / len(train_loader)))

    for epoch in range(epoch_offset, 100):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            mel, c = batch
            mel = torch.autograd.Variable(mel.cuda())
            c = torch.autograd.Variable(c.cuda())
            output = model(mel)

            loss = criterion(output, c)
            reduced_loss = loss.item()
            loss.backward()

            optimizer.step()
            print("{}:\t{:.9f}".format(iteration, reduced_loss))

            iteration += 1
            if (iteration % 40 == 0):
                print(output.detach().numpy())
                print(c.detach().numpy())

if __name__ == "__main__":
    train()