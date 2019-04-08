import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads source(audio,text), target(audio) pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_source_target_pair(self, audiopath_and_text):
        # separate filename and text
        source_audiopath, source_text, target_audiopath, target_text = audiopath_and_text
        source_mel = self.get_mel(source_audiopath) # []
        source_text = self.get_text(source_text)
        target_mel = self.get_mel(target_audiopath) # []
        #target_text = self.get_text(target_text)
        return (source_mel, target_mel, source_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_source_target_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from source (mel-spectrogram,text) and target (mel-spectrogram)
        PARAMS
        ------
        batch: [[source_mel_normalized, target_mel_normalized, source_text, target_text], ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        # print('source_mel_normalized:', batch[0][0].shape)
        # print('target_mel_normalized:', batch[0][1].shape)
        # print('source_text:', batch[0][2].shape)

        source_mel_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_source_text_len = max([len(x[2]) for x in batch])
        #print('max_source_text_len: ', max_source_text_len)

        source_text_padded = torch.LongTensor(len(batch), max_source_text_len).zero_()

        for i in range(len(ids_sorted_decreasing)):
            x = batch[ids_sorted_decreasing[i]]
            source_text = x[2]
            source_text_padded[i, :source_text.size(0)] = source_text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_source_len = max([x[0].size(1) for x in batch])
        if max_source_len % self.n_frames_per_step != 0:
            max_source_len += self.n_frames_per_step - max_source_len % self.n_frames_per_step
            assert max_source_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        source_mel_padded = torch.FloatTensor(len(batch), num_mels, max_source_len)
        source_mel_padded.zero_()

        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        target_mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        target_mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        #source_mel_lengths = torch.LongTensor(len(batch))
        source_text_lengths = torch.LongTensor(len(batch))
        for j in range(len(batch)):
            i = ids_sorted_decreasing[j]
            source_mel = batch[i][0]
            target_mel = batch[i][1]
            source_text = batch[i][2]
            source_mel_padded[j, :, :source_mel.size(1)] = source_mel
            target_mel_padded[j, :, :target_mel.size(1)] = target_mel
            gate_padded[j, target_mel.size(1)-1:] = 1
            output_lengths[j] = target_mel.size(1)
            #source_mel_lengths[j] = source_mel.size(1)
            source_text_lengths[j] = source_text.size(0)


        return source_mel_padded, source_mel_lengths, target_mel_padded, gate_padded, \
            output_lengths, source_text_padded, source_text_lengths
