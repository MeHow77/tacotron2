import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import argparse
import os

import numpy as np
import time
import torch

from hparams import create_hparams
from layers import TacotronSTFT
from audio_processing import griffin_lim, mel_denormalize
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write

def plot_data(data, index, output_dir="", figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig(os.path.join(output_dir, 'sentence_{}.png'.format(index)))

def generate_mels(hparams, checkpoint_path, sentences, cleaner, silence_mel_padding, is_GL, output_dir=""):
    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
    _ = model.eval()

    output_mels = []
    encoder_outputs = []
    for i, s in enumerate(sentences):
        sequence = np.array(text_to_sequence(s, cleaner))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        stime = time.time()
        _, mel_outputs_postnet, _, alignments, encoder_output = model.inference(sequence)
        if(is_GL):
            plot_data((mel_outputs_postnet.data.cpu().numpy()[0],
                   alignments.data.cpu().numpy()[0].T), i, output_dir)
        inf_time = time.time() - stime
        print("{}th sentence, Infenrece time: {:.2f}s, len_mel: {}".format(i, inf_time, mel_outputs_postnet.size(2)))
        output_mels.append(mel_outputs_postnet[:,:,:-silence_mel_padding].squeeze(0).data.cpu().numpy())
        encoder_outputs.append(encoder_output.squeeze(0).data.cpu().numpy())
    return output_mels, encoder_outputs

def mels_to_wavs_GL(hparams, mels, taco_stft, output_dir="", ref_level_db = 0, magnitude_power=1.5):
    for i, mel in enumerate(mels):
        stime = time.time()
        mel_decompress = mel_denormalize(torch.from_numpy(mel).cuda().unsqueeze(0))
        mel_decompress = taco_stft.spectral_de_normalize(mel_decompress + ref_level_db) ** (1/magnitude_power)
        mel_decompress_ = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress_[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :]),
                               taco_stft.stft_fn, 60)
        waveform = waveform[0].data.cpu().numpy()
        dec_time = time.time() - stime
        len_audio = float(len(waveform)) / float(hparams.sampling_rate)
        str = "{}th sentence, audio length: {:.2f} sec,  mel_to_wave time: {:.2f}".format(i, len_audio, dec_time)
        print(str)
        write(os.path.join(output_dir,"sentence_{}.wav".format(i)), hparams.sampling_rate, waveform)

def run(hparams, checkpoint_path, sentence_path, clenaer, silence_mel_padding, is_GL, is_melout, is_metaout, is_encout, output_dir):
    f = open(sentence_path, 'r')
    sentences = [x.strip() for x in f.readlines()]
    print('All sentences to infer:',sentences)
    f.close()
    os.makedirs(output_dir, exist_ok=True)

    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

    mels, encs = generate_mels(hparams, checkpoint_path, sentences, clenaer, silence_mel_padding, is_GL, output_dir)
    if(is_GL): mels_to_wavs_GL(hparams, mels, stft, output_dir)

    mel_paths = []
    enc_paths = []
    if is_melout:
        mel_dir = os.path.join(output_dir, 'mels')
        os.makedirs(mel_dir, exist_ok=True)

        for i, mel in enumerate(mels):
            mel_path = os.path.join(output_dir, 'mels/', "mel-{}.npy".format(i))
            mel_paths.append(mel_path)
            if(list(mel.shape)[1] >=  hparams.max_decoder_steps - silence_mel_padding):
                continue
            np.save(mel_path, mel)

    if is_encout:
        enc_dir = os.path.join(output_dir, 'encs')
        os.makedirs(enc_dir, exist_ok=True)

        for i, enc in enumerate(encs):
            mel = mels[i]
            enc_path = os.path.join(output_dir, 'encs/', "enc-{}.npy".format(i))
            enc_paths.append(enc_path)
            if (list(mel.shape)[1] >= hparams.max_decoder_steps - silence_mel_padding):
                continue
            np.save(enc_path, enc)

    if is_metaout:
        with open(os.path.join(output_dir, 'metadata.csv'), 'w', encoding='utf-8') as file:
            lines = []
            for i, s in enumerate(sentences):
                mel_path = mel_paths[i]
                enc_path = enc_paths[i]
                if (list(mels[i].shape)[1] >= hparams.max_decoder_steps - silence_mel_padding):
                    continue
                if is_encout: lines.append('{}|{}|{}\n'.format(mel_path,s,enc_path))
                else: lines.append('{}|{}\n'.format(mel_path,s))
            file.writelines(lines)

if __name__ == '__main__':
    """
    usage
    python inference.py -o=synthesis/80000 -c=nam_h_ep8/checkpoint_80000 -s=test.txt --silence_mel_padding=3 --is_GL 
        -> wave, figure
    python inference.py -o=kss_mels_given_park_text -c=kakao_kss_model_checkpoint_23500 -s=skip_review_percentile_metadata_n.csv --silence_mel_padding=3 --is_melout --is_metaout 
        -> mels, metadata.csv
    python inference.py -o=kss_mels_given_park_text -c=kakao_kss_model_checkpoint_23500 -s=skip_review_percentile_metadata_n.csv --silence_mel_padding=3 --is_melout --is_metaout --is_encout 
        -> mels, encs, metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save wave and fig')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=True, help='checkpoint path')
    parser.add_argument('-s', '--sentence_path', type=str, default=None,
                        required=True, help='sentence path')
    parser.add_argument('--silence_mel_padding', type=int, default=1,
                        help='silence audio size is hop_length * silence mel padding')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--is_GL', action="store_true", help='Whether to do Giffin & Lim inference or not ')
    parser.add_argument('--is_melout', action="store_true", help='Whether to save files for melspectrogram  or not ')
    parser.add_argument('--is_metaout', action="store_true", help='Whether to save metadata.csv for (mel, text) tuple or not ')
    parser.add_argument('--is_encout', action="store_true", help='Whether to save files for encoder features or not ')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    run(hparams, args.checkpoint_path, args.sentence_path, hparams.text_cleaners, args.silence_mel_padding, args.is_GL, args.is_melout, args.is_metaout, args.is_encout, args.output_directory)


