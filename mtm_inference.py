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
from scipy.io.wavfile import write

def plot_data(data, index, output_dir="", figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig(os.path.join(output_dir, 'sentence_{}.png'.format(index)))

def get_mel(filename, silence_mel_padding):
    # print(np.load(filename).shape, np.load(filename).max(), np.load(filename).min())
    # print((np.ones((1,80,silence_mel_padding),dtype=np.float32)*-4.0).shape)
    melspec = torch.from_numpy(np.append(np.load(filename), np.ones((80,silence_mel_padding),dtype=np.float32)*-4.0, axis=1)).unsqueeze(0)
    # print(melspec.shape)
    return melspec

def generate_mels(hparams, checkpoint_path, mel_paths, silence_mel_padding, stft, output_dir=""):
    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})
    _ = model.eval()
    output_mels = []
    for i, a in enumerate(mel_paths):
        source_mel = get_mel(a, silence_mel_padding).cuda()
        stime = time.time()
        _, mel_outputs_postnet, _, alignments = model.inference(source_mel)
        plot_data((source_mel.data.cpu().numpy()[0], mel_outputs_postnet.data.cpu().numpy()[0],
                   alignments.data.cpu().numpy()[0].T), i, output_dir)
        inf_time = time.time() - stime
        print("{}th sentence, Infenrece time: {:.2f}s, len_mel: {}".format(i, inf_time, mel_outputs_postnet.size(2)))
        output_mels.append(mel_outputs_postnet[:,:,:-silence_mel_padding])
    return output_mels

def mels_to_wavs_GL(hparams, mels, taco_stft, output_dir="", ref_level_db = 0, magnitude_power=1.5):
    map = []
    for i, mel in enumerate(mels):
        stime = time.time()
        mel_decompress = mel_denormalize(mel)
        mel_decompress = taco_stft.spectral_de_normalize(mel_decompress + ref_level_db) ** (1/magnitude_power)
        mel_decompress_ = mel_decompress.transpose(1, 2).data.cpu()
        mel_decompress = mel_decompress.data.cpu().numpy()
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
        mel = torch.clamp(mel, -4, 4)
        mel = mel.squeeze(0).transpose(0,1).data.cpu().numpy()
        mel_path = os.path.join(output_dir, "mel_{}.npy".format(i))
        np.save(mel_path, mel)
        map.append("|{}|\n".format(mel_path))
    f = open(os.path.join(output_dir,'map.txt'),'w',encoding='utf-8')
    f.writelines(map)
    f.close()

def run(hparams, checkpoint_path, audio_path_file, silence_mel_padding, output_dir):
    f = open(audio_path_file, 'r')
    os.makedirs(output_dir,exist_ok=True)
    mel_paths = [x.strip() for x in f.readlines()]
    print('All mel to infer:',mel_paths)
    f.close()

    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

    mels = generate_mels(hparams, checkpoint_path, mel_paths, silence_mel_padding, stft, output_dir)
    mels_to_wavs_GL(hparams, mels, stft, output_dir)
    pass

if __name__ == '__main__':
    """
    usage
    python mtm_inference.py -o=sts_output -c=vc2/checkpoint_35000 -m=mtm_test.txt
    python mtm_inference.py -o=. -c=vc_cho_to_park_checkpoint -m=mtm_test.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='',
                        help='directory to save wave and fig')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=True, help='checkpoint path')
    parser.add_argument('-m', '--mel_path_file', type=str, default=None,
                        required=True, help='melspectrogram paths')
    parser.add_argument('--silence_mel_padding', type=int, default=3,
                        help='silence audio size is hop_length * silence mel padding')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    run(hparams, args.checkpoint_path, args.mel_path_file, args.silence_mel_padding ,args.output_directory)