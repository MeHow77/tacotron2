from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(self.n_symbols,
                         self.n_symbols,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.n_symbols))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.n_symbols,
                            int(self.n_symbols / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_rnn_dim = hparams.encoder_rnn_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_rnn_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_rnn_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_rnn_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_rnn_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_rnn_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_rnn_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        ## -- start Attention related block
        # s = output(final representation),
        # a = attention context
        # f(,) = additive rnn attention
        #
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        # 왜 결합하는데? location based attention 이라서, attend(s_(i-1), a_(i-1)) -> W^T sigma(W^(1)s_(i-1),W^(2)a_(i-1)) 이니 W[s_(i-1),a_(i-1)] -> 해결
        self.attention_hidden, self.attention_cell = self.attention_rnn( # additive attention rnn
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        # attention_hidden: f(,)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)
        # attention_cell: 다음 step 계산을 위해 유지되는 값

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        ## -- end Attention related block

        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        #memory = memory.transpose(1,2) #memory must be [self.batch_size, output_lengths, dim], so make [self.batch_size, mel_dim, output_lengths] to [self.batch_size, output_lengths, dim]
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        #memory = memory.transpose(1, 2)  # memory must be [self.batch_size, output_lengths, mel_dim], so make [self.batch_size, mel_dim, output_lengths] to [self.batch_size, output_lengths, mel_dim]
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

class MelLinEncoder(nn.Module):
    def __init__(self, hparams):
        super(MelLinEncoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fc_dim = hparams.mel_fc_dim
        self.mel_rnn_dim = hparams.mel_rnn_dim
        self.encoder_rnn_dim = hparams.encoder_rnn_dim
        self.encoder_attention_rnn_dim = hparams.encoder_attention_rnn_dim
        self.linguistic_feature_dim = hparams.encoder_embedding_dim

        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_encoder_dropout = hparams.p_encoder_dropout
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.mel_fc_dim, hparams.mel_fc_dim])

        #cell_input = torch.cat((mel_input, self.attention_context), -1)
        self.mel_rnn = nn.LSTMCell(
            hparams.mel_fc_dim + self.linguistic_feature_dim,
            hparams.mel_rnn_dim) # -> mel_rnn

        self.attention_layer = Attention(
            hparams.mel_rnn_dim, self.linguistic_feature_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        # 차원 맞춰서 수정 ( encoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.encoder_rnn = nn.LSTMCell(
            hparams.encoder_rnn_dim + self.linguistic_feature_dim,
            hparams.encoder_rnn_dim)

    def initialize_encoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.encoder_attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.encoder_attention_rnn_dim).zero_())

        self.encoder_hidden = Variable(memory.data.new(
            B, self.encoder_rnn_dim).zero_())
        self.encoder_cell = Variable(memory.data.new(
            B, self.encoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.linguistic_feature_dim).zero_())

        #print('memory.type()', memory.type())
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        # 왜 한번 계산되고 치우는거지? -> 당연 다 만들어진 값이라 recursive 하게 게산될 필요가 없음.
        # 따라서 한번 계산한 값을 계속 유지
        self.mask = mask
        pass

    def parse_encoder_outputs(self, encoder_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        encoder_outputs:
        alignments:

        RETURNS
        -------
        encoder_outputs:
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        # print('torch.stack(alignments): ', torch.stack(alignments).shape)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B, encoder_rnn_dim) -> (B, T_out, encoder_rnn_dim)
        # print('torch.stack(encoder_outputs): ', torch.stack(encoder_outputs).shape)
        mel_outputs = torch.stack(encoder_outputs).transpose(0, 1).contiguous() # ?

        return mel_outputs, alignments

    def encode(self, mel_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        mel_input: melspectrogram vector at time t

        RETURNS
        self.encoder_hidden, self.attention_weights
        -------

        """
        # print('mel_input: ', mel_input.shape, ' attention_context: ', self.attention_context.shape)
        cell_input = torch.cat((mel_input, self.attention_context), -1)
        # print('cell_input: ', cell_input.shape)
        # 왜 결합하는데? location based attention 이라서, attend(s_(i-1), a_(i-1)) -> W^T sigma(W^(1)s_(i-1),W^(2)a_(i-1)) 이니 W[s_(i-1),a_(i-1)] -> 해결
        # ? 위 생각은 좀 잘못 생각한 것 같음. 입력단부터 추가적인 정보를 하나 더가지고 시작해서 representation을 만들자는 접근으로 보임
        self.attention_hidden, self.attention_cell = self.mel_rnn( # additive attention rnn
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)
        # attention_cell: 다음 step 계산을 위해 유지되는 값
        # print('attention_hidden: ', self.attention_hidden.shape)
        # print('attention_cell: ', self.attention_cell.shape)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        encoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        # print('attention_hidden: ', self.attention_hidden.shape)
        # print('attention_context: ', self.attention_context.shape)
        # print('encoder_input: ', encoder_input.shape)
        self.encoder_hidden, self.encoder_cell = self.encoder_rnn(
            encoder_input, (self.encoder_hidden, self.encoder_cell))
        ## -- end Attention related block

        self.encoder_hidden = F.dropout(
            self.encoder_hidden, self.p_encoder_dropout, self.training)
        self.encoder_cell = F.dropout(
            self.encoder_cell, self.p_encoder_dropout, self.training)

        return self.encoder_hidden, self.attention_weights

    def forward(self, linguistic, melspectrogram, linguistic_lengths, melspectrogram_lengths):
        representation_x = self.prenet(melspectrogram)
        # pack_padded_sequence 적용 실패, attention output에 masking 적용해야 됨

        self.initialize_encoder_states(
            linguistic, mask=~get_mask_from_lengths(linguistic_lengths))

        encoder_outputs, alignments = [], []

        while len(encoder_outputs) < melspectrogram.size(1):
            # matrix to vector,
            # print('representation_x: ',representation_x.shape)
            # print('encoder_outputs: ', encoder_outputs.shape)
            # print('len(encoder_outputs): ', len(encoder_outputs))
            encoder_input = representation_x[:,len(encoder_outputs),:]
            #print('encoder_input: ', encoder_input.shape)
            encoder_output, attention_weights = self.encode(encoder_input)
            # print('encoder_outputs: ', encoder_output.shape)
            encoder_outputs += [encoder_output.squeeze(1)]
            alignments += [attention_weights]

        # print('len: ', len(encoder_outputs))
        encoder_outputs, alignments = self.parse_encoder_outputs(encoder_outputs, alignments)
        #encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        # print('encoder_outputs: ', encoder_outputs.shape)
        # print('alignments: ', alignments.shape)
        return encoder_outputs, alignments

    def inference(self, linguistic, melspectrogram):
        representation_x = self.prenet(melspectrogram)

        self.initialize_encoder_states(
            linguistic, mask=None)

        encoder_outputs, alignments = [], []
        while len(encoder_outputs) < melspectrogram.size(1):
            encoder_input = representation_x[:, len(encoder_outputs), :]
            encoder_output, attention_weights = self.encode(encoder_input)
            values, indices = torch.max(attention_weights, 1)
            print('alignments', indices)
            encoder_outputs += [encoder_output.squeeze(1)]
            alignments += [attention_weights]

        encoder_outputs, alignments = self.parse_encoder_outputs(encoder_outputs, alignments)
        return encoder_outputs, alignments
        pass

class MelToMel(nn.Module):
    def __init__(self, hparams):
        super(MelToMel, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder = MelLinEncoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        source_mel_padded, input_lengths, target_mel_padded, gate_padded, \
        output_lengths,  source_text_padded, source_text_lengths = batch
        max_len = list(source_mel_padded.shape)[2]
        source_mel_padded = to_gpu(source_mel_padded).float()
        target_mel_padded = to_gpu(target_mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        input_lengths = to_gpu(input_lengths).long()
        output_lengths = to_gpu(output_lengths).long()
        source_text_padded = to_gpu(source_text_padded).long()
        source_text_lengths = to_gpu(source_text_lengths).long()

        return (
            (source_mel_padded, input_lengths, target_mel_padded, max_len, output_lengths, source_text_padded, source_text_lengths),
            (target_mel_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        source_mel_padded, input_lengths, target_mel_padded, max_len, output_lengths, linguistic_features, source_text_lengths = self.parse_input(inputs)
        output_lengths = output_lengths.data
        source_mel_shape = list(source_mel_padded.shape)
        source_mel_padded = source_mel_padded.transpose(1, 2)
        encoded_feature, mel_lin_alignments =  self.encoder(linguistic_features, source_mel_padded, source_text_lengths, input_lengths) #linguistic, melspectrogram, linguistic_lengths, melspectrogram_lengths
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoded_feature, target_mel_padded, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        source_mel_padded, linguistic_features = self.parse_input(inputs)
        source_mel_padded = source_mel_padded.transpose(1, 2)
        encoded_feature, mel_lin_alignments = self.encoder.inference(linguistic_features, source_mel_padded)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoded_feature)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_lin_alignments])

        return outputs
