""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if not bptt:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec

class ContextModel(BaseModel):
    def __init__(self, src_encoder, decoder, des_encoder, rep_encoder, exp_encoder, oth_encoder, context_ratio):
        super(ContextModel, self).__init__(src_encoder, decoder)
        self.src_encoder = src_encoder
        self.des_encoder = des_encoder
        self.rep_encoder = rep_encoder
        self.exp_encoder = exp_encoder
        self.oth_encoder = oth_encoder
        self.decoder = decoder
        self.context_ratio = context_ratio

    def forward(self, src, tgt, lengths, des, des_lengths,
                        rep, rep_lengths, exp, exp_lengths, oth, oth_lengths, bptt=False, with_align=False):
        tgt_in = tgt[:-1]  # exclude last target from inputs

        src_state, memory_bank, lengths = self.src_encoder(src, lengths)
        des_state, des_memory_bank, des_lengths = self.des_encoder(des, des_lengths)
        rep_state, rep_memory_bank, rep_lengths = self.rep_encoder(rep, rep_lengths)
        exp_state, exp_memory_bank, exp_lengths = self.exp_encoder(exp, exp_lengths)
        oth_state, oth_memory_bank, oth_lengths = self.oth_encoder(oth, oth_lengths)
        
        des_memory_bank = torch.transpose(des_memory_bank, 0, 1)
        rep_memory_bank = torch.transpose(rep_memory_bank, 0, 1)
        exp_memory_bank = torch.transpose(exp_memory_bank, 0, 1)
        oth_memory_bank = torch.transpose(oth_memory_bank, 0, 1)

        context_state = ((des_state[0]+rep_state[0]+exp_state[0]+oth_state[0])/4, 
                         (des_state[1]+rep_state[1]+exp_state[1]+oth_state[1])/4)
        # print("src_state:{}".format(src_state[0].shape))
        # print("context_state:{}".format(context_state[0].shape))    
        src_state = (src_state[0] * (1-self.context_ratio) + context_state[0] * self.context_ratio,
                     src_state[1] * (1-self.context_ratio) + context_state[1] * self.context_ratio)

        zeros = torch.zeros_like(des_memory_bank[:, -1, :]).unsqueeze(1)
        for id in range(len(lengths)):
            des_hidden = des_memory_bank[:, :des_lengths[id], :]
            if des_lengths[id] != 1:
                des_hidden = torch.cat([des_hidden, zeros], dim=1)
            rep_hidden = rep_memory_bank[:, :rep_lengths[id], :]
            if rep_lengths[id] != 1:
                rep_hidden = torch.cat([rep_hidden, zeros], dim=1)
            exp_hidden = exp_memory_bank[:, :exp_lengths[id], :]
            if exp_lengths[id] != 1:
                exp_hidden = torch.cat([exp_hidden, zeros], dim=1)
            oth_hidden = oth_memory_bank[:, :oth_lengths[id], :]
            if oth_lengths[id] != 1:
                oth_hidden = torch.cat([oth_hidden, zeros], dim=1)
            memory_bank[:,id,:] = memory_bank[:,id,:] * (1-self.context_ratio) 
            + torch.transpose(torch.cat([des_hidden, rep_hidden, exp_hidden, oth_hidden], dim=1), 0, 1) * self.context_ratio
            
        # print("src_state:{}, memory_bank:{}, lengths:{}".format(src_state[0].shape, memory_bank.shape,lengths))
        # print("context_state:{}, memory_bank:{}, lengths:{}".format(context_state[0].shape, context_memory_bank.shape, des_lengths))
        # print("des_state:{}, memory_bank:{}, lengths:{}".format(des_state[0].shape, des_memory_bank.shape, des_lengths))
        # print("rep_state:{}, memory_bank:{}, lengths:{}".format(rep_state[0].shape, rep_memory_bank.shape, rep_lengths))
        # print("exp_state:{}, memory_bank:{}, lengths:{}".format(exp_state[0].shape, exp_memory_bank.shape, exp_lengths))
        # print("oth_state:{}, memory_bank:{}, lengths:{}".format(oth_state[0].shape, oth_memory_bank.shape, oth_lengths))

        if not bptt:
            self.decoder.init_state(src, memory_bank, src_state)
        tgt_out, attns = self.decoder(tgt_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return tgt_out, attns

    def update_dropout(self, dropout):
        self.src_encoder.update_dropout(dropout)
        self.des_encoder.update_dropout(dropout)
        self.rep_encoder.update_dropout(dropout)
        self.exp_encoder.update_dropout(dropout)
        self.oth_encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec

class LanguageModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic decoder only model.
    Currently TransformerLMDecoder is the only LM decoder implemented
    Args:
      decoder (onmt.decoders.TransformerLMDecoder): a transformer decoder
    """

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used"
                             "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (Tensor): A source sequence passed to decoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on decoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, memory_bank=None, memory_lengths=lengths,
            with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout):
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).
        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec
