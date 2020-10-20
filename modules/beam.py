import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence
import time

"""
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
"""


class Beam:
    def __init__(self, prevNode, wordId, logProb, length):

        self.prevNode = prevNode
        self.wordId = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        T = 0.1
        # prevent overflow
        return self.logp / float((self.length - 1)**T + 1e-6) + alpha*reward


def beam_decode(x, encoder, decoder,
                embeddings, classifier,
                beam_width, n_top):
    bs, _, _ = x.size()
    device = x.device

    decoded_batch = []
    sos_token = 0
    eos_token = 9
    max_length = 22
    encoder_outputs = encoder(x).permute(1, 0, 2)

    # decoding goes sentence by sentence
    for idx in range(bs):
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(
            1)  # (time_step, 1, out_channels)
        decoder_input = torch.LongTensor([[sos_token]]).to(device)

        # number of sentences to generate
        endnodes = []
        number_required = min(n_top+1, n_top-len(endnodes))

        # starting node - prevNode, wordId, logProb, length
        node = Beam(None, decoder_input, 0, 1)
        beam_nodes = PriorityQueue()
        # add length to avoid tie-breaking
        beam_nodes.put((-node.eval(), time.time(), node.length, node))
        qsize = 1

        # start beam search
        while True:
            if qsize > 3000:
                break

            scores, t, l, n = beam_nodes.get()
            decoder_input = n.wordId

            if n.wordId[-1].item() == eos_token and n.prevNode:
                endnodes.append((scores, n))

                # if we reached maximum sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # feeding to decoder
            embedding = embeddings(decoder_input)  # (num_words,1, out_channel)
            decoder_output = decoder(
                embedding, encoder_output, attention_mask=None)  # (1,n, out_channel)
            decoder_output = F.log_softmax(classifier(decoder_output), dim=-1)

            # get beam_width most value
            log_prob, indexes = decoder_output[-1].data.topk(beam_width)
            next_nodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[-1][new_k].view(1, -1)
                log_p = log_prob[-1][new_k].view(1, -1)
                node = Beam(n, torch.cat([decoder_input, decoded_t], dim=0),
                            n.logp+log_p, n.length+1)
                score = -node.eval()
                next_nodes.append((score, node))

            # put nodes into queue
            for i in range(len(next_nodes)):
                score, nn = next_nodes[i]
                beam_nodes.put((score, time.time(), nn.length, nn))
            qsize += len(next_nodes)-1

        # choose best paths, back trace
        if len(endnodes) == 0:
            endnodes = [beam_nodes.get() for _ in range(n_top)]
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordId)
        for i in range(n_top):
            decoded_batch.append(utterances[i])
    return pad_sequence(decoded_batch, batch_first=False, padding_value=eos_token)[:, :, -1]

# EOF
