import math
import torch
import torch.nn.functional as F
from utils import Vocab


def greedy(decoder, x0, state0, enc_out, max_len=18,
           pad=Vocab.pad(), eos=Vocab.eos(), unk=Vocab.unk(), no_unk=True, normalize_scores=True):
    x = x0
    state = state0
    tokens = []
    scores = []
    for _ in range(max_len + 1):  # +1 for eos
        logits, state = decoder(x, state, enc_out)
        probs = F.log_softmax(logits, dim=-1) if normalize_scores else logits
        probs[:, pad] = -math.inf  # never select pad
        if no_unk:
            probs[:, unk] = -math.inf
        score, x = torch.max(probs, dim=1, keepdim=True)
        tokens.append(x)
        scores.append(score)
    tokens = torch.cat(tokens, dim=1).tolist()
    scores = torch.cat(scores, dim=1).tolist()
    outputs = []
    for token, score in zip(tokens, scores):
        try:
            seq_len = token.index(eos) + 1
        except ValueError:
            seq_len = len(token)
        outputs.append([{
            'tokens': token[:seq_len],
            'positional_scores': score[:seq_len],
            'score': sum(score[:seq_len]),
        }])
    return outputs


def beam_search(decoder, x0, state0, enc_out, max_len=18, beam_size=3,
                pad=Vocab.pad(), eos=Vocab.eos(), unk=Vocab.unk(),
                stop_early=True, normalize_scores=True, sampling=False,
                len_penalty=0., dup_penalty=0., no_unk=True, mono_penalty=0., min_len=1,
                forbidden_words=None, forbidden_penalties=None):
    bsz = x0.size(0)

    # initialize buffers: avoid allocating new memories repeatedly
    # current scores in each step
    scores = x0.data.new(bsz * beam_size, max_len + 1).float().fill_(0)
    # rearranged scores based on the lastest
    scores_buf = scores.clone()
    tokens = x0.data.new(bsz * beam_size, max_len + 2).fill_(pad)
    tokens_buf = tokens.clone()
    tokens[:, :1] = eos

    def beam_repeat(x):
        return x.repeat(1, beam_size, *(1 for _ in range(x.dim() - 2))).view(-1, *x.size()[1:])

    state = [beam_repeat(x) for x in state0]
    enc_out = [beam_repeat(x) for x in enc_out]

    # list of completed sentences
    finalized = [[] for _ in range(bsz)]
    finished = [False for _ in range(bsz)]
    worst_finalized = [{'idx': None, 'score': -math.inf} for _ in range(bsz)]
    num_remaining_sent = bsz

    # number of candidate hypos per step
    cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

    # offset arrays for converting between different indexing schemes
    bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
    cand_offsets = torch.arange(0, cand_size).type_as(tokens)

    # helper function for allocating buffers on the fly
    buffers = {}

    def buffer(name, type_of=tokens):  # noqa
        if name not in buffers:
            buffers[name] = type_of.new()
        return buffers[name]

    def is_finished(sent, step, unfinalized_scores=None):
        """
        Check whether we've finished generation for a given sentence, by
        comparing the worst score among finalized hypotheses to the best
        possible score among unfinalized hypotheses.
        """
        assert len(finalized[sent]) <= beam_size
        if len(finalized[sent]) == beam_size:
            if stop_early or step == max_len or unfinalized_scores is None:
                return True
            # stop if the best unfinalized score is worse than the worst
            # finalized one
            best_unfinalized_score = unfinalized_scores[sent].max()
            if normalize_scores:
                best_unfinalized_score /= max_len
            if worst_finalized[sent]['score'] >= best_unfinalized_score:
                return True
        return False

    def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
        """
        Finalize the given hypotheses at this step, while keeping the total
        number of finalized hypotheses per sentence <= beam_size.

        Note: the input must be in the desired finalization order, so that
        hypotheses that appear earlier in the input are preferred to those
        that appear later.

        Args:
            step: current time step
            bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                indicating which hypotheses to finalize
            eos_scores: A vector of the same size as bbsz_idx containing
                scores for each hypothesis
            unfinalized_scores: A vector containing scores for all
                unfinalized hypotheses
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)
        tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
        tokens_clone[:, step] = eos

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if normalize_scores:
            eos_scores /= (step + 1) ** len_penalty

        sents_seen = set()
        for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
            sent = idx // beam_size
            sents_seen.add(sent)

            def get_hypo():
                return {
                    'tokens': tokens_clone[i].tolist(),
                    'score': score,
                    'positional_scores': pos_scores[i].tolist(),
                }

            if len(finalized[sent]) < beam_size:
                finalized[sent].append(get_hypo())
            elif not stop_early and score > worst_finalized[sent]['score']:
                # replace worst hypo for this sentence with new/better one
                worst_idx = worst_finalized[sent]['idx']
                if worst_idx is not None:
                    finalized[sent][worst_idx] = get_hypo()

                # find new worst finalized hypo for this sentence
                idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                worst_finalized[sent] = {
                    'score': s['score'],
                    'idx': idx,
                }

        # return number of hypotheses finished this step
        num_finished = 0
        for sent in sents_seen:
            # check termination conditions for this sentence
            if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                finished[sent] = True
                num_finished += 1
        return num_finished

    def columnwise_scatter_(x, index, addend):
        """
        This function is in-place, only supporting 2D tensors and dim=1.
        for i in x.size(0):
            for j in index.size(1):
                x[i][index[i][j]] += addend
        :param x: 2D torch.Tensor
        :param index: 2D torch.LongTensor, index.size(0) == x.size(0), 0 <= elements < x.size(1)
        :param addend: float
        :return: x itself
        """
        flat_index = index + index.new_tensor(torch.arange(x.size(0)).unsqueeze(1)) * x.size(1)
        return x.put_(flat_index, x.new_full(flat_index.size(), addend), accumulate=True)

    reorder_state = None
    for step in range(max_len + 1):  # one extra step for EOS marker
        # reorder decoder internal states based on the prev choice of beams
        if reorder_state is not None:
            state = (s.index_select(0, reorder_state) for s in state)

        logits, state = decoder(tokens[:, step:step + 1], state, enc_out)
        probs = F.log_softmax(logits, dim=-1)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
            scores = scores.type_as(probs)
            scores_buf = scores_buf.type_as(probs)
        elif not sampling:
            # make probs contain cumulative scores for each hypothesis
            probs.add_(scores[:, step - 1].view(-1, 1))
            # apply duplication penalty
            columnwise_scatter_(probs, index=tokens[:, 1: step + 1], addend=-dup_penalty)

        probs[:, pad] = -math.inf  # never select pad
        if no_unk:
            probs[:, unk] = -math.inf  # never select unk

        # penalize forbidden words
        if forbidden_words:
            for fw, fp in zip(forbidden_words, forbidden_penalties):
                probs[:, fw].sub_(fp)

        cand_scores = buffer('cand_scores', type_of=scores)
        cand_indices = buffer('cand_indices')
        per_cand_scores = buffer('per_cand_scores', type_of=scores)
        per_cand_indices = buffer('per_cand_indices')
        cand_beams = buffer('cand_beams')
        eos_bbsz_idx = buffer('eos_bbsz_idx')
        eos_scores = buffer('eos_scores', type_of=scores)

        if step < max_len:
            if sampling:
                assert unk > pad, 'here we assume first tokens in vocab can be ignored'
                start_idx = (unk if no_unk else pad) + 1
                exp_probs = probs.exp_().view(-1, probs.size(-1))
                if step == 0:
                    # we exclude the first two vocab items,
                    torch.multinomial(exp_probs[:, start_idx:], beam_size, replacement=True, out=cand_indices)
                    cand_indices.add_(start_idx)
                else:
                    torch.multinomial(exp_probs[:, start_idx:], 1, replacement=True, out=cand_indices)
                    cand_indices.add_(start_idx)
                torch.gather(exp_probs, dim=1, index=cand_indices, out=cand_scores)
                cand_scores.log_()
                cand_indices = cand_indices.view(bsz, -1).repeat(1, 2)
                cand_scores = cand_scores.view(bsz, -1).repeat(1, 2)
                if step == 0:
                    cand_beams = torch.zeros(bsz, cand_size).type_as(cand_indices)
                else:
                    cand_beams = torch.arange(0, beam_size).repeat(bsz, 2).type_as(cand_indices)
                    # make scores cumulative
                    cand_scores.add_(
                        torch.gather(
                            scores[:, step - 1].view(bsz, beam_size), dim=1,
                            index=cand_beams,
                        )
                    )
            else:
                # take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                torch.topk(probs, k=min(cand_size, probs.numel() // bsz - 1 - no_unk),
                           out=(per_cand_scores, per_cand_indices))
                if mono_penalty and step > 0:
                    per_cand_scores -= torch.arange(cand_size).unsqueeze(0).to(probs) * mono_penalty
                torch.topk(per_cand_scores.view(bsz, -1),
                           k=cand_size,
                           out=(cand_scores, cand_indices))
                torch.div(cand_indices, cand_size, out=cand_beams)
                torch.gather(per_cand_indices.view(bsz, -1), 1, cand_indices, out=cand_indices)
        else:
            # finalize all active hypotheses once we hit max_len
            # pick the hypothesis with the highest prob of EOS right now
            torch.sort(
                probs[:, eos],
                descending=True,
                out=(eos_scores, eos_bbsz_idx),
            )
            num_remaining_sent -= finalize_hypos(
                step, eos_bbsz_idx, eos_scores)
            assert num_remaining_sent == 0
            break

        # cand_bbsz_idx contains beam indices for the top candidate
        # hypotheses, with a range of values: [0, bsz*beam_size),
        # and dimensions: [bsz, cand_size]
        cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

        # finalize hypotheses that end in eos
        eos_mask = cand_indices.eq(eos)
        if step >= min_len:
            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                num_remaining_sent -= finalize_hypos(
                    step, eos_bbsz_idx, eos_scores, cand_scores)

        assert num_remaining_sent >= 0
        if num_remaining_sent == 0:
            break
        assert step < max_len

        # set active_mask so that values > cand_size indicate eos hypos
        # and values < cand_size indicate candidate active hypos.
        # After, the min values per row are the top candidate active hypos
        active_mask = buffer('active_mask')
        torch.add(
            eos_mask.type_as(cand_offsets) * cand_size,
            cand_offsets[:eos_mask.size(1)],
            out=active_mask,
        )

        # get the top beam_size active hypotheses, which are just the hypos
        # with the smallest values in active_mask
        active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
        torch.topk(
            active_mask, k=beam_size, dim=1, largest=False,
            out=(_ignore, active_hypos)
        )
        active_bbsz_idx = buffer('active_bbsz_idx')
        torch.gather(
            cand_bbsz_idx, dim=1, index=active_hypos,
            out=active_bbsz_idx,
        )
        active_scores = torch.gather(
            cand_scores, dim=1, index=active_hypos,
            out=scores[:, step].view(bsz, beam_size),
        )
        active_bbsz_idx = active_bbsz_idx.view(-1)
        active_scores = active_scores.view(-1)

        # copy tokens and scores for active hypotheses
        torch.index_select(
            tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
            out=tokens_buf[:, :step + 1],
        )
        torch.gather(
            cand_indices, dim=1, index=active_hypos,
            out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
        )
        if step > 0:
            torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx,
                out=scores_buf[:, :step],
            )
        torch.gather(
            cand_scores, dim=1, index=active_hypos,
            out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
        )

        # swap buffers
        old_tokens = tokens
        tokens = tokens_buf
        tokens_buf = old_tokens
        old_scores = scores
        scores = scores_buf
        scores_buf = old_scores

        # reorder incremental state in decoder
        reorder_state = active_bbsz_idx

    # sort by score descending
    for sent in range(bsz):
        finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

    return finalized
