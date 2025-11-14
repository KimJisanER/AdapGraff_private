from contextlib import nullcontext
import torch
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence

from esm.sdk.api import ESMProtein, LogitsConfig

BOS_ID = 4098
EOS_ID = 4097
PAD_ID = 0            # <- ESMC 토크나이저의 실제 pad id로 교체
MASK_ID = 4096        # 구조 트랙 마스킹 id (네 예시 값)

def bucketize_indices_by_length(lengths, max_tokens_per_batch=65536, sort_desc=True):
    """
    lengths: List[int]  (BOS/EOS 포함 길이)
    max_tokens_per_batch: 토큰 예산(길이합 상한). GPU 메모리 한계 따라 조절.

    Returns: List[List[int]]  (각 배치에 들어갈 샘플 인덱스 리스트)
    """
    idxs = list(range(len(lengths)))
    idxs.sort(key=lambda i: lengths[i], reverse=sort_desc)
    batches = []
    cur = []
    cur_tokens = 0
    for i in idxs:
        L = lengths[i]
        if cur and (cur_tokens + L > max_tokens_per_batch):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += L
    if cur:
        batches.append(cur)
    return batches


def esmc_encode_batched(esmc_model, seqs, tokenizer,
                        max_len_cap=1024, max_tokens_per_batch=65536,
                        device="cuda",
                        requires_grad=False,
                        use_autocast=False):
    """
    seqs: List[str]  (B개 시퀀스)
    tokenizer: ESMC 토크나이저
    return_embeddings=True이면 sequence-wise 임베딩 리턴.

    Returns:
      per_seq_embeddings: List[Tensor[L_i-2, D]]  (BOS/EOS 제거된 형태로 반환)
    """
    # 1) 우선 한 번 토크나이즈/패딩해서 길이 정보만 얻자(버킷을 위해)
    seq_ids, _, _, true_lens = build_sequence_tokens(seqs, tokenizer, max_len_cap=max_len_cap)

    batches = bucketize_indices_by_length(true_lens, max_tokens_per_batch=max_tokens_per_batch)
    per_seq_embeddings = [None] * len(seqs)

    for batch_idxs in batches:
        # 배치 시퀀스 뽑기
        batch_seqs = [seqs[i] for i in batch_idxs]

        # 다시 해당 배치만 토크나이즈/패딩
        ids, attn_mask, struct, lens = build_sequence_tokens(batch_seqs, tokenizer, max_len_cap=max_len_cap)

        ids      = ids.to(device)
        attn_mask= attn_mask.to(device)
        struct   = struct.to(device)

        ctx_grad = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx_grad:
            if use_autocast:
                ac = autocast()
            else:
                # no-op context
                from contextlib import nullcontext
                ac = nullcontext()

            with ac:
                try:
                    proteins = [ESMProtein(sequence=s) for s in batch_seqs]
                    outputs = esmc_model.encode(proteins)
                    logits = esmc_model.logits(outputs, LogitsConfig(sequence=True, return_embeddings=True))
                    emb = logits.embeddings  # [B, L_max, D]
                except Exception:
                    # 폴백(개별 호출): 그래도 grad가 필요하면 grad 켠 상태로
                    emb_list = []
                    for s in batch_seqs:
                        pt = esmc_model.encode(ESMProtein(sequence=s))
                        lg = esmc_model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
                        emb_list.append(lg.embeddings)  # [1, L_i, D]
                    # pad_sequence는 autograd-safe
                    from torch.nn.utils.rnn import pad_sequence
                    emb = pad_sequence([e.squeeze(0) for e in emb_list], batch_first=True)

        # BOS/EOS 제거 시에도 detach 금지
        for j, L in enumerate(lens):
            per_seq_embeddings[batch_idxs[j]] = emb[j, 1:L - 1, :].contiguous()  # ★ detach 없음

    return per_seq_embeddings

def masked_mean(x, valid_lengths):
    # x: (B, L, D), valid_lengths: (B,)
    B, L, D = x.size()
    mask = torch.arange(L, device=x.device).unsqueeze(0) < valid_lengths.unsqueeze(1)  # (B,L)
    mask = mask.unsqueeze(-1).float()  # (B,L,1)
    s = (x * mask).sum(dim=1)  # (B,D)
    denom = mask.sum(dim=1).clamp_min(1.0)  # (B,1)
    return s / denom

def build_sequence_tokens(seqs, tokenizer, max_len_cap=1024):
    seq_ids = []
    structure = []
    true_lens = []

    for s in seqs:
        s_trim = s[: max_len_cap - 2]  # BOS/EOS 2토큰 고려

        ids = tokenizer.encode(s_trim, add_special_tokens=True)  # [BOS] ... [EOS]
        ids = torch.tensor(ids, dtype=torch.long)

        # 3) 구조 트랙 (네 예시를 배치 형태로 일반화)
        st = torch.full_like(ids, MASK_ID)
        st[0] = BOS_ID
        st[-1] = EOS_ID

        seq_ids.append(ids)
        structure.append(st)
        true_lens.append(int(ids.size(0)))

    # 4) 패딩 (좌/우 패딩 모두 가능하나, 여기선 우측 PAD)
    seq_ids_padded = pad_sequence(seq_ids, batch_first=True, padding_value=PAD_ID)          # [B, L_max]
    structure_pad  = pad_sequence(structure, batch_first=True, padding_value=MASK_ID)       # [B, L_max]

    # 5) 어텐션 마스크 (PAD=False)
    attn_mask = (seq_ids_padded != PAD_ID)                                                  # [B, L_max]

    return seq_ids_padded, attn_mask, structure_pad, true_lens