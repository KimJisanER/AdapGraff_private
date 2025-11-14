# ==================== Standard library ====================
import os
import sys
import math
import gc

# ==================== Scientific stack ====================
import numpy as np
import matplotlib.pyplot as plt

# ==================== tqdm (notebook/terminal 자동 선택) ====================
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ==================== PyTorch ====================
import torch

# ==================== PyTorch Geometric (utilities only) ====================
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    coalesce,
    to_undirected,
)

# ==================== ESM (ESM3 + SDK) ====================
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.structure_tokenizer import StructureTokenizer

# ==================== Biopython (for PDB parsing) ====================
from Bio.PDB import PDBParser

########################################################################################################################

def protein_init(seqs, contact_threshold=8.0, batch_size=2,
                 weight_mode: str = 'inverse', sigma: float | None = None, esm_structure = False):
    """
    weight_mode: 'inverse' | 'gaussian' | 'linear' | 'binary' | 'raw'
      - inverse : 1 / (d + eps)
      - gaussian: exp(-(d / sigma)^2),  sigma 미지정 시 threshold/2
      - linear  : (threshold - d) / threshold, [0,1]로 클램프
      - binary  : 접촉이면 1
      - raw     : 원거리 d 그대로 (권장 X, 특별한 경우만)
    """
    from math import ceil
    result_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: ESM3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    model.eval()

    structure_decoder = model.get_structure_decoder().to(device)
    tokenizer = StructureTokenizer()
    bos_id = tokenizer.vq_vae_special_tokens["BOS"]
    eos_id = tokenizer.vq_vae_special_tokens["EOS"]

    if sigma is None:
        sigma = contact_threshold / 2.0
    eps = 1e-6

    num_batches = ceil(len(seqs) / batch_size)

    for b in tqdm(range(num_batches), desc="Processing Proteins in Batches"):
        batch_seqs = seqs[b * batch_size: (b + 1) * batch_size]

        for seq in batch_seqs:
            try:
                with torch.no_grad():
                    seq_feat = seq_feature(seq)

                    # Step 1: Encode input
                    protein_input = ESMProtein(sequence=seq)
                    protein_tensor = model.encode(protein_input)

                    # Step 2: Logits & Embedding
                    config = LogitsConfig(
                        return_embeddings=True,
                        sequence=True,
                        structure=True,
                        secondary_structure=True,
                        sasa=True,
                        function=True,
                        residue_annotations=True
                    )
                    output = model.logits(protein_tensor, config=config)

                    if output.embeddings is None:
                        print(f"[Warning] No embedding for sequence: {seq[:10]}...")
                        continue

                    # 🔻 Remove BOS/EOS from all residue-level outputs
                    token_repr = output.embeddings.squeeze(0)[1:-1].to(torch.float16)
                    ss = output.logits.secondary_structure.squeeze(0)[1:-1].to(torch.float32)
                    sasa = output.logits.sasa.squeeze(0)[1:-1].to(torch.float32)
                    func_logits = output.logits.function.squeeze(0)[1:-1].to(torch.float32)
                    num_residues = token_repr.shape[0]

                    # Step 3: Structure tokens to coordinates
                    structure_tokens = output.logits.structure.argmax(dim=-1)
                    bos_tensor = torch.full((1, 1), bos_id, dtype=structure_tokens.dtype,
                                            device=structure_tokens.device)
                    eos_tensor = torch.full((1, 1), eos_id, dtype=structure_tokens.dtype,
                                            device=structure_tokens.device)
                    structure_tokens = torch.cat([bos_tensor, structure_tokens, eos_tensor], dim=1)

                    structure_output = structure_decoder.decode(structure_tokens)
                    coords = structure_output["bb_pred"].to(torch.float32)
                    coords = coords[:, 1:-1]  # remove BOS/EOS

                    # Step 4: Contact map
                    contact_map, dist_matrix = get_contact_map_from_coords(
                        coords, atom_index=1, threshold=contact_threshold
                    )
                    contact_map = contact_map[0]
                    dist_matrix = dist_matrix[0]

                    edge_index = (contact_map > 0).nonzero(as_tuple=False).T  # (2, E)

                    # ✅ Filter invalid edges
                    valid_mask = (edge_index[0] < num_residues) & (edge_index[1] < num_residues)
                    edge_index = edge_index[:, valid_mask]

                    if edge_index.numel() == 0:
                        print(f"[Warning] Empty edge_index after filtering for sequence: {seq[:20]}...")
                        continue

                    try:
                        # ---- distance & weight ----
                        edge_length = dist_matrix[edge_index[0], edge_index[1]].to(torch.float32)

                        if weight_mode == 'inverse':
                            edge_weight = 1.0 / (edge_length + eps)
                        elif weight_mode == 'gaussian':
                            edge_weight = torch.exp(- (edge_length / float(sigma)) ** 2)
                        elif weight_mode == 'linear':
                            edge_weight = (float(contact_threshold) - edge_length) / float(contact_threshold)
                            edge_weight = edge_weight.clamp(min=0.0, max=1.0)
                        elif weight_mode == 'binary':
                            edge_weight = torch.ones_like(edge_length)
                        elif weight_mode == 'raw':
                            edge_weight = edge_length.clone()
                        else:
                            raise ValueError(f"Unknown weight_mode: {weight_mode}")

                    except IndexError:
                        print(f"[Error] Indexing dist_matrix failed. Skipping: {seq[:20]}")
                        continue

                    # ---- remove self-loops (동일 마스크 적용) ----
                    noself_mask = edge_index[0] != edge_index[1]
                    edge_index = edge_index[:, noself_mask]
                    edge_length = edge_length[noself_mask]
                    edge_weight = edge_weight[noself_mask]

                    # ---- coalesce: 같은 (i,j) 중복 병합 (edge_length/edge_weight 각각 동일 방식) ----
                    ei0 = edge_index
                    edge_index, edge_length = coalesce(ei0, edge_length, num_nodes=num_residues)
                    _,         edge_weight = coalesce(ei0, edge_weight, num_nodes=num_residues)

                    if edge_index.numel() == 0:
                        print(f"[Warning] Empty edge_index after coalesce for sequence: {seq[:20]}...")
                        continue

                    # Step 5: Save results
                    result_dict[seq] = {
                        'seq': seq,
                        'seq_feat': torch.from_numpy(seq_feat),
                        'token_representation': token_repr,
                        'num_nodes': num_residues,
                        'num_pos': torch.arange(num_residues).reshape(-1, 1),
                        'edge_index': edge_index,
                        'edge_length': edge_length,   # 원 거리(Å)
                        'edge_weight': edge_weight,   # 변환된 가중치(모델 투입용)
                        'secondary_structure': ss,
                        'sasa': sasa,
                        'function_logits': func_logits,
                        'weight_mode': weight_mode,
                        'sigma': float(sigma) if sigma is not None else None,
                        'contact_threshold': float(contact_threshold),
                    }

            except RuntimeError as e:
                print(f"[Error] Failed for sequence: {seq[:10]}... - {str(e)}")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        gc.collect()

    if esm_structure:
        return result_dict, structure_output
    else:
        return result_dict

def get_contact_map_from_coords(coords, atom_index=1, threshold=8.0):
    """
    Compute contact map and distance matrix from predicted coordinates.
    Args:
        coords: Tensor (B, L, 3, 3) predicted structure coordinates.
        atom_index: Index of atom to use (0:N, 1:CA, 2:C).
        threshold: Distance threshold (Å) to define a contact.
    Returns:
        contact_map: Tensor (B, L, L)
        dist_matrix: Tensor (B, L, L)
    """
    atom_coords = coords[:, :, atom_index, :]  # (B, L, 3)
    dist_matrix = torch.cdist(atom_coords, atom_coords)  # (B, L, L)
    contact_map = (dist_matrix < threshold).float()
    return contact_map, dist_matrix

def plot_3d_structure(coords: torch.Tensor, seq: str, save_path: str = None):
    """
    3D 구조를 시각화합니다. (CA 좌표 기준)

    Args:
        coords (torch.Tensor): shape (1, L, 3, 3)
        seq (str): 시퀀스 문자열
        save_path (str, optional): 저장 경로. None이면 화면에 표시
    """
    ca_coords = coords[0, :, 1, :]  # CA atom: (L, 3)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = ca_coords[:, 0].cpu(), ca_coords[:, 1].cpu(), ca_coords[:, 2].cpu()
    ax.plot(x, y, z, c='blue', linewidth=2)
    ax.scatter(x, y, z, c='red', s=10)

    ax.set_title(f"3D Backbone (CA) Structure: {seq[:10]}...")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_distance_matrix(dist_matrix: torch.Tensor, seq: str, save_path: str = None, vmax: float = 30.0):
    """
    거리 행렬 시각화 (0 ~ vmax Å 스케일 고정)

    Args:
        dist_matrix (torch.Tensor): shape (L, L), 실수 거리 행렬
        seq (str): 시퀀스 문자열 (파일명용)
        save_path (str, optional): 저장 경로. None이면 화면에 표시
        vmax (float): 최대 거리 스케일 (default: 30Å)
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(dist_matrix.cpu(), cmap="viridis", interpolation="none", vmin=0, vmax=vmax)
    plt.title("Distance Matrix (Å)")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    cbar = plt.colorbar()
    cbar.set_label("Distance (Å)", rotation=270, labelpad=15)

    if save_path is None:
        save_path = f"distance_matrix_{seq[:10]}.png"

    plt.savefig(save_path, dpi=150)
    plt.close()

########################################################################################################################

# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def seq_feature(pro_seq):    
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U','X').replace('B','X')
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_map_proba[row, col].float()
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1],(torch.arange(num_residues)+1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:],(torch.arange(num_residues)-1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1],dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2],(torch.arange(num_residues)+2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:],(torch.arange(num_residues)-2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) *contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2],dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2],dim=-1)
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight,fill_value=1)
    
    return edge_index, edge_weight



def esm_extract(model, batch_converter, seq, layer=36, approach='mean',dim=2560):
    pro_id = 'A'
    if len(seq) <= 700:
        data = []
        data.append((pro_id, seq))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        logits = results["logits"][0].cpu().numpy()[1: len(seq) + 1]
        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer

        if approach == 'last':
            token_representation = token_representation[-1]
        elif approach == 'sum':
            token_representation = token_representation.sum(dim=0)
        elif approach == 'mean':
            token_representation = token_representation.mean(dim=0)

        token_representation = token_representation.cpu().numpy()
        token_representation = token_representation[1: len(seq) + 1]
    else:
        contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
        token_representation = np.zeros((len(seq), dim))
        logits = np.zeros((len(seq),layer))
        interval = 350
        i = math.ceil(len(seq) / interval)
        # ======================
        # =                    =
        # =                    =
        # =          ======================
        # =          =*********=          =
        # =          =*********=          =
        # ======================          =
        #            =                    =
        #            =                    =
        #            ======================
        # where * is the overlapping area
        # subsection seq contact map prediction
        for s in range(i):
            start = s * interval  # sub seq predict start
            end = min((s + 2) * interval, len(seq))  # sub seq predict end
            sub_seq_len = end - start

            # prediction
            temp_seq = seq[start:end]
            temp_data = []
            temp_data.append((pro_id, temp_seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

            # insert into the global contact map
            row, col = np.where(contact_prob_map[start:end, start:end] != 0)
            row = row + start
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
            
            logits[start:end] += results['logits'][0].cpu().numpy()[1: len(temp_seq) + 1]
            logits[row] = logits[row]/2.0

            ## TOKEN
            subtoken_repr = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == 'last':
                subtoken_repr = subtoken_repr[-1]
            elif approach == 'sum':
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == 'mean':
                subtoken_repr = subtoken_repr.mean(dim=0)

            subtoken_repr = subtoken_repr.cpu().numpy()
            subtoken_repr = subtoken_repr[1: len(temp_seq) + 1]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0

            if end == len(seq):
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map), torch.from_numpy(logits)

from Bio.PDB import PDBParser
biopython_parser = PDBParser()

one_to_three = {"A" : "ALA",
              "C" : "CYS",
              "D" : "ASP",
              "E" : "GLU",
              "F" : "PHE",
              "G" : "GLY",
              "H" : "HIS",
              "I" : "ILE",
              "K" : "LYS",
              "L" : "LEU",
              "M" : "MET",
              "N" : "ASN",
              "P" : "PRO",
              "Q" : "GLN",
              "R" : "ARG",
              "S" : "SER",
              "T" : "THR",
              "V" : "VAL",
              "W" : "TRP",
              "Y" : "TYR",
              "B" : "ASX",
              "Z" : "GLX",
              "X" : "UNK",
              "*" : " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):

    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash X.')
        
    return seq, chain_str