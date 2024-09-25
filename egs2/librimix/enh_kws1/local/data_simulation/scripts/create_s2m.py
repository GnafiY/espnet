import os
import numpy as np
import argparse
import torch
import torchaudio

from pathlib import Path
from loguru import logger
from typing import List
from tqdm import tqdm

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_num', type=int, required=True, default=100, 
                        help="Num of mixed wav file(s)")
    # attention that target_num may be greater than len(os.listdir(librispeech_dir))
    parser.add_argument('--spk1_scp', type=str, required=True, 
                        help="Directory to scp of spk1.")
    parser.add_argument('--spk2_scp', type=str, required=True, 
                        help="Directory to scp of spk2.")
    parser.add_argument('--noise_scp', type=str, required=True, 
                        help="Directory to scp of noise.")
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help="root path of output csv file")
    parser.add_argument('--snr', type=float, required=True,
                        help="SNR of spk1")
    parser.add_argument('--modes', type=str, required=True, default="max min",
                        help="mix mode")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed.")

    return parser

def init_seed(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_scp(scp_dir:str) -> list:
    lines = open(scp_dir).readlines()
    return [line.split() for line in lines]

def get_mix_list(spk1_scp_content:list, 
                 spk2_scp_content:list,
                 noise_scp_content:list,
                 target_num:int,
                 ) -> list:

    logger.info("Getting mix index list.")
    mix_list = []
    produced = set()
    while len(mix_list) < target_num:
        spk1_idx = np.random.randint(0, len(spk1_scp_content))
        spk2_idx = np.random.randint(0, len(spk2_scp_content))
        noise_idx = np.random.randint(0, len(noise_scp_content))

        # uid of spk1, spk2 utterances should be different
        while spk2_scp_content[spk2_idx] == spk1_scp_content[spk1_idx]:
            spk1_idx = np.random.randint(0, len(spk1_scp_content))
            spk2_idx = np.random.randint(0, len(spk2_scp_content))
    
        mix_idx = [spk1_idx, spk2_idx, noise_idx]
        uid = "{}_{}".format(
            spk1_scp_content[spk1_idx][0],
            spk2_scp_content[spk2_idx][0],
        )
        if uid in produced:
            continue
        else:
            mix_list.append(mix_idx)
            produced.add(uid)
    logger.info("Mix index list collected.")
    return mix_list

def process_wav_length(spk1_wav: torch.Tensor, 
                       spk2_wav: torch.Tensor,
                       noise_wav: torch.Tensor,
                       mode):
    # max -> fill
    if mode == 'max':
        ret_wav_length = max(spk1_wav.shape[1], spk2_wav.shape[1], noise_wav.shape[1])
        pad_spk1_wav = torch.cat((spk1_wav, torch.zeros(1, ret_wav_length - spk1_wav.shape[1])), 1)
        pad_spk2_wav = torch.cat((spk2_wav, torch.zeros(1, ret_wav_length - spk2_wav.shape[1])), 1)
        pad_noise_wav = torch.cat((noise_wav, torch.zeros(1, ret_wav_length - noise_wav.shape[1])), 1)
        return pad_spk1_wav, pad_spk2_wav, pad_noise_wav
    # max -> fill
    elif mode == 'min':
        ret_wav_length = min(spk1_wav.shape[1], spk2_wav.shape[1], noise_wav.shape[1])
        return spk1_wav[:, :ret_wav_length], spk2_wav[:, :ret_wav_length], noise_wav[:, :ret_wav_length]
    else:
        logger.error(f"Unexpected mixture mode `{mode}`!")
        import ipdb; ipdb.set_trace()

def main(args):
    logger.add(Path(args.output_base_dir) / "speechmix.log")
    logger.warning(args)

    init_seed(args.seed)

    spk1_scp_content = read_scp(args.spk1_scp)
    spk2_scp_content = read_scp(args.spk2_scp)
    noise_scp_content = read_scp(args.noise_scp)

    mix_list = get_mix_list(
        spk1_scp_content=spk1_scp_content,
        spk2_scp_content=spk2_scp_content,
        noise_scp_content=noise_scp_content,
        target_num=args.target_num,
    )
    
    create_speechmix(
        spk1_scp_content=spk1_scp_content,
        spk2_scp_content=spk2_scp_content,
        noise_scp_content=noise_scp_content,
        mix_list=mix_list,
        output_base_dir=args.output_base_dir,
        snr=float(args.snr),
        modes=args.modes.split(),
    )

def create_speechmix(spk1_scp_content:list,
                     spk2_scp_content:list,
                     noise_scp_content:list,
                     mix_list:list,
                     output_base_dir:str,
                     snr: float,
                     modes: List[str],
    ):
    output_dir = Path(output_base_dir)
    for mode in modes:
        logger.info(f"Creating wav file at SNR `{snr}` with `{mode}` mode for `{str(output_dir).split('/')[-1]}`.")
        output_base_dir = output_dir / "wav16k" / mode
        for folder in ["mixed", "s1", "s2", "noise", "scps"]:
            if not os.path.exists(output_base_dir / folder):
                (output_base_dir / folder).mkdir(parents=True)

        scp_lines = []

        for mix_idx in tqdm(mix_list):
            spk1_idx, spk2_idx, noise_idx = mix_idx

            spk1_wav, _ = torchaudio.load(spk1_scp_content[spk1_idx][1])
            spk2_wav, _ = torchaudio.load(spk2_scp_content[spk2_idx][1])
            noise_wav, _ = torchaudio.load(noise_scp_content[noise_idx][1])

            if noise_wav.shape[0] >= 2:
                noise_wav = noise_wav[0].unsqueeze(0)

            spk1_wav, spk2_wav, noise_wav = process_wav_length(spk1_wav=spk1_wav,
                                                        spk2_wav=spk2_wav,
                                                        noise_wav=noise_wav,
                                                        mode=mode)
            # ls_alpha = calculate_librispeech_gain(ls_wav, ns_wav, 0)
            # hs_alpha = calculate_librispeech_gain(hs_wav, ns_wav + ls_alpha * ls_wav, snr)
            spk2_alpha = calculate_librispeech_gain(spk2_wav, noise_wav, 0)
            spk1_alpha = calculate_librispeech_gain(spk1_wav, noise_wav + spk2_alpha * spk2_wav, snr)

            mixed_file_id = "{}_{}".format(
                spk1_scp_content[spk1_idx][0],
                spk2_scp_content[spk2_idx][0],
            )
            scp_lines.append(mixed_file_id)

            mixed_wav = spk1_alpha * spk1_wav + spk2_alpha * spk2_wav + noise_wav

            torchaudio.save(output_base_dir / "mixed" / f"{mixed_file_id}.wav", mixed_wav, RATE)
            torchaudio.save(output_base_dir / "s1" / f"{mixed_file_id}.wav", spk1_alpha * spk1_wav, RATE)
            torchaudio.save(output_base_dir / "s2" / f"{mixed_file_id}.wav", spk2_alpha * spk2_wav, RATE)
            torchaudio.save(output_base_dir / "noise" / f"{mixed_file_id}.wav", noise_wav, RATE)
        
            with open(output_base_dir / "scps" / "spk1.scp", 'w') as spk1_scp:
                with open(output_base_dir / "scps" / "spk2.scp", 'w') as spk2_scp:
                    with open(output_base_dir / "scps" / "wav.scp", 'w') as wav_scp:
                        with open(output_base_dir / "scps" / "noise.scp", 'w') as noise_scp:
                            for scp_line in scp_lines:
                                wav_scp.write(f"{scp_line} {output_base_dir}/mixed/{scp_line}.wav\n")
                                spk1_scp.write(f"{scp_line} {output_base_dir}/s1/{scp_line}.wav\n")
                                spk2_scp.write(f"{scp_line} {output_base_dir}/s2/{scp_line}.wav\n")
                                noise_scp.write(f"{scp_line} {output_base_dir}/noise/{scp_line}.wav\n")

def si_snr_xy(x, y, eps=1e-8):
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    y_zm = y - torch.mean(y, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * y_zm, dim=-1,
        keepdim=True) * y_zm / (l2norm(y_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def snr_xy(x: np.array, y: np.array):
    return 10 * np.log10(np.mean(x ** 2) / (np.mean(y ** 2) + EPS) + EPS)

def calculate_librispeech_gain(clean: torch.Tensor, noise: torch.Tensor, t):
    l, r = EPS, 10000
    # wav, ns_wav = wav.numpy(), ns_wav.numpy()
    def f(alpha):
        mixed_wav = noise + alpha * clean
        return si_snr_xy(alpha * clean, mixed_wav) - t
    while r - l >= EPS:
        mid = (l + r) / 2
        if f(mid) * f(l) > 0:
            l = mid
        else:
            r = mid
    return l

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
