import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time
import re

import argparse
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
# from scipy import linalg

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
import copy

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

# 4s
PROMPTS = [
    "a muffin with a burning candle and a love sign by a ceramic mug", # food
    "a group of friend place doing hand gestures of agreement", # humantmux
    "aerial view of snow piles", # scenery
    "yacht sailing through the ocean", # vehicle
]

def load_ref_videos(ref_videos_folder):
    ref_videos = []
    for i in range(4):
        video_path = os.path.join(ref_videos_folder, f"{i}.pt")
        video = torch.load(video_path)
        # video_normalized = video.float() / 255.0
        ref_videos.append(video)
    return ref_videos

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

class EvolutionSearcher(object):

    def __init__(self, opt, full_timesteps, time_step, ref_videos, ref_sigma, device, dpm_params=None):
        self.opt = opt
        self.full_timesteps = full_timesteps
        self.time_step = time_step
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.use_ddim_init_x = opt.use_ddim_init_x

        # TODO: Load ref_latent
        self.ref_videos = load_ref_videos(ref_videos_folder=ref_videos) # torch.load(ref_latent)
        self.ref_sigma = None

        self.dpm_params = dpm_params
        self.device = device

        if opt.load_log_path:
            self.load_log(opt.load_log_path)

        # init self.model
        self._init_t2v_model(opt)

    def _init_t2v_model(self, args):
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        _init_logging(rank)

        if args.offload_model is None:
            args.offload_model = False if world_size > 1 else True
            logging.info(
                f"offload_model is not specified, set to {args.offload_model}.")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                args.t5_fsdp or args.dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                args.ulysses_size > 1 or args.ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."

        if args.ulysses_size > 1 or args.ring_size > 1:
            assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (initialize_model_parallel,
                                                init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )

        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]

        if "t2v" in args.task:
            logging.info("Creating WanT2V pipeline.")
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )
            self.model = wan_t2v
    
    def load_log(self, log_file_path):
        # Regex pattern to extract cand and mse
        pattern = r"cand: (\[.*?\]), mse: ([\d.]+)"
        
        with open(log_file_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    cand_str = match.group(1)
                    mse = float(match.group(2))
                    
                    # Add to vis_dict
                    if cand_str not in self.vis_dict:
                        self.vis_dict[cand_str] = {
                            'visited': True,
                            'mse': mse
                        }
        
        # Sort all candidates by MSE and populate keep_top_k
        sorted_candidates = sorted(
            self.vis_dict.items(),
            key=lambda x: x[1]['mse']
        )
        
        # Extract top-k candidates (select_num and 50)
        top_select_num = [cand for cand, _ in sorted_candidates[:self.select_num]]
        top_50 = [cand for cand, _ in sorted_candidates[:50]]
        
        self.keep_top_k[self.select_num] = top_select_num
        self.keep_top_k[50] = top_50

        print("Loaded keep_top_k:", self.keep_top_k)

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand, reverse=True)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['mse'] = self.get_cand_mse(cand=eval(cand))
        logging.info('cand: {}, mse: {}'.format(cand, info['mse']))

        info['visited'] = True
        return True
    
    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand, reverse=True)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['mse'] = self.get_cand_mse(cand=eval(cand))
        logging.info('cand: {}, mse: {}'.format(cand, info['mse']))

        info['visited'] = True
        return True
    
    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            selected = set()  # Track unique selections

            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5 and cand1[i] not in selected:
                    new_cand.append(cand1[i])
                    selected.add(cand1[i])
                elif cand2[i] not in selected:
                    new_cand.append(cand2[i])
                    selected.add(cand2[i])

            # Ensure new_cand has the same length as original sequences
            remaining = [x for x in cand1 + cand2 if x not in selected]
            new_cand.extend(remaining[:len(cand1) - len(new_cand)])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            # for i in range(self.sampler.ddpm_num_timesteps):
            for i in self.full_timesteps: # TODO
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            # for i in range(self.sampler.ddpm_num_timesteps):
            for i in self.full_timesteps:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        # TODO: Swap the init timesteps with rf timesteps
        # original_num_steps = self.sampler.ddpm_num_timesteps
        # use_timestep = [i for i in range(original_num_steps)]
        original_timestep = self.full_timesteps
        random.shuffle(original_timestep)
        use_timestep = original_timestep[:self.time_step] # time_step is set by ea searcher
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    # TODO
    def get_cand_mse(self, cand=None, device='cuda'):
        mse_scores = []
        for i, prompt in enumerate(PROMPTS):
            cand_video = self.model.generate(
            input_prompt=prompt,
            size=SIZE_CONFIGS[self.opt.size],
            frame_num=self.opt.frame_num,
            shift=self.opt.sample_shift,
            sample_solver=self.opt.sample_solver,
            guide_scale=self.opt.sample_guide_scale,
            seed=self.opt.base_seed,
            offload_model=self.opt.offload_model,
            ea_timesteps=cand,
            )
            # cand_video_float = cand_video.float() / 255.0
            ref_video = self.ref_videos[i] # normalized alrd
            mse_loss = F.mse_loss(cand_video, ref_video)
            mse_scores.append(mse_loss.item())

            # ## only for test
            # video_filename = f"{prompt}.mp4"  # Format index as 4 digits (e.g., 0000, 0001, etc.)
            # save_dir = "/home/yuge/src/tmp/wan33"
            # save_path = os.path.join(save_dir, video_filename)
                
            # logging.info(f"Saving generated video[{i}] to {save_path}")
            # cache_video(
            #     tensor=cand_video[None],
            #     save_file=save_path,
            #     fps=16,
            #     nrow=1,
            #     normalize=True,
            #     value_range=(-1, 1))
        
        mean_mse = np.mean(mse_scores)
        print("Mean MSE Loss:", mean_mse)
        return mean_mse

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            raise NotImplementedError("not implemented sampler")

        # TODO: Update the metric evaluation method
        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['mse'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['mse'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} mse = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['mse']))
            
            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            if self.opt.dpm_solver:
                mutation = self.get_mutation_dpm(
                    self.select_num, self.mutation_num, self.m_prob)
            else:
                mutation = self.get_mutation(
                    self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            self.epoch += 1
