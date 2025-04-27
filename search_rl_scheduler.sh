PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 \
python search_ea.py \
--outdir 'outputs/65x480p_step25_search50_cache' \
--n_samples 6 \
--num_sample 1000 \
--time_step 25 \
--max_epochs 30 \
--population_num 50 \
--mutation_num 20 \
--crossover_num 15 \
--seed 1024 \
--use_ddim_init_x false \
--frame_num 65 \
--ref_videos 'assets/65x480p' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \
--task 't2v-1.3B' \
--size '832*480' \
--ckpt_dir '/data/models/wan2.1/Wan2.1-T2V-1.3B' \
--sample_shift 8 \
--sample_guide_scale 6 \
--sample_steps 50