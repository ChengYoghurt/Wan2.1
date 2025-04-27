from videosys import OpenSoraPlanConfig, VideoSysEngine
import argparse

# prompts_1s = [
#     "a black dog wearing halloween costume", # animal
#     "an apartment building with balcony", # archi
#     "freshly baked finger looking cookies", # food
#     "people carving a pumpkin", # human
#     # "scenic video of sunset", # scenery
# ]

prompts_4s = [
    "a muffin with a burning candle and a love sign by a ceramic mug", # food
    "a group of friend place doing hand gestures of agreement", # human
    "aerial view of snow piles", # scenery
    "yacht sailing through the ocean", # vehicle
]

def save_ref_video(video, i, prompt):
    import os
    import torch
    # Check if `video` is already a PyTorch tensor
    if not isinstance(video, torch.Tensor):
        # Convert to a PyTorch tensor if it's not already one
        video = torch.tensor(video)
    ref_video_folder = f"examples/open_sora_plan/assets/93x480p"
    # Save the video tensor to a .pt file
    os.makedirs(ref_video_folder, exist_ok=True)
    ref_video_path = os.path.join(ref_video_folder, f"{i}.pt")
    torch.save(video, ref_video_path)

    print(f"Video saved as {ref_video_path}")

def run_base(save_ref_videos=False, load_ea_timesteps=False):
    print(f"Running with save_ref_videos={save_ref_videos}, load_ea_timesteps={load_ea_timesteps}")
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v120", transformer_type="93x480p", num_gpus=1)
    engine = VideoSysEngine(config)

    ea_timesteps_list = []
    prompts = ["a drone flying over a snowy forest.", ]
    # seed=-1 means random seed. >0 means fixed seed.
    #File path
    # prompt_file_path = "/home/yuge/src/Vbench/prompts/all_dimension.txt"
   
    # # Read all prompts
    # with open(prompt_file_path, "r") as f:
    #     prompts = [line.strip() for line in f.readlines()]
    #     # prompts = [line.strip() for i, line in enumerate(f.readlines()) if i % 16 == 0]

    for i, prompt in enumerate(prompts):

        if load_ea_timesteps:
            import yaml
            # Load YAML file
            ea_timesteps_path = "examples/open_sora_plan/outputs/93x480p_step70_search100_cache/ea_timesteps.yaml"
            with open(ea_timesteps_path, "r") as file:
                ea = yaml.safe_load(file)  # Use safe_load to avoid execution risks

            ea_timesteps_list = ea["ea_timesteps_list"]
        
        if ea_timesteps_list:
            from pathlib import Path
            import os
            # Original YAML file path
            ea_path = Path(ea_timesteps_path)

            # Construct new folder path
            videos_folder = ea_path.parent / "videos_test"

            # Create the folder if it doesn't exist
            videos_folder.mkdir(parents=True, exist_ok=True)
            for idx, ea_timesteps in enumerate(ea_timesteps_list):
                video = engine.generate(
                    prompt=prompt,
                    guidance_scale=7.5,
                    num_inference_steps=100,
                    seed=1024,
                    ea_timesteps=ea_timesteps
                ).video[0]
                
                # TODO: modify save name
                # prompt_prefix = prompt[:20]
                video_filename = f"{prompt}.mp4"  # TODO
                video_save_path = os.path.join(videos_folder, video_filename)
                engine.save_video(video, video_save_path)
                print(f"Saved video with EA timesteps to {video_save_path}")
                exit(0)
        else:
            print("Generating videos WITHOUT EA...")
            video = engine.generate(
                prompt=prompt,
                guidance_scale=7.5,
                num_inference_steps=100,
                seed=1024,
            ).video[0]

            if save_ref_videos:
                print(f"Saving reference video {i} for prompt '{prompt}'")
                save_ref_video(video, i, prompt)

            import os
            video_filename = f"{prompt}.mp4"
            videos_folder = f"./outputs/osp/93x480p_ref"
            # prompt_suffix = prompt[:20] if len(prompt) > 20 else prompt
            # engine.save_video(video, f"./outputs/category_93x480p_org/{prompt_suffix}.mp4")
            video_save_path = os.path.join(videos_folder, video_filename)
            engine.save_video(video, video_save_path)
            




if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run base function with customizable parameters.')
    
    # Add arguments with default values matching your original function call
    parser.add_argument('--save_ref_videos', type=bool, default=False,
                       help='Whether to save reference videos (default: False)')
    parser.add_argument('--load_ea_timesteps', type=bool, default=False,
                       help='Whether to load EA timesteps (default: False)')
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    run_base(save_ref_videos=args.save_ref_videos, 
             load_ea_timesteps=args.load_ea_timesteps)


