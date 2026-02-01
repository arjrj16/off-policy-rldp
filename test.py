import json
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import numpy as np

# Initialize obs modality
obs_modality_dict = {
    "low_dim": ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object'],
}
ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

# Load env config
with open('cfg/robomimic/env_meta/transport.json', 'r') as f:
    env_meta = json.load(f)

# Create env
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)

# Check observation dimensions
obs = env.reset()
for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
            'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object']:
    print(f"{key}: {obs[key].shape}")

total = sum(obs[key].shape[0] for key in obs_modality_dict['low_dim'])
print(f"\nTotal observation dim: {total}")