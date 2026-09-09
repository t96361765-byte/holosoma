# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplh, mocap, lafan) and task types including robot-only motion, object interaction, and climbing.

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```
**Note**: Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Data Preparation

We provide `demo_data/` for fast testing. To test on more motion sequences, please follow the instructions below to download and prepare the data.

### OMOMO

Our pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `demo_data/OMOMO_new`

The data should contain `.pt` files.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `demo_data/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan --output_dir ../demo_data/lafan
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. The SMPL-X models structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
# Install dependencies
cd holosoma_retargeting/data_utils/
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
cd ../

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

This converts each AMASS sequence into an `.npz` containing 22 global body joint positions and subject height.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path demo_data/amass_smplx_processed --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/amass_smplx_processed --task-type robot_only --data_format smplx --save_dir demo_results_parallel/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

```bash
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/amass_smplx/HumanEva_S1_Box_1_stageii_original.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above (Single Sequence Motion Retargeting or Batch Processing for Motion Retargeting).

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training. The conversion script takes the retargeted `.npz` files as input and outputs converted files with the specified frame rate and format.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)



## AMASS G1 *OmniRetarget*

### Prepare AMASS SMPL-X for OmniRetarget

Input: AMASS-120Hz, Output: KeyPoints-30Hz

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\data_utils
& D:\anaconda3\envs\omniretarget\python.exe `
  .\prep_amass_smplx_for_rt.py `
  --input-file "D:\track_dataset\CMU_dataset\85\85_10_stageii.npz" `
  --output-folder "D:\track_dataset\omniretarget_processed"
```

### AMASS CMU retargeting

Remember to choose differents kinds of constraints:

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\examples\robot_retarget.py `
  --data-path "D:\track_dataset\omniretarget_processed" `
  --task-type robot_only `
  --task-name "CMU_dataset_90_90_29_stageii" `
  --data-format smplx `
  --robot g1 `
  --robot-config.robot-urdf-file "models/g1_29dof_fist_pan/g1_29dof_fist_pan.urdf" `
  --save-dir "D:\track_dataset\omniretarget_results" `
  --task-config.ground-range -1.8 1.8 `
  --task-config.ground-size 23 `
  --retargeter.hand-sticking.enable `
  --retargeter.arm-straightness.enable `
  --retargeter.arm-straightness.weight 2.5
```

### Play AMASS G1 with Viser

CMU-85: 01/flip 02/flip 05/handstand 06/webster 08/thomas 13/handstand 14/dance(Hard)
CMU-88: 06/flykick 08/handstand 09/handstand
CMU-90: 06/flykick 08/cartwheel 19/flip 29/dance(Hard) 30/Russian

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\viser_player.py `
  --robot-urdf ".\models\g1_29dof_fist_pan\g1_29dof_fist_pan.urdf" `
  --qpos-npz "D:\track_dataset\omniretarget_results\CMU_dataset_90_90_29_stageii.npz" `
  --no-assume-object-in-qpos `
  --loop
```

## NOKOV G1-Mushroom *OmniRetarget*

### Side/Front support retargeting

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\examples\robot_retarget.py `
  --data-path "D:\track_dataset\nokov_bvh\0822cc_lt_2" `
  --task-type climbing `
  --task-name "0822cc_lt_2" `
  --data-format nokov `
  --robot g1 `
  --robot-config.robot-urdf-file "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_29dof_fist_pan\g1_29dof_fist_pan.urdf" `
  --task-config.object-name mushroom `
  --task-config.object-scale 1.0 `
  --save-dir "D:\track_dataset\omniretarget_results"
```

### Play NOKOV G1 and mushroom with Viser

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\viser_player.py `
  --qpos-npz "D:\track_dataset\omniretarget_results\0822cc_lt_2_original.npz" `
  --robot-urdf "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_29dof_fist_pan\g1_29dof_fist_pan.urdf" `
  --show-object `
  --loop
```

## NOKOV G1 *OmniRetarget*

### Side/Front support retargeting

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\examples\robot_retarget.py `
  --data-path "D:\track_dataset\nokov_bvh\260630_lt" `
  --task-type robot_only `
  --task-name "lt_cc3" `
  --data-format nokov `
  --robot g1 `
  --robot-config.robot-urdf-file "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_29dof_fist_pan\g1_29dof_fist_pan.urdf" `
  --save-dir "D:\track_dataset\omniretarget_results" `
  --task-config.ground-range -1.5 1.5 `
  --task-config.ground-size 20 `
  --retargeter.hand-sticking.enable `
  --retargeter.hand-sticking.demo-joint-names LeftHand RightHand `
  --retargeter.arm-straightness.enable `
  --retargeter.arm-straightness.weight 10
```

### Play NOKOV G1 with Viser

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& D:\anaconda3\envs\omniretarget\python.exe `
  .\viser_player.py `
  --qpos-npz "D:\track_dataset\omniretarget_results\0822cc_lt_2_original.npz" `
  --robot-urdf "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_29dof_fist_pan\g1_29dof_fist_pan.urdf" `
  --loop
```

## Mushroom-Flare *OmniRetarget*

### G1 flare retargeting (mushroom)

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& "D:\anaconda3\envs\omniretarget\python.exe" `
  .\examples\robot_retarget.py `
  --data-path "D:\track_dataset\marker53_optimized_ground.bvh" `
  --task-type climbing `
  --task-name "pommel_bvh-omniretarget" `
  --data-format pommel `
  --robot g1 `
  --robot-config.robot-urdf-file "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_theshy\g1_theshy.urdf" `
  --save-dir "D:\track_dataset\omniretarget_results" `
  --task-config.object-name mushroom `
  --task-config.object-mesh "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\mushroom\mushroom_visual.obj" `
  --task-config.pommel-reference-position 0.02315461 -0.64678669 0.24471219 `
  --task-config.pommel-reference-scale 0.85 `
  --task-config.pommel-z-compression 1 `
  --task-config.pommel-human-radial-offset 0.02 `
  --task-config.fixed-object-ground-shape 9 9 `
  --task-config.fixed-object-ground-range -1.2 1.2 `
  --task-config.no-fixed-object-surface-points-enabled `
  --retargeter.activate-obj-non-penetration `
  --retargeter.hand-tracking-point-offset 0 0 0 `
  --retargeter.hand-sticking.no-enable `
  --retargeter.arm-straightness.no-enable
```

### Play G1-mushroom with Viser

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& "D:\anaconda3\envs\omniretarget\python.exe" `
  .\viser_player.py `
  --robot-urdf "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_theshy\g1_theshy.urdf" `
  --qpos-npz "D:\track_dataset\omniretarget_results\mushroom_R1_original.npz" `
  --show-object `
  --loop
```

### R1 flare retargeting (mushroom)

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& "D:\anaconda3\envs\omniretarget\python.exe" `
  .\examples\robot_retarget.py `
  --data-path "D:\track_dataset\marker53_optimized_ground.bvh" `
  --task-type climbing `
  --task-name "mushroom_R1" `
  --data-format pommel `
  --robot r1 `
  --robot-config.robot-urdf-file "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\r1\r1_26dof.urdf" `
  --save-dir "D:\track_dataset\omniretarget_results" `
  --task-config.object-name mushroom `
  --task-config.object-mesh "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\mushroom\mushroom_visual.obj" `
  --task-config.pommel-reference-position 0.02315461 -0.64678669 0.24471219 `
  --task-config.pommel-reference-scale 0.85 `
  --task-config.pommel-z-compression 1 `
  --task-config.pommel-human-radial-offset 0.02 `
  --task-config.fixed-object-ground-shape 9 9 `
  --task-config.fixed-object-ground-range -1.2 1.2 `
  --task-config.fixed-object-surface-points-enabled `
  --retargeter.activate-joint-limits `
  --retargeter.activate-obj-non-penetration `
  --retargeter.hand-tracking-point-offset 0 0 0 `
  --retargeter.hand-sticking.no-enable `
  --retargeter.arm-straightness.no-enable
```

### Play R1-mushroom with Viser

```powershell
cd D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting
& "D:\anaconda3\envs\omniretarget\python.exe" `
  .\viser_player.py `
  --robot-urdf "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\r1\r1_26dof.urdf" `
  --qpos-npz "D:\track_dataset\omniretarget_results\mushroom_R1_original.npz" `
  --show-object `
  --loop
```

## Convert npz to RGMT_50Hz (require Extreme-RGMT virtual environment)

```powershell
cd D:\GitHub\RGMT_1
& D:\anaconda3\envs\env_isaaclab\python.exe `
  .\scripts\convert_omniretarget_npz.py `
  --input "D:\track_dataset\omniretarget_results\pommel_bvh-omniretarget_original.npz" `
  --output "D:\track_dataset\Extreme-RGMT_50Hz\flare_mushroom_0.85_50Hz.npz" `
  --xml "D:\GitHub\holosoma\src\holosoma_retargeting\holosoma_retargeting\models\g1_theshy\g1_theshy.xml" `
  --target-fps 50
```