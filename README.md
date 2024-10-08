# activeINR-S

## Installation

Our environment has been tested on 20.04 (CUDA 11.8).

Install `ROS Noetic` following the [instructions](http://wiki.ros.org/noetic/Installation/Ubuntu).

```bash
sudo apt install -y ros-noetic-rviz-imu-plugin
```

Initialize ROS Workspace.

```bash
mkdir -p ~/Workspace/active_inr_s_ws/src && cd ~/Workspace/active_inr_s_ws/src && catkin_init_workspace
```

Clone the repo and create conda environment

```bash
git clone git@github.com:kzj18/activeINR-S.git ~/Workspace/active_inr_s_ws/src/activeINR-S && cd ~/Workspace/active_inr_s_ws/src/activeINR-S
git submodule update --init --progress

conda env create -f environment.yml
conda activate activeINR_S
```

Install pytorch by following the [instructions](https://pytorch.org/get-started/locally/). For torch 2.0.1 with CUDA version 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Ubuntu 20.04
pip install -r requirements.txt

pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git

pip install git+ssh://git@github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Preparation

### Simulated environment

[Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) need to be installed for simulation. We use v0.2.3 (`git checkout tags/v0.2.3`) for habitat-sim & habitat-lab and install the habitat-sim with the flag `--with-cuda`.

```bash
cd ~/Workspace/active_inr_s_ws/src/activeINR-S/habitat/habitat-lab && git checkout tags/v0.2.3
pip install -e habitat-lab
pip install -e habitat-baselines
cd ~/Workspace/active_inr_s_ws/src/activeINR-S/habitat/habitat-sim && git checkout tags/v0.2.3

git submodule update --init --progress --recursive
python setup.py install --with-cuda
```

## Build activeINR-S

```bash
# For Ubuntu 20.04
cd ~/Workspace/active_inr_s_ws && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
echo "source ~/Workspace/active_inr_s_ws/devel/setup.bash" >> ~/.bashrc
```

## Run activeINR-S

### Config Datasets Path

Copy config template from `config/.templates/user_config.json` to `config/user_config.json` and modify the path to the dataset.

### Single Scene

1. To run our method on the `Denmark` scene of `Gibson` dataset, run the following command.

    ```bash
    roslaunch active_inr_s habitat.launch
    ```

2. To run our method on the `Pablo` scene of `Gibson` dataset, run the following command.

    ```bash
    roslaunch active_inr_s habitat.launch scene_id:=Pablo
    ```

3. To run our method on the `zsNo4HB9uLZ` scene of `MP3D` dataset, run the following command.

    ```bash
    roslaunch active_inr_s habitat.launch config:=config/datasets/mp3d.json
    ```

4. To run our method on the `YmJkqBEsHnH` scene of `MP3D` dataset, run the following command.

    ```bash
    roslaunch active_inr_s habitat.launch config:=config/datasets/mp3d.json scene_id:=YmJkqBEsHnH
    ```

### Run IROS Results

```bash
python scripts/entry_points/batch/iros_run.py
```

### Eval Results

```bash
python scripts/entry_points/batch/eval_results_actions.py
```

## Citation

```
@inproceedings{Kuang2024iros,
  title={Active Neural Mapping at Scale},
  author={Kuang, Zijia and Yan, Zike and Zhao, Hao and Zhou, Guyue and Zha, Hongbin},
  booktitle={IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS)},
  year={2024}
}
```
