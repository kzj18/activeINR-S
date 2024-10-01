#!/usr/bin/env python
import os
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
import sys
sys.path.append(WORKSPACE)

from scripts.configer import get_repo_status

if __name__ == '__main__':
    commit_id, repo_dirty_flag = get_repo_status()
    assert not repo_dirty_flag, f'Please commit your changes before running this script. (commit_id: {commit_id})'
    
    datasets_config = [
        ('gibson.json', 'gibson_small.txt', 1000),
        ('gibson.json', 'gibson_big.txt', 2000),
        ('mp3d.json', 'mp3d_small.txt', 1000),
        ('mp3d.json', 'mp3d_big.txt', 2000),
        ('mp3d.json', 'mp3d_huge.txt', 10000)]
    
    for config_file_name, scenes_file_name, step_num in datasets_config:
        config_file_url = os.path.join(WORKSPACE, 'config', 'datasets', config_file_name)
        with open(os.path.join(WORKSPACE, 'scripts', 'entry_points', 'batch', scenes_file_name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            scene_id = line.strip()
            os.system(f'roslaunch active_inr_s habitat.launch config:={config_file_url} scene_id:={scene_id} hide_planner_windows:=1 hide_mapper_windows:=1 step_num:={step_num}')