#!/bin/bash

ShellScriptFolder=$(cd $(dirname "$0"); pwd)
cd $ShellScriptFolder/../../..

set -x

# 读取txt文件中的每一行
while IFS= read -r scene_id
do
  # 执行roslaunch命令
  roslaunch active_inr_s habitat.launch config:=$3 scene_id:=$scene_id hide_planner_windows:=1 hide_mapper_windows:=1 step_num:=$2
done < "$1"