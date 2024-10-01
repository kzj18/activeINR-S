#!/bin/bash

ShellScriptFolder=$(cd $(dirname "$0"); pwd)
cd $ShellScriptFolder/../../..
WorkspaceFolder=$(pwd)
echo $WorkspaceFolder

set -x

echo "Testing gibson small"
$ShellScriptFolder/gibson.sh $ShellScriptFolder/gibson_small.txt 1000

echo "Testing gibson big"
$ShellScriptFolder/gibson.sh $ShellScriptFolder/gibson_big.txt 2000

echo "Testing mp3d small"
$ShellScriptFolder/mp3d.sh $ShellScriptFolder/mp3d_small.txt 1000

echo "Testing mp3d big"
$ShellScriptFolder/mp3d.sh $ShellScriptFolder/mp3d_big.txt 2000

echo "Testing mp3d huge"
$ShellScriptFolder/mp3d.sh $ShellScriptFolder/mp3d_huge.txt 10000