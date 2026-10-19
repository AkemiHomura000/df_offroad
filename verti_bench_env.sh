# source this: source /home/rm/Desktop/df_offroad/verti_bench_env.sh
source /home/rm/miniconda3/etc/profile.d/conda.sh
conda activate chrono9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CHRONO_DATA_DIR=/home/rm/Desktop/df_offroad/verti_bench/envs/data/
export PYTHONPATH=$PYTHONPATH:/home/rm/Desktop/df_offroad
# ROS2 Humble if present
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
