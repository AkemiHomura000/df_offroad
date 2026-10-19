# source /home/rm/Desktop/df_offroad/ddv2_env.sh
source /home/rm/miniconda3/etc/profile.d/conda.sh
conda activate ddv2
export DDV2_ROOT=/home/rm/Desktop/df_offroad/DiffusionDriveV2
export PYTHONPATH=$DDV2_ROOT${PYTHONPATH:+:$PYTHONPATH}
cd "$DDV2_ROOT"
