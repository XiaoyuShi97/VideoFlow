PYTHONPATH=/mnt/cache/shixiaoyu1/.local/lib/python3.6/site-packages
export CXX=/mnt/lustre/share/gcc/gcc-5.4/bin/g++
export CC=/mnt/lustre/share/gcc/gcc-5.4/bin/gcc
export CUDA_HOME=/mnt/lustre/share/cuda-11.2
srun  --cpus-per-task=5 --ntasks-per-node=1 -p ISPCodec -n1 --gres=gpu:1 python setup.py install --user
