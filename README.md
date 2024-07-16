# pyvideoloader

Simple Impl to accel video data loader 

> My own experiment to identify the best way for efficient video data loader in PyTorch
> WIP

## Methods

This mainly focus on decoding compressed video data (e.g. mp4)

1. Native CPU based dataloader (with pyav)
2. GPU accelerated dataloader (with pyav)
3. VQGAN based dataloader (with pytorch)
    - Use [Open-MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) with provided pytorch checkpoint

## Installation 

Installation of pyav with gpu acceleration is a bit tricky. Here is the steps to install it:

https://www.cyberciti.biz/faq/how-to-install-ffmpeg-with-nvidia-gpu-acceleration-on-linux/

NOTE: we need to switch branch to `release/6.1` to have the correct version of ffmpeg, to be compatible with pyav compilation

## Troubleshooting

1. When api error when installing pyav with custom ffmpeg via: `pip install av --no-binary av`

default ffmpeg is version 7. make ffmpeg with release/6.1

Rememeber to always clear pip cache `pip cache purge` and `pip uninstall av`. just to have a clean installation

2. error on static ffmpeg
```bash
(lerobot) youliang@youliang-All-Series:~/nvidia/ffmpeg$ pip install av --no-binary av --verbose
Using pip 24.0 from /home/youliang/anaconda3/envs/lerobot/lib/python3.10/site-packages/pip (python 3.10)
Collecting av
  Using cached av-12.2.0.tar.gz (3.8 MB)
  Running command pip subprocess to install build dependencies
  Collecting setuptools
    Using cached setuptools-70.3.0-py3-none-any.whl.metadata (5.8 kB)
  Collecting cython
    Using cached Cython-3.0.10-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
  Using cached setuptools-70.3.0-py3-none-any.whl (931 kB)
  Using cached Cython-3.0.10-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
  Installing collected packages: setuptools, cython
  Successfully installed cython-3.0.10 setuptools-70.3.0
  Installing build dependencies ... done
  Running command Getting requirements to build wheel
  pkg-config returned flags we don't understand: -pthread -pthread -pthread
  Building PyAV against static FFmpeg libraries is not supported.
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
```

Need to explicitly specify `--disable-static` in ./configure during ffmpeg

```bash
./configure --enable-shared --disable-static --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --enable-libx265 --enable-gpl
```
