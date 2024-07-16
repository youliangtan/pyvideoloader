import av
import time
from more_itertools import one
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from tqdm import tqdm
import glob
import argparse

from utils import Timer
from typing import List, Optional


############################################################################################


class VideoReader:
    """
    Video reader class that reads frames from a video file

    Can read or refer to for further improvements:
     - https://github.com/pytorch/vision/blob/main/torchvision/io/video_reader.py
     - https://github.com/dmlc/decord
     - https://github.com/pytorch/vision/issues/5720
    """
    def __init__(self, filename: str, use_gpu=False, multi_threaded=True):
        self.container = av.open(filename)
        # according to https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/video_container.py
        if multi_threaded:
            self.container.streams.video[0].thread_type = "AUTO"
        self.stream = one(self.container.streams.video)
        # NOTE: HEVC_CUVID is an Nvidia CUVID HEVC decoder codec.
        # CUVID is a hardware-acc encoding and decoding API used by NVIDIA.
        self.ctx_gpu = None
        if use_gpu:
            self.ctx_gpu = av.Codec('hevc_cuvid', 'r').create() # or h264_cuvid, or av1_cuvid
            self.ctx_gpu.extradata = self.stream.codec_context.extradata
        self._height = self.stream.height
        self._width = self.stream.width
        self._channels = 3
        self._frames = torch.zeros((5, self._channels, self._height, self._width))

    def read_frame(self, idx: int) -> av.VideoFrame:
        """Get the frame at the given index"""
        target_pts = idx * int(self.stream.duration / self.stream.frames)
        self.container.seek(target_pts, backward=True, any_frame=False, stream=self.stream)
        if self.ctx_gpu:
            for packet in self.container.demux(self.stream):
                for frame in self.ctx_gpu.decode(packet):
                    if frame.pts == target_pts:
                        # this will release memory after each readframe, but makes it slow =(
                        # self.ctx_gpu.close()
                        # NOTE: directly output tensor to avoid memory copy
                        frame_tensor = self._to_tensor(frame)
                        return frame_tensor
        else:
            for frame in self.container.decode(video=0):
                if frame.pts == target_pts:
                    # return frame.to_image()
                    return self._to_tensor(frame)
        raise ValueError(f'Could not find frame with pts {target_pts}')

    def read_batch_frames(self, start_idx: int, target_size: int) -> torch.Tensor:
        """
        Get a batch of frames starting from start_idx
        """
        # prevent re-allocating memory every time
        if len(self._frames) != target_size:
            self._frames = torch.zeros((target_size, self._channels, self._height, self._width))

        target_pts = start_idx * int(self.stream.duration / self.stream.frames)
        self.container.seek(target_pts, backward=True, any_frame=False, stream=self.stream)
        frame_count = 0
        if self.ctx_gpu:
            for packet in self.container.demux(self.stream):
                for frame in self.ctx_gpu.decode(packet):
                    if frame.pts >= target_pts:
                        tensor_frame = self._to_tensor(frame)
                        self._frames[frame_count] = tensor_frame
                        frame_count += 1
                    if frame_count == target_size:
                        return self._frames

        for frame in self.container.decode(video=0):
            if frame.pts >= target_pts:
                tensor_frame = self._to_tensor(frame)
                self._frames[frame_count] = tensor_frame
                frame_count += 1
            if frame_count == target_size:
                return self._frames
        raise ValueError(f'Could not find frame with pts {target_pts}')

    def _to_tensor(self, frame: av.VideoFrame) -> torch.Tensor:
        """convert frame to tensor and move to GPU"""
        # use as_tensor to avoid memory copy https://github.com/pytorch/vision/issues/8172
        return torch.as_tensor(frame.to_rgb().to_ndarray(), dtype=torch.float).permute(2, 0, 1)
        # return torch.as_tensor(frame.to_rgb().to_ndarray()).float().permute(2, 0, 1).to(TORCH_DEVICE)

############################################################################################


class VideoDataset(Dataset):
    def __init__(self,
                 video_files: List[str],
                 transform: transforms.Compose,
                 use_gpu: bool = False,
                 prefetch_size: Optional[int] = None,
                 torch_device: str = 'cuda'
                 ):
        """
        Args:
            video_files: list of video file paths
            transform: torchvision.transforms.Compose
            use_gpu: whether to use GPU for video decoding
            prefetch_size: whether to batch frames together, Optional[int]
            torch_device: torch device to use after reading frames
        """
        self.video_files = video_files
        self.transform = transform
        self.use_gpu = use_gpu
        self.torch_device = torch_device
        self._debug_timer = Timer()
        self._count = 0

        self.prefetch_size = prefetch_size  # whether to batch frames together
        if self.prefetch_size:
            print(f'Prefetch by batching frames during reading with size {self.prefetch_size}')
            # create a map which maps video file to current frame index
            self.video_frame_indices = {video: 0 for video in video_files}
            self.video_prefetched_frames = {video: None for video in video_files}

        # a map of video_file to video reader
        self.video_readers = {}

        # get a map of video file to frame count
        self.video_frame_counts = {
            video: self.get_num_frames(video) for video in video_files
        }

        # generate a prefix sum of frame counts for faster lookup of which video to use
        self.prefix_sum = [0]  # sequence is based on self.video_files
        for video in video_files:
            self.prefix_sum.append(self.prefix_sum[-1] + self.video_frame_counts[video])

        # get frame count
        self.total_frames = self.prefix_sum[-1]
        print(f'Loaded {len(video_files)} videos with a total of {self.total_frames} frames')

    @staticmethod
    def get_num_frames(video_file: str) -> int:
        container = av.open(video_file)
        num = container.streams.video[0].frames
        # close this and have each worker re-open handle to container in its own process
        container.close()
        return max(num - 10, 0)  # Ignore last 10 frames, according to original code

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx: int):
        # new __getitem__ to handle multiple video files
        # Determine which video file and frame index to read
        target_vid_idx = 0  # Target
        while self.prefix_sum[target_vid_idx] <= idx:
            target_vid_idx += 1
        target_vid_idx -= 1
        target_frame_idx = idx - self.prefix_sum[target_vid_idx]  # Target
        video_file = self.video_files[target_vid_idx]

        # get the frame
        with self._debug_timer('get_frame'):
            image = self.get_frame(target_frame_idx, video_file)
        with self._debug_timer('to_device'):
            image = image.to(self.torch_device)
        # apply transform
        with self._debug_timer('transform'):
            image = self.transform(image)

        # for debuging
        # if self._count % 200 == 0:
        #     print(self._debug_timer.get_average_times())
        #     self._count = 0
        # self._count += 1
        return image

    def get_frame(self, idx: int, video_file: str):
        """
        This reads the frame of a video_file using the VideoReader class
        Args:
            idx: frame index
            video_file: video file path
        """
        if video_file not in self.video_readers:
            self.video_readers[video_file] = VideoReader(video_file, self.use_gpu)

        if not self.prefetch_size or self.prefetch_size <= 1:
            return self.video_readers[video_file].read_frame(idx)
        else:
            # NOTE: fancy prefetching
            curr_frame_idx = self.video_frame_indices[video_file]
            if curr_frame_idx == 0:
                # get batch of frames
                self.video_prefetched_frames[video_file] = \
                    self.video_readers[video_file].read_batch_frames(
                        idx, self.prefetch_size
                )
            # return frame of the current index
            frame = self.video_prefetched_frames[video_file][curr_frame_idx]
            # update frame index
            new_curr_frame_idx = (curr_frame_idx + 1) % self.prefetch_size
            self.video_frame_indices[video_file] = new_curr_frame_idx
            return frame

############################################################################################


def run_dataloader():
    parser = argparse.ArgumentParser(description='Load video files and run dataloader')
    parser.add_argument('--cpu_decoder', action='store_true', help='Use CPU for video decoding')
    parser.add_argument('--cpu_model', action='store_true', help='Use CPU for torch model')
    parser.add_argument('--run_model', action='store_true', help='Run a model on the frames')
    parser.add_argument('--videos_dir', type=str, default='videos', help='Directory containing video files')
    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), antialias=True),
        # transforms.ToTensor(),  # CHW, float NOTE: converted to tensor, thus not required
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    ])

    # Keep configs here for now, can move to argparse
    TORCH_DEVICE = 'cuda:1'
    PREFETCH_SIZE = 5
    NUM_WORKERS = 4
    VIDEO_FILES = glob.glob(f'{args.videos_dir}/*.mp4')
    # VIDEO_FILES = ['video.mp4', 'video2.mp4']
    print("loading videos: ", VIDEO_FILES)

    ds = VideoDataset(
        VIDEO_FILES,
        transform=train_transform,
        use_gpu=not args.cpu_decoder,
        prefetch_size=PREFETCH_SIZE,
        torch_device=TORCH_DEVICE if not args.cpu_model else 'cpu',
    )
    print("ds length: ", len(ds))

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_dataloader = DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
        num_workers=NUM_WORKERS,  # default in code 10
        prefetch_factor=2,  # default in pytorch with 2
    )
    print("done creating dataloader")

    device = TORCH_DEVICE if not args.cpu_model else 'cpu'
    run_model = args.run_model

    if run_model:
        net = nn.Conv2d(3, 64, 3, 2)

    it = 0
    is_init = False
    start = time.time()

    for imgs in tqdm(train_dataloader):
        # NOTE: this is to move the model to gpu after the dataloader has been created,
        # to avoid the "RuntimeError: Cannot re-initialize CUDA" error from pyav
        # when using GPU decoding
        if run_model and it == 0 and not is_init:
            is_init = True
            net.to(device)

        if is_init:
            with torch.no_grad():
                net(imgs)
        it += 1

    print(f'it/sec: {it/(time.time() - start)}')

############################################################################################


def test_video_reader():
    """NOTE: Testing code for VideoReader class"""
    video_file = 'video.mp4'
    reader = VideoReader(video_file, use_gpu=True)
    print(f'Video {video_file} has {reader.stream.frames} frames')
    for i in range(5):
        timer = Timer()
        for i in range(50):
            timer.tick('read_frame')
            frame = reader.read_frame(i)
            timer.tock('read_frame')

            timer.tick('read_multi_frames')
            frame = reader.read_batch_frames(i, 5)
            timer.tock('read_multi_frames')

        print(timer.get_average_times())
    print('Done')


def test_torch_streamer():
    """NOTE: Testing code, yet still consumes high gpu ram during decoding"""
    # https://pytorch.org/audio/stable/tutorials/nvdec_tutorial.html
    # https://pytorch.org/audio/main/generated/torio.io.StreamingMediaDecoder.html#torio.io.StreamingMediaDecoder
    from torchaudio.io import StreamReader

    def get_frame(idx, video_file='video.mp4', device='cuda:1'):
        """Get frame at index idx from a video file"""
        s = StreamReader(video_file)
        num_frames = s.get_src_stream_info(0).num_frames
        frame_rate = s.get_src_stream_info(0).frame_rate
        timestamp = frame_rate*idx/num_frames
        s.seek(timestamp)
        s.add_video_stream(5, decoder="hevc_cuvid", hw_accel=device)
        s.fill_buffer()
        (video,) = s.pop_chunks()
        return video

    # get total frames
    for i in range(100):
        video = get_frame(i)
        print(video.shape, video.dtype, video.device)
    return video


if __name__ == "__main__":
    # test_video_reader()
    # test_torch_streamer()
    run_dataloader()
