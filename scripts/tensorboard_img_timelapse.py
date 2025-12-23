import argparse
from tbparse import SummaryReader
import imageio


argparser = argparse.ArgumentParser()

argparser.add_argument("--log_dir", type=str, required=True, help="TensorBoard log directory")
argparser.add_argument("--tag", type=str, required=True, help="Tag of the images to convert to timelapse")
argparser.add_argument("--output_path", type=str, required=True, help="Output path for the timelapse (gif or mp4)")
argparser.add_argument("--fps", type=int, default=5, help="Frames per second for the output timelapse")

args = argparser.parse_args()


def tensorboard_images_to_gif(log_dir: str, tag: str, output_path: str, fps: int = 10):
    reader = SummaryReader(log_dir)
    
    images_df = reader.images
    tag_images = images_df[images_df['tag'] == tag].sort_values('step')
    
    frames = [row['value'] for _, row in tag_images.iterrows()]
    
    if output_path.endswith('.gif'):
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"Saved {len(frames)} frames to {output_path}")

# Usage
tensorboard_images_to_gif(
    log_dir=args.log_dir,
    tag=args.tag,
    output_path=args.output_path,
    fps=args.fps
)
