from torch.utils.tensorboard import SummaryWriter

import argparse

argparser = argparse.ArgumentParser(description="Add notes to a file.")
argparser.add_argument('--run_dir', type=str, required=True, help='Directory of the run to add a text note to.')
argparser.add_argument('--note', type=str, required=True, help='Text note to add to the run.')

args = argparser.parse_args()

writer = SummaryWriter(args.run_dir)

writer.add_text('notes', args.note, 0)
writer.close()
print(f"Note '{args.note}' added to {args.run_dir}")
