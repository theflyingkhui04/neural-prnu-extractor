import argparse
import ffdnet.processors as processors

def get_args():
  r"""Parse command line arguments."""

  parser = argparse.ArgumentParser(
    prog='ffdnet photo response non-uniformity',
    description='FFDNet for image denoising',
  )

  # subparsers
  subparsers = parser.add_subparsers(help='sub-commands help')
  processors.train_ffdnet.configure_subparsers(subparsers)
  processors.test_ffdnet.configure_subparsers(subparsers)
  processors.prnu_ffdnet.configure_subparsers(subparsers)
  processors.prepare_patches.configure_subparsers(subparsers)
  processors.prepare_vision_dataset.configure_subparsers(subparsers)
  processors.prepare_prnu_vision.configure_subparsers(subparsers)

  # parse arguments
  parsed_args = parser.parse_args()

  if 'func' not in parsed_args:
    parser.print_usage()
    parser.exit(1)

  return parsed_args

def main(args):
  r"""Main function."""
  args.func(
    args,
  )

if __name__ == '__main__':
  main(get_args())