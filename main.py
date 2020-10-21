import argparse
from unittest.mock import patch
import sys
from tempfile import TemporaryDirectory
from pathlib import Path


def _patch_argv(args):
    return patch.object(sys, 'argv', args)


def main():
    parser = argparse.ArgumentParser(description='Inter-dataset agreement calculator for relations')
    parser.add_argument('gold_directory', help='First data folder path (gold)')
    parser.add_argument('system_a_directory', help='Second data folder path (system_a)')
    parser.add_argument('system_b_directory', help='Third data folder path (system_b)')
    parser.add_argument('-n', '--num_shuffles', type=int, default=50_000, help='number of shuffles')
    args = parser.parse_args()

    gold_dir = Path(args.gold_directory)
    sys1_dir = Path(args.system_a_directory)
    sys2_dir = Path(args.system_b_directory)

    _dif_dir = TemporaryDirectory()
    dif_dir = Path(_dif_dir.name)

    new_args = ['createSignificanceTestFiles.py', str(gold_dir), str(sys1_dir), str(sys2_dir), str(dif_dir)]
    with _patch_argv(new_args):
        import createSignificanceTestFiles

    print(list(dif_dir.iterdir()))

    # python art.py -c $gold_c -n$numShuffles -v -r -a  sys1_c sys2_c
    new_args = ['art.py', '-c', str(dif_dir / 'gold_c'), '-n', str(args.num_shuffles), '-v', '-r', '-a',
                str(dif_dir / 'sys1_c'), str(dif_dir / 'sys2_c')]
    with _patch_argv(new_args):
        print('importing')
        import art

    print('cleanup')
    _dif_dir.cleanup()


if __name__ == '__main__':
    main()
