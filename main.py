import argparse
from unittest.mock import patch
import sys


def main():
    parser = argparse.ArgumentParser(description='Inter-dataset agreement calculator for relations')
    parser.add_argument('gold_directory', help='First data folder path (gold)')
    parser.add_argument('system_a_directory', help='Second data folder path (system_a)')
    parser.add_argument('system_b_directory', help='Third data folder path (system_b)')
    parser.add_argument('-f', '--format', default='plain', help='format to print the table (options include grid, github, and latex)')
    parser.add_argument('-d', '--decimal', type=int, default=3, help='number of decimal places to round to')
    args = parser.parse_args()

    gold_c = args.gold_directory
    num_shuffles = 0
    sys1_c = args.system_a_directory
    sys2_c = args.system_b_directory

    new_args = ['art.py', gold_c, '-n', str(100), '-v', '-r', '-a', sys1_c, sys2_c]

    with patch.object(sys, 'argv', new_args):
        import art


if __name__ == '__main__':
    main()
