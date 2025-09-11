#!/usr/bin/env python
import argparse
from diser.viz.figio import show_figure_pickle


def parse_args():
    p = argparse.ArgumentParser(description='Open a saved matplotlib figure with labels/annotations')
    p.add_argument('pickle_path', help='Path to .mplfig.pkl file')
    return p.parse_args()


def main():
    args = parse_args()
    show_figure_pickle(args.pickle_path)


if __name__ == '__main__':
    main()

