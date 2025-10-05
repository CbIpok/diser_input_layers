#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

# Ensure project root on sys.path when executed as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from save_subduction import save_functions


def parse_args():
    p = argparse.ArgumentParser(description='Compute functions.wave from areas and config')
    p.add_argument('-c', '--config', default='data/config.json', help='Path to config.json')
    p.add_argument('-o', '--output', default='data/functions.wave', help='Output file path')
    return p.parse_args()


def main():
    args = parse_args()
    save_functions(args.config, args.output)


if __name__ == '__main__':
    main()

