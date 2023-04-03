#!/usr/bin/env bash

python surf_baseline_multi_main.py '../data/CASIA-SURF' 'average' 'multi_full_' 3 0 --p [1,1,1]
python surf_baseline_multi_main.py '../data/CASIA-SURF' 'average' 'multi_full_' 3 1 --p [1,1,1]
python surf_baseline_multi_main.py '../data/CASIA-SURF' 'average' 'multi_full_'  3 2 --p [1,1,1]


