#!/bin/bash
python main.py -f images/*.png \
-m gpu \
--dump_pts false \
--dump_res false \
--dump_depth false \
--dump_pncc false \
--dump_pose false \
--dump_obj false
