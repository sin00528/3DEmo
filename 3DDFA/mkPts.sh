#!/bin/bash
python main.py -f images/*.png \
-m gpu \
--dump_pts true \
--dump_res false \
--dump_depth false \
--dump_pncc false \
--dump_pose false \
--dump_obj false \
--dump_ply false
