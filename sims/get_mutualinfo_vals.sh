#!/bin/sh
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_4areas'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_1area'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_2areas'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas'

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_correctdale_ffi=0p01'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_correctdale_ffi=0p02'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_correctdale_ffi=0p05'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_correctdale_ffi=0p1'

python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_fb=0p0'
python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_fb=0p1'

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_ff=0p2'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_ff=0p3'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_ff=0p5'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_ff=1'

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_nodale_ff=0p01' --use_dale False
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_nodale_ff=0p1' --use_dale False
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_nodale_ff=0p5' --use_dale False
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_nodale_ff=1' --use_dale False

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_nodale_ff=0p1' --suffix '_outputpos_lambdaw=1' --use_dale False

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_units=120' --num_units 120
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_units=150' --num_units 150
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_units=210' --num_units 210
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_units=600' --num_units 600

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lambdar=1'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lambdar=1e-1'

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lambdaw=1e-1'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lambdaw=1e-2'

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lr=1e-4'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lr=1e-5'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lr=2.5e-4'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lr=5e-4'
# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --suffix '_lr=7.5e-4'

# python analyze_info_layers.py --modelpath '2020-04-10_cb_simple_4areas' --rnn_areas 4
# python analyze_info_layers.py --modelpath '2020-04-10_cb_simple_3areas' --rnn_areas 3
# python analyze_info_layers.py --modelpath '2020-04-10_cb_simple_2areas' --rnn_areas 2
# python analyze_info_layers.py --modelpath '2020-04-10_cb_simple_1area' --rnn_areas 1

# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lambdaw=1e-1' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lambdaw=1e-2' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lambdar=1' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lambdar=1e-1' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lr=1e-4' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lr=1e-5' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lr=2.5e-4' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lr=5e-4' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'
# python analyze_context_info_layers.py --num_seeds 2 --suffix '_lr=7.5e-4' --rnn_areas 1 --modelpath '2020-04-10_cb_simple_1area'



# STOP HERE

# python analyze_color_info_layers.py --modelpath '2020-04-10_cb_simple_3areas_perf80'

# python analyze_info_layers.py --rnn_datapath '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/' --modelpath '2018-08-29_cb_3areas_ff0p1_fb0p05'
# python analyze_color_info_layers.py --rnn_datapath '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/' --modelpath '2018-08-29_cb_3areas_ff0p2_fb0p05'
# python analyze_color_info_layers.py --rnn_datapath '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/' --modelpath '2018-08-29_cb_3areas_ff0p3_fb0p05'
# python analyze_color_info_layers.py --rnn_datapath '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/' --modelpath '2018-08-29_cb_3areas_ff0p1_fb0p0'
# python analyze_color_info_layers.py --rnn_datapath '/Users/michael/Desktop/tibi_backup/tibi/saved_rnns/three_rnns/' --modelpath '2018-08-29_cb_3areas_ff0p1_fb0p1'

