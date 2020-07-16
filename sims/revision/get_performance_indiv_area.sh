#!/bin/sh
python noisy-perturb-rec-indiv-area.py --rnn_areas 4 --diff_area_noise True --area_noise 1 --modelpath 2020-04-10_cb_simple_4areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 4 --diff_area_noise True --area_noise 2 --modelpath 2020-04-10_cb_simple_4areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 4 --diff_area_noise True --area_noise 3 --modelpath 2020-04-10_cb_simple_4areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 4 --diff_area_noise True --area_noise 4 --modelpath 2020-04-10_cb_simple_4areas

python noisy-perturb-rec-indiv-area.py --rnn_areas 3 --diff_area_noise True --area_noise 1 --modelpath 2020-04-10_cb_simple_3areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 3 --diff_area_noise True --area_noise 2 --modelpath 2020-04-10_cb_simple_3areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 3 --diff_area_noise True --area_noise 3 --modelpath 2020-04-10_cb_simple_3areas

python noisy-perturb-rec-indiv-area.py --rnn_areas 2 --diff_area_noise True --area_noise 1 --modelpath 2020-04-10_cb_simple_2areas
python noisy-perturb-rec-indiv-area.py --rnn_areas 2 --diff_area_noise True --area_noise 2 --modelpath 2020-04-10_cb_simple_2areas

python noisy-perturb-rec-indiv-area.py --rnn_areas 1 --diff_area_noise True --area_noise 1 --modelpath 2020-04-10_cb_simple_1area
