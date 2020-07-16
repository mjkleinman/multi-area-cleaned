cfg_mk = {
    'diff_area_noise': False,
    'area_noise': 3,

    'linear_mut_info': True,

    'path': '/Users/michael/Documents/GitHub/multi-area/',
    'suffix': '',  # _outputpos
    'use_dale': True,  # only matters for analysis of mi
    'rnn_areas': 3,  # only matters for analysis
    'modelpath': '2020-04-10_cb_simple_3areas',  # _nodale_ff=0p1',  # make sure this matches num_rnn_areas
    'num_units': 300,
    'rnn_datapath': '/Users/michael/Documents/GitHub/multi-area/saved_rnns_server_apr/data/',
    'num_seeds': 8,  # -1 if did not specify

    'gamma_rec': True,  # True if recurrent distribution for Crec is gamma
    'random_mask_wrec': False,  # default is False
    'random_mask_win': False,  # default is False
    'random_mask_wout': False,  # default is False
    'make_positive': True,  # default is true (for training)
    'positive_ic': True,  # default is true
    'make_positive_output': True  # default is true
}
