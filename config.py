import os



data_dir = os.path.join(os.path.curdir, 'data')
lbl_data_dir = os.path.join(data_dir, 'bijankhan_lbl_small')
log_dir = os.path.join(os.path.curdir, 'logs')
save_dir = os.path.join(os.path.curdir, 'save')
save_models_dir = os.path.join(save_dir, 'models')

lexicon_file = os.path.join(save_dir, 'lexicon.npy')
saved_data_file = os.path.join(save_dir, 'data.npy')
w2v_model_file = os.path.join(save_dir, 'word2vec.model')

