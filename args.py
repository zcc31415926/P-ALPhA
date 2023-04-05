# attack
attack_lr = 1e-4
attack_iter = 500
benign_ratio = 0.8
benign_weight = 0.1
clip_grad = 10
adv_class = 'plus1' # plus1, minus1, random
out_dir = './outputs'
# testing under simulated conditions
conds = ['size'] # 'illumination', 'size', 'noise'
illumination_strength = [100, 200]
illumination_radius_ratio = [0.2, 0.3]
size_ratio = [0.25, 0.875]
gaussian_amp = [0.02, 0.03]
snp_ratio = [1e-4, 3e-4]
test_dir = './outputs'
# basics
data_dir = './data'
img_size = 128
batch_size = 32
num_classes = 3
device = 'cuda'

