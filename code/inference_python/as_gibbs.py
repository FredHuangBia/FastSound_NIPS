'''
Gibbs sampling for infereing latent variables of a given sound.

Gibbs + MH for conditional prob.
'''


import os
import sys
from subprocess import call, check_output
import argparse
import numpy as np
import scipy as sp
import scipy.signal
from cv2.cv import CalcEMD2, CV_DIST_L1, CV_DIST_L2
from copy import copy
from tqdm import tqdm
from utils import cd, mkdir

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_test'
SOUND_SCRIPTS = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v1/'
SCRIPTS = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/'
PI = np.pi

PRIM = [62, 187, 312, 437, 553, 679, 805, 931, 1057, 1183, 1309, 1435, 1593, 1713]
H_MAX = 5
H_MIN = 3
ALPHA_MAX = -5
ALPHA_MIN = -8
BETA_MAX = 5
BETA_MIN = 0
RES_MAX = 0.9
RES_MIN = 0.3
VAR = 1/3

def change_dir_by_user():
    ''' chage ROOT by username '''
    user = check_output('whoami')
    if user == 'ztzhang\n':
        global ROOT
        ROOT = '/data/vision/billf/object-properties/sound/ztzhang/primitives/exp/random_test'

def sample_quat_uniform():
    '''uniformly sample quaternion
    x,y,z,w, bullet convention '''

    # Pick axis uniformly over S2
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    # rotation
    angle = PI*np.random.rand(1)
    q_x = axis[0]*np.sin(angle/2)
    q_y = axis[1]*np.sin(angle/2)
    q_z = axis[2]*np.sin(angle/2)
    q_w = np.cos(angle/2)
    return [q_x[0], q_y[0], q_z[0], q_w[0]]

def quat_to_matrix(quat):
    '''convert quat to matrix'''
    pass

def label_to_val(label):
    '''I guess I don;t need this now '''
    shape = prim[label[0]]
    prim = [62, 187, 312, 437, 553, 679, 805, 931, 1057, 1183, 1309, 1435, 1593, 1713]
    modulus = label[1]-1

def feature_raw(sound):
    '''raw'''
    return np.array(sound)

def feature_mfcc(sound):
    '''mfcc'''
    return mfcc

def feature_stft(sound):
    '''stft distance'''
    freq, time, stft = scipy.signal.spectrogram(np.array(sound), 44100, scaling='spectrum', nperseg=5001, noverlap=2000, nfft=44100, mode='magnitude')
    return stft

def feature_corl(sound):
    '''cross correlation of two signals'''
    pass


def linear_random(min, max):
    '''linear random in [min,max)'''
    return (max+min)/2 + (np.random.random(1)-0.5)*(max-min)

def gen_sound(label, path, prefix=""):
    '''Generate Sound using label'''
    output_dir = os.path.join(path, prefix)
    mkdir(output_dir)
    gen_config_script = os.path.join(SCRIPTS, 'gen_config.sh')
    config_arguments = [output_dir, str(label[3]), \
                        '[%.4f, %.4f, %.4f, %.4f]'%tuple(label[2]), \
                        str(label[4]), str(label[5]), str(label[6])]
    call(['sh', gen_config_script]+config_arguments)
    gen_sound_script = os.path.join(SOUND_SCRIPTS, 'gen_sound_new.py')
    sound_arguments = "%s 0 0 100000 10 10 10 %d%d 0 0 0 > %s" \
        %(output_dir, label[0], label[1], os.path.join(path, prefix, 'gen_sound.log'))
    call('python %s %s'%(gen_sound_script, sound_arguments), shell=True)

def gen_random_label():
    '''generate random label for sound synthesis'''
    shape = np.random.randint(14, size=1)
    prim_type = PRIM[shape[0]]
    youngs = np.random.randint(10, size=1)[0]
    quat = sample_quat_uniform()
    height = linear_random(H_MIN, H_MAX)[0]
    alpha = 10**linear_random(ALPHA_MIN, ALPHA_MAX)[0]
    beta = 2**linear_random(BETA_MIN, BETA_MAX)[0]
    res = linear_random(RES_MAX, RES_MIN)[0]
    return [prim_type, youngs, quat, height, alpha, beta, res]

def save_label(label, out_dir, prefix=''):
    '''save label for generated sound'''
    file_name = os.path.join(out_dir, 'label.txt')
    with open(file_name, 'w') as label_file:
        label_file.write(' '.join(str(e) for e in label[0:2])+' ')
        label_file.write(' '.join(str(e) for e in label[2])+' ')
        label_file.write(' '.join(str(e) for e in label[3:]))

def stft_to_pointsample(feature_1, base=1e-4, type='1d'):
    '''convert stft matrix to point samples with weight.'''
    if type == '1d':
        pass
    



def calc_distance(feature_1, feature_2i, type='l2'):
    '''Calculate Feature Space distance'''
    if type == 'l2':
        if feature_1.size != feature_2.size:
            raise ValueError('Feature Size Mismatch!')
        return np.sum(np.power(feature_1-feature_2, 2))
    if type == 'stft_emd1':
        c1 = stft_to_pointsample(feature_1, type='1d')
        c2 = stft_to_pointsample(feature_2, type='1d')
        distance_emd2 = CalcEMD2(c1, c2, distance_type=CV_DIST_L2)
        return distance_emd2
    


def gen_random_sound(num):
    ''' Generate random sound '''
    print 'Generating %d random sounds'%num
    sounddir = os.path.join(ROOT, 'test_sound')
    mkdir(sounddir)
    for k in tqdm(range(num)):
    # for k in range(num):
        label = gen_random_label()
        gen_sound(label, sounddir, str(k))
        save_label(label, sounddir, str(k))

def read_raw_wav(sound_file_path):
    ''' read raw wav file '''
    target_sound = []
    with open(sound_file_path, 'r') as raw_wav_file:
        for line in raw_wav_file.readlines():
            target_sound.append(float(line))
    return target_sound

def read_gen_sound_raw(label, dir, prefix=''):
    '''load a sound generated by label'''
    gen_sound(label, dir, prefix)
    return read_raw_wav(os.path.join(dir, prefix, 'sound.raw'))

def calc_feature_distance(sound_1, sound_2, feature_func):
    '''calc feature space distace of 2 sounds'''
    return calc_distance(feature_func(sound_1), feature_func(sound_2))

def label_to_str(label):
    ''' conver label to string '''
    label_str = ''
    label_str = label_str+' '.join(str(e) for e in label[0:2])+' '
    label_str = label_str+' '.join(str(e) for e in label[2])+' '
    label_str = label_str+' '.join(str(e) for e in label[3:])
    return label_str

class Logger:
    def __init__(self, path):
        self.distance_log = open(os.path.join(path, 'distance_log.txt'), 'w', 0)
        self.label_log = open(os.path.join(path, 'label_log.txt'), 'w', 0)
        self.ar_log = open(os.path.join(path, 'ar_log.txt'), 'w', 0)
        self.prob_log = open(os.path.join(path, 'prob_log.txt'), 'w', 0)
    def log_distance(self, distance):
        self.distance_log.write('%.4f\n'%distance)
    def log_label(self, label):
        self.label_log.write('%s\n'%label_to_str(label))
    def log_ar(self, status):
        self.ar_log.write('%s\n'%status)
    def log_prob(self, prob):
        self.prob_log.write('%.3E\n'%prob)

    def log(self, distance, label, status, prob):
        self.log_distance(distance)
        self.log_label(label)
        self.log_ar(status)
        self.log_prob(prob)
    def close(self):
        self.distance_log.close()
        self.label_log.close()
        self.ar_log.close()
        self.prob_log.close()

#def map_to_prob(distance):
#    return 1/(distance)

def map_to_prob(distance, time):
    t = float(time)
    return np.exp(-distance * (0.5+0.3*t*t/15))

def gaussian_random(max, min):
    return np.random.randn(1)*VAR*(max-min)/2

def update_label(label, id):
    newlabel = copy(label)
    if id == 0:
        newlabel[0] = PRIM[np.random.randint(14, size=1)[0]]
    if id == 1:
        newlabel[1] = np.random.randint(10, size=1)[0]
    if id == 2:
        newlabel[2] = sample_quat_uniform()
    if id == 3:
        newlabel[3] = gaussian_random(H_MAX, H_MIN)[0]+label[3]
    if id == 4:
        newlabel[4] = gaussian_random(ALPHA_MAX, ALPHA_MIN)[0] + label[4]
    if id == 5:
        newlabel[5] = gaussian_random(BETA_MAX, BETA_MIN)[0] + label[5]
    if id == 6:
        newlabel[6] = gaussian_random(RES_MAX, RES_MIN)[0]+label[6]
    return newlabel

def load_label(path):
    with open(path, 'r') as f:
        raw_label = f.readline()

METRIC_DICT = {'raw':feature_raw, 'mfcc': feature_mfcc, 'stft':feature_stft}
#DIST_DICT = {'inverse':inverse_distance, 'expneg':expneg}
# pylint: disable=C0103
if __name__ == '__main__':
    #INPUT: ID
    #OPTIONAL FLAG: root, #iter, #sampling, metric, is_gen
    
    change_dir_by_user()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', help='Generate Random sound.', type=int, default=0)
    parser.add_argument('test_id', type=int, help='specify the id of wavefile to be tested')
    parser.add_argument('-m', '--metric', help='specify metric for sound: raw, stft, mfcc.', choices=['raw', 'mfcc', 'stft'], default='stft')
    parser.add_argument('-n', '--niter', help='specify #iterations of MH sampling for conditional prob.', type=int, default=30)
    parser.add_argument('-N', '--nsample', help='Specify #samples for gibbs sampling.', type=int, default=10)
    parser.add_argument('-t', '--type', help='mapping l2 distance to probability: inverse, expneg', choices=['inverse', 'expneg'], default='inverse')
    parser.add_argument('-T', '--temp', help='factor for expneg. default = 1000', type=float, default=1000.0)
    args = parser.parse_args()
    if args.generate != 0:
        gen_random_sound(args.generate)
    else:
        test_id = args.test_id
        sound_file_path = os.path.join(ROOT, 'test_sound', str(test_id), 'sound.raw')
        target_sound = read_raw_wav(sound_file_path)
        feature_function = METRIC_DICT[args.metric]
        dir_name = '%s_%s_%04d_%02d' %(args.metric, args.type, args.niter, args.nsample)
        result_dir = os.path.join(ROOT, 'result', dir_name, str(test_id))
        mkdir(result_dir)
        # Prepare init
        init_label = gen_random_label()
        init_sound = read_gen_sound_raw(init_label, result_dir, 'init')
        init_distance = calc_feature_distance(target_sound, init_sound, feature_function)
        print 'INIT label: %s' % label_to_str(init_label)
        print 'INIT distance: %.4f' % init_distance
        print 'INIT prob: %.5f' % map_to_prob(init_distance, 0.0)
        log = Logger(result_dir)
        #mapping = DIST_DICT[args.type]
        label = init_label
        distance = init_distance
        counter = 0
        acc_counter = 0
        log.log(init_distance, init_label, 'init', map_to_prob(init_distance, 0))
        record = {'label':init_label, 'distance':init_distance}
        ## Gibbs + MH
        for t in tqdm(range(args.niter)):
            for label_id in range(7):
                for sample in tqdm(range(args.nsample)):
                    status = 'reject'
                    newlabel = update_label(label, label_id)
                    newsound = read_gen_sound_raw(newlabel, result_dir, str(counter))
                    newdistance = calc_feature_distance(target_sound, newsound, feature_function)
                    alpha = map_to_prob(newdistance, t)/map_to_prob(distance, t)
                    if alpha >= 1:
                        status = 'accept'
                        acc_counter += 1
                        label = newlabel
                        distance = newdistance
                        if newdistance < record['distance']:
                            record['label'] = newlabel
                            record['distance'] = newdistance
                            print '***Current BEST!***'
                    elif alpha > 1e-2:
                        if np.random.binomial(1, alpha, 1) == 1:
                            label = newlabel
                            distance = newdistance
                            status = 'accept %.3f' %alpha
                            acc_counter += 1
                    log.log(distance, label, status, map_to_prob(distance, t))
                    counter += 1
                    print 'sample %d: distance %.4f %s'%(counter, newdistance, status)
                    print 'Current Acc Rate: %.4f'%(float(acc_counter)/float(counter))
        log.close()
        save_label(record['label'], result_dir)
