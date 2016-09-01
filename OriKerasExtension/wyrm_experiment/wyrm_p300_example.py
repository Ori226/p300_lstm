
# coding: utf-8

# In[1]:

from __future__ import division

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl

from wyrm import plot
plot.beautify()
from wyrm.types import Data
from wyrm import processing as proc
from wyrm.io import load_bcicomp3_ds2


# In[2]:

TRAIN_A = r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\ipytho_notebook\follow_wyrm_tutorial\data\BCI_Comp_III_Wads_2004\Subject_A_Train.mat'
TRAIN_B = r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\ipytho_notebook\follow_wyrm_tutorial\data\BCI_Comp_III_Wads_2004\Subject_B_Train.mat'

TEST_A = r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\ipytho_notebook\follow_wyrm_tutorial\data\BCI_Comp_III_Wads_2004/Subject_A_Test.mat'
TEST_B = r'C:\Users\ORI\Documents\IDC-non-sync\Thesis\PythonApplication1\ipytho_notebook\follow_wyrm_tutorial\data\BCI_Comp_III_Wads_2004/Subject_B_Test.mat'

TRUE_LABELS_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
TRUE_LABELS_B = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'

MATRIX = ['abcdef',
          'ghijkl',
          'mnopqr',
          'stuvwx',
          'yz1234',
          '56789_']

MARKER_DEF_TRAIN = {'target': ['target'], 'nontarget': ['nontarget']}
MARKER_DEF_TEST = {'flashing': ['flashing']}

SEG_IVAL = [0, 700]

JUMPING_MEANS_IVALS_A = [150, 220], [200, 260], [310, 360], [550, 660] # 91%
JUMPING_MEANS_IVALS_B = [150, 250], [200, 280], [280, 380], [480, 610] # 91%


# In[3]:

def preprocessing_simple(dat, MRK_DEF, *args, **kwargs):
    """Simple preprocessing that reaches 97% accuracy.
    """
    fs_n = dat.fs / 2
    b, a = proc.signal.butter(5, [10 / fs_n], btype='low')
    dat = proc.filtfilt(dat, b, a)
   
    dat = proc.subsample(dat, 20)
    epo = proc.segment_dat(dat, MRK_DEF, SEG_IVAL)
    fv = proc.create_feature_vectors(epo)
    return fv, epo


# In[4]:

def preprocessing(dat, MRK_DEF, JUMPING_MEANS_IVALS):
    dat = proc.sort_channels(dat)
    
    fs_n = dat.fs / 2
    b, a = proc.signal.butter(5, [30 / fs_n], btype='low')
    dat = proc.lfilter(dat, b, a)
    b, a = proc.signal.butter(5, [.4 / fs_n], btype='high')
    dat = proc.lfilter(dat, b, a)
    
    dat = proc.subsample(dat, 60)
    epo = proc.segment_dat(dat, MRK_DEF, SEG_IVAL)
    
    fv = proc.jumping_means(epo, JUMPING_MEANS_IVALS)
    fv = proc.create_feature_vectors(fv)
    return fv, epo


# In[7]:

epo = [None, None]
acc = 0
for subject in range(2):
    if subject == 0:
        training_set = TRAIN_A
        testing_set = TEST_A
        labels = TRUE_LABELS_A
        jumping_means_ivals = JUMPING_MEANS_IVALS_A
    else:
        training_set = TRAIN_B
        testing_set = TEST_B
        labels = TRUE_LABELS_B
        jumping_means_ivals = JUMPING_MEANS_IVALS_B
    
    # load the training set
    print "before loading"
    dat = load_bcicomp3_ds2(training_set)
    print "after loading "
    fv_train, epo[subject] = preprocessing(dat, MARKER_DEF_TRAIN, jumping_means_ivals)
    
    # train the lda
    print "before training"
    cfy = proc.lda_train(fv_train)

    print "after training"
    
    # load the testing set
    dat = load_bcicomp3_ds2(testing_set)
    fv_test, _ = preprocessing(dat, MARKER_DEF_TEST, jumping_means_ivals)
    
    # predict
    lda_out_prob = proc.lda_apply(fv_test, cfy)
    
    # unscramble the order of stimuli
    unscramble_idx = fv_test.stimulus_code.reshape(100, 15, 12).argsort()
    static_idx = np.indices(unscramble_idx.shape)
    lda_out_prob = lda_out_prob.reshape(100, 15, 12)
    lda_out_prob = lda_out_prob[static_idx[0], static_idx[1], unscramble_idx]
    
    #lda_out_prob = lda_out_prob[:, :5, :]
    
    # destil the result of the 15 runs
    #lda_out_prob = lda_out_prob.prod(axis=1)
    lda_out_prob = lda_out_prob.sum(axis=1)
        
    # 
    lda_out_prob = lda_out_prob.argsort()
    
    cols = lda_out_prob[lda_out_prob <= 5].reshape(100, -1)
    rows = lda_out_prob[lda_out_prob > 5].reshape(100, -1)
    text = ''
    for i in range(100):
        row = rows[i][-1]-6
        col = cols[i][-1]
        letter = MATRIX[row][col]
        text += letter
    print
    print 'Result for subject %d' % (subject+1)
    print 'Constructed labels: %s' % text.upper()
    print 'True labels       : %s' % labels
    a = np.array(list(text.upper()))
    b = np.array(list(labels))
    accuracy = np.count_nonzero(a == b) / len(a)
    print 'Accuracy: %.1f%%' % (accuracy * 100)
    acc += accuracy
print
print 'Overal accuracy: %.1f%%' % (100 * acc / 2)


# In[ ]:



