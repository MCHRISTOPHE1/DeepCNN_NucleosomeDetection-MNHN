import numpy as np
from myfunc import loadnp, consecutive
import tensorflow as tf
import math
from myfunc import create_ranges, gauss, reshape_bin
import os
import tempfile
import time

try : 
    from sklearn.utils.class_weight import compute_sample_weight
except (ImportError, ModuleNotFoundError) as e:
    print("sklearn could not load....")
    pass


class DNAmulti5H():

    """
    Generate batch for keras.fit()
    """

    def __init__(self,seq,lab=None, winsize = 2001, frac=1., 
    zeros=False, sample = True, apply_weights= -1, reverse=False, N=0, truncate = 3_000_000):
        """
        :param str seq: path to a numpy array stored as npz in ["arr_0"], the array contains the one-hot encoded DNA/RNA sequence
        :param str lab:  path to a numpy array stored as npz in ["arr_0"], the array contains labels for the center of the considered window
        :param str weights: path to a numpy array stored as npz in ["arr_0"], the array contains weights for each label
        :param int winsize: size of the window, must be odd
        :param float frac: fraction of the label to use, must be in [0:1]
        :param bool zeros: include labels 0 in the training if True
        :param bool sample: sample the labels if true
        :param int apply_weights: 0: No weights, 1: array of sample weights, 2: dict of sample weights, 3: undersampling
        :param int reverse: 0:No reverse, 1: both, 2: reverse on<sly
        :param int N: 0:remove N, 1:keep N
        :param func fun: allow user to pass a function to select specific label ex : def func(arr): \n arr[arr<0.5] = 0 \n return arr 
        """
        # Assertions
        #assert winsize%2==1, "Window size must be odd"
        assert 0.<=frac<=1., "frac must be in [0:1]"
      
        # Variables
        self.len, self.lab, self.seq = [0],[],[]
        self.frac = frac
        self.zeros = zeros
        self.apply_weight = apply_weights
        self.winsize = winsize
        self.sample=sample
        self.reverse = reverse
        self.truncate = truncate
        print("winsize :", self.winsize)
        # Read files
        """Check wether the file is a npz or npy and extract ["arr_0"] in case of npz
        Each label file must match with the sequence file (no double check programmed)"""

        if type(lab) == type(None):
            assert len(seq) == 1, "you cannot predict multiple chromosome at once"
            for i, s in enumerate(seq):
                self.seq.append(loadnp(s))
                self.lab.append(np.zeros(len(self.seq[-1])))
                self.len.append(len(self.lab[-1]))

        else:
            for i,l in enumerate(lab):
                self.lab.append(loadnp(l)[truncate:])
                self.len.append(len(self.lab[-1]))
                self.seq.append(loadnp(seq[i]).reshape((-1,4))[truncate:self.len[-1]+truncate,:])


                # N ([0,0,0,0]) removal 
                if N==0:
                    zidx = np.sum(self.seq[i], axis=1).ravel()
                    zidx = zidx == 0
                    zidx = np.convolve(zidx, np.ones(self.winsize), mode="same")                #Remove all indices where the window would contain any N ([0,0,0,0])
                    zidx = np.where(zidx>=1)[0]
                    self.lab[-1][zidx] = 0

        # Shape lab and seq
        self.lab = np.concatenate(self.lab).ravel()
        self.seq = np.concatenate(self.seq).reshape((-1,4))


    def generate_split_indexes(self):
        """Generate indices for sequence and label and process data (weights, randomization, zeros"""
        print("split")

        #Raw indices generation (0:N)
        idx = []
        cs = np.cumsum(self.len) #Cumulative sum of lab length

        for i,l in enumerate(self.len[1:]):
            s = cs[i]
            #np.arange((self.winsize//2),l-(self.winsize//2)-1))
            ltmp = np.unique(np.clip(np.arange(l), (self.winsize//2),l-(self.winsize//2)-1)) #Remove indices such as the windows can always fit
            # With zeros
            if self.zeros == True:
                if self.sample == True:
                    a = np.random.choice(ltmp,size=int(len(ltmp)*self.frac), replace=False) #Randomize indices order and apply fractionnement (frac)

                else:
                    a = ltmp[:int(len(ltmp)*self.frac)]

            #Remove zeros
            else:
                zz = self.lab[s:s+l]!=0
                zz = zz[(self.winsize//2):(l-(self.winsize//2))]
                ret = ltmp[zz]
                if self.sample:
                    a = np.random.choice(a=ret, size=int(len(ret)*self.frac), replace=False)

                else:
                    a = ret[:int(len(ret)*self.frac)]
            a += s
            idx.append(a)
        idx = np.concatenate(idx)
        
        if self.apply_weight: #sklearn
            
            self.weights_f = np.zeros(self.lab.shape)
            weights_tmp = compute_sample_weight("balanced", np.round(self.lab[idx],2)).ravel()
            self.weights_f[idx] = weights_tmp
            print("number of weights : ", len(np.unique(self.weights_f)))
        # elif self.apply_weight > 1: #hist
        #     hist , nbin = np.histogram(self.lab[idx], bins=self.apply_weight, density=True)
        #     digit = np.digitize(self.lab, nbin[:-1], right = False)-1
        #     self.weights_f = hist[digit]
            
        else:
            self.weights_f = None
        print("number of 0's in labels : ", np.sum(self.lab[idx]==0))
        return idx


    def generate_images(self, image_idx, is_training=True, batch_size=4096, step=[-500,-250,0,250,500]):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        :param arr image_idx: index produced by generate_split_indexes
        :param bool is_training: indicate if the generator is used to train (True) or else (False)
        :param int batch_size: number of data in a single batch

        steps are indexed from the stard of the window
        """
        print("Data generation....")

        assert len(image_idx) >= batch_size, "not enough index for this batch_size"

        if self.sample:
            idx = np.random.choice(image_idx, replace=False, size=len(image_idx))

        images, classes, weights = [], [], []

        while True:
            for idx in image_idx:
                try:
                    im = self.seq[idx - (self.winsize//2) : idx+(self.winsize//2)+self.winsize%2,:]
                    classe = self.lab[step+idx]
                    assert im.shape == (self.winsize, 4)
                    try:
                        weight = self.weights_f[step+idx]
                    except TypeError:
                        pass

                except (ValueError, AssertionError) as e:
                    # print("issue at index {}, value ignored".format(idx))
                    continue
                
                if self.reverse == 1: #reverse complement and std
                    im2 = im.copy()[::-1, ::-1]

                    images.append(im)
                    classes.append(classe)
                    
                    #Reverse complement
                    images.append(im2)
                    classes.append(classe.copy()[::-1])

                    try:
                        weights.append(weight)
                        weights.append(weight.copy()[::-1])
                    except  (TypeError, UnboundLocalError) as e:
                        pass

                elif self.reverse == 2: #reverse complement only
                        a = im.copy()[::-1, ::-1]
                        try:
                            a.shape == im.shape
                        except AssertionError:
                            continue

                        classes.append(classe.copy()[::-1])
                        images.append(im.copy()[::-1, ::-1])

                        try:
                            weights.append(weight.copy()[::-1])
                        except (TypeError, UnboundLocalError) as e:
                            pass
                elif self.reverse == 3: #random reverse
                    rand = np.random.rand()
                    if rand>0.5:
                        classes.append(classe)
                        images.append(im)
                        try:
                            weights.append(weight)
                        except (TypeError, UnboundLocalError) as e:
                            pass
                    else:
                        classes.append(classe.copy()[::-1])
                        images.append(im.copy()[::-1, ::-1])
                        try:
                            weights.append(weight.copy()[::-1])
                        except (TypeError, UnboundLocalError) as e:
                            pass
                else:
                    classes.append(classe)
                    images.append(im)

                    try:
                        weights.append(weight)
                    except (TypeError, UnboundLocalError) as e:
                        pass

                # yielding condition
                if len(images) >= batch_size:
                    classes = np.array(classes)
                    images = np.vstack(images).reshape(-1, self.winsize, 4)
                    # print(f"classes : {classes.shape}\
                    #       images : {images.shape}")

                    if self.apply_weight:
                        weights = np.array(weights)
                        yield images, classes, weights
                    else:
                        yield images, classes
                    images, classes, weights = [], [], []
                    
            if not is_training:
                print("END")
                return None


class KDNAmulti(tf.keras.utils.Sequence):
    #UPDATED 14/06/2022
    """
    Generate batch for keras.fit()
    """

    def __init__(self,seq,lab=None, winsize = 2001, frac=10_000_000, 
                reverse="", truncate = 3_000_000, batch_size=2048, mask = None,
                headsteps = np.array([-500, -250, 0, 250, 500]), weights = True, index_selection=""):
        """
        :param str seq: path to a numpy array stored as npz in ["arr_0"], the array contains the one-hot encoded DNA/RNA sequence
        :param str lab:  path to a numpy array stored as npz in ["arr_0"], the array contains labels for the center of the considered window
        :param str weights: path to a numpy array stored as npz in ["arr_0"], the array contains weights for each label
        :param int winsize: size of the window, must be odd
        :param float frac: fraction of the label to use, must be in [0:1]
        :param bool zeros: include labels 0 in the training if True
        :param bool sample: sample the labels if true
        :param int apply_weights: 0: No weights, 1: array of sample weights, 2: dict of sample weights, 3: undersampling
        :param int reverse: 0:No reverse, 1: both, 2: reverse only
        :param int N: 0:remove N, 1:keep N
        :param func fun: allow user to pass a function to select specific label ex : def func(arr): \n arr[arr<0.5] = 0 \n return arr 
        """
        # Variables
        ztoy = lambda x : x.replace(".npz", "_temp.npy")
        for i in range(len(lab)):
            try:
                assert os.path.isfile(ztoy(lab[i]))
            except AssertionError:
                sig = loadnp(lab[i])
                np.save(ztoy(lab[i]), sig)

            try:
                assert os.path.isfile(ztoy(seq[i]))
            except AssertionError:
                oeseq = loadnp(seq[i])
                np.save(ztoy(seq[i]), oeseq)

        
        self.chr = np.array([ztoy(x) for x in lab])
        self.one_hot = np.array([ztoy(x) for x in seq])

        # self.length, self.lab, self.seq = [0],[],[]
        self.winsize = winsize
        self.reverse = reverse
        self.batch_size = batch_size
        # self.index = []
        # self.weights_f = []
        self.step = headsteps
        self.frac =frac
        self.truncate = truncate
        self.weights = weights
        self.index_selection = index_selection
        self.mask = mask

        self.select_chr()


    def select_chr(self):
        self.lab, self.seq = [],[]
        #self.index = []
        self.weights_f = []

        # Read files
        """Check wether the file is a npz or npy and extract ["arr_0"] in case of npz
        Each label file must match with the sequence file (no double check programmed)"""
        random_idx = np.random.choice(np.arange(0, len(self.chr)), size=min(3, len(self.chr)), replace=None)

        for ridx in random_idx:
            self.lab.append(loadnp(self.chr[ridx])[self.truncate:])
            self.seq.append(loadnp(self.one_hot[ridx]).reshape((-1,4))[self.truncate:len(self.lab[-1])+self.truncate,:])
            self.lab[-1][:self.winsize+1] = 0
            self.lab[-1][-self.winsize:] = 0

            if self.mask is not None:
                print("masking labels...")
                m = self.mask[ridx] - self.truncate
                m = m[m>=0]
                self.lab[-1][m] = 0 #remove selected part of signal, usually used for repeats
                

        self.lab = np.concatenate(self.lab, axis = None)
        self.seq = np.concatenate(self.seq, axis = 0)

        #N
        n_pos = np.sum(self.seq, axis = 1)==0
        self.lab[np.convolve(n_pos, np.ones(self.winsize), mode = "same")!=0] = 0

        # if self.index_selection != "validation":
        #     binidx = np.convolve(self.lab, np.ones(1000), "same")
        #     self.index = np.where((self.lab!=0) & (binidx>=500))[0]
        # else:
        self.index = np.where(self.lab!=0)[0]

        if self.index_selection == "local":
            self.index_epoch = np.random.choice(self.index, replace=False, size=(self.frac//self.batch_size)+1)
            self.index_epoch = np.repeat(self.index_epoch, self.batch_size)
            self.index_epoch += np.tile(np.arange(self.batch_size), (self.frac//self.batch_size)+1)
        else:
            try:
                self.index_epoch = np.random.choice(self.index, replace=False, size = min(self.frac, len(self.index)))
                self.index = self.index[np.isin(self.index, self.index_epoch, invert=True)]
            except (IndexError, ValueError) as e:
                print(e, "New indices generated")

    def __len__(self):
        size = len(self.index_epoch)//self.batch_size
        if self.reverse == "augmentation":
            return 2*size
        else:
            return size
    
    def on_epoch_end(self):
        if self.index_selection == "validation":
            pass
        elif len(self.chr)>3:
            self.select_chr()
        else:
            if self.index_selection=="local":
                self.index_epoch = np.random.choice(self.index, replace=False, size=(self.frac//self.batch_size)+1)
                self.index_epoch = np.repeat(self.index_epoch, self.batch_size)
                self.index_epoch += np.tile(np.arange(self.batch_size), (self.frac//self.batch_size)+1)
            else:
                self.index_epoch = np.random.choice(self.index, replace=False, size=self.frac)


    def __getitem__(self, index):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        :param arr image_idx: index produced by generate_split_indexes
        :param bool is_training: indicate if the generator is used to train (True) or else (False)
        :param int batch_size: number of data in a single batch

        steps are indexed from the stard of the window
        """

        images, classes, weights = [], [], []
        halfsize = self.winsize//2
        odd = self.winsize%2

        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.index_epoch))
        batch = self.index_epoch[low:high]

        #Sequences
        ranges = create_ranges(batch-halfsize, batch + halfsize + odd)
        

        #Labels
        heads = np.repeat(batch, len(self.step)).reshape((-1, len(self.step)))
        heads += self.step

        fixidx = np.unique(np.where(heads<len(self.lab))[0])
        ranges = ranges.reshape((-1, self.winsize))[fixidx,:].ravel()
        heads = heads[fixidx,:].ravel()
        images = self.seq[ranges,:].reshape(-1, self.winsize, 4)
        classes = self.lab[heads].reshape((-1, len(self.step)))
        weights = np.zeros(classes.shape)
        idx_weights = classes!=0
        if self.weights:
            weights[idx_weights] = compute_sample_weight("balanced", np.round(classes[idx_weights], 2).ravel())
        else:
            weights = np.ones(classes.shape)
        
        if self.reverse == "random":
            randidx = np.random.choice(np.arange(len(images)), size=len(images)//2, replace=False)
            images[randidx,...] = images[randidx,::-1, ::-1]
            classes[randidx,:] = classes[randidx,::-1]
            weights[randidx,:] = weights[randidx, ::-1]

        elif self.reverse == "reverse":
            images = images[:,::-1, ::-1]
            classes = classes[:,::-1]
            weights = weights[:, ::-1]

        elif self.reverse == "augmentation":
            images = np.vstack([images, images[:,::-1, ::-1]])
            classes = np.vstack([classes, classes[:,::-1]])
            weights = np.vstack([weights, weights[:, ::-1]])


        # print(images.shape)
        # print(classes.shape)
        # print(weights.shape)
        
        if self.weights:
            return images, classes, weights
        else:
            return images, classes



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# import pyBigWig as pwb

# class generator_from_bigwig(tf.keras.utils.Sequence):
#     #UPDATED 14/06/2022
#     """
#     Generate batch for keras.fit()
#     """

#     def __init__(self,bw, chr, seq,lab=None, winsize = 2001, frac=10_000_000, 
#                 reverse="", truncate = 3_000_000, batch_size=2048,
#                 headsteps = np.array([-500, -250, 0, 250, 500]), weights = True, index_selection=""):
#         """
#         :param Iter str chr: MUST BE IN FORMAT "chrK"
#         :param str seq: path to a numpy array stored as npz in ["arr_0"], the array contains the one-hot encoded DNA/RNA sequence
#         :param str lab:  path to a numpy array stored as npz in ["arr_0"], the array contains labels for the center of the considered window
#         :param str weights: path to a numpy array stored as npz in ["arr_0"], the array contains weights for each label
#         :param int winsize: size of the window, must be odd
#         :param float frac: fraction of the label to use, must be in [0:1]
#         :param bool zeros: include labels 0 in the training if True
#         :param bool sample: sample the labels if true
#         :param int apply_weights: 0: No weights, 1: array of sample weights, 2: dict of sample weights, 3: undersampling
#         :param int reverse: 0:No reverse, 1: both, 2: reverse only
#         :param int N: 0:remove N, 1:keep N
#         :param func fun: allow user to pass a function to select specific label ex : def func(arr): \n arr[arr<0.5] = 0 \n return arr 
#         """

#         self.bw = pwb.open(bw)
#         self.chr = chr
#         self.one_hot = np.array([x for x in seq])

#         # self.length, self.lab, self.seq = [0],[],[]
#         self.winsize = winsize
#         self.reverse = reverse
#         self.batch_size = batch_size
#         # self.index = []
#         # self.weights_f = []
#         self.step = headsteps
#         self.frac =frac
#         self.truncate = truncate
#         self.weights = weights
#         self.index_selection = index_selection

#         self.select_chr()


#     def select_chr(self):
#         self.lab, self.seq = [],[]
#         #self.index = []
#         self.weights_f = []


#         sizes = np.array([self.bw.chroms()[x] for x in self.chr])
#         rel_sizes = sizes/np.sum(sizes)
#         nb_sizes = self.

#         random_idx = np.random.choice
#         random_idx = np.random.choice(np.arange(0, len(self.chr.chroms())), size=min(3, len(self.chr)), replace=None)

#         for ridx in random_idx:
#             self.lab.append(loadnp(self.chr[ridx])[self.truncate:])
#             self.seq.append(loadnp(self.one_hot[ridx]).reshape((-1,4))[self.truncate:len(self.lab[-1])+self.truncate,:])
#             self.lab[-1][:self.winsize+1] = 0
#             self.lab[-1][-self.winsize:] = 0
                

#         self.lab = np.concatenate(self.lab, axis = None)
#         self.seq = np.concatenate(self.seq, axis = 0)

#         #N
#         n_pos = np.sum(self.seq, axis = 1)==0
#         self.lab[np.convolve(n_pos, np.ones(self.winsize), mode = "same")!=0] = 0

#         # if self.index_selection != "validation":
#         #     binidx = np.convolve(self.lab, np.ones(1000), "same")
#         #     self.index = np.where((self.lab!=0) & (binidx>=500))[0]
#         # else:
#         self.index = np.where(self.lab!=0)[0]

#         if self.index_selection == "local":
#             self.index_epoch = np.random.choice(self.index, replace=False, size=(self.frac//self.batch_size)+1)
#             self.index_epoch = np.repeat(self.index_epoch, self.batch_size)
#             self.index_epoch += np.tile(np.arange(self.batch_size), (self.frac//self.batch_size)+1)
#         else:
#             try:
#                 self.index_epoch = np.random.choice(self.index, replace=False, size = min(self.frac, len(self.index)))
#                 self.index = self.index[np.isin(self.index, self.index_epoch, invert=True)]
#             except (IndexError, ValueError) as e:
#                 print(e, "New indices generated")

#     def __len__(self):
#         size = len(self.index_epoch)//self.batch_size
#         if self.reverse == "augmentation":
#             return 2*size
#         else:
#             return size
    
#     def on_epoch_end(self):
#         if self.index_selection == "validation":
#             pass
#         elif len(self.chr)>3:
#             self.select_chr()
#         else:
#             if self.index_selection=="local":
#                 self.index_epoch = np.random.choice(self.index, replace=False, size=(self.frac//self.batch_size)+1)
#                 self.index_epoch = np.repeat(self.index_epoch, self.batch_size)
#                 self.index_epoch += np.tile(np.arange(self.batch_size), (self.frac//self.batch_size)+1)
#             else:
#                 self.index_epoch = np.random.choice(self.index, replace=False, size=self.frac)


#     def __getitem__(self, index):
#         """
#         Used to generate a batch with images when training/testing/validating our Keras model.
#         :param arr image_idx: index produced by generate_split_indexes
#         :param bool is_training: indicate if the generator is used to train (True) or else (False)
#         :param int batch_size: number of data in a single batch

#         steps are indexed from the stard of the window
#         """

#         images, classes, weights = [], [], []
#         halfsize = self.winsize//2
#         odd = self.winsize%2

#         low = index * self.batch_size
#         high = min(low + self.batch_size, len(self.index_epoch))
#         batch = self.index_epoch[low:high]

#         #Sequences
#         ranges = create_ranges(batch-halfsize, batch + halfsize + odd)
        

#         #Labels
#         heads = np.repeat(batch, len(self.step)).reshape((-1, len(self.step)))
#         heads += self.step

#         fixidx = np.unique(np.where(heads<len(self.lab))[0])
#         ranges = ranges.reshape((-1, self.winsize))[fixidx,:].ravel()
#         heads = heads[fixidx,:].ravel()
#         images = self.seq[ranges,:].reshape(-1, self.winsize, 4)
#         classes = self.lab[heads].reshape((-1, len(self.step)))
#         weights = np.zeros(classes.shape)
#         idx_weights = classes!=0
#         if self.weights:
#             weights[idx_weights] = compute_sample_weight("balanced", np.round(classes[idx_weights], 2).ravel())
#         else:
#             weights = np.ones(classes.shape)
        
#         if self.reverse == "random":
#             randidx = np.random.choice(np.arange(len(images)), size=len(images)//2, replace=False)
#             images[randidx,...] = images[randidx,::-1, ::-1]
#             classes[randidx,:] = classes[randidx,::-1]
#             weights[randidx,:] = weights[randidx, ::-1]

#         elif self.reverse == "reverse":
#             images = images[:,::-1, ::-1]
#             classes = classes[:,::-1]
#             weights = weights[:, ::-1]

#         elif self.reverse == "augmentation":
#             images = np.vstack([images, images[:,::-1, ::-1]])
#             classes = np.vstack([classes, classes[:,::-1]])
#             weights = np.vstack([weights, weights[:, ::-1]])


#         # print(images.shape)
#         # print(classes.shape)
#         # print(weights.shape)
        
#         if self.weights:
#             return images, classes, weights
#         else:
#             return images, classes
