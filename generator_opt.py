import numpy as np
from myfunc import loadnp

try : 
    from sklearn.utils.class_weight import compute_sample_weight
except (ImportError, ModuleNotFoundError) as e:
    print("sklearn could not load....")
    pass


class DNAmulti5H():
    #UPDATED 14/06/2022
    """
    Generate batch for keras.fit()
    """

    def __init__(self,seq,lab=None, winsize = 2001, frac=1., 
    zeros=False, sample = True, apply_weights= True, reverse=False, N=0, truncate = 3_000_000):
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
        # Assertions
        assert winsize%2==1, "Window size must be odd"
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
                self.seq.append(loadnp(seq[i])[truncate:self.len[-1]].reshape((-1,4)))


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
            weights_tmp = compute_sample_weight("balanced", self.lab[idx]).ravel()
            self.weights_f[idx] = weights_tmp
            
        else:
            self.weights_f = None

        return idx


    def generate_images(self, image_idx, is_training=True, batch_size=4096, step=[-500,-250,0,250,500]):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        :param arr image_idx: index produced by generate_split_indexes
        :param bool is_training: indicate if the generator is used to train (True) or else (False)
        :param int batch_size: number of data in a single batch

        steps are indexed from the stard of the window
        """
        assert len(image_idx)>=batch_size, "not enough index for this batch_size"

        if self.sample:
            idx = np.random.choice(image_idx, replace=False, size=len(image_idx))

        images, classes, weights = [], [], []

        while True:
            for idx in image_idx:
                try:
                    im = self.seq[idx - (self.winsize//2) : idx+(self.winsize//2)+1,:]
                    classe = self.lab[[idx + x for x in step]]

                    try:
                        weight = self.weights_f[[idx + x for x in step]]
                    except TypeError:
                        pass

                except ValueError:
                    print("issue at index idx, value ignored")
                    continue
                
                if self.reverse == 1: #reverse complement
                    im2 = im.copy()[::-1, [3, 2, 1, 0]]
                    try:
                        assert im2.shape==im.shape==(self.winsize, 4)
                    except AssertionError:
                        continue
                    classes.append(classe)
                    classes.append(classe[::-1])
                    images.append(im)
                    images.append(im2)

                    try:
                        weights.append(weight)
                        weights.append(weight[::-1])
                    except  (TypeError, UnboundLocalError) as e:
                        pass

                elif self.reverse == 2: #reverse complement only
                        classes.append(classe[::-1])
                        images.append(im[::-1, [3,2,1,0]])

                        try:
                            weights.append(weight[::-1])
                        except (TypeError, UnboundLocalError) as e:
                            pass

                else:
                    classes.append(classe)
                    images.append(im)
                    weights.append(weight)

                # yielding condition
                if len(images) >= batch_size:
                    classes = np.array(classes)
                    images = np.vstack(images).reshape(-1, self.winsize, 4)



                    # print("Classes", classes.shape)
                    # print("Images", images.shape)
                    # print("Weights", weights.shape)
                    if self.apply_weight:
                        weights = np.array(weights)
                        yield images, classes, weights
                    else:
                        yield images, classes
                    images, classes, weights = [], [], []
                    
            if not is_training:
                print("END")
                return None