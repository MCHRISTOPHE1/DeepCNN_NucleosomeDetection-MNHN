import numpy as np


class DNAseqgenerator():
    """
    Generate batch for keras.fit()
    """

    def __init__(self,seq,lab,weights = None, winsize = 2001, frac=1., zeros=False, sample = True, apply_weights= False):
        """
        :param str seq: path to a numpy array stored as npz in ["arr_0"], the array contains the one-hot encoded DNA/RNA sequence
        :param str lab:  path to a numpy array stored as npz in ["arr_0"], the array contains labels for the center of the considered window
        :param str weights: path to a numpy array stored as npz in ["arr_0"], the array contains weights for each label
        :param int winsize: size of the window, must be odd
        :param float frac: fraction of the label to use, must be in [0:1]
        :param bool zeros: include labels 0 in the training if True
        :param bool sample: sample the labels if true
        :param bool apply_weights: if true, apply_selected weights
        """
        assert winsize%2==1, "Window size must be odd"
        assert 0.<=frac<=1., "frac must be in [0:1]"

        self.lab = np.load(lab)["arr_0"]
        self.seq = np.load(seq)["arr_0"].reshape((-1,4))
        self.frac = frac
        self.zeros = zeros

        if apply_weights:
            self.weights_f = np.load(weights)["arr_0"]
        else:
            self.weights_f = np.ones(len(self.lab))

        self.winsize = winsize
        self.sample=sample

    def generate_split_indexes(self):
        print("split")

        l = len(self.lab)
        ltmp = np.unique(np.clip(np.arange(l), self.winsize//2,len(self.seq)-self.winsize//2))

        if self.zeros:
            if self.sample:
                a = np.random.choice(ltmp,size=int(len(ltmp)*self.frac), replace=False)
                print("With zeros, shuffled:", len(a))
                return a

            else:
                a = ltmp[:int(len(ltmp)*self.frac)]
                print("With zeros, not shuffled:", len(a))
                return a

        else:
            ret = ltmp[self.lab[:len(ltmp)]!=0]

            if self.sample:
                a = np.random.choice(a=ret, size=int(len(ret)*self.frac), replace=False)
                print("With no zeros, shuffled", len(a))
                return a
                
            else:
                a = ret[:int(len(ret)*self.frac)]
                print("With no zeros, not shuffled", len(a))
                return a

    def generate_images(self, image_idx, is_training=True, batch_size=4096):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        :param arr image_idx: index produced by generate_split_indexes
        :param bool is_training: indicate if the generator is used to train (True) or else (False)
        :param int batch_size: number of data in a single batch
        """
        assert len(image_idx)>=batch_size, "not enough index for this batch_size"
        print(image_idx)
        print()
        # arrays to store our batched data

        images, classes, weights = [], [], []
        while True:
            for idx in image_idx:
                im = self.seq[int(idx)-self.winsize//2:int(idx+self.winsize//2)+1,:].reshape((self.winsize,4))
                classe = self.lab[idx]
                weight = self.weights_f[idx]

                classes.append(classe)
                images.append(im)
                weights.append(weight)
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), np.array(classes), np.array(weights)
                    images, classes, weights = [], [], []
                    
            if not is_training:
                print("END")
                return None