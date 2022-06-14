import numpy as np


class DNAmulti():
    #UPDATED 14/06/2022
    """
    Generate batch for keras.fit()
    """

    def __init__(self,seq,lab, winsize = 2001, frac=1., zeros=False, sample = True, apply_weights= 1, reverse=False, N=0, fun = None):
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

        # Read files
        """Check wether the file is a npz or npy and extract ["arr_0"] in case of npz
        Each label file must match with the sequence file (no double check programmed)"""
        for i,l in enumerate(lab):
            if l[-1] == "y":
                self.lab.append(np.load(l))
            elif l[-1] == "z":
                self.lab.append(np.load(l)["arr_0"])
            else:
                raise ValueError("file: '{}' is nor .npz nor .npy".format(lab))

            self.len.append(len(self.lab[-1]))

            if seq[i][-1] == "y":
                self.seq.append(np.load(seq[i])[:self.len[-1]].reshape((-1,4)))             #Sequence is truncated to fit the labels
            elif seq[i][-1] == "z":
                self.seq.append(np.load(seq[i])["arr_0"][:self.len[-1]].reshape((-1,4)))
            else:
                raise ValueError("file: '{}' is nor .npz nor .npy".format(seq[i]))

            # N ([0,0,0,0]) removal 
            if N==0:
                zidx = np.sum(self.seq[i], axis=1).ravel()
                zidx = zidx == 0
                zidx = np.convolve(zidx, np.ones(self.winsize), mode="same")                #Remove all indices where the window would contain any N ([0,0,0,0])
                zidx = np.where(zidx>=1)[0]
                self.lab[-1][zidx] = 0


            # Data selection
            """Data selection can be made with any function of your choice, returning labels, labels 0 will be ignored, this function is passed as a parameter
            example:
            --------
            def func(arr):
                arr[arr<0.5] = 0
                return arr"""
            if type(fun)!=type(None):
                self.zeros = False #Mandatory to remove zeros if fun is passed
                self.lab[-1] = fun(self.lab[-1])

        # Shape lab and seq
        self.lab = np.concatenate(self.lab).ravel()
        self.seq = np.concatenate(self.seq).reshape((-1,4))


        # Reverse
        """Construct reverse complement according to specified parameter
        0: Original sequence only 
        1: Reverse complement only
        2: Both reverse complement and original sequence"""
        if reverse == 1:
            self.rseq = self.seq[:,[3,2,1,0]]

            self.reverse=1

        elif reverse == 0:
            self.reverse = 0
        
        elif reverse == 2:
            self.seq = self.seq[:,[3,2,1,0]]
            self.reverse = 2
        else:
            raise ValueError("Wrong reverse value")


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

        # Apply weights
        if self.apply_weight == 2: # Discreet signal (low number of unique values)
            uni, cou = np.unique(self.lab[idx], return_counts=True)
            if self.zeros:
                w = np.max(cou[1:])/cou[1:]
                dicw = dict(zip(uni[1:],w))
            else:
                w = np.max(cou)/cou
                dicw = dict(zip(uni,w))
            dicw[0.] = 0.
            self.weights_f = dicw
        
        elif self.apply_weight == 3: #undersampling
            uni, cou = np.unique(self.lab[idx], return_counts=True)

            if self.zeros:
                min_cou = np.min(cou[1:])
            else:
                min_cou = np.min(cou)

            print(min_cou, "*", len(uni), "=", min_cou*len(uni))
            tmp_idx = []
            for i,v in enumerate(uni): #Get same number as the min class
                print(i,"/",len(uni))
                subidx = np.where(self.lab[idx]==v)[0]
                subidx = np.random.choice(idx[subidx], size=min_cou, replace=False)
                tmp_idx.append(subidx)
            idx = np.concatenate(tmp_idx)
            self.weights_f = None
        
        elif self.apply_weight == 1: #For binned signal
            _ , nbin = np.histogram(self.lab[idx], bins="doane")
            digit = np.digitize(self.lab[idx], nbin)
            uni, cou = np.unique(digit, return_counts=True)
            w = np.max(cou)/cou
            dicw = dict(zip(uni,w))
            digit = np.vectorize(dicw.__getitem__)(digit)
            self.weights_f = digit

        else:
            self.weights_f = None

        return idx


    def generate_images(self, image_idx, is_training=True, batch_size=4096):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        :param arr image_idx: index produced by generate_split_indexes
        :param bool is_training: indicate if the generator is used to train (True) or else (False)
        :param int batch_size: number of data in a single batch
        """
        assert len(image_idx)>=batch_size, "not enough index for this batch_size"
        reverse_idx = image_idx
        if self.sample:
            image_idx = np.random.choice(image_idx, replace=False, size=len(image_idx))
            reverse_idx = np.random.choice(image_idx, replace=False, size=len(image_idx))

        print(image_idx)
        print()
        # arrays to store our batched data
        if self.apply_weight==2: #Weights dict
            images, classes = [], []
            while True:
                for i,idx in enumerate(image_idx):
                    
                    try:
                        im = self.seq[int(idx)-(self.winsize//2):int(idx+(self.winsize//2))+1,:].reshape((self.winsize,4))
                        classe = self.lab[idx]
                    except ValueError:
                        continue

                    if self.reverse==1:
                        try:
                            rdx = reverse_idx[i]
                            im2 = self.rseq[int(rdx)-(self.winsize//2):int(rdx+self.winsize//2)+1,:].reshape((self.winsize,4))[::-1,:]
                            class2 = self.lab[rdx]
                        except ValueError:
                            continue
                        classes.extend([classe, class2])
                        images.extend([im,im2])

                        
                    elif self.reverse==2:
                        im = im[::-1,:]
                        classes.append(classe)
                        images.append(im)
                        

                    else:
                        classes.append(classe)
                        images.append(im)
                
                    # yielding condition
                    if len(images) >= batch_size:
                        classes = np.array(classes)
                        yield np.array(images), classes, np.vectorize(self.weights_f.__getitem__)(classes)
                        images, classes = [], []
                        
                if not is_training:
                    print("END")
                    return None
        
        elif self.apply_weight==1: #weights list
            images, classes, weights = [], [], []
            while True:
                for i,idx in enumerate(image_idx):
                    try:
                        im = self.seq[int(idx)-(self.winsize//2):int(idx+(self.winsize//2))+1,:].reshape((self.winsize,4))
                        classe = self.lab[idx]
                        weight = self.weights_f[idx]
                    except ValueError:
                        continue
                    
                    if self.reverse == 1: #reverse complement
                        try:
                            rdx = reverse_idx[i]
                            im2 = self.rseq[int(rdx)-(self.winsize//2):int(rdx+(self.winsize//2))+1,:].reshape((self.winsize,4))[::-1]
                            class2 = self.lab[rdx]
                            weight2 = self.weights_f[rdx]
                            classes.extend([classe, class2])
                            images.extend([im,im2])
                            weights.extend([weight, weight2])
                        except ValueError:
                            continue
                        

                    elif self.reverse == 2: #reverse complement only
                            im = im[::-1]
                            classes.append(classe)
                            images.append(im)
                            weights.append(weight)

                    else:
                        classes.append(classe)
                        images.append(im)
                        weights.append(weight)

                    # yielding condition
                    if len(images) >= batch_size:
                        classes = np.array(classes)
                        yield np.array(images), classes, np.array(weights)
                        images, classes, weights = [], [], []
                        
                if not is_training:
                    print("END")
                    return None

        else:
            images, classes = [], []
            while True:
                for i,idx in enumerate(image_idx):
                    try:
                        im = self.seq[int(idx)-self.winsize//2:int(idx+self.winsize//2)+1,:].reshape((self.winsize,4))
                        classe = self.lab[idx]
                    except ValueError:
                        continue
                    
                    if self.reverse == 1: #reverse complement
                        rdx = reverse_idx[i]
                        im2 = self.rseq[int(rdx)-self.winsize//2:int(rdx+self.winsize//2)+1,:].reshape((self.winsize,4))[::-1]
                        class2 = self.lab[rdx]
                        weight2 = self.weights_f[rdx]
                        classes.extend([classe, class2])
                        images.extend([im,im2])
                    
                        """
                            elif self.reverse == 2: #reverse complement only
                            im = im[::-1]
                            classes.append(classe)
                            images.append(im)
                        """



                    else:
                        classes.append(classe)
                        images.append(im)

                    # yielding condition
                    if len(images) >= batch_size:
                        classes = np.array(classes)
                        yield np.array(images), classes
                        images, classes = [], []
                        
                if not is_training:
                    print("END")
