'''
Topic:          Application of demographic handwriting rec to gender, handedness, 
                and gender + handedness classification. 
                "Individuality of Handwriting"

Data:           IAM dataset 

Model:          CNN, extract good features

Targets:        Gender, age, nationality, handedness

Problem Type:   Binary or multiclassification. 
                Binary = gender or handedness. 
                Multiclass = age intervals, nationalities. 

Forensics:      Crime scene, auto recognize anonymous piece of handwritten 
                text is a left-handed woman.  

Physocology     Handwritting and attributes of personality. 

Biometric
Security        Combine with other biometric modalities in order to improve 
                security when accessing compyter systems. 



Prior Studies   Hecker, 1996, 71.5% accuracy predicting gender
                Koppel, 2003 85% accuracy, gender,
                Bouadjenek, 2015 IAM database, 75% for IAM, used SVM
                Siddiqi, 2015, focused on features slant, orientation, 
                roundedness, curvature, neatness, legibility, writing
                texture. Classified using ANN and SVM.  73%

Questions       Does it make any difference that we are not using the same
                text (same sentence) written by difffernt authors?  Wouldn't
                that be a better test?

Thoughts:       Maybe we could rank all of the authors results in a graph
                and show where ours lands. 



Author's Proposed Approach
                Automatically detenxt features using some type of NN or CNN
                Purpose of CNN is to detect the features of the writer. 


Input:          Order-3 tensors, ie monochannel image with H rows and 
                W columns.  These inputs are processed sequentially through
                all network layers and produce as output a c-dimensional
                vector for classification with c classes. 

Preprocessing   Authors talk about the challenges associated with continous
                handwriting, especially around segmentation.
                Down sampling (lower res images)
                Max pooling: smaller kernel than the feature map
                that calculates the maximum or largest value in each patch 
                of the feature map. 
                https://computersciencewiki.org/index.php/Max-pooling_/_Pooling


Architecture    - 6 trainable layers, grouped in 2 stacks of vonclutional and 
                  subsampling (or max-pooling) layers, 2 final dense layers
                - Network receives input images with spatial resolution of 
                  30/100. 
                - Used kernels of size 5x% for the convolutional layers
                  and of 2x2 for the maxpooling layers. 
                  Smaller kernels produce worse results and bigger kernels did 
                  not improve significantly the results. 
                - Zero padding to preserve the spacial size. (smart?)
                - Relu as activation function
                - Softmax as output
                - Dropout regularization with value of 0.25 to each convolutional                  layer and 0.5 for the dense layer. 
                - Binary classification problems were trained using SGD
                  and multiclass using Adam. 

Input           - Sentence by the author.  They split the line into component                      words.  They use image processing techniques to get
                  bounding boxes around each word to separate them. 
                - Authors believe that the advantage is to give the network
                  more training examples. 
                - Somehow they combine the prediction of all words for a single
                  line of text and take the majority vote. 


Dataset         http://www.iapr-tc11.org/mediawiki/index.php?title=Datasets_List

                https://ieeexplore.ieee.org/document/6424486
                https://www.kaggle.com/c/wic2011/data


'''
