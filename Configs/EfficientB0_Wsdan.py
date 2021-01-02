##################################################
# Training Config
##################################################
learning_rate = 1e-3        # initial learning rate
##################################################
# Model Config
##################################################
net = 'efficientnet-b0'     # feature extractor
num_attentions = 8          # number of attention maps
beta = 5e-2                 # param for update feature centers
num_classes = 2