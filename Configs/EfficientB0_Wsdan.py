##################################################
# Training Config
##################################################
epochs = 60                 # number of epochs
batch_size = 16             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
net = 'efficientnet-b0'     # feature extractor
num_attentions = 8          # number of attention maps
beta = 5e-2                 # param for update feature centers
num_classes = 2

##################################################
# Dataset/Path Config
##################################################
pretrained = 'yes'
# saving directory of .ckpt models
save_dir = '/content/drive/My Drive/Colab Notebooks/Output'
model_name = 'model.ckpt'
log_name = 'train.log'
