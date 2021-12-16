import os
import numpy as n
from matplotlib import pyplot as plt
import pickle
import numpy.lib.recfunctions as rfn
import copy
from modules import utils
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
# from tqdm import tqdm  # For a nice progress bar!

#Load the data________________________________
data_file = 'data/all_data.p'
f = open(data_file, 'rb')
all_data = pickle.load(f)
experiment = all_data['Female1']
ts = experiment['t']
annotations = experiment['ann']

#Calc some features______________________________
vectors = {'rear_1' : ('ImplantedTailBase_1', 'BackCenter_1'),
           'rear_2' : ('InteracteeTailBase_2', 'BackCenter_2'),
           'front_1': ('BackCenter_1', 'NapeCenter_1'),
           'front_2': ('BackCenter_2', 'NapeCenter_2'),
           'head_1' : ('NapeCenter_1', ('GreenTape_1', 'RedTape_1')),
           'head_2' : ('NapeCenter_2', ('YellowEar_2', 'OrangeEar_2')),
           'front_1_to_front_2' : ('NapeCenter_1', 'NapeCenter_2'),
           'front_2_to_front_1' : ('NapeCenter_2', 'NapeCenter_1'),
           'front_1_to_rear_2' : ('NapeCenter_1', 'InteracteeTailBase_2'),
           'front_2_to_rear_1' : ('NapeCenter_2', 'ImplantedTailBase_1'),
           'rear_1_to_rear_2' : ('BackCenter_1','BackCenter_2'),
           'rear_2_to_rear_1' : ('BackCenter_2','BackCenter_1')}
velocities_to_calculate = [
    'BackCenter_1',
    'BackCenter_2',
]
exp = utils.compute_and_add_vectors(experiment,vectors)
exp = utils.compute_and_add_velocities(exp, velocities_to_calculate)

#Pre_process data_________________________________________
labels = exp['ann']
unique_labels = n.unique(labels)
ts = exp['t'] #time steps
features_to_use = [
    'rear_1_ang',
    'front_1_ang',
    'head_1_ang',
    'rear_1_ang',
    'front_1_ang',
    'head_1_ang',
    'BackCenter_1_vel_mag',
    'BackCenter_1_vel_ang',
    'rear_2_ang',
    'front_2_ang',
    'head_2_ang',
    'rear_2_ang',
    'front_2_ang',
    'head_2_ang',
    'BackCenter_2_vel_mag',
    'BackCenter_2_vel_ang',
]
n_timepoints = len(ts)
n_features = len(features_to_use)
n_labels = len(unique_labels)
Xs = n.zeros((n_timepoints, n_features))
Ys = n.zeros(n_timepoints)

#process label set
label_encoder = preprocessing.LabelEncoder() #Convert non-numerical labels into numberical ones
label_encoder.fit(unique_labels) #Convert non-numerical labels into numberical ones
Ys = label_encoder.transform(labels) #Convert non-numerical labels into numberical ones

#Process feature set
for i in range(n_features):
    Xs[:,i] = exp[features_to_use[i]]
Xs[n.isnan(Xs)] = 0 #Nans are converted into zeros, could also drop the frames

#split into train and test for cross val
n_splits = 2
splitter = StratifiedKFold(n_splits=n_splits, shuffle=False)

#Plot data split
# i = 0
# plt.figure(figsize=(10,1))
# for train, test in splitter.split(Xs, Ys):
#     plt.scatter(train, [i]*len(train),s=1, color='navy')
#     plt.scatter(test, [i]*len(test),s=1, color='salmon')
#     i += 1
# plt.show()

#Build architecture of neural net______________________________________________
#Check GPU is set up
# # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
# is_cuda = torch.cuda.is_available()
#
# # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

#Build architecture
# Hyperparameters
hidden_size = 256
num_layers = 2
# sequence_length = 16
N_batch = 64
n_epochs = 50
learning_rate = 1e-4
print_interval = 50
train_scores = []
test_scores = []

#Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0) # _ is hidden state output but ignore
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return(out)

#Initiallize network
model = RNN(Xs.shape[1] , hidden_size, num_layers, n_labels)
Ys_pred = n.zeros(Ys.shape)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to tensors
Xs = torch.tensor(Xs).float()
Ys = torch.tensor(Ys)
Ys = Ys.type(torch.LongTensor)

train_loss = []

for train_idxs,test_idxs in splitter.split(Xs,Ys):

    #Split training data and test data
    Xs_train = Xs[train_idxs]
    Ys_train = Ys[train_idxs]
    Xs_test = Xs[test_idxs]
    Ys_test = Ys[test_idxs]

    # Convert data to tensors
    features_train = torch.tensor(Xs_train).float()
    features_test  = torch.tensor(Xs_test).float()
    targets_train  = torch.tensor(Ys_train).type(torch.LongTensor)
    targets_test   = torch.tensor(Ys_test).type(torch.LongTensor)

    # targets_train = torch.tensor(Y_train)
    # targets_train = targets_train.type(torch.LongTensor)
    # targets_test = torch.tensor(Y_test)
    # targets_test = targets_test.type(torch.LongTensor)

    for epoch in range(n_epochs):
        epoch_loss = []
        for i in range(N_batch):
            local_X, local_y = Xs[i*N_batch:(i+1)*N_batch,], Ys[i*N_batch:(i+1)*N_batch,]

            # Get data to cuda if possible
            data = local_X.unsqueeze(1)
            targets = local_y

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()
            train_loss.append(loss.detach().numpy())
            epoch_loss.append(loss.detach().numpy())
        epoch_loss = n.average(epoch_loss)
        print("Epoch #:{}, Test Loss Avg:{}".format(epoch + 1, epoch_loss))

plt.plot(train_loss)
plt.show()

# for train_idxs,test_idxs in splitter.split(Xs,Ys):
#
#     # Load scaled data
#     X_train = Xs[train_idxs]
#     Y_train = Ys[train_idxs]
#
#     X_test = Xs[test_idxs]
#     Y_test = Ys[test_idxs]
#
#     # scaler = preprocessing.StandardScaler().fit(X_train)
#     # X_train_scaled = scaler.transform(X_train)
#     # X_test_scaled = scaler.transform(X_test)
#
#     # Set up the model
#     trackers = {'test_loss' : [], 't' : [], 'train_loss' : [], 'test_frac_correct': []}
#     best_model = None
#
#     def track():
#         for key in trackers.keys():
#             trackers[key].append(globals()[key])
#
#     t = 0
#     end = False
#
#     # Convert data to tensors
#     features_train = torch.tensor(X_train).float()
#     features_test = torch.tensor(X_test).float()
#
#     targets_train = torch.tensor(Y_train)
#     targets_train = targets_train.type(torch.LongTensor)
#
#     targets_test = torch.tensor(Y_test)
#     targets_test = targets_test.type(torch.LongTensor)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # Train Network
    # for epoch in range(n_epochs):
    #     for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
    #         # Get data to cuda if possible
    #         data = data.to(device=device).squeeze(1)
    #         targets = targets.to(device=device)
    #
    #         # forward
    #         scores = model(data)
    #         loss = criterion(scores, targets)
    #
    #         # backward
    #         optimizer.zero_grad()
    #         loss.backward()
    #
    #         # gradient descent update step/adam step
    #         optimizer.step()


#     #Batch processing
#     for epoch_idx in range(n_epochs):
#         print("### EPOCH {:2d} ###".format(epoch_idx))
#         indices = n.random.choice(range(targets_train.shape[0]), targets_train.shape[0], False)
#         num_batches = len(indices) // N_batch + 1
#         for batch_idx in range(num_batches):
#             batch_train_x = features_train[indices[batch_idx*N_batch :(batch_idx+1)*N_batch]]
#             batch_train_y = targets_train[indices[batch_idx*N_batch : (batch_idx+1)*N_batch]]
#             batch_train_x = batch_train_x.to(device=device).squeeze(1)
#             batch_train_y = batch_train_y.to(device=device)
#
#             #prints
#             print('Input size dimensions', batch_train_y.size())
#
#             #Forward pass
#             batch_train_y_pred = model(batch_train_x)
#             loss = criterion(batch_train_y_pred, batch_train_y)
#
#             # backward
#             optimizer.zero_grad()
#             loss.backward()
#
#             # gradient descent update step/adam step
#             optimizer.step()
#
#             #Printing
#             if batch_idx % print_interval == 0:
#                 prediction_test = model(features_test)
#                 train_loss = loss.item()
#                 test_loss= loss_fn(prediction_test, targets_test).item()
#
#
#                 pred_labels = n.argmax(prediction_test.detach().numpy(),axis=1)
#                 true_labels = targets_test.detach().numpy()
#                 correct_preds = n.array(pred_labels == true_labels, n.int)
#                 test_frac_correct = n.mean(correct_preds)
#
#                 if len(trackers['test_loss']) == 0 or test_loss < min(trackers['test_loss']):
#                     best_model = copy.deepcopy(model)
#                 print("Batch {:3d}, Train Loss: {:5f}, Test Loss: {:5f}, Test Correct Frac: {:.3f}".format(batch_idx, train_loss, test_loss, test_frac_correct))
#                 track()
#
#             t += 1
#         if end: break
#
#     train_score = 0
#     test_score = test_frac_correct
#     prediction_test = best_model(features_test)
#
#     pred = n.argmax(prediction_test.detach().numpy(),axis=1)
#
# #     true = targets_test.detach().numpy()
#     Ys_pred[test_idxs] = pred
#
#     train_scores.append(train_score)
#     test_scores.append(test_score)
#
#     plt.plot(trackers['t'],trackers['train_loss'], label='train')
#     plt.plot(trackers['t'],trackers['test_loss'], label='test')
#     plt.title("Cross Entropy Loss through training")
#     plt.legend()
#     plt.show()
#     plt.plot(trackers['t'], trackers['test_frac_correct'])
#     plt.title("Fraction of correct labels on test set through training")
#     plt.show()
#
# #Plot confusion matrix
# plt.imshow(confusion_matrix((Ys),(Ys_pred), normalize='true'), vmin=0)
# # plt.xticks(unique_labels)
# plt.colorbar()
# plt.xlabel("Predicted label")
# plt.ylabel("True Label")
# plt.xticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=90);
# plt.yticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=0);
# plt.show()
