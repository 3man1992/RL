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
import copy

best_model = None

#Load the data________________________________
data_file = 'data/all_data.p'
f = open(data_file, 'rb')
all_data = pickle.load(f)
names = ['Female1', 'Female2', 'Female4', 'Male1', 'Male2', 'Male3']

print(all_data.keys())

for x in range(len(all_data)):
    print(names[x])
    experiment = names[x]
    experiment = all_data[experiment]
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

    #Build architecture of neural net______________________________________________
    # Hyperparameters
    hidden_size = 128
    num_layers = 2
    # sequence_length = 16
    N_batch = 64
    n_epochs = 50
    learning_rate = 0.01
    print_interval = 5
    train_scores = []
    test_scores = []

    #architecture
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.rnn(x, h0,) # _ is hidden state output but ignore
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

        # Set up the model
        trackers = {'test_loss' : [], 't' : [], 'train_loss' : [], 'test_frac_correct': []}
        def track():
            for key in trackers.keys():
                trackers[key].append(globals()[key])
        t=0

        for epoch in range(n_epochs):
            epoch_loss = []
            for i in range(N_batch):
                local_X, local_y = features_train[i*N_batch:(i+1)*N_batch,], targets_train[i*N_batch:(i+1)*N_batch,]

                # Get data to cuda if possible
                data = local_X.unsqueeze(1)
                targets = local_y

                # forward
                if best_model:
                    scores = best_model(data)
                else:
                    scores = model(data)
                loss = criterion(scores, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent update step/adam step
                optimizer.step()
                t += 1
            #Print interval
            if epoch % print_interval == 0:
                prediction_test = model(features_test.unsqueeze(1))
                train_loss = loss.item()
                test_loss = criterion(prediction_test, targets_test).item()
                pred_labels = n.argmax(prediction_test.detach().numpy(),axis=1)
                true_labels = targets_test.detach().numpy()
                correct_preds = n.array(pred_labels == true_labels, n.int)
                test_frac_correct = n.mean(correct_preds)

                if len(trackers['test_loss']) == 0 or test_loss < min(trackers['test_loss']):
                    best_model = copy.deepcopy(model)
                print("Epoch {:3d}, Train Loss: {:5f}, Test Loss: {:5f}, Test Correct Frac: {:.3f}".format(epoch, train_loss, test_loss, test_frac_correct))
                track()

        prediction_test = best_model(features_test.unsqueeze(1))
        pred = n.argmax(prediction_test.detach().numpy(),axis=1)
        Ys_pred[test_idxs] = pred
        # plt.plot(trackers['t'],trackers['train_loss'], label='train')
        # plt.plot(trackers['t'],trackers['test_loss'], label='test')
        # plt.title("Cross Entropy Loss through training")
        # plt.legend()
        # plt.show()
        # plt.plot(trackers['t'], trackers['test_frac_correct'])
        # plt.title("Fraction of correct labels on test set through training")
        # plt.show()
#Plot confusion matrix
plt.imshow(confusion_matrix((Ys),(Ys_pred), normalize='true'), vmin=0)
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.xticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=90);
plt.yticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=0);
plt.show()
