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

#Load the data________________________________
data_file = 'data/all_data.p'
f = open(data_file, 'rb')
all_data = pickle.load(f)
experiment = all_data['Female1']
# Save timestamps into variable
ts = experiment['t']
# Save annotations into variable
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
#print(list(exp.dtype.fields.keys())) #Print the feature list

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
# Set the architecture parameters
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in = Xs.shape[1]
D_out = n_labels
hidden_dims = [300,300,300]

# Set the learning parameter
N_batch = 16
n_epochs = 3
learning_rate = 1e-4
print_interval = 50

train_scores = []
test_scores = []
Ys_pred = n.zeros(Ys.shape)
for train_idxs,test_idxs in splitter.split(Xs,Ys):

    # Load scaled data
    X_train = Xs[train_idxs]
    Y_train = Ys[train_idxs]

    X_test = Xs[test_idxs]
    Y_test = Ys[test_idxs]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up the model
    trackers = {'test_loss' : [], 't' : [], 'train_loss' : [], 'test_frac_correct': []}
    best_model = None
    def track():
        for key in trackers.keys():
            trackers[key].append(globals()[key])
    layers = []
    prev_dim = D_in
    for dim in hidden_dims:
        layers += [torch.nn.Linear(prev_dim, dim),
                   torch.nn.ReLU()]
        prev_dim = dim
    layers += [torch.nn.Linear(prev_dim, D_out)]

    loss_fn = torch.nn.CrossEntropyLoss()
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    t = 0
    end = False

    # Convert data to tensors
    features_train = torch.tensor(X_train).float()
    targets_train = torch.tensor(Y_train)
    targets_train = targets_train.type(torch.LongTensor)
    features_test = torch.tensor(X_test).float()
    targets_test = torch.tensor(Y_test)
    targets_test = targets_test.type(torch.LongTensor)

    for epoch_idx in range(n_epochs):
        print("### EPOCH {:2d} ###".format(epoch_idx))
        # Figure out what batch to look at
        indices = n.random.choice(range(targets_train.shape[0]), targets_train.shape[0], False)
        num_batches = len(indices) // N_batch + 1
        for batch_idx in range(num_batches):
            # extract the batch
            batch_train_x = features_train[indices[batch_idx*N_batch :(batch_idx+1)*N_batch]]
            batch_train_y = targets_train[indices[batch_idx*N_batch : (batch_idx+1)*N_batch]]
            # Forward pass: compute predicted y by passing x to the model.
            batch_train_y_pred = model(batch_train_x)

            # Compute and print loss.
            loss = loss_fn(batch_train_y_pred, batch_train_y)
            if batch_idx % print_interval == 0:
                prediction_test = model(features_test)
                train_loss = loss.item()
                test_loss= loss_fn(prediction_test, targets_test).item()


                pred_labels = n.argmax(prediction_test.detach().numpy(),axis=1)
                true_labels = targets_test.detach().numpy()
                correct_preds = n.array(pred_labels == true_labels, n.int)
                test_frac_correct = n.mean(correct_preds)

                if len(trackers['test_loss']) == 0 or test_loss < min(trackers['test_loss']):
                    best_model = copy.deepcopy(model)
                print("Batch {:3d}, Train Loss: {:5f}, Test Loss: {:5f}, Test Correct Frac: {:.3f}".format(batch_idx, train_loss, test_loss, test_frac_correct))
                track()

            t += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if end: break

    train_score = 0
    test_score = test_frac_correct
    prediction_test = best_model(features_test)

    pred = n.argmax(prediction_test.detach().numpy(),axis=1)

#     true = targets_test.detach().numpy()
    Ys_pred[test_idxs] = pred

    train_scores.append(train_score)
    test_scores.append(test_score)

    plt.plot(trackers['t'],trackers['train_loss'], label='train')
    plt.plot(trackers['t'],trackers['test_loss'], label='test')
    plt.title("Cross Entropy Loss through training")
    plt.legend()
    plt.show()
    plt.plot(trackers['t'], trackers['test_frac_correct'])
    plt.title("Fraction of correct labels on test set through training")
    plt.show()

#Plot confusion matrix
plt.imshow(confusion_matrix((Ys),(Ys_pred), normalize='true'), vmin=0)
# plt.xticks(unique_labels)
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True Label")
plt.xticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=90);
plt.yticks(n.arange(len(unique_labels)),label_encoder.inverse_transform(n.arange(len(unique_labels))), rotation=0);
plt.show()
