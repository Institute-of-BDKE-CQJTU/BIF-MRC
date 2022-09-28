import pickle

f = open('./data/XSJModel/bridge/train_43.pkl', 'rb')
features = pickle.load(f)
print(len(features[0][0]))