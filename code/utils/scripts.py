import pickle

cnn_file = "C:\\Users\\vedire\\Desktop\\Stack-GANs-for-Text-to-Image-Synthesis\\data\coco\\train\\char-CNN-RNN-embeddings.pickle"
char_cnn_subfile = "C:\\Users\\vedire\\Desktop\\Stack-GANs-for-Text-to-Image-Synthesis\\data\\coco\\train\\char-CNN-RNN-embeddings-subset.pickle"
data = []
file = open(char_cnn_subfile, 'wb')
with open(cnn_file, 'rb') as f:
    embeddings = pickle.load(f,encoding='latin1')
for i in range(0,10):
    data.append(embeddings[i])
pickle.dump(data, file)



