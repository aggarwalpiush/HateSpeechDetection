from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np

def load_emb(json_file, hdf5_file):
	vlmo = Elmo(json_file, hdf5_file, 1, dropout=0) 
	return vlmo


def word_to_sentence(embeddings):
	return embeddings.mean(axis=1)

def embed(sentences,loaded_emb):
	# use batch_to_ids to convert sentences to character ids
	sents = []

	for sent in sentences:
		
		sents.append(sent.split())
	

	character_ids = batch_to_ids(sents)# 2,2,50
	
	embeddings = loaded_emb(character_ids)

	embeddings = embeddings['elmo_representations'][0]
	
	embeddings = embeddings.data.numpy()
	
	return embeddings

def get_embeddings(sentences,loaded_vlmo):
	
	emb = embed(sentences,loaded_vlmo)
	
	return word_to_sentence(emb)
def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
	
	json_file = "./VELMO.30k.1ep/velmo_options.json"

	hdf5_file = "./VELMO.30k.1ep/velmo_weights.hdf5"
	import sys

	sentences = [sys.argv[1], sys.argv[2]]
	
	loaded_vlmo = load_emb(json_file,hdf5_file)
	
	em = embed(sentences,loaded_vlmo)
	(vec_orig, vec_pert) = get_embeddings(sentences,loaded_vlmo)
	print("%s,%s,%.2f"%(sentences[0], sentences[1],  cos_sim(vec_orig, vec_pert)))
	#from scipy.spatial.distance import cdist

	#print(1. - cdist(list(em[0][0]), list(em[1][0]), 'cosine'))
	#from sklearn.metrics.pairwise import cosine_similarity
	
	#print(cosine_similarity(np.array(list(em[0][0])), np.array(list(em[1][0]))))
	#print(list(em[0][0]))
	#print(list(em[1][0]))
	#print(em.shape)

	#print(em[:,:,:5])
