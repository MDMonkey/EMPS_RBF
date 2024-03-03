import argparse


def load_args():

	parser = argparse.ArgumentParser()
	
    #Location
	parser.add_argument('--PATH_TRAIN', default=r"./Data/Organized/DATA_EMPS_TEST.mat")
	parser.add_argument('--PATH_TEST', default=r"./Data/Organized/DATA_EMPS_TRAIN.mat")

	#Aditional settings
	parser.add_argument('--CUDA', default=True, type=bool)
	parser.add_argument('--TRANSFER_LEARNING', default=False, type=bool)
    
	#Pre-Processing
	parser.add_argument('--NORMALIZE_INPUT', default=False, type=bool)
	parser.add_argument('--VAL_SPLIT', default=0.2, type=float)
	parser.add_argument('--TEST_SPLIT', default=0.2, type=float)
	

	#RBF
	parser.add_argument('--IN_FEATURES', default=1, type=int)
	parser.add_argument('--OUT_FEATURES', default=1)
	parser.add_argument('--NUM_KERNELS', default=4, type=int)


	#Training
	parser.add_argument('--BATCH_SIZE', default=1000, type=int)
	parser.add_argument('--N_EPOCHS', default=1, type=int)
	parser.add_argument('--LEARNING_RATE', default=1e-2, type=float)

	

	
	args = parser.parse_args()
	

	return args