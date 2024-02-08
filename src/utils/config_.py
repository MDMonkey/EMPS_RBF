import argparse


def load_args():

	parser = argparse.ArgumentParser()
	
    #Location
	parser.add_argument('--PATH_TRAIN', default=r"./Data/Organized/DATA_EMPS_TEST.mat")
	parser.add_argument('--PATH_TEST', default=r"./Data/Organized/DATA_EMPS_TRAIN.mat")
    
	#Pre-Processing
	parser.add_argument('--NORMALIZE_INPUT', default=False, type=bool)
	parser.add_argument('--VAL_SPLIT', default=0.2, type=float)
	parser.add_argument('--TEST_SPLIT', default=0.2, type=float)

	#RBF
	parser.add_argument('--IN_FEATURES', default=1, type=int)
	parser.add_argument('--OUT_FEATURES', default=1)
	parser.add_argument('--NUM_KERNELS', default=8, type=int)


	#Training
	parser.add_argument('--BATCH_SIZE', default=400, type=int)
	parser.add_argument('--N_EPOCHS', default=10, type=int)
	parser.add_argument('--LEARNING_RATE', default=0.01, type=float)

	

	
	args = parser.parse_args()
	

	return args