import os
import sys
global server_mode
from src.experiments import *


if __name__ == '__main__':
	if len(sys.argv)>1:
		server_mode = True if (str(sys.argv[1]) =='server') else False
	else:
		server_mode=False
	if server_mode is True :
		print("Server mode is ON!")
	# exp = NIN_Dropout(1)

	augment_rate = 0.2
	for augment_rate in [0,0.1,0.2,0.3,0.4,0.5,0.6]:
		for alpha in [0.5,1,2,3,4]:
			exp = Trainings_SimpleCIFAR10()
			exp.train_cifar10_SimpleCIFAR_labellikelihood(alpha,augment_rate)
			exp.train_cifar10_SimpleCIFAR_jointlikelihood(alpha,augment_rate)

	# exp.run()
	# exp = VGG_PMAP_CIFAR10_Try(1)
	# exp = Synthetic_PMaps(1)
	# exp.run()

