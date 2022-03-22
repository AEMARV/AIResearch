import os
import sys
global server_mode
from src.experiments import *

if __name__ == '__main__':


	# exp = NIN_Dropout(1)
	''' Detect panc on NIH
	1. create
	
	'''
	''' Detect Panc on Mayo'''
	''' Train EBM on NIH and MAYO'''
	''' TEST Phase'''
	desc = "This experiment is designed to test variation of alpha param and multiple loss functions" \
		   "\nThe loss functions are as followed:\n" \
		   "\n1. Cross Entropy Conditional/Joint\n" \
		   "\nCross Entropy means that each sample affects the network independantly\n" \
		   "\nCond/Joint reflect that the network is trained on input/label ( joint) or just the label (Cond) " \
		   "" \
		   "\n2. Probabilistic Conditional/Joint" \
		   "\nIn Probabilistic Training each sample affects the gradient based on the probability of correctness." \
		   "\n In other words, the marginalized probability, is optimized" \
		   "" \
		   "" \
		   "In addition to the previous objecrtive function a v2 for Joint version is experimented:\n" \
		   "where in hyper normalization as opposed to the original, the original input is not used."
	augment_rate = 0
	datasets= [(nih_pancreas,2)]
	alpha_list=[2,4,16,1.5,2.5,3,3.5]
	l1_list = [0]
	l2_list = l1_list
	exp_list = []
	for trial in range(1):
		for batch_size in [1]:
			for alpha in alpha_list:
				for numlayers in [8,10,12,14]:
					for dataset,classnum in datasets:
						for scale in [1,2]:
							for lr in [0.001,0.005]:
								for init_coef in [0.1,0.01,0.001,1]:
									for optimizer in [Subset_Seg_Balanced]:
										for model in [
													Panc_Seg_Bottled,
													# Panc_Segmentation,
													  ]:
											exp_desc = dict(trial=trial,
															lr=lr,
															l2=0,
															l1=0,
															init_coef=init_coef,
															filter_scale=scale,
															momentum=0.9,
															batchsz=batch_size,
															alpha=alpha,
															model=model,
															num_layer=numlayers,
															dataset=dataset,
															classnum=classnum,
															optimizer=optimizer)
										exp_list = exp_list +[exp_desc]

	worker_id= 1
	total_workers=1
	if sys.argv.__len__()>1:
		worker_id= int(sys.argv[1])-1
		total_workers= int(sys.argv[2])
		exp_list=  exp_list[worker_id::total_workers]
		print("The script is running with batch sz "+ str(alpha_list))

	print(exp_list.__len__())
	for exp_dict in exp_list:
		exp = Trainings_SimpleCIFAR10('PancSegmentBottled', description=desc,worker_id=worker_id)
		exp.generic_train(**exp_dict)
	#





	# exp.run()
	# exp = VGG_PMAP_CIFAR10_Try(1)
	# exp = Synthetic_PMaps(1)
	# exp.run()

