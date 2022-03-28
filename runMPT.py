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
	exp_list = []
	datasets= [(cifar10,10),(cifar100,100),(mnist,10)]

	alpha_list=[1,2,3,4,5,6]
	l1_list = [0]
	l2_list = l1_list

	for trial in range(2):
		for batch_size in [128]:
			for alpha in alpha_list:
				for numlayers in [8,12,16]:
					for dataset,classnum in datasets:
						for scale in [1]:
							for l1 in l1_list:
								for l2 in l2_list:
									for lr in [0.01,0.001,0.0001]:
										for init_coef in [1,0.1,10]:
											for optimizer in [SGD]:
												for loss in [Conditional_Cross,
															 Joint_Cross,
															 Joint_Intersection_Subset,
															 Joint_Intersection_Indpt,
															 ]:
													for model in [
														BottleNet,
														SimpleCIFAR10,
														resnet_cifar_nmnist
																  ]:
														exp_desc = dict(trial=trial,
																		lr=lr,
																		l2=l2,
																		l1=l1,
																		init_coef=init_coef,
																		filter_scale=scale,
																		momentum=0.9,
																		batchsz=batch_size,
																		alpha=alpha,
																		model=model,
																		num_layer=numlayers,
																		dataset=dataset,
																		classnum=classnum,
																		optimizer=optimizer,
																		loss=loss)
													exp_list = exp_list +[exp_desc]






	worker_id= 1
	total_workers=1
	print("Experimenting on total of %d settings"%exp_list.__len__())
	if sys.argv.__len__()>1:
		worker_id= int(sys.argv[1])-1
		total_workers= int(sys.argv[2])
		exp_list=  exp_list[worker_id::total_workers]
	print("This worker is experimenting on total of %d settings"%exp_list.__len__())
	for exp_dict in exp_list:
		exp = Trainings_SimpleCIFAR10('MassMPTExp', description=desc,worker_id=worker_id)
		exp.generic_train(**exp_dict)
	#





	# exp.run()
	# exp = VGG_PMAP_CIFAR10_Try(1)
	# exp = Synthetic_PMaps(1)
	# exp.run()

