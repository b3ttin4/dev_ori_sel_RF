import os
# import argparse
import numpy as np
# from subprocess import Popen
import time
import sys
from itertools import product

from bettina.modeling.ori_dev_model import data_dir
from bettina.modeling.ori_dev_model.tools import misc



def runjobs_oridevModel_Cluster():
	"""
		Function to be run in slurm queuing system to start list of jobs
		
	"""
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--test", required=True, type=int, default=0)
	# args = parser.parse_args()
	# if (args.test):
	#     print ("testing commands")
	
	# Vectors for iteration
	script_name = "run_twolayer"
	SI = np.array([1.0])#([0.2,0.5,1.0,1.4])#0.5,1.0,1.5,2.5])
	RC = np.array([0.3])#([0.1,0.2,0.5,0.95])#np.array([0.2,0.3,0.5,0.8])
	r_A = np.array([0.15])
	
	betaP = np.array([0.0005])
	runtime = np.array([100000])
	saving_stepsize = np.array([2])
	avg_no_inp = np.array([1])
	gamma_lgn = np.array([0.005])

	## Recurrent layer
	pattern_duration = np.array([2500])
	nonlin = np.array(["powerlaw"]) ##(default: "rectifier")
	## 1pop: "Gaussian_inv", "Gaussian"
	## 2pop: "Gaussian2pop", "Gaussian_prob2pop", "Gaussian_prob_cluster2pop"
	## 	     "Gaussian_prob2pop","Gaussian_sparse2pop","Gaussian_prob_density2pop"
	w4to4 = np.array(["Gaussian2pop"])
	# density = np.array([0.6])
	# ncluster = np.array([3,5,10])
	Nvert = np.array([1])
	# w23to4_ampl = np.array([2./0.9,4./0.9])
	w4to23_width = np.array([0.075/0.1])


	## FF parameters
	onoff_rel_weight = np.array([1.])
	#"Mexican-hat"#Gaussian#"Gaussian_broadOFF"
	Wret_to_LGN_profile = np.array(["Gaussian"])
	# MH_strength = np.array([1.])
	arbor_profile = np.array(["gaussian"])
	# lgn_corr_het = np.array(["independent"])
	connectivity_type = np.array(["EI"])
	# Wlim = np.array([4])

	iterator_dict = {
					"sI" : SI,
					"rC" : RC,
					"rA" : r_A,
					"betaP" : betaP,
					"gamma_lgn" : gamma_lgn,
					
					"runtime" : runtime,
					"avg_no_inp" : avg_no_inp,
					"patt_dur" : pattern_duration,
					# "onoff_rel_weight" : onoff_rel_weight,

					"nonlin" : nonlin,
					"Nvert" : Nvert,
					"wrec" : w4to4,
					# "density" : density,
					# "ncluster" : ncluster,
					# "w23to4" : w23to4,
					# "w4to23" : w4to23,
					# "w23to23" : w23to23,
					# "w23to4_ampl" : w23to4_ampl,
					"w4to23_width" : w4to23_width,

					"Wret_to_LGN_profile" : Wret_to_LGN_profile,
					"arbor_profile" : arbor_profile,
					# "MH_strength" : MH_strength,
					# "lgn_corr_het"	: lgn_corr_het,
					"connectivity_type" : connectivity_type,
					# "Wlim" : Wlim,

					"saving_stepsize" : saving_stepsize,
					}


	currwd = os.getcwd()
	current_user = os.environ["USER"]
	print("currwd",currwd,current_user);sys.stdout.flush()
	if current_user=="hein":
		ofilesdir = "/home/hein/data_mpif2/code/"
		inpath = currwd + "/bettina/modeling/ori_dev_model/{}.py ".format(script_name)
	elif current_user=="bh2757":
		ofilesdir = currwd + "/projects/ori_dev_model/Ofiles/"
		# ofilesdir = "/rigel/theory/users/bh2757/columbia/projects/ori_dev_model/Ofiles/"
		# ofilesdir = "/burg/theory/users/bh2757/columbia/projects/ori_dev_model/Ofiles/"
		inpath = currwd + "/code/bettina/modeling/ori_dev_model/{}.py ".format(script_name)
		# data_dir = "/rigel/theory/users/bh2757/columbia/projects/ori_dev_model/data/"


	var_keys = sorted(iterator_dict.keys())
	n_variables = len(var_keys)
	count = 0
	var_list = []
	var_string = ""
	for key in var_keys:
		var_list.append(iterator_dict[key])
		var_string += " --{:s}".format(key) + " {}"

	count=0;
	for var_setting in product(*var_list):
		time.sleep(0.2)
		Version = misc.get_version(data_dir + "two_layer/",version=None,readonly=False)
		print("Version={}".format(Version));sys.stdout.flush()
		c1 = "{:s} --V {:.0f}".format(inpath,Version) + var_string.format(*var_setting)

		jobname="v{}_twolayer".format(Version)
		jobnameDir = os.path.join(ofilesdir, jobname)

		text_file = open(jobnameDir, "w");
		os. system("chmod u+x "+ jobnameDir)
		text_file.write("#!/bin/sh \n")


		if current_user=="hein":
			text_file.write("#SBATCH --job-name="+jobname+ "\n")
			text_file.write("#SBATCH -t 0-05:00  \n")


			# text_file.write("#SBATCH --mem=120gb \n")
			# text_file.write("#SBATCH --gres=gpu:1 \n")

			text_file.write("#SBATCH -p sleuths \n")
			text_file.write("#SBATCH --gres gpu \n")
			text_file.write("#SBATCH --mem=60gb \n")
			text_file.write("#SBATCH --nodelist=turbine \n")
			

			text_file.write("#SBATCH -o "+ ofilesdir + "%j.o # STDOUT \n")
			text_file.write("#SBATCH -e "+ ofilesdir +"%j.e # STDERR \n")
			text_file.write("source /home/hein/data_mpif2/code/bettina/tfenv/bin/activate \n") 
			text_file.write("srun python " + c1 + " \n")
			text_file.write("echo $PATH  \n")
			text_file.write("exit 0  \n")
			text_file.close()

		elif current_user=="bh2757":
			text_file.write("#SBATCH --account=theory \n")
			text_file.write("#SBATCH --job-name="+jobname+ "\n")
			text_file.write("#SBATCH -t 1-24:00  \n")## four days for 32x32

			# text_file.write("#SBATCH --mem=120gb \n")
			# text_file.write("#SBATCH --gres=gpu:1 \n")

			text_file.write("#SBATCH -c 1 \n")
			text_file.write("#SBATCH --mem=40gb \n")## 60gb for 32x32,30gb for 25x25

			text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
			text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
			# text_file.write("source /rigel/home/bh2757/.bashrc \n")
			text_file.write("source /burg/home/bh2757/.bashrc \n")
			text_file.write("module load anaconda \n")
			text_file.write("module load singularity \n")
			# text_file.write("singularity exec tensorflow_latest.sif python " + c1 + " \n")
			text_file.write("singularity exec intel-optimized-tensorflow_latest.sif python " + c1 + " \n")
			text_file.write("echo $PATH  \n")
			text_file.write("exit 0  \n")
			text_file.close()

		os.system("sbatch " + jobnameDir);
		#Popen(c1, shell=True)
		time.sleep(0.2)
		
		count += 1


	print("{} jobs started".format(count));sys.stdout.flush()


if __name__ == "__main__":
	runjobs_oridevModel_Cluster()