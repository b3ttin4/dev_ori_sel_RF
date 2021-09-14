#!/usr/bin/python

"""
layer 4 receiving ff input from lgn receiving input from retina
"""

import os
import sys
import numpy as np
import tensorflow as tf
import logging
from scipy import linalg

from bettina.modeling.ori_dev_model import integrator_tf, connectivity,\
dynamics,data_dir,network
from bettina.modeling.ori_dev_model.tools import misc



if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def set_tf_loglevel(level):
	if level >= logging.FATAL:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	if level >= logging.ERROR:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	if level >= logging.WARNING:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
	else:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
	logging.getLogger('tensorflow').setLevel(level)

set_tf_loglevel(logging.FATAL)

def parameter_sweep_twolayer(Version,config_dict,**kwargs):
	## Parameters
	Nvert = config_dict["Nvert"]
	N4 = config_dict["N4"]
	N23 = config_dict["N23"]
	Nlgn = config_dict["Nlgn"]
	Nret = config_dict["Nret"]
	dt = config_dict["dt"]
	t = np.arange(0,config_dict["Inp_params"]["pattern_duration"],1).astype("float64")

	last_timestep = config_dict["runtime"]/dt
	## number of input patterns needed
	T_pd = config_dict["Inp_params"]["pattern_duration"]
	T_exp = config_dict["Inp_params"]["expanse_time"]
	config_dict["Inp_params"].update({"Nsur" : int(np.ceil(1.*last_timestep/T_pd/(T_exp+1) ))})
	# print("# of stimuli: {}".format(config_dict["Inp_params"]["Nsur"]));sys.stdout.flush()

	config_dict.update({
					"last_timestep" : last_timestep,
					"RF_mode" : "initialize",
					"system" : "two_layer",
					"Version" : Version,
					})
	n = network.Network(Version,config_dict)
	Wret_to_lgn,Wlgn_to_4,arbor,arbor2,init_weights,W4to4,W23to23,W4to23,arbor4to23,\
		init_weights_4to23,W23to4 = n.system
	Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
	max_ew = config_dict["W4to4_params"]["max_ew"]
	arbor_hstack = np.hstack([arbor4to23,0*arbor4to23])
	arbor4to23_full = np.concatenate([arbor_hstack,arbor_hstack])


	### Normalisation projector for ff connectivity
	if config_dict["normalisation_mode"]=="xalpha":
		arbor_profile = "heaviside"#config_dict["Wlgn_to4_params"]["arbor_profile"]
		try:
			constraint_vec = np.load(data_dir + "layer4/P_orth/N4{}_Nlgn{}{}_rA{}_{}.npy".format(\
									 N4,Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
									 np.around(config_dict["Wlgn_to4_params"]["r_A"],2),\
									 arbor_profile))
			c_orth = constraint_vec[:constraint_vec.shape[0]//2,:]
			s_orth = constraint_vec[constraint_vec.shape[0]//2:,:]
		except Exception as e:
			print(e)
			print("No file found for projection operator\n generating operator now...")
			sys.stdout.flush()
			c_orth,s_orth = dynamics.generate_simIO_normalisation(Nlgn,N4,arbor,Nvert=Nvert)
			np.save(data_dir + "layer4/P_orth/N4{}_Nlgn{}{}_rA{}_{}.npy".format(N4,\
					Nlgn,"" if Nvert==1 else "_Nvert{}".format(Nvert),
					np.around(config_dict["Wlgn_to4_params"]["r_A"],2),arbor_profile),\
					np.concatenate([c_orth,s_orth]))
	else:
		c_orth,s_orth = np.array([]),np.array([])


	### Normalisation projector for connectivity from L4 to L23
	if config_dict["W4to23_params"]["plastic"]:
		arbor_profile = "heaviside"#config_dict["W4to23_params"]["arbor_profile"]
		r_A = config_dict["W4to23_params"]["r_A"]
		try:
			constraint_vec = np.load(data_dir + "two_layer/P_orth/N23{}_N4{}{}_rA{}_{}.npy".format(\
									 N23,N4,"" if Nvert==1 else "_Nvert{}".format(Nvert),\
									 np.around(r_A,2),arbor_profile))
			c_orth_4to23 = constraint_vec[:constraint_vec.shape[0]//2,:]
			s_orth_4to23 = constraint_vec[constraint_vec.shape[0]//2:,:]
		except Exception as e:
			print(e)
			print("No file found for projection operator\n generating operator now...")
			sys.stdout.flush()
			c_orth_4to23,s_orth_4to23 = dynamics.generate_simIO_normalisation_oneUnittype(N4,\
										N23,arbor4to23,Nvert=1)
			np.save(data_dir + "two_layer/P_orth/N23{}_N4{}{}_rA{}_{}.npy".format(N23,\
					N4,"" if Nvert==1 else "_Nvert{}".format(Nvert),
					np.around(r_A,2),arbor_profile),np.concatenate([c_orth_4to23,s_orth_4to23]))
	else:
		c_orth_4to23,s_orth_4to23 = np.array([]),np.array([])
	
	# exit()
	##================================= initialization ====================================
	# tf.random.set_seed(20200128)
	tf.random.set_seed(config_dict["random_seed"]*113)
	l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=1, dtype=tf.float32)*0.1
	l230 = tf.random.uniform([N23*N23*2], minval=0, maxval=1, dtype=tf.float32)*0.1
	
	if config_dict["tau"]!=1:
		tau = np.ones((N4**2*2*Nvert),dtype=float)
		tau[N4**2*Nvert:] *= config_dict["tau"]
	else:
		tau = 1.
		
	## run network
	params_dict = {
				"Version" : tf.constant(Version, dtype=tf.int32),
				"Nlgn" : tf.constant(Nlgn, dtype=tf.int32),
				"N4" : tf.constant(N4, dtype=tf.int32),
				"N23" : tf.constant(N23, dtype=tf.int32),
				"Nret" : tf.constant(Nret, dtype=tf.int32),
				"Nvert" : tf.constant(Nvert, dtype=tf.int32),
				
				"Wret_to_lgn" : tf.convert_to_tensor(Wret_to_lgn,dtype=tf.float32),
				"init_weights" : tf.convert_to_tensor(init_weights,dtype=tf.float32),
				"W4to4" : tf.convert_to_tensor(W4to4, dtype=tf.float32),
				"W23to23" : tf.convert_to_tensor(W23to23, dtype=tf.float32),
				"W4to23" : tf.convert_to_tensor(W4to23, dtype=tf.float32),
				"W23to4" : tf.convert_to_tensor(W23to4, dtype=tf.float32),
				"init_weights_4to23" : tf.convert_to_tensor(init_weights_4to23,dtype=tf.float32),
				
				"arbor" : tf.convert_to_tensor(arbor,name="arbor",dtype=tf.float32),
				"arbor2" : tf.convert_to_tensor(arbor2,dtype=tf.float32),
				"arbor4to23" : tf.convert_to_tensor(arbor4to23,dtype=tf.float32),
				"arbor4to23_full" : tf.convert_to_tensor(arbor4to23_full,dtype=tf.float32),
				
				"gamma_4" : tf.constant(config_dict["gamma_4"], dtype=tf.float32),
				"gamma_lgn" : tf.constant(config_dict["gamma_lgn"], dtype=tf.float32),
				"beta_P" : tf.constant(config_dict["beta_P"], dtype=tf.float32),
				"beta_O" : tf.constant(config_dict["beta_O"], dtype=tf.float32),
				"tau" : tf.convert_to_tensor(tau, name='tau', dtype=tf.float32),
				"Wlim" : tf.constant(config_dict["Wlgn_to4_params"]["Wlim"], dtype=tf.float32),
				"pattern_duration" : tf.constant(T_pd,dtype=tf.float64),
				"expanse_time" : tf.constant(T_exp,dtype=tf.float64),
				"avg_no_inp" : config_dict["Inp_params"]["avg_no_inp"],
				
				"normalisation_mode" : tf.constant(config_dict["normalisation_mode"]),
				"c_orth" : tf.convert_to_tensor(c_orth,dtype=tf.float32),
				"s_orth" : tf.convert_to_tensor(s_orth,dtype=tf.float32),
				"c_orth_4to23" : tf.convert_to_tensor(c_orth_4to23,dtype=tf.float32),
				"s_orth_4to23" : tf.convert_to_tensor(s_orth_4to23,dtype=tf.float32),

				"integrator" : config_dict["integrator"],
				"config_dict" : config_dict,
				}

	s = N4*N4*Nlgn*Nlgn*Nvert
	print("Starting simulation. This might take a while...")
	print("...")
	sys.stdout.flush()
	if config_dict["W4to23_params"]["plastic"]:
		y0 = tf.concat([Wlgn_to_4.flatten(), l40, l230, W4to23.flatten()], axis=0)
	else:
		y0 = tf.concat([Wlgn_to_4.flatten(), l40, l230], axis=0)
	yt,l4_23_t = integrator_tf.odeint_new(dynamics.dynamics_twolayer,\
											y0, t, dt, params_dict, mode="dynamic")
	print("yt",yt.shape,l4_23_t.shape);sys.stdout.flush()
	l4t = l4_23_t[:,:2*N4**2*Nvert]
	l23t = l4_23_t[:,2*N4**2*Nvert:]
	y = yt[-1,:]
	l4 = l4t[-1,:]
	l23 = l23t[-1,:]
	print("CHEKC RESULT",l4t.shape,l23t.shape,y.shape,l4.shape,l23.shape)
	#################################################################################
	############################# SAVE PARAMS AND DATA ##############################
	if not os.path.exists(data_dir + "two_layer/v{v}".format(v=Version)):
		os.makedirs(data_dir + "two_layer/v{v}".format(v=Version))
	print("Version",Version,l4t.shape,yt.shape,yt[:,2*s+2*N4**2*Nvert:].shape)
	sys.stdout.flush()
	
	data_dict_time = {
					"Wt"		:	yt[:,:config_dict["num_lgn_paths"]*s],\
					"l4t"		:	l4t,\
					"l23t"		:	l23t,
					}
	if config_dict["W4to23_params"]["plastic"]:
		L23_size = N23**2*2
		L4_size = N4**2*2*Nvert
		print("w4to23.shape",yt[:,config_dict["num_lgn_paths"]*s+L4_size+L23_size:].shape)
		data_dict_time.update({\
			"W4to23t" : yt[:,config_dict["num_lgn_paths"]*s+L4_size+L23_size:]})
	filename = "two_layer/v{v}/yt_v{v}.npz".format(v=Version)
	misc.save_data(Version, filename, data_dict_time)
	## save ff connections and activity of last timestep separately
	filename = "two_layer/v{v}/y_v{v}.npz".format(v=Version)
	data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "l4" : l4, "l23" : l23}
	if config_dict["W4to23_params"]["plastic"]:
		data_dict.update({"W4to23" : y[config_dict["num_lgn_paths"]*s+L4_size+L23_size:]})
	misc.save_data(Version, filename, data_dict)


	filename = "two_layer/v{v}/config_v{v}".format(v=Version)
	misc.save_params(Version,filename,config_dict)
	#################################################################################
	#################################################################################




if __name__=="__main__":
	from bettina.modeling.ori_dev_model import config_dict
	from bettina.modeling.ori_dev_model.tools import parse_args,update_params_dict

	args_dict = vars(parse_args.args)
	print("args_dict",args_dict)

	default_dict = config_dict
	default_dict = update_params_dict.update_params_dict(default_dict,args_dict)

	if args_dict["V"] is not None:
		Version = args_dict["V"]
	else:
		Version = misc.get_version(data_dir + "layer4/",version=None,readonly=False)

	print("Version",Version)
	print("args_dict",args_dict)
	print(" ")
	print("config_dict, Wret_to_lgn_params",default_dict["Wret_to_lgn_params"])
	print("config_dict, W4to4_params",default_dict["W4to4_params"])

	print(" ")
	print("config_dict, Wlgn_to4_params",default_dict["Wlgn_to4_params"])
	print(" ")


	default_dict["W4to4_params"]["sigma_EE"] = 0.1
	default_dict["W4to4_params"]["sigma_IE"] = 0.1
	default_dict["W4to4_params"]["sigma_EI"] = 0.07
	default_dict["W4to4_params"]["sigma_II"] = 0.07
	default_dict["W4to4_params"]["aEE"] = 11.4
	default_dict["W4to4_params"]["aIE"] = 11.
	default_dict["W4to4_params"]["aEI"] = 10.9
	default_dict["W4to4_params"]["aII"] = 10.2

	default_dict["W4to4_params"]["max_ew"] = 0.3
	default_dict["W23_params"]["max_ew"] = 0.9#0.013#
	default_dict["W4to23_params"]["max_ew"] = "orig"#10
	default_dict["W23to4_params"]["max_ew"] = "orig"#2
	default_dict["tau"] = 0.25
	config_dict["W4to23_params"]["sigma_EE"] *= 15
	config_dict["W4to23_params"]["sigma_IE"] *= 15
	print("W4to23 width",config_dict["W4to23_params"]["sigma_EE"])
	
	print("4",default_dict["W4to4_params"])
	print("23",default_dict["W23_params"])
	print("23to4",default_dict["W23to4_params"])
	print("4to23",default_dict["W4to23_params"])
	print("default_dict",default_dict["tau"],default_dict["gamma_lgn"])
	parameter_sweep_twolayer(Version,default_dict,**args_dict)
	print("done")