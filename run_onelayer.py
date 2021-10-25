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

from bettina.modeling.ori_dev_model import integrator_tf,\
dynamics, data_dir, network
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


# def parameter_sweep_layer4(Version,sigma_rec,sigma_cc,r_A):
def parameter_sweep_layer4(Version,config_dict,**kwargs):
	## Parameters
	Nvert = config_dict["Nvert"]
	N4 = config_dict["N4"]
	Nlgn = config_dict["Nlgn"]
	Nret = config_dict["Nret"]
	dt = config_dict["dt"]
	# t = np.arange(0,config_dict["runtime"]/dt,1).astype("float64")
	t = np.arange(0,config_dict["Inp_params"]["pattern_duration"],1).astype("float64")


	last_timestep = config_dict["runtime"]/dt
	## number of input patterns needed
	T_pd = config_dict["Inp_params"]["pattern_duration"]
	T_exp = config_dict["Inp_params"]["expanse_time"]
	config_dict["Inp_params"].update({"Nsur" : int(np.ceil(1.*last_timestep/T_pd/(T_exp+1) ))})
	print("# of stimuli: {}".format(config_dict["Inp_params"]["Nsur"]));sys.stdout.flush()
	
	config_dict.update({
					"last_timestep" : last_timestep,
					"RF_mode" : "initialize",
					"system" : "one_layer",
					"Version" : Version,
					})
	n = network.Network(Version,config_dict)
	Wret_to_lgn,Wlgn_to_4,arbor_on,arbor_off,arbor2,init_weights,W4to4 = n.system
	Wrec_mode = config_dict["W4to4_params"]["Wrec_mode"]
	max_ew = config_dict["W4to4_params"]["max_ew"]

	print("W4to4",W4to4.shape)

	# ================== Normalisation projector for ff connectivity =======================
	c_orth,s_orth = misc.get_projection_operators(config_dict,config_dict["Wlgn_to4_params"],\
												  arbor_on,arbor_off,\
												  config_dict["Wlgn_to4_params"]["constraint_mode"],\
												  "layer4")

	if config_dict["Wlgn_to4_params"]["mult_norm"]=="xalpha":
		num_pops = Wlgn_to_4.shape[0]//2
		init_weights = 0
		## do normalisation separately for E and I population
		for i in range(num_pops):
			Wpop = Wlgn_to_4[i*2:(i+1)*2,...]
			arbpop = arbor2[i*2:(i+1)*2,...]

			dot_product = np.dot(c_orth,Wpop[arbpop>0])
			if isinstance(init_weights,int):
				init_weights = 1.*dot_product
			else:
				init_weights = np.stack([init_weights,dot_product])

	##================================= initialization ====================================
	tf.random.set_seed(config_dict["random_seed"]*113)
	if "2pop" in Wrec_mode:
		l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=1, dtype=tf.float32)
	else:
		l40 = tf.random.uniform([N4*N4*Nvert], minval=0, maxval=1, dtype=tf.float32)

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
					"Nret" : tf.constant(Nret, dtype=tf.int32),
					"Nvert" : tf.constant(Nvert, dtype=tf.int32),
					
					# "gamma_4" : tf.constant(config_dict["gamma_4"], dtype=tf.float32),
					# "gamma_lgn" : tf.constant(config_dict["gamma_lgn"], dtype=tf.float32),
					# "beta_P" : tf.constant(config_dict["beta_P"], dtype=tf.float32),
					# "beta_O" : tf.constant(config_dict["beta_O"], dtype=tf.float32),
					# "tau" : tf.convert_to_tensor(tau, name='tau', dtype=tf.float32),

					"init_weights" : tf.convert_to_tensor(init_weights,dtype=tf.float32),
					"Wret_to_lgn" : tf.convert_to_tensor(Wret_to_lgn,dtype=tf.float32),
					"W4to4" : tf.convert_to_tensor(W4to4, dtype=tf.float32),
					"W23to23" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
					"W4to23" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
					"W23to4" : tf.convert_to_tensor(np.array([]), dtype=tf.float32),
					"init_weights_4to23" : None,
					"arbor4to23" : None,
					"arbor4to23_full" : None,

					# "c_coeff" : tf.convert_to_tensor(c_coeff,dtype=tf.float32),
					# "c_vec" : tf.convert_to_tensor(c_vec,dtype=tf.float32),

					"arbor_on" : tf.convert_to_tensor(arbor_on,dtype=tf.float32),
					"arbor_off" : tf.convert_to_tensor(arbor_off,dtype=tf.float32),
					"arbor2" : tf.convert_to_tensor(arbor2,dtype=tf.float32),
					"arbor4to23" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
					"arbor4to23_full" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
					"arbor4to4" : None,
					"arbor23to23" : None,

					# "pattern_duration" : tf.constant(T_pd,dtype=tf.float32),
					# "expanse_time" : tf.constant(T_exp,dtype=tf.float32),
					# "avg_no_inp" : config_dict["Inp_params"]["avg_no_inp"],
					
					# "normalisation_mode" : tf.constant(config_dict["normalisation_mode"]),
					# "Wlim" : tf.constant(config_dict["Wlgn_to4_params"]["Wlim"], dtype=tf.float32),
					"c_orth" : tf.convert_to_tensor(c_orth,dtype=tf.float32),
					"s_orth" : tf.convert_to_tensor(s_orth,dtype=tf.float32),
					"c_orth_4to23" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
					"s_orth_4to23" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),

					"Corr" : tf.convert_to_tensor(np.array([]),dtype=tf.float32),
					
					"integrator" : config_dict["integrator"],
					"config_dict" : config_dict,
					}


	s = N4*N4*Nlgn*Nlgn*Nvert
	print("Starting simulation. This might take a while...")
	print("...")
	sys.stdout.flush()
	if config_dict["Inp_params"]["simulate_activity"]:
		if not kwargs["not_saving_temp"]:
			y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
			if config_dict["test_lowDsubset"]:
				yt,l4t = integrator_tf.odeint_new(dynamics.lowD_GRF_l4,\
											 	y0, t, dt, params_dict, mode="dynamic")
			else:
				yt,l4t = integrator_tf.odeint_new(dynamics.dynamics_l4_new,\
											 	y0, t, dt, params_dict, mode="dynamic")
			# yt,l4t = integrator_tf.odeint(dynamics.dynamics_l4,\
			# 								 y0, t, dt, params_dict, mode="dynamic")
		
			y = yt[-1,:]
			l4 = l4t[-1,:]
			print("yt",yt.shape,l4t.shape);sys.stdout.flush()
			print("yt",yt[-1,:])

			Wfinal = y[:config_dict["num_lgn_paths"]*s].numpy().reshape(config_dict["num_lgn_paths"],\
																N4**2*Nvert,Nlgn**2)
			# lgn_norm = (lgn - np.nanmean(lgn,axis=1)[:,None,:])/np.nanstd(lgn,axis=1)[:,None,:]
			# l4_norm = (l4t[:,:N4*N4*Nvert] - np.nanmean(l4t[:,:N4*N4*Nvert],axis=1)[:,None]) /\
			#  			np.nanstd(l4t[:,:N4*N4*Nvert],axis=1)[:,None]
			# cc_recff = []
			# for it in range(min([l4t.shape[0],lgn_norm.shape[2]])):
			# 	cc_recff.append( np.nanmean(l4_norm[it,None,::Nvert] * lgn_norm[:,:,it],axis=1) )
			# cct = np.array(cc_recff)
		else:
			t = t[:-config_dict["Inp_params"]["pattern_duration"]]
			y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
			y = integrator_tf.odeint(dynamics.dynamics_l4_sgl, y0, t, dt, params_dict,\
										mode="single_stim_update")
			l4 = y[2*s:]
			print("y,l4",y.shape,l4.shape);sys.stdout.flush()

	else:
		y0 = tf.concat([Wlgn_to_4.flatten(), l40], axis=0)
		if config_dict["test_lowDsubset"]:
			print("run test_lowDsubset")
			yt,l4t = integrator_tf.odeint_new(dynamics.lowD_GRF_l4,\
										 	y0, t, dt, params_dict, mode="dynamic")
		else:
			yt,l4t = integrator_tf.odeint_new(dynamics.dynamics_l4_new,y0,t,dt,params_dict,\
										 		mode="static")
		print("yt",yt.shape);sys.stdout.flush()
		y = yt[-1,:]
		l4 = l4t[-1,:]
	
	#################################################################################
	############################# SAVE PARAMS AND DATA ##############################
	if not os.path.exists(data_dir + "layer4/v{v}".format(v=Version)):
		os.makedirs(data_dir + "layer4/v{v}".format(v=Version))
	print("Version",Version,s);sys.stdout.flush()
	filename = "layer4/v{v}/yt_v{v}.npz".format(v=Version)
	if config_dict["Inp_params"]["simulate_activity"]:
		if not kwargs["not_saving_temp"]:
			data_dict_time = {
				"Wt"		:	yt[:,:config_dict["num_lgn_paths"]*s],\
				#optional:
				# "lgn_inp"	:	lgn,\
				# "cct"		:	cct,\
				"l4t"		:	l4t
			}
		data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "l4" : l4}
	else:
		data_dict_time = {
				"Wt"		:	yt[:,:config_dict["num_lgn_paths"]*s],\
				"l4t"		:	l4t
		}
		data_dict = {"W" : y[:config_dict["num_lgn_paths"]*s], "l4" : l4}
	## save time development of ff connections and activity
	if not kwargs["not_saving_temp"]:
		misc.save_data(Version, filename, data_dict_time)

	## save ff connections and activity of last timestep separately
	filename = "layer4/v{v}/y_v{v}.npz".format(v=Version)
	misc.save_data(Version, filename, data_dict)

	## save parameter settings
	filename = "layer4/v{v}/config_v{v}".format(v=Version)
	config_dict.update({
				"maxew"		: np.array([max_ew])\
		})
	misc.save_params(Version,filename,config_dict)
	#################################################################################
	#################################################################################


	try:
		del yt
	except:
		pass




if __name__=="__main__":
	# import argparse
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
	print(" ")
	print("config_dict, Wret_to_lgn_params",default_dict["Wret_to_lgn_params"])
	print("config_dict, W4to4_params",default_dict["W4to4_params"])

	print(" ")
	print("config_dict, Wlgn_to4_params",default_dict["Wlgn_to4_params"])
	print(" ")

	parameter_sweep_layer4(Version,default_dict,**args_dict)
	print("done")







