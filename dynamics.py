import tensorflow as tf
import numpy as np
from bettina.modeling.ori_dev_model import inputs#, image_dir



def dynamics_twolayer(y, t, params_dict, **kwargs):
	"""Layer 4 and Layer 23 dynamics with vertical units"""
	nl = params_dict["nonlinearity"]
	lgn = params_dict["lgn"]

	##network params
	N23 = params_dict["N23"]
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	## cortical layer parameters
	gamma_lgn = params_dict["gamma_lgn"]
	W4to4 = params_dict["W4to4"]
	W23to4 = params_dict["W23to4"]
	W4to23 = params_dict["W4to23"]
	W23to23 = params_dict["W23to23"]
	tau = params_dict["tau"]
	gamma_4 = params_dict["gamma_4"]

	s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
	l4_size = N4**2*Nvert * 2
	l23_size = N23**2 * 2
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

	l4 = y[s:s+l4_size]
	l23 = y[s+l4_size:s+l4_size+l23_size]
	rhs_l4EI = kwargs["rhs_l4"]
	dl4 = rhs_l4EI(t, l4, l23, N4*N4*Nvert, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4,\
					 W23to4, tau, nl)

	
	rhs_l23EI = kwargs["rhs_l23"]
	dl23 = rhs_l23EI(t, l23, l4, N23**2, 1., 1., W4to23, W23to23, tau, nl)

	dW = tf.zeros_like(Wlgn_to_4)
	dW = tf.reshape(dW, [s])
	# out = tf.concat([dW, dl4, dl23], axis=0)
	out = tf.concat([dW, dl4, dl23, 0*y[s+l4_size+l23_size:]], axis=0)
	return out


def dynamics_l4_sgl(y, t, params_dict, yt):
	"""ff plasticity update after each stimlus, with vertical cortical units"""
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]

	T_pd = params_dict["pattern_duration"]
	T_exp = params_dict["expanse_time"]
	avg_no_inp = params_dict["avg_no_inp"]


	if params_dict["config_dict"]["nonlinearity"]=="rectifier":
		def nl(x):
			out = tf.where(tf.greater(x, 0), x, tf.zeros(tf.shape(x),dtype=tf.float32) )
			# out = x
			return out
	elif params_dict["config_dict"]["nonlinearity"]=="linear":
		def nl(x):
			out = x
			return out

	# lgn = params_dict["lgn"][...,int(tf.math.floor(t/T_pd))]
	if (int(tf.math.floor(t/T_pd))%2)==1:
		lgn = tf.zeros((2,Nlgn*Nlgn))
		# print("zero",t,int(tf.math.floor(t/T_pd)))
	else:
		# rng_seed = params_dict["Version"].numpy()*1000 + int(tf.math.floor(t/T_pd))
		rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(tf.math.floor(t/T_pd))
		lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
			  rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
			  "white_noise_online", params_dict["Wret_to_lgn"].numpy())
		lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)

	arbor2 = params_dict["arbor2"]
	Wlim = params_dict["Wlim"]
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	gamma_lgn = params_dict["gamma_lgn"]
	W4to4 = params_dict["W4to4"]
	tau = params_dict["tau"]
	beta_P = params_dict["beta_P"]
	beta_O = params_dict["beta_O"]
	gamma_4 = params_dict["gamma_4"]

	s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

	l4 = y[s:]

	# dl4 = rhs_l4(t, l4, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
	dl4 = rhs_l4EI(t, l4, N4*N4*Nvert, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
	## update ff weights after activity has converged for specific input
	if (t%T_pd)==(T_pd-1):
		notfrozen = tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
		mask = tf.math.logical_and( notfrozen, arbor[tf.newaxis,:,:]>0 )
		mode = params_dict["normalisation_mode"]
		if mode=="xalpha":
			c_orth = params_dict["c_orth"]
			s_orth = params_dict["s_orth"]
		else:
			c_orth,s_orth = None,None
		dW = unconstrained_plasticity(t, l4[:N4**2*Nvert], lgn, Wlgn_to_4, beta_P, beta_O)
		mask_fl = tf.cast(mask, tf.float32)
		dW = constrain_plasticity_update(dW*arbor[tf.newaxis,:,:],mask_fl,arbor2,mode,\
										 c_orth,s_orth)
	else:
		dW = tf.zeros_like(Wlgn_to_4)

	dW = tf.reshape(dW, [s])
	out = tf.concat([dW, dl4], axis=0)
	return out


def dynamics_l4_new(y, t, params_dict, **kwargs):
	"""only Layer 4 dynamics after number of stimuli with vertical units"""
	nl = params_dict["nonlinearity"]
	lgn = params_dict["lgn"]

	##network params
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	## cortical layer parameters
	gamma_lgn = params_dict["gamma_lgn"]
	W4to4 = params_dict["W4to4"]
	tau = params_dict["tau"]
	gamma_4 = params_dict["gamma_4"]

	s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])
	l4 = y[s:]
	rhs_l4EI = kwargs["rhs_l4"]
	dl4 = rhs_l4EI(t, l4, N4*N4*Nvert, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)

	dW = tf.zeros_like(Wlgn_to_4)
	dW = tf.reshape(dW, [s])
	out = tf.concat([dW, dl4], axis=0)
	return out


def plasticity_dynamics(y, t, params_dict, **kwargs):
	"""ff plasticity for Lgn to L4 connectivity with vertical units"""
	lgn = params_dict["lgn"]

	##network params
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]	
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	## feedforward parameters
	beta_P = params_dict["beta_P"]
	beta_O = params_dict["beta_O"]

	s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])

	## take E and I units of L4 for plasticity update
	l4 = y[s:s+N4*N4*Nvert*num_lgn_paths//2]
	unconstrained_plasticity_fct = kwargs["unconst_plasticity_update"]
	dW = unconstrained_plasticity_fct(t,l4,lgn,Wlgn_to_4,beta_P,beta_O,N4*N4*Nvert)
	dW = tf.reshape(dW, [s])
	return dW

def plasticity_dynamics_all_ff(y, t, params_dict, **kwargs):
	"""ff plasticity for LGN to L4 and L4 to L23 connectivity with vertical units"""
	lgn = params_dict["lgn"]

	##network params
	N4 = params_dict["N4"]
	N23 = params_dict["N23"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]	
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	## feedforward parameters
	beta_P = params_dict["config_dict"]["beta_P"]
	beta_O = params_dict["config_dict"]["beta_O"]
	beta_4to23 = params_dict["config_dict"]["beta_4to23"]

	s = num_lgn_paths * N4*N4*Nlgn*Nlgn*Nvert
	l4_size = N4**2*Nvert * 2
	l23_size = N23**2 * 2

	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])
	W4to23 = y[s+l4_size+l23_size:]
	W4to23 = tf.reshape(W4to23, [N23**2*2, N4**2*Nvert*2])

	## take E and I units of L4 for plasticity update
	l4 = y[s:s+l4_size]
	l23 = y[s+l4_size:s+l4_size+l23_size]
	unconstrained_plasticity_fct = kwargs["unconst_plasticity_update"]
	dWlgn_to_4 = unconstrained_plasticity_fct(t,l4,lgn,Wlgn_to_4,beta_P,beta_O,N4*N4*Nvert)
	dWlgn_to_4 = tf.reshape(dWlgn_to_4, [s])

	s = N4*N4*N23*N23*Nvert*4
	dW4to23 = l23[:,None] * l4[None,:] * beta_4to23 * params_dict["arbor4to23_full"]
	dW4to23 = tf.reshape(dW4to23, [s])
	
	dW = tf.concat([dWlgn_to_4, dW4to23], axis=0)
	return dW


def dynamics_l4(y, t, params_dict, yt):
	"""ff plasticity after number of stimuli with vertical units"""
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]

	T_pd = params_dict["pattern_duration"]
	T_exp = params_dict["expanse_time"]
	avg_no_inp = params_dict["avg_no_inp"]

	if params_dict["config_dict"]["nonlinearity"]=="rectifier":
		def nl(x):
			out = tf.where(tf.greater(x, 0), x, tf.zeros(tf.shape(x),dtype=tf.float32) )
			# out = x
			return out
	elif params_dict["config_dict"]["nonlinearity"]=="linear":
		def nl(x):
			out = x
			return out

	num_lgn_paths = 2	## default: only on/off input to E units
	if params_dict["config_dict"]["Wlgn_to4_params"]["connectivity_type"]=="EI":
		num_lgn_paths = 4	## separate on/off input to E and I units
	# lgn = params_dict["lgn"][...,int(tf.math.floor(t/T_pd))]
	if (int(tf.math.floor(t/T_pd))%2)==1:
		lgn = tf.zeros((num_lgn_paths,Nlgn*Nlgn))
		# print("zero",t,int(tf.math.floor(t/T_pd)))
	else:
		# rng_seed = params_dict["Version"].numpy()*1000 + int(tf.math.floor(t/T_pd))
		rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(tf.math.floor(t/T_pd))
		lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
			  rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
			  "white_noise_online", params_dict["Wret_to_lgn"].numpy())
		lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)

	arbor2 = params_dict["arbor2"]
	Wlim = params_dict["Wlim"]

	gamma_lgn = params_dict["gamma_lgn"]
	W4to4 = params_dict["W4to4"]
	tau = params_dict["tau"]
	beta_P = params_dict["beta_P"]
	beta_O = params_dict["beta_O"]
	gamma_4 = params_dict["gamma_4"]

	s = 2 * N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [2, N4*N4*Nvert, Nlgn*Nlgn])

	l4 = y[s:]
	# print("dynamics_l4",y.shape,l4.shape,s)

	# dl4 = rhs_l4(t, l4, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
	dl4 = rhs_l4EI(t, l4, N4*N4*Nvert, lgn, gamma_lgn, gamma_4, Wlgn_to_4, W4to4, tau, nl)
	## update ff weights after activity has converged for specific input
	# if (t%T_pd)==(T_pd-1):
	# 	notfrozen = tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
	# 	mask = tf.math.logical_and( notfrozen, arbor[tf.newaxis,:,:]>0 )
	# 	mode = params_dict["normalisation_mode"]
	# 	if mode=="xalpha":
	# 		c_orth = params_dict["c_orth"]
	# 		s_orth = params_dict["s_orth"]
	# 	else:
	# 		c_orth,s_orth = None,None
	# 	dW = constrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O, mask, arbor, arbor2\
	# 		 mode, c_orth, s_orth)
	# 	print("constrained_plasticity",t,dW.shape,np.nanmax(dW),np.nanmin(dW),\
	# 			np.nanmax(l4),np.nanmax(lgn))

	if ((((t+1)/T_pd)%avg_no_inp)==0 and t>0):
		dW = 0
		notfrozen = tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
		mask = tf.math.logical_and( notfrozen, arbor2>0 )
		mode = params_dict["normalisation_mode"]
		if mode=="xalpha":
			c_orth = params_dict["c_orth"]
			s_orth = params_dict["s_orth"]
			# P_orth = params_dict["P_orth"]
		else:
			c_orth,s_orth = None,None
			P_orth = None
		for it in range(avg_no_inp):
			## second lgn input is always zero, don't need to calc update
			if ((int(tf.math.floor((t+1)/T_pd)) - it)%2)==0:
				# print("CONTINUE",t,it,(t)/T_pd,it,((int(tf.math.floor((t+1.)/T_pd)) - it)%2))
				continue
			# lgn = params_dict["lgn"][...,int(tf.math.floor(t/T_pd))-it]
			# rng_seed = params_dict["Version"].numpy()*1000 + int(tf.math.floor(t/T_pd)) - it
			rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(tf.math.floor(t/T_pd)) - it
			# print("rng_seed",rng_seed,it,t)
			lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
					rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
					"white_noise_online", params_dict["Wret_to_lgn"].numpy())
			lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)
			## take only activity from exc units for plasticity update
			if it==0:
				l4 = l4[:N4*N4*Nvert]
			else:
				# print("l4",yt.shape,it,rng_seed,s,N4*N4)
				l4 = yt[-it,s:s+N4*N4*Nvert]

			# if False:
			# 	l4_filt = tf.linalg.matvec(W4to4[:N4**2,:N4**2], tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:])) +\
			# 		tf.linalg.matvec(W4to4[:N4**2,:N4**2], tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:]))
			# 	l4_filt = l4_filt * gamma_lgn
			# 	import matplotlib.pyplot as plt
			# 	fig = plt.figure()
			# 	ax = fig.add_subplot(121)
			# 	im=ax.imshow((l4_filt.numpy()).reshape(N4,N4),interpolation="nearest",cmap="binary")
			# 	plt.colorbar(im,ax=ax)
			# 	ax = fig.add_subplot(122)
			# 	ax.set_title("ON,rng={}".format(rng_seed))
			# 	im=ax.imshow((lgn.numpy())[0,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
			# 		cmap="binary")
			# 	plt.colorbar(im,ax=ax)
			# 	plt.savefig(image_dir +\
			# 	 "layer4/tmp/v96_rA2.0_srec1.00_scc0.04_xalpha_AVG_test/activity_pattern/l4sim_t{}_it{}.pdf".format(t,it))
			# 	plt.close(fig)
			
			# dW += constrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O, mask,\
			# 	arbor, arbor2,mode, c_orth, s_orth)
			dW += unconstrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O)
			# print("dW",np.nanmax(dW),np.nanmean(dW),np.nanmin(dW))

		mask_fl = tf.cast(mask, tf.float32)
		dW = constrain_plasticity_update(dW*arbor2,mask_fl,arbor2,mode,c_orth,s_orth)

	else:
		dW = tf.zeros_like(Wlgn_to_4)

	dW = tf.reshape(dW, [s])
	out = tf.concat([dW, dl4], axis=0)
	return out


def dynamics_Wonly(y, t, params_dict, yt):
	"""plasticity dynamics only, cortical activity assumed to be converged
	in linear approximation"""
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nret = params_dict["Nret"]
	Nvert = params_dict["Nvert"]

	arbor2 = params_dict["arbor2"]
	Wlim = params_dict["Wlim"]
	avg_no_inp = params_dict["avg_no_inp"]

	gamma_lgn = params_dict["gamma_lgn"]
	beta_P = params_dict["beta_P"]
	beta_O = params_dict["beta_O"]
	gamma_4 = params_dict["gamma_4"]
	W4to4 = params_dict["W4to4"]

	s = 2 * N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [2, N4*N4*Nvert, Nlgn*Nlgn])

	## update ff weights every 
	if (((t+1)%avg_no_inp)==0 and t>0):
		dW = 0
		notfrozen = arbor2>0#tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim)
		mask = tf.math.logical_and( notfrozen, arbor2>0 )
		mode = params_dict["normalisation_mode"]
		if mode=="xalpha":
			c_orth = params_dict["c_orth"]
			s_orth = params_dict["s_orth"]
			# P_orth = params_dict["P_orth"]
		else:
			c_orth,s_orth = None,None
			P_orth = None
		for it in range(avg_no_inp):
			rng_seed = params_dict["config_dict"]["random_seed"]*1000 + int(t)*2 - it*2
			# print("rng_seed",rng_seed,it,t)
			lgn = inputs.Inputs_lgn((Nret,Nret),params_dict["Version"].numpy(),\
				rng_seed).create_lgn_input(params_dict["config_dict"]["Inp_params"],\
				"white_noise_online", params_dict["Wret_to_lgn"].numpy())
			lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)

			## assume here that W4to4 is Gaussian as in Francescos derivation
			## ## take only activity from exc units for plasticity update, that means here
			## W4to4 is E to E connectivity matrix
			l4 = tf.linalg.matvec(W4to4, tf.linalg.matvec(Wlgn_to_4[0,:,:],lgn[0,:])) +\
					tf.linalg.matvec(W4to4, tf.linalg.matvec(Wlgn_to_4[1,:,:],lgn[1,:]))
			l4 = l4 * gamma_lgn

			# if True:
			# 	import matplotlib.pyplot as plt
			# 	fig = plt.figure(figsize=(12,5))
			# 	ax = fig.add_subplot(121)
			# 	im=ax.imshow((l4.numpy()).reshape(N4,N4),interpolation="nearest",cmap="binary")
			# 	plt.colorbar(im,ax=ax)
			# 	ax = fig.add_subplot(122)
			# 	ax.set_title("ON,rng={}".format(rng_seed))
			# 	im=ax.imshow((lgn.numpy())[0,:].reshape(Nlgn,Nlgn),interpolation="nearest",\
			# 		cmap="binary")
			# 	plt.colorbar(im,ax=ax)
			# 	plt.savefig("/home/bettina/physics/columbia/projects/ori_dev_model/image/layer4/tmp/v99/l4filt_t{}_it{}.pdf".format(t,it))
			# 	plt.close(fig)
			
			# dW += constrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O, mask,\
			# 	arbor, mode, c_orth, s_orth)
			dW += unconstrained_plasticity(t, l4, lgn, Wlgn_to_4, beta_P, beta_O)
			# print("dW",np.nanmean(dW),np.nanmean(l4),t)

		mask_fl = tf.cast(mask, tf.float32)
		dW = constrain_plasticity_update(dW*arbor2,mask_fl,arbor2,mode,\
										c_orth,s_orth)
			
	else:
		dW = tf.zeros_like(Wlgn_to_4)

	# print("dW after constriant",np.nanmean(dW),np.nanmax(dW),np.nanmin(dW))
	dW = tf.reshape(dW, [s])
	# print("dW",dW.shape)
	return dW
	

def rhs_l4(t,l_act,inp,gamma_FF,gamma_rec,Wff_to_l,W_rec,tau,nl):
	'''
	input:
	inp			:	input to cortical layer
	l_act		:	activity of purely E crt layer 
	'''
	arg = gamma_FF * tf.linalg.matvec(Wff_to_l[0,:,:],inp[0,:]) +\
		  gamma_FF * tf.linalg.matvec(Wff_to_l[1,:,:],inp[1,:]) +\
		  gamma_rec * tf.linalg.matvec(W_rec, l_act)

	return 1./tau * (nl(arg) - l_act)

def arg_l4_1pop(l_act,inp_on,inp_off,gamma_FF,gamma_rec,Won_to_l,Woff_to_l,W_rec,tau):
	arg_ff = gamma_FF * tf.linalg.matvec(Won_to_l,inp_on) +\
			gamma_FF * tf.linalg.matvec(Woff_to_l,inp_off)
	arg_rec = gamma_rec * tf.linalg.matvec(W_rec, l_act)
	return arg_ff + arg_rec

def rhs_l4EI(t,l_act,N,inp,gamma_FF,gamma_rec,Wff_to_l,W_rec,tau,nl):
	'''
	input:
	inp			:	input to cortical layer
	l_act		:	activity of E/I crt layer 
	'''
	argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
						Wff_to_l[1,:,:],W_rec[:N,:],tau)
	argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], l_act)
	# argE = gamma_FF * tf.linalg.matvec(Wff_to_l[0,:,:],inp[0,:]) +\
	# 	   gamma_FF * tf.linalg.matvec(Wff_to_l[1,:,:],inp[1,:]) +\
	# 	   gamma_rec * tf.linalg.matvec(W_rec[:N,:], l_act)
	# argI = gamma_rec * tf.linalg.matvec(W_rec[N:,:], l_act)
	return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - l_act)

def rhs_l4EI_full_LGN_input(t,l_act,N,inp,gamma_FF,gamma_rec,Wff_to_l,W_rec,tau,nl):
	'''
	input:
	inp			:	input to cortical layer
	l_act		:	activity of E/I crt layer 
	'''
	argE = arg_l4_1pop(l_act,inp[0,:],inp[1,:],gamma_FF,gamma_rec,Wff_to_l[0,:,:],\
						Wff_to_l[1,:,:],W_rec[:N,:],tau)
	argI = arg_l4_1pop(l_act,inp[2,:],inp[3,:],gamma_FF,gamma_rec,Wff_to_l[2,:,:],\
						Wff_to_l[3,:,:],W_rec[N:,:],tau)
	return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - l_act)


def rhs_twolayer_l4EI(t,
					act,
					inp_fb,
					Nrec,
					inp_ff,
					gamma_ff,
					gamma_rec,
					Wff_to_l,
					W_rec,
					Wfb_to_l,
					tau,
					nl):
	argE = gamma_ff * tf.linalg.matvec(Wff_to_l[0,:,:],inp_ff[0,:]) +\
		   gamma_ff * tf.linalg.matvec(Wff_to_l[1,:,:],inp_ff[1,:]) +\
		   gamma_rec * tf.linalg.matvec(W_rec[:Nrec,:], act) +\
		   tf.linalg.matvec(Wfb_to_l[:Nrec,:], inp_fb)

	argI = gamma_rec * tf.linalg.matvec(W_rec[Nrec:,:], act) +\
		   tf.linalg.matvec(Wfb_to_l[Nrec:,:], inp_fb)
	return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)

def rhs_twolayer_l4EI_full_LGN_input(t,
									act,
									inp_fb,
									Nrec,
									inp_ff,
									gamma_ff,
									gamma_rec,
									Wff_to_l,
									W_rec,
									Wfb_to_l,
									tau,
									nl):
	
	arg_ff = gamma_ff * tf.linalg.matvec(Wff_to_l[0,:,:],inp_ff[0,:]) +\
			gamma_ff * tf.linalg.matvec(Wff_to_l[1,:,:],inp_ff[1,:])
	argE =  arg_ff +\
			gamma_rec * tf.linalg.matvec(W_rec[:Nrec,:], act) +\
			tf.linalg.matvec(Wfb_to_l[:Nrec,:], inp_fb)

	argI = gamma_ff * tf.linalg.matvec(Wff_to_l[2,:,:],inp_ff[2,:]) +\
			gamma_ff * tf.linalg.matvec(Wff_to_l[3,:,:],inp_ff[3,:]) +\
			gamma_rec * tf.linalg.matvec(W_rec[Nrec:,:], act) +\
			tf.linalg.matvec(Wfb_to_l[Nrec:,:], inp_fb)
	# print("argL4",np.nanmax(argE),np.nanmax(argI),tau)
	return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)


def rhs_twolayer_l23EI(t,
			act,
			inp_ff,
			Nrec,
			gamma_ff,
			gamma_rec,
			Wff_to_l,
			W_rec,
			tau,
			nl):
	argE = gamma_ff * tf.linalg.matvec(Wff_to_l[:Nrec,:],inp_ff) +\
		   gamma_rec * tf.linalg.matvec(W_rec[:Nrec,:], act)
	# argI = gamma_rec * tf.linalg.matvec(W_rec[Nrec:,:], act)
	argI = gamma_ff * tf.linalg.matvec(Wff_to_l[Nrec:,:],inp_ff) +\
		   gamma_rec * tf.linalg.matvec(W_rec[Nrec:,:], act)
	# print("argL23",np.nanmax(argE),np.nanmax(argI))
	return 1./tau * (nl( tf.concat([argE,argI], axis=0) ) - act)


def rhs_l23(t,
			l_act,
			inp,
			gamma_FF,
			gamma_rec,
			Wff_to_l,
			W_rec,
			tau,
			nl):
	'''
	input:
	inp		:	input to cortical layer
	l_act		:	activity of crt layer 
	'''
	arg = gamma_FF * tf.linalg.matvec(Wff_to_l,inp) +\
		  gamma_rec * tf.linalg.matvec(W_rec,l_act)
	# print("ff",np.nanmax(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)),np.nanmin(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
	# print("winp",np.nanstd(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
	# print("rec",np.nanmax(tf.linalg.matvec(W_rec_23, l23)),np.nanmin(tf.linalg.matvec(W_rec_23, l23)))
	return 1./tau * (nl(arg) - l_act)

def rhs_l4_wfb(t,
			l_act,
			inp,
			fb_act,
			gamma_FF,
			gamma_rec,
			gamma_FB,
			Wff,
			Wrec,
			Wfb,
			tau,
			nl):
	'''
	input:
	inp			:	input to cortical layer
	l_act		:	activity of crt layer 
	'''
	arg = gamma_FF * tf.linalg.matvec(Wff[0,:,:],inp[0,:,:]) +\
		  gamma_FF * tf.linalg.matvec(Wff[1,:,:],inp[1,:,:]) +\
		  gamma_rec * tf.linalg.matvec(Wrec, l_act) +\
		  gamma_FB * tf.linalg.matvec(Wfb,fb_act)
	# print("ff",np.nanmax(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)),np.nanmin(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
	# print("winp",np.nanstd(gamma_FF * tf.linalg.matvec(Wff_to_23,inp)))
	# print("rec",np.nanmax(tf.linalg.matvec(W_rec_23, l23)),np.nanmin(tf.linalg.matvec(W_rec_23, l23)))
	return 1./tau * (nl(arg) - l_act)


def unconstrained_plasticity(t,postsyn_act,presyn_act,W,beta_P,beta_O,N=None):
	update = postsyn_act[None,:,None] * (beta_P * presyn_act[:,None,:] ) #-\
			# beta_O * tf.linalg.matvec(tf.transpose(Wff[0,:,:]),l4)[None,None,:] -\
			# beta_O * tf.linalg.matvec(tf.transpose(Wff[1,:,:]),l4)[None,None,:])
	## normalise learning rate such that learning rate is approximately indep of l4
	## activity amplitude
	# update = update/np.nanmean(postsyn_act) * 307.
	return update

def unconstrained_plasticity_full_LGN_input(t,postsyn_act,presyn_act,W,beta_P,beta_O,N):
	E_update = postsyn_act[None,:N,None] * (beta_P * presyn_act[:2,None,:] )
	I_update = postsyn_act[None,N:,None] * (beta_P * presyn_act[2:,None,:] )
	update = tf.concat([E_update,I_update],0)
	## normalise learning rate such that learning rate is approximately indep of l4
	## activity amplitude
	# update = update/np.nanmean(postsyn_act) * 307.
	return update


def constrain_plasticity_update_wrapper(y,dt,dWlgn_to_4,arbor2,Wlim,params_dict,\
	mode="xalpha",c_orth=None,s_orth=None):
	"""Plasticity update averaged over avg_no_inp number of input patterns"""
	##network params
	N4 = params_dict["N4"]
	Nlgn = params_dict["Nlgn"]
	Nvert = params_dict["Nvert"]
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	s = num_lgn_paths*N4*N4*Nlgn*Nlgn*Nvert
	Wlgn_to_4 = y[:s]
	Wlgn_to_4 = tf.reshape(Wlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])
	notfrozen = tf.math.logical_and(Wlgn_to_4>0, Wlgn_to_4<Wlim*arbor2)
	mask = tf.math.logical_and( notfrozen, arbor2>0 )
	mask_fl = tf.cast(mask, tf.float32)
	
	
	dWlgn_to_4 = tf.reshape(dWlgn_to_4, [num_lgn_paths, N4*N4*Nvert, Nlgn*Nlgn])
	dWE = constrain_plasticity_update(dWlgn_to_4[:2,:,:]*arbor2[:2,:,:],\
									mask_fl[:2,:,:],arbor2[2:,:,:],mode,c_orth,s_orth)
	
	if num_lgn_paths==4:
		dWI = constrain_plasticity_update(dWlgn_to_4[2:,:,:]*arbor2[2:,:,:],\
							 			mask_fl[2:,:,:],arbor2[2:,:,:],mode,c_orth,s_orth)
		dWlgn_to_4 = tf.concat([dWE,dWI], 0)
	else:
		dWlgn_to_4 = dWE
	print("dWlgn_to_4",np.nanmax(dWlgn_to_4),np.nanmin(dWlgn_to_4))
	Wlgn_to_4 += dt*dWlgn_to_4
	return Wlgn_to_4


def constrain_plasticity_update(delta,mask_fl,arbor2,mode="xalpha",c_orth=None,s_orth=None):
	if mode=="x":
		## sum over x
		norm = tf.reduce_sum(arbor2,axis=1)
		norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
		eps = 1.*tf.reduce_sum(delta,axis=1)/norm
		delta2 = (delta - eps[:,None,:] * arbor2) * mask_fl

	elif mode=="alpha":
		## sum over alpha and on/off
		norm = tf.reduce_sum(arbor2,axis=(0,2))
		norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
		eps = 1.*tf.reduce_sum(delta,axis=(0,2))/norm
		delta2 = (delta - eps[None,:,None]*arbor2) * mask_fl

	elif mode=="xalpha":
		if False:
			## first sum over alpha and on/off, then over x
			norm = tf.reduce_sum(arbor2,axis=(0,2))
			norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
			eps = 1.*tf.reduce_sum(delta,axis=(0,2))/norm
			delta2 = (delta - eps[None,:,None]*arbor2) * mask_fl

			norm = tf.reduce_sum(arbor2,axis=1)
			norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
			eps = 1.*tf.reduce_sum(delta2,axis=1)/norm
			delta2 = (delta2 - eps[:,None,:] * arbor2) * mask_fl
		else:
			delta_mask = delta[arbor2>0] ## complete update incl multiplied by arbor
			mask_fl = mask_fl[arbor2>0]	## boolean mask as type float
			# delta_mask *= mask_fl	## inserted to see whether that incr conservation of weights

			delta_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,delta_mask)[:,None],axis=0)
			delta_mask *= mask_fl
			delta2 = tf.scatter_nd(tf.where(arbor2>0),delta_mask,arbor2.shape)
			delta2 = tf.reshape(delta2, tf.shape(delta))

			print("norm after delta2 update",\
				np.abs(np.nansum(delta2,axis=(0,2)))[:5],\
				np.abs(np.nansum(delta2,axis=1))[0,:5])

	return delta2


def constrain_plasticity_update_W4to23(y,dt,dW4to23,arbor,arbor2,Wlim,params_dict,\
	c_orth,s_orth):
	"""Plasticity update averaged over avg_no_inp number of input patterns"""
	##network params
	N4 = params_dict["N4"]
	N23 = params_dict["N23"]
	Nlgn = params_dict["Nlgn"]
	Nvert = params_dict["Nvert"]
	num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]

	s = num_lgn_paths*N4*N4*Nlgn*Nlgn*Nvert
	W4to23 = y[s+N4**2*2*Nvert+N23**2*2:]
	W4to23 = tf.reshape(W4to23, [N23**2*2, N4*N4*Nvert*2])
	notfrozen = tf.math.logical_and(W4to23>0, W4to23<Wlim*arbor2)
	mask = tf.math.logical_and( notfrozen, arbor2>0 )
	mask_fl = tf.cast(mask, tf.float32)

	"""normalise plasticity update via orthogonalised arbor vectors"""
	dW4to23 = tf.reshape(dW4to23, [N23**2*2, N4*N4*Nvert*2])
	print("dW4to23",np.nanmax(dW4to23))
	delta_mask = dW4to23[arbor2>0] ## complete update incl multiplied by arbor
	mask_fl = mask_fl[arbor2>0]	## boolean mask as type float

	delta_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,delta_mask)[:,None],axis=0)
	print("delta_mask",np.nanmax(delta_mask))
	delta_mask *= mask_fl
	
	delta2 = tf.scatter_nd(tf.where(arbor2>0),delta_mask,arbor2.shape)
	delta2 = tf.reshape(delta2, [N23**2*2, N4*N4*Nvert*2])
	# print("check norm",np.sum(delta2,axis=0),np.sum(delta2,axis=1))

	print("W4to23, before",np.nanmax(W4to23),np.nanmin(W4to23))
	W4to23 += dt*delta2
	print("W4to23, after",np.nanmax(W4to23),np.nanmin(W4to23))
	return W4to23


def constrained_plasticity(t,
				l4,
				inp,
				Wff,
				beta_P,
				beta_O,
				mask,
				arbor,
				arbor2,
				mode="xalpha",
				c_orth=None,
				s_orth=None
				):
	"""
	incorporates conservation of synaptic strength, and leads to competition
	mode:
	x		:	normalize over all neurons x (normalize per on/off axon)
	alpha	:	normalize over all lgn units alhpha (normalize per crt neuron dendritic tree)
	xalpha	:	normalize first over alpha, then over x
	"""
	mask_fl = tf.cast(mask, tf.float32)
	# print("arbor",(arbor.numpy()).size,np.sum(arbor.numpy()<0.001))
	delta = unconstrained_plasticity(t,l4,inp,Wff,beta_P,beta_O) * arbor[tf.newaxis,:,:]# * mask_fl

	### set filled = 1 since otherwise if all synapses to one cortical\
	#neuron are frozen we get RuntimeWarning due to dividing by zero.
	### since a_frzn_on.filled(0)/a_frzn_off.filled(0) is only nonzero for\
	#active synapses, eps gets 0 nonetheless if all synapses to one cortical neuron are frozen.
	## normalized per thalamical unit (sum over cortex units)

	constrained_delta = constrain_plasticity_update(delta,mask_fl,arbor2,mode,\
													c_orth,s_orth)


	# print("delta2",t.numpy(),np.nanmin(delta2),np.nanmax(delta2),\
	# 	np.sum(delta2.numpy(),axis=1)[0,:5], np.sum(mask_fl==0) )
	return constrained_delta
	


def rhs_W(t, l23, inp, Wff_to_23, beta_P, beta_post):
	# arbor = tf.where(tf.greater(Wff_to_23, 0), tf.ones(tf.shape(Wff_to_23),dtype=tf.float32),\
	#  tf.zeros(tf.shape(Wff_to_23),dtype=tf.float32) )
	arbor = tf.ones(Wff_to_23.shape)
	## simple Hebb:
	# update = beta_P/tf.reduce_sum(arbor) * l23[:,None]*inp[None,:] * arbor
	## Oja's rule:
	update = arbor * l23[:,None]*(beta_P*inp[None,:])# -\
		# beta_post* tf.linalg.matvec(tf.transpose(Wff_to_23),l23)[None,:])
	norm = tf.reduce_sum(update,axis=0)/tf.reduce_sum(arbor)
	update -= norm[None,:] * arbor

	## subtractive normalization:
	## normalization for each input loc sepearately (ala francesco)
	# update = update - tf.reduce_sum(update,axis=0)[None,:]
	# print("update",np.nanmax(update),np.nanmin(update),np.nanmin(Wff_to_23),np.nanmin(l23))
	

	## normalization for each unit in layer 23 (miller 94)
	# update = update - tf.reduce_sum(update,axis=1)

	# # divisive normalization:
	# # normalised incoming connections to each cortical unit
	# norm = tf.reduce_sum(Wff_to_23 + update,axis=1)[:,None]
	
	# # normalised outgoing connections from each thalamic unit
	# norm = tf.reduce_sum(W_ff+update,axis=0)[None,:]
	# update = update/norm

	return update



if __name__=="__main__":
	pass