import sys
import tensorflow as tf
import numpy as np
from bettina.modeling.ori_dev_model import image_dir,inputs,network_tools,dynamics,\
normalisation_constraints

from bettina.modeling.ori_dev_model.tools import gen_gaussian_random_field as ggrf,calc_tools,\
update_params_dict


def tf_check_type(t, y0): 
	# Ensure Input type is Correct
	if not (y0.dtype.is_floating and t.dtype.is_floating):
		raise TypeError('Error in Datatype')


def clip_weights(Wlgn_to_4,Wlim,arbor):
	return tf.clip_by_value(Wlgn_to_4,0,arbor*Wlim)

def check_for_frozen_weights(Wlgn_to_4,Wlim,arbor):
	frozen = tf.math.logical_or(Wlgn_to_4[arbor>0]>=(Wlim*arbor[arbor>0]), Wlgn_to_4[arbor>0]<=0)
	return np.sum(frozen)>0


def synaptic_normalization(W_clipped,arbor,Wlim,init_W,c_orth=None,axis=1,mode="xalpha"):
	if c_orth is None:
		frozen = tf.math.logical_or(W_clipped>=(Wlim*arbor), W_clipped<=0)
		frozen_fl = tf.cast(frozen,tf.float32)
		gamma = np.ones_like(W_clipped,dtype=np.float32)
		# ## norm over on/off and alpha
		# Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
		# 			W, tf.zeros(tf.shape(W),dtype=tf.float32)), axis=(0,2))
		# Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
		#  			W, tf.zeros(tf.shape(W),dtype=tf.float32)), axis=(0,2))
		# gamma[...] =  (-Wfrozen + tf.reduce_sum(arbor_fl,axis=1)[None,:]*2.)/Wactive
		# W = gamma * W

		if isinstance(axis,tuple):
			init_WE = tf.reduce_sum(init_W[:2,...],axis=0)
			init_WI = tf.reduce_sum(init_W[2:,...],axis=0)
			# norm over outgoing connections (sum over L4 cortical space x for conn LGN to L4)
			Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
						W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=2)
			WfrozenE = tf.reduce_sum(Wfrozen[:2,:],axis=0)
			WfrozenI = tf.reduce_sum(Wfrozen[2:,:],axis=0)
			Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
						W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=2)
			WactiveE = tf.reduce_sum(Wactive[:2,:],axis=0)
			WactiveI = tf.reduce_sum(Wactive[2:,:],axis=0)

			gamma[:2,...] = tf.expand_dims(tf.expand_dims((-WfrozenE + init_WE)/WactiveE,\
							axis=axis[0]),axis=axis[1])
			gamma[2:,...] = tf.expand_dims(tf.expand_dims((-WfrozenE + init_WE)/WactiveE,\
							axis=axis[0]),axis=axis[1])
		else:
			# norm over outgoing connections (sum over L4 cortical space x for conn LGN to L4)
			Wfrozen = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 1.),\
						W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=axis)
			Wactive = tf.reduce_sum(tf.where(tf.equal(frozen_fl, 0.),\
						W_clipped, tf.zeros(tf.shape(W_clipped),dtype=tf.float32)), axis=axis)
			gamma[...] = tf.expand_dims((-Wfrozen + init_W)/Wactive,axis=axis)
		gamma[frozen.numpy()] = 1.0
		gamma = tf.clip_by_value(gamma,0.8,1.2)
		W_new = gamma * W_clipped
	else:
		num_pops = W_clipped.shape[0]//2
		W_new = 0
		n_orth = tf.cast(c_orth>0,tf.float32)
		## do normalisation separately for E and I population
		for i in range(num_pops):
			Wpop = W_clipped[i*2:(i+1)*2,...]
			arbpop = arbor[i*2:(i+1)*2,...]
			initWpop = init_W[i,...]


			frozen = tf.math.logical_or(Wpop>=(Wlim*arbpop), Wpop<=0)
			frozen_fl = tf.cast(frozen,tf.float32)


			W_orth = Wpop[arbpop>0] - tf.linalg.matvec(c_orth,\
													tf.linalg.matvec(c_orth,Wpop[arbpop>0]),\
													transpose_a=True)
			Wfrozen = tf.where(tf.equal(frozen_fl, 1.),\
								Wpop, tf.zeros(tf.shape(Wpop),dtype=tf.float32))
			Wactive = tf.where(tf.equal(frozen_fl, 0.),\
								Wpop, tf.zeros(tf.shape(Wpop),dtype=tf.float32))
			Wfrozen_vec = tf.linalg.matvec(c_orth,Wfrozen[arbpop>0])
			Wactive_vec = tf.linalg.matvec(c_orth,Wactive[arbpop>0])
			gamma_vec = (initWpop - Wfrozen_vec)/Wactive_vec ## gamma_k			
			gamma_vec = tf.where(tf.equal(Wactive_vec, 0),\
								 tf.ones(tf.shape(Wactive_vec),dtype=tf.float32), gamma_vec)
			# #gamma = tf.clip_by_value(gamma,0.8,1.2)

			if isinstance(W_new,int):
				W_new = tf.linalg.matvec(c_orth,Wactive_vec*gamma_vec + Wfrozen_vec,\
										transpose_a=True) + W_orth
				W_new = tf.where(tf.equal(frozen_fl[arbpop>0], 1.), Wpop[arbpop>0], W_new)
			else:
				# W_tmp = Wactive*gamma + Wfrozen
				W_tmp = tf.linalg.matvec(c_orth,Wactive_vec*gamma_vec + Wfrozen_vec,\
										transpose_a=True) + W_orth
				W_tmp = tf.where(tf.equal(frozen_fl[arbpop>0], 1.), Wpop[arbpop>0], W_tmp)
				W_new = tf.concat([W_new,W_tmp], 0)

		W_new = tf.scatter_nd(tf.where(arbor>0),W_new,W_clipped.shape)

		## doesnt conserve frozen weights, because in the end i ADD normalised weights to 
		## clipped weight matrix again
		# _,Nout,Ninp = W_clipped.shape
		# dWeff = W_clipped - W_old
		# W_new = 0
		# num_pops = W_clipped.shape[0]//2
		# ## do normalisation separately for E and I population
		# for i in range(num_pops):
		# 	Wpop = W_clipped[i*2:(i+1)*2,...]
		# 	arbpop = arbor[i*2:(i+1)*2,...]
		# 	dW = dWeff[i*2:(i+1)*2,...]
		# 	dW_1d = dW[arbpop>0]

		# 	w_orth = normalisation_constraints.expand_weights_to_orthog_space(dW.numpy(),\
		# 			 c_orth.numpy(),Nout,Ninp,arbpop.numpy())
		# 	notfrozen = tf.math.logical_or(Wpop<(Wlim*arbpop), Wpop>0)
		# 	notfrozen_fl = tf.cast(notfrozen,tf.float32)
		# 	mask_1d = notfrozen_fl[arbpop>0]

		# 	dW_1d -= tf.reduce_sum(w_orth*tf.linalg.matvec(c_orth,dW_1d)[:,None],axis=0)
		# 	dW_1d *= mask_1d
		# 	dW_2d = tf.scatter_nd(tf.where(arbpop>0),dW_1d,arbpop.shape)
		# 	dW_2d = tf.reshape(dW_2d, tf.shape(dW))

		# 	if isinstance(W_new,int):
		# 		W_new = W_old[i*2:(i+1)*2,...] + dW_2d
		# 	else:
		# 		W_new = tf.concat([W_new,W_old[i*2:(i+1)*2,...] + dW_2d], 0)

	return W_new,None


def homeostatic_normalization(W_clipped,H,l4,l4_target,dt):
	H_new = H + dt * H * (1 - l4/l4_target)
	W_new = W_clipped * tf.expand_dims(H_new,axis=(0,2))
	return W_new, H_new


class Tf_integrator_new:
	def __init__(self, params_dict):
		self.params_dict = params_dict
		self.scheme = params_dict["integrator"]
		self.avg_no_inp = params_dict["avg_no_inp"]
		self.num_inputs = int(self.params_dict["config_dict"]["runtime"]/\
							self.params_dict["config_dict"]["dt"]/\
							self.params_dict["config_dict"]["Inp_params"]["pattern_duration"])
		self.num_plasticity_steps = int(self.num_inputs/self.avg_no_inp)
		self.frozen = False
		self.nonlinearity_rule = params_dict["config_dict"]["nonlinearity"]
		self.num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]
		self.saving_stepsize = params_dict["config_dict"]["saving_stepsize"]
		self.system = params_dict["config_dict"]["system"]
		self.connectivity_type = params_dict["config_dict"]["Wlgn_to4_params"]["connectivity_type"]
		self.multiplicative_normalisation = params_dict["config_dict"]["multiplicative_normalisation"]

		self._init_nonlinearity()
		self.params_dict.update({"nonlinearity" : self.tf_nonlinearity})

		self._init_cortical_layer_dynamics()
		self._init_plasticity_update()
		self._init_multiplicative_normalisation()
		
		self.divergence_limit = 10000000.



	def _init_nonlinearity(self):
		if self.nonlinearity_rule == 'rectifier': 
			self.tf_nonlinearity = network_tools.nl_rect
		elif self.nonlinearity_rule == 'linear': 
			self.tf_nonlinearity = network_tools.nl_linear
		elif self.nonlinearity_rule == "powerlaw":
			self.tf_nonlinearity = network_tools.nl_powerlaw
		else:
			raise Exception('Unknown nonlinearity rule')
	

	def _init_cortical_layer_dynamics(self):
		if self.system=="one_layer":
			self.rhs_l23 = None
			if self.num_lgn_paths==2:
				self.rhs_l4 = dynamics.rhs_l4EI

			elif self.num_lgn_paths==4:
				self.rhs_l4 = dynamics.rhs_l4EI_full_LGN_input

			else:
				raise Exception('Number of LGN paths not defined. Choose either 2 or 4.')

		elif self.system=="two_layer":
			self.rhs_l23 = dynamics.rhs_twolayer_l23EI
			if self.num_lgn_paths==2:
				self.rhs_l4 = dynamics.rhs_twolayer_l4EI

			elif self.num_lgn_paths==4:
				self.rhs_l4 = dynamics.rhs_twolayer_l4EI_full_LGN_input

			else:
				raise Exception('Number of LGN paths not defined. Choose either 2 or 4.')
		else:
			raise Exception('System not defined. Choose either one_layer or two_layer.')


	def _init_plasticity_update(self):
		if self.connectivity_type=="EI":
			self.unconst_plasticity_update = dynamics.unconstrained_plasticity_full_LGN_input

		elif self.connectivity_type=="E":
			self.unconst_plasticity_update = dynamics.unconstrained_plasticity

		else:
			raise Exception('connectivity_type not defined. Choose either "E" or "EI".')

		if self.params_dict["config_dict"]["W4to23_params"]["plastic"]:
			self.plasticity_dynamics = dynamics.plasticity_dynamics_all_ff

		else:
			self.plasticity_dynamics = dynamics.plasticity_dynamics

	
	def _init_multiplicative_normalisation(self):
		self.l4_target = None
		if self.multiplicative_normalisation=="x":
			self.mult_normalization =\
			 lambda Wnew,arb,Wlim,initW,c_orth,H,l4,l4_target: synaptic_normalization(\
				 													Wnew,arb,Wlim,\
																	initW,c_orth=None,axis=1)

		elif self.multiplicative_normalisation=="alpha":
			self.mult_normalization =\
			 lambda Wnew,arb,Wlim,initW,c_orth,H,l4,l4_target: synaptic_normalization(\
				 													Wnew,arb,Wlim,\
																	initW,c_orth=None,\
																	axis=(0,2))

		elif self.multiplicative_normalisation=="xalpha":
			self.mult_normalization =\
			 lambda Wnew,arb,Wlim,initW,c_orth,H,l4,l4_target: synaptic_normalization(\
				 													Wnew,arb,Wlim,\
																	initW,c_orth=c_orth,\
																	axis=None)

		elif self.multiplicative_normalisation=="homeostatic":
			self.mult_normalization =\
			 lambda Wnew,arb,Wlim,initW,c_orth,H,l4,l4_target: homeostatic_normalization(\
			 														Wnew,H,l4,l4_target,\
			 														self.params_dict["dt"])

		else:
			raise Exception('multiplicative_normalisation not defined.\
							Choose either "x", "alpha" or "xalpha" "homeostatic".')

	
	def yield_init_conditions(self,y):
		"""initiate conditions for cortical layer(s)"""
		N4 = self.params_dict["N4"]
		Nvert = self.params_dict["Nvert"]
		Nlgn = self.params_dict["Nlgn"]
		lim = Nlgn*Nlgn*N4*N4*Nvert
		if self.system=="one_layer":
			if "2pop" in self.params_dict["config_dict"]["W4to4_params"]["Wrec_mode"]:
				l40 = tf.random.uniform([N4**2*2*Nvert],minval=0,maxval=0.1,dtype=tf.float32)
			else:
				l40 = tf.random.uniform([N4**2*Nvert],minval=0,maxval=0.1,dtype=tf.float32)
			y0 = tf.concat([y[:self.num_lgn_paths*lim], l40], axis=0)

		elif self.system=="two_layer":
			N23 = self.params_dict["N23"]
			l40 = tf.random.uniform([N4*N4*2*Nvert], minval=0, maxval=1, dtype=tf.float32)*0.1
			l230 = tf.random.uniform([N23*N23*2], minval=0, maxval=1, dtype=tf.float32)*0.1
			
			if self.params_dict["config_dict"]["W4to23_params"]["plastic"]:
				y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230,\
								y[self.num_lgn_paths*lim+N4*N4*2*Nvert+N23*N23*2:]], axis=0)
			else:
				y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230], axis=0)
					
		else:
			raise Exception('System not defined. Choose either one_layer or two_layer.')
		return y0


	def integrate(self, l4_dynamics, y0, t, dt):
		## NETWORK PARAMETERS
		Nret = self.params_dict["Nret"]
		N4 = self.params_dict["N4"]
		Nlgn = self.params_dict["Nlgn"]
		Nvert = self.params_dict["Nvert"]
		lim = Nlgn*Nlgn*N4*N4*Nvert
		l4_size = N4**2*2*Nvert
		crt_size = N4**2*2*Nvert
		if self.system=="two_layer":
			crt_size = N4**2*Nvert*2 + self.params_dict["N23"]**2*2

		arbor2 = self.params_dict["arbor2"]
		arbor4to23 = self.params_dict["arbor4to23"]
		# arbor2 = tf.stack([arbor]*2)

		init_weights = self.params_dict["init_weights"]
		# c_coeff = self.params_dict["c_coeff"]
		# c_vec = self.params_dict["c_vec"]
		Wlim = self.params_dict["Wlim"]
		num_lgn_paths = self.params_dict["config_dict"]["num_lgn_paths"]

		T_pd = self.params_dict["pattern_duration"]
		mode = self.params_dict["normalisation_mode"]
		c_orth = self.params_dict["c_orth"]
		s_orth = self.params_dict["s_orth"]
		c_orth_4to23 = self.params_dict["c_orth_4to23"]
		s_orth_4to23 = self.params_dict["s_orth_4to23"]
		
		
		def scan_func(func, y, t_dt, **kwargs):
			t, dt = t_dt
			dy = self._step_func(func, y, t, dt, **kwargs)
			out = y + dy
			return out

		## init norm
		Wlgn_to_4 = y0[:lim*num_lgn_paths].numpy().reshape(num_lgn_paths,N4**2*Nvert,Nlgn**2)
		init_norm_alpha = np.nansum(Wlgn_to_4,axis=(0,2))
		init_norm_x = np.nansum(Wlgn_to_4,axis=1)
		print("init_sum",init_norm_alpha[:5],init_norm_x[0,:5])
		H = 1.


		## init activity
		y = y0
		yt = tf.convert_to_tensor([y0])
		l4t = tf.convert_to_tensor([y0[num_lgn_paths*lim:num_lgn_paths*lim+crt_size]])
		tf.random.set_seed(self.params_dict["config_dict"]["random_seed"]*113)
		print("Number of plasticity steps: {}".format(self.num_plasticity_steps))
		for istep in range(self.num_plasticity_steps):
			dW = 0
			for jinput in range(self.avg_no_inp):
				print("")
				print("Plasticity step: {}, input: {}".format(istep,jinput))
				sys.stdout.flush()

				update_params_dict.update_developing_parameters(istep,self.params_dict["config_dict"])
				# print("PARAM CHECK 1",self.params_dict["config_dict"]["beta_P"])
				# print("PARAM CHECK 2",self.params_dict["config_dict"]["Inp_params"]["off_bias_strength"])
				# print("PARAM CHECK 3",self.params_dict["config_dict"]["Inp_params"]["off_bias_region_size"])
				# print("PARAM CHECK 4",self.params_dict["config_dict"]["Inp_params"]["onoff_rel_weight"])

				####### generate LGN input per L4 simulation #######
				rng_seed = self.params_dict["config_dict"]["random_seed"]*1000 +\
						   jinput + istep*self.avg_no_inp

				inp = inputs.Inputs_lgn((Nret,Nret),self.params_dict["Version"].numpy(),rng_seed)

				lgn = inp.create_lgn_input(\
					  self.params_dict["config_dict"]["Inp_params"],\
					  self.params_dict["config_dict"]["Inp_params"]["input_type"],\
					  self.params_dict["Wret_to_lgn"].numpy()
					  )
				lgn = inp.apply_ONOFF_bias(lgn,self.params_dict["config_dict"]["Inp_params"])
				## use same inputs for E and I units
				if num_lgn_paths==4:
					lgn = np.concatenate([lgn,lgn])
				lgn = tf.convert_to_tensor(lgn,dtype=tf.float32)
				self.params_dict.update({"lgn" : lgn})

				y0 = self.yield_init_conditions(y)
				y = y0
				####### L4 activity dynamics #######
				for kt in t:
					y = scan_func(l4_dynamics, y, (kt, dt), rhs_l4=self.rhs_l4,\
								  rhs_l23=self.rhs_l23)

				####### Test if plasticity update still works if L4 activity is ############
				####### effectively decoupled from LGN input ###############################
				# radius = 2.
				# width = 0.4
				# rng = np.random.RandomState(self.params_dict["config_dict"]["random_seed"]*10000+\
				# 							jinput+istep*self.avg_no_inp)
				# cmap = ggrf.generate_topology_map(N4.numpy(),N4.numpy(),radius,width,rng=rng,\
		 		# 									  return_complex=True, symmetrise=True)
				# # l4 = np.real(cmap).flatten()*500 + 1000
				# l4 = np.clip(np.real(cmap).flatten(),0,np.nanmax(np.real(cmap)))
				# l4 = tf.convert_to_tensor(l4, dtype=tf.float32)
				# l4 = tf.concat([l4,l4],0)
				# # l4 = tf.clip_by_value(l4,0.,100000.)
				# y = tf.concat([y[:lim*num_lgn_paths],l4], 0)
				###########################################################################
				############################################################################

				if (istep%self.saving_stepsize)==0:
					l4t = tf.concat([l4t,[y[lim*num_lgn_paths:num_lgn_paths*lim+crt_size]]], 0)

				if np.nanmean(y[lim*num_lgn_paths:])>self.divergence_limit:
					dW += self.plasticity_dynamics(y*0, 0, self.params_dict,\
							unconst_plasticity_update=self.unconst_plasticity_update)
					print("Activity diverged (={}), skip plasticity update step: {},\
						   input: {}".format(np.nanmean(y[lim*num_lgn_paths:]),istep,jinput))
					sys.stdout.flush()
				
				####### ff plasticity update #######
				dW += self.plasticity_dynamics(y, 0, self.params_dict,\
											   unconst_plasticity_update=\
											   self.unconst_plasticity_update)


			if np.isnan(np.nanmedian(dW[dW>0])):
				print("Median plasticity update is NaN at t={}, exit integration!".format(\
					  jinput))
				break

			
			####### Plasticity update averaged over avg_no_inp number of input patterns ######
			s = num_lgn_paths*lim
			print("dW",np.nanmax(dW[:s]),np.nanmin(dW[:s]))			
			Wlgn_to_4 = dynamics.constrain_plasticity_update_wrapper(y,dt,dW[:s],arbor2,\
																		Wlim,self.params_dict,\
																		mode,c_orth,s_orth)
		
		
			####### synaptic normalisation #######
			print("norm after update",\
				np.sum(np.abs(np.nansum(Wlgn_to_4,axis=(0,2))-init_norm_alpha)),\
				np.sum(np.abs(np.nansum(Wlgn_to_4,axis=1)-init_norm_x)))
			print("sum",np.nansum(Wlgn_to_4,axis=(0,2))[:5],np.nansum(Wlgn_to_4,axis=1)[0,:5])
			Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor2)

			if check_for_frozen_weights(Wlgn_to_4,Wlim,arbor2):
				l4 = l4t[-1]
				if self.l4_target is None:
					self.l4_target = l4t[1]
				Wlgn_to_4,H_new = self.mult_normalization(Wlgn_to_4,\
														  arbor2,Wlim,\
														  init_weights,c_orth,\
														  H,l4,self.l4_target)
				H = H_new
				print("norm after clip/div norm",\
					np.sum(np.abs(np.nansum(Wlgn_to_4,axis=(0,2))-init_norm_alpha)),\
					np.sum(np.abs(np.nansum(Wlgn_to_4,axis=1)-init_norm_x)))
			print("sum",np.nansum(Wlgn_to_4,axis=(0,2))[:5],np.nansum(Wlgn_to_4,axis=1)[0,:5])
			print("Wlgn_to_4_new",Wlgn_to_4[arbor2>0].numpy()[:5])


			####### check fraction of synaptic ff weights are frozen #######
			frozen = np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)<=0) + \
					 np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)>=Wlim*(arbor2[arbor2>0]))
			if frozen>(0.9*np.sum(arbor2.numpy())):
				self.frozen = True
				print("0.9 of all ff weights are frozen at t={}, exit integration!".format(\
					  jinput))
				break
			Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim*num_lgn_paths])


			## if w4to23 is plastic do multiplicative normalisation
			if self.params_dict["config_dict"]["W4to23_params"]["plastic"]:
				l23_size = self.params_dict["N23"]**2*2
				W4to23 = y[s+crt_size:]
				arbor4to23_full = self.params_dict["arbor4to23_full"]
				print("dW4to23",np.nanmax(dW[s:]),np.nanmin(dW[s:]))
				W4to23 = dynamics.constrain_plasticity_update_W4to23(y,dt,dW[s:],arbor4to23,\
																arbor4to23_full,Wlim,\
																self.params_dict,c_orth_4to23,\
																s_orth_4to23)
				W4to23 = clip_weights(W4to23,Wlim,arbor4to23_full)
				init_weights_4to23 = self.params_dict["init_weights_4to23"]
				W4to23,_ = synaptic_normalization(W4to23,arbor4to23_full,Wlim,init_weights_4to23,\
												axis=0)
				W4to23 = tf.reshape(W4to23,[l23_size*l4_size])
				####### concatenate updated connectivity to y vector #######
				y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:lim*num_lgn_paths+crt_size],\
								W4to23],0)
			else:
				y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:]],0)


			if (istep%self.saving_stepsize)==0:
				yt = tf.concat([yt,[y]], 0)
		return yt,l4t
		

	def _step_func(self, func, y, t, dt, **kwargs):
		if self.scheme=="Euler":
			dy = dt*func(y, t, self.params_dict, **kwargs)
		elif self.scheme=="RK4":
			k1 = func(y, t, self.params_dict, **kwargs)
			half_step = t + dt/2.
			dt_cast = tf.cast(dt, y.dtype)
			k2 = func(y + dt_cast * k1 / 2., half_step, self.params_dict, **kwargs)
			k3 = func(y + dt_cast * k2 / 2., half_step, self.params_dict, **kwargs)
			k4 = func(y + dt_cast * k3, t + dt, self.params_dict, **kwargs)
			dy = tf.add_n([k1, 2*k2, 2*k3, k4]) * (dt_cast / 6.)
		return dy


def odeint_new(func1, y0, t, dt, params_dict, mode="dynamic"):
	t = tf.convert_to_tensor(t, name='t')
	y0 = tf.convert_to_tensor(y0, name='y0')
	dt = tf.convert_to_tensor(dt,name="dt")
	tf_check_type(t, y0)
	if mode=="dynamic":
		return Tf_integrator_new(params_dict).integrate(func1,y0,t,dt)
	elif mode=="static":
		print("mode static not implemented yet")
		return None
	elif mode=="single_stim_update":
		print("mode single_stim_update not implemented yet")
		return None




class Tf_integrator:
	def __init__(self, params_dict):
		self.params_dict = params_dict
		self.scheme = params_dict["integrator"]
		self.avg_no_inp = params_dict["avg_no_inp"]
		self.frozen = False
	
	def integrate(self, func, y0, t, dt):
		# time_delta_grid = dt*tf.ones(t.shape,dtype=tf.float32)

		def scan_func(y, t_dt, yt):
			t, dt = t_dt
			dy = self._step_func(func, y, t, dt, yt)
			out = y + dy
		
			T_pd = self.params_dict["pattern_duration"]
			

			## update ff weights after activity has converged for specific input
			# if (t%T_pd)==(T_pd-1):
			if ((((t+1)/T_pd)%self.avg_no_inp)==1 and t>0):
				N4 = self.params_dict["N4"]
				Nlgn = self.params_dict["Nlgn"]
				Nvert = self.params_dict["Nvert"]
				arbor = self.params_dict["arbor"]
				lim = 2*Nlgn*Nlgn*N4*N4*Nvert

				weight_change = dy[:lim].numpy()
				arbor2 = np.stack([arbor,arbor]).flatten()
				init_weights = self.params_dict["init_weights"]
				Wlim = self.params_dict["Wlim"]
				Wlgn_to_4 = out[:lim]
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
				Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

				Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
				if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
				   (1.8*np.sum(arbor.numpy())):
					self.frozen = True
				out = tf.concat([Wlgn_to_4,out[lim:]],0)
			# if (t%T_pd)==(T_pd-1):
			# 	out = tf.concat([out[:lim],0*out[lim:]],0)

			return out

		# y = tf.scan(scan_func, (t, time_delta_grid), y0, swap_memory=True)
		# y = y[::100,:]

		y = y0
		yt = tf.convert_to_tensor([y0])
		N4 = self.params_dict["N4"]
		Nlgn = self.params_dict["Nlgn"]
		Nvert = self.params_dict["Nvert"]
		lim = 2*Nlgn*Nlgn*N4*N4*Nvert
		l4t = tf.convert_to_tensor([y0[lim:]])
		# cc_lgn_l4_t = []
		T_pd = self.params_dict["pattern_duration"]
		for it in t:
			if self.frozen:
				print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
				break
			y = scan_func(y, (it, dt), yt)

			## set yt to y again to only store values since last W update
			if ((((it+1)/T_pd)%self.avg_no_inp)==0 and it>0):
				no_update_step = int(((it+1)/T_pd) / self.avg_no_inp)
				print(no_update_step,int(tf.math.floor(it/T_pd)));sys.stdout.flush()
				# yt = tf.convert_to_tensor([y0])
				# for jt in range(no_update_step):
				# 	print("jt",jt,yt[jt,10])
				# 	yt = tf.concat([yt,[yt[jt,:]]], 0)
				yt = yt[:no_update_step,:]

				# if False:
				# 	print("l4t",(l4t.numpy()).shape)
				# 	import matplotlib.pyplot as plt
				# 	fig = plt.figure()
				# 	ax = fig.add_subplot(121)
				# 	ax.plot(np.nanmean(l4t,axis=1),'-k')
				# 	ax.plot(l4t[:,100],'-m')
				# 	ax = fig.add_subplot(122)
				# 	im=ax.imshow((l4t[-500,:N4**2].numpy()).reshape(N4,N4),interpolation="nearest",cmap="binary")
				# 	plt.colorbar(im,ax=ax)
				# 	plt.savefig(image_dir + \
				# 		"layer4/tmp/v96_rA2.0_srec1.00_scc0.04_xalpha_AVG_test/activity_pattern/act_{}_ev95_t-1.pdf".format(it))
				# 	plt.close()

				# print("set yt to zero again",yt.shape)

			## store values of l4 act and W as fct of timesteps
			## TODO: only keep values of l4 act (needed for plasticity step)
			if (it%T_pd)==(T_pd-1):
				# print("append yt",it,yt.shape)
				yt = tf.concat([yt,[y]], 0)
				l4t = tf.concat([l4t,[y[lim:]]],0)

		no_update_step = int(np.ceil(((it+1)/T_pd) / self.avg_no_inp))
		yt = yt[:no_update_step+1,:]

		return yt,l4t
		
	def _step_func(self, func, y, t, dt, yt):
		if self.scheme=="Euler":
			dy = dt*func(y, t, self.params_dict, yt)
		elif self.scheme=="RK4":
			k1 = func(y, t, self.params_dict)
			half_step = t + dt/2.
			dt_cast = tf.cast(dt, y.dtype)
			k2 = func(y + dt_cast * k1 / 2., half_step, self.params_dict)
			k3 = func(y + dt_cast * k2 / 2., half_step, self.params_dict)
			k4 = func(y + dt_cast * k3, t + dt, self.params_dict)
			dy = tf.add_n([k1, 2*k2, 2*k3, k4]) * (dt_cast / 6.)
		return dy


	def integrate_sgl(self,func,y0,t,dt):
		"""use this function if plasticity update is done after each new stimulus,
		   and time dependence of W and L4 is not saved"""
		def scan_func_sgl(y, t_dt):
			t, dt = t_dt
			dy = self._step_func(func, y, t, dt, None)
			# print("out",y.shape,dy.shape)
			out = y + dy

			T_pd = self.params_dict["pattern_duration"]
			## update ff weights after activity has converged for specific input
			if ((((t+1)/T_pd)%self.avg_no_inp)==1 and t>0):
				N4 = self.params_dict["N4"]
				Nlgn = self.params_dict["Nlgn"]
				Nvert = self.params_dict["Nvert"]
				arbor = self.params_dict["arbor"]
				lim = 2*Nlgn*Nlgn*N4*N4*Nvert

				weight_change = dy[:lim].numpy()
				arbor2 = np.stack([arbor,arbor]).flatten()
				init_weights = self.params_dict["init_weights"]
				Wlim = self.params_dict["Wlim"]
				Wlgn_to_4 = out[:lim]
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
				Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

				Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
				if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
				   (1.8*np.sum(arbor.numpy())):
					self.frozen = True
				out = tf.concat([Wlgn_to_4,out[lim:]],0)
			return out

		y = y0
		N4 = self.params_dict["N4"]
		Nlgn = self.params_dict["Nlgn"]
		Nvert = self.params_dict["Nvert"]
		lim = 2*Nlgn*Nlgn*N4*N4*Nvert
		T_pd = self.params_dict["pattern_duration"]
		for it in t:
			if self.frozen:
				print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
				break
			y = scan_func_sgl(y, (it, dt))

		return y

	def integrate_static(self, func, y0, t, dt):
		"""use this function if layer 4 activity is not simulated"""
		def scan_func_static(y, t_dt, yt):
			t, dt = t_dt
			dy = self._step_func(func, y, t, dt, yt)
			# print("out",y.shape,dy.shape)
			out = y + dy

			## update ff weights after activity has converged for specific input
			if (((t+1)%self.avg_no_inp)==0 and t>0):
				N4 = self.params_dict["N4"]
				Nlgn = self.params_dict["Nlgn"]
				Nvert = self.params_dict["Nvert"]
				arbor = self.params_dict["arbor"]
				lim = 2*Nlgn*Nlgn*N4*N4*Nvert

				init_weights = self.params_dict["init_weights"]
				Wlim = self.params_dict["Wlim"]
				Wlgn_to_4 = out
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[2,N4*N4*Nvert,Nlgn*Nlgn])
				Wlgn_to_4 = clip_weights(Wlgn_to_4,Wlim,arbor)

				Wlgn_to_4 = synaptic_normalization(Wlgn_to_4,arbor,Wlim,init_weights)
				Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim])
				arbor2 = np.stack([arbor,arbor]).flatten()
				print("check lims",np.sum(Wlgn_to_4[arbor2>0]==0),\
					np.sum(Wlgn_to_4[arbor2>0]==Wlim));sys.stdout.flush()
				if (np.sum(Wlgn_to_4[arbor2>0]==0)+np.sum(Wlgn_to_4[arbor2>0]==Wlim))>\
				(1.8*np.sum(arbor.numpy())):
					self.frozen = True
				out = Wlgn_to_4

			return out

		y = y0
		yt = tf.convert_to_tensor([y0])
		# N4 = self.params_dict["N4"]
		# Nlgn = self.params_dict["Nlgn"]
		# Nvert = self.params_dict["Nvert"]
		# lim = 2*Nlgn*Nlgn*N4*N4*Nvert
		for it in t:
			if self.frozen:
				print("0.9 of all ff weights are frozen at t={}, exit integration!".format(it))
				break
			y = scan_func_static(y, (it, dt), yt)

			## set yt to y again to only store values since last W update
			if (((it+1)%self.avg_no_inp)==0 and it>0):
				print("it","append",yt.shape);sys.stdout.flush()
				yt = tf.concat([yt,[y]], 0)

		return yt

		
def odeint(func, y0, t, dt, params_dict, mode="dynamic"):
	t = tf.convert_to_tensor(t, name='t')
	y0 = tf.convert_to_tensor(y0, name='y0')
	dt = tf.convert_to_tensor(dt,name="dt")
	tf_check_type(t, y0)
	if mode=="dynamic":
		return Tf_integrator(params_dict).integrate(func,y0,t,dt)
	elif mode=="static":
		return Tf_integrator(params_dict).integrate_static(func,y0,t,dt)
	elif mode=="single_stim_update":
		return Tf_integrator(params_dict).integrate_sgl(func,y0,t,dt)
		
	#tf.placeholder(tf.float32)

