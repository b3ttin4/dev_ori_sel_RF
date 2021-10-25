import sys
import tensorflow as tf
import numpy as np
from collections import defaultdict

from bettina.modeling.ori_dev_model import image_dir,inputs,network_tools,dynamics,\
normalisation_constraints,plasticity_dynamics
from bettina.modeling.ori_dev_model.tools import calc_tools,\
get_spatial_wavelength,gen_lowd_activity_patterns


def tf_check_type(t, y0): 
	# Ensure Input type is Correct
	if not (y0.dtype.is_floating and t.dtype.is_floating):
		raise TypeError('Error in Datatype')


# def clip_weights(Wlgn_to_4,Wlim,arbor):
# 	return tf.clip_by_value(Wlgn_to_4,0,arbor*Wlim)

def check_for_frozen_weights(Wlgn_to_4,Wlim,arbor):
	frozen = tf.math.logical_or(Wlgn_to_4[arbor>0]>=(Wlim*arbor[arbor>0]), Wlgn_to_4[arbor>0]<=0)
	return np.sum(frozen)>0



class Tf_integrator_new:
	def __init__(self, params_dict):
		self.params_dict = params_dict
		self.scheme = params_dict["config_dict"]["integrator"]
		self.avg_no_inp = params_dict["config_dict"]["Inp_params"]["avg_no_inp"]
		self.num_inputs = int(self.params_dict["config_dict"]["runtime"]/\
							self.params_dict["config_dict"]["dt"]/\
							self.params_dict["config_dict"]["Inp_params"]["pattern_duration"])
		self.num_plasticity_steps = int(self.num_inputs/self.avg_no_inp)
		self.expanse_steps = int(self.params_dict["config_dict"]["Inp_params"]["expanse_time"])
		self.frozen = False
		
		self.nonlinearity_rule_l4 = params_dict["config_dict"]["nonlinearity_l4"]
		self.nonlinearity_rule_l23 = params_dict["config_dict"]["nonlinearity_l23"]

		self.num_lgn_paths = params_dict["config_dict"]["num_lgn_paths"]
		self.saving_stepsize = params_dict["config_dict"]["saving_stepsize"]
		self.system = params_dict["config_dict"]["system"]
		self.connectivity_type = params_dict["config_dict"]["Wlgn_to4_params"]["connectivity_type"]


		self._init_inverse_cortical_interaction_function()
		nl = self._init_nonlinearity(self.nonlinearity_rule_l4)
		self.params_dict.update({"nonlinearity_l4" : nl})
		nl = self._init_nonlinearity(self.nonlinearity_rule_l23)
		self.params_dict.update({"nonlinearity_l23" : nl})

		self.l4_target = None
		self._init_cortical_layer_dynamics()
		
		tf.random.set_seed(self.params_dict["config_dict"]["random_seed"]*114)
		self.divergence_limit = 10000000.

		self._init_running_avg()

		self.estimate_spatial_wavelength_activity_from_conn()

		# self.plastic = plasticity_dynamics.Plasticity(self.params_dict)
		self._init_plasticity_dynamics()


		
	def _init_nonlinearity(self,nonlinearity_rule):
		# TODO: add two different nonlinearities for L 4 and 2/3
		if nonlinearity_rule == 'rectifier': 
			tf_nonlinearity = network_tools.nl_rect
		elif nonlinearity_rule == 'linear': 
			tf_nonlinearity = network_tools.nl_linear
		elif nonlinearity_rule == "powerlaw":
			tf_nonlinearity = network_tools.nl_powerlaw
		else:
			raise Exception('Unknown nonlinearity rule')
		return tf_nonlinearity
	

	def _init_cortical_layer_dynamics(self):
		if self.system=="one_layer":
			self.rhs_l23 = None
			if self.connectivity_type=="E":
				self.rhs_l4 = dynamics.rhs_l4EI

			elif self.connectivity_type=="EI":
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


	def yield_init_conditions(self,y):
		"""initiate conditions for cortical layer(s)"""
		if self.params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="dynamics":
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
				
				if self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"] is not None:
					y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230,\
									y[self.num_lgn_paths*lim+N4*N4*2*Nvert+N23*N23*2:]], axis=0)
				else:
					y0 = tf.concat([y[:self.num_lgn_paths*lim], l40, l230], axis=0)

			else:
				raise Exception('System not defined. Choose either one_layer or two_layer.')
		else:
			y0 = y	
		return y0


	def _init_inverse_cortical_interaction_function(self):
		if self.params_dict["config_dict"]["Inp_params"]["simulate_activity"]=="steady_state":
			if self.system=="one_layer":
				N4 = self.params_dict["N4"]
				Nvert = self.params_dict["Nvert"]
				W4to4 = self.params_dict["W4to4"]
				I_crt = np.linalg.inv(np.diagflat(np.ones(N4*N4*2*Nvert)) - W4to4)
				self.I_crt = tf.convert_to_tensor(I_crt, dtype=tf.float32)
				self.I_crt23 = None

			elif self.system=="two_layer":
				N4 = self.params_dict["N4"]
				N23 = self.params_dict["N23"]
				Nvert = self.params_dict["Nvert"]
				W4to4 = self.params_dict["W4to4"]
				W4to23 = self.params_dict["W4to23"]
				W23to4 = self.params_dict["W23to4"]
				W23to23 = self.params_dict["W23to23"]

				I = np.diagflat(np.ones(N23**2*2))
				Ir = np.linalg.inv(I - W4to4)
				Wtmp = np.dot(W4to23,Ir)
				self.I_crt23 = np.dot(np.linalg.inv(I - W23to23 - np.dot(Wtmp,W23to4)),Wtmp)

				I = np.diagflat(np.ones(N4**2*2*Nvert))
				Ix = np.linalg.inv(I - W23to23)
				self.I_crt = np.linalg.inv(I - W4to4 - np.dot(np.dot(W23to4,Ix),W4to23))

				ew23,_ = np.linalg.eig(W23to23 + np.dot(Wtmp,W23to4))
				ew4,_ = np.linalg.eig(W4to4 + np.dot(np.dot(W23to4,Ix),W4to23))
				print("Eigvals, L23: {}".format(np.nanmax(np.real(ew23))))
				print("Eigvals, L4: {}".format(np.nanmax(np.real(ew4))))
				sys.stdout.flush()

		else:
			self.I_crt = None
			self.I_crt23 = None


	def _init_running_avg(self):
		self.count = 1.
		self.avg = None
		self.tau = 1.
		self.len_running_avg = 2
		self.avg_ampl = 2.
		N4 = self.params_dict["N4"]
		Nvert = self.params_dict["Nvert"]
		l4_size = N4**2*Nvert
		self.running_l4_avg = None

	
	def running_avg(self,new_value):
		new_value = tf.reshape(new_value,[2,-1])
		if self.running_l4_avg is None:
			self.running_l4_avg = new_value
			self.avg = new_value
		else:
			self.count =  (self.count * np.exp(-1./self.tau) + 1)
			self.avg = (self.avg * np.exp(-1./self.tau) + new_value)
			self.running_l4_avg = self.avg/self.count
		# self.running_l4_avg = tf.roll(self.running_l4_avg,shift=-1,axis=0)
		# indices = tf.constant([[self.len_running_avg-1],])
		# updates = [y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size],]
		# self.running_l4_avg = tf.tensor_scatter_nd_update(self.running_l4_avg, indices, updates)


	def estimate_spatial_wavelength_activity_from_conn(self):
		# estimate_spatial_wavelength_activity_from_conn
		k = np.arange(0,60,0.1)
		aee = self.params_dict["config_dict"]["W4to4_params"]["aEE"]
		aie = self.params_dict["config_dict"]["W4to4_params"]["aIE"]
		aei = self.params_dict["config_dict"]["W4to4_params"]["aEI"]
		aii = self.params_dict["config_dict"]["W4to4_params"]["aII"]
		sigma_factor = self.params_dict["config_dict"]["W4to4_params"]["sigma_factor"]
		see = self.params_dict["config_dict"]["W4to4_params"]["sigma_EE"] * sigma_factor
		sei = self.params_dict["config_dict"]["W4to4_params"]["sigma_EI"] * sigma_factor
		sii = self.params_dict["config_dict"]["W4to4_params"]["sigma_II"] * sigma_factor
		eigvals = get_spatial_wavelength.eigval_max(k,aee,aei*aie,aii,see,sei,sii,tau=1,alpha=0)
		max_freq = k[np.nanargmax(eigvals)]
		print("max_freq",max_freq,1./max_freq,2*np.pi/max_freq)

		ndim = 4
		width = 0.2
		size = self.params_dict["config_dict"]["N4"]
		freq = max_freq
		lowD_subset = gen_lowd_activity_patterns.create_activity_patterns(freq,size,ndim,width)
		cc = np.corrcoef(lowD_subset)
		lowD_subset = np.clip(1000 * lowD_subset,0,100000)
		lowD_subset = tf.convert_to_tensor(lowD_subset, dtype=tf.float32)
		lowD_subset = tf.concat([lowD_subset,lowD_subset],0)
		self.params_dict["lowD_subset"] = lowD_subset

		self.params_dict["Corr"] = tf.convert_to_tensor(cc,dtype=tf.float32)

		
	def _init_plasticity_dynamics(self):
		p_dict = defaultdict(lambda:None)

		dt = self.params_dict["config_dict"]["dt"]

		c_orth = self.params_dict["c_orth"]
		s_orth = self.params_dict["s_orth"]
		beta_P = self.params_dict["config_dict"]["Wlgn_to4_params"]["beta_P"]
		Wlim = self.params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
		plasticity_rule = self.params_dict["config_dict"]["Wlgn_to4_params"]["plasticity_rule"]
		constraint_mode = self.params_dict["config_dict"]["Wlgn_to4_params"]["constraint_mode"]
		mult_norm = self.params_dict["config_dict"]["Wlgn_to4_params"]["mult_norm"]
		init_weights = None
		if mult_norm in ("x","alpha"):
			print(self.params_dict["init_weights"].shape)
			init_weights = self.params_dict["init_weights"][:2,:]
		
		p_dict["p_lgn_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
							plasticity_rule,constraint_mode,mult_norm,Wlim,init_weights)

		if self.connectivity_type=="EI":
			init_weights = None
			if mult_norm in ("x","alpha"):
				init_weights = self.params_dict["init_weights"][2:,:]
			p_dict["p_lgn_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm,Wlim,init_weights)

		if self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"] is not None:
			c_orth = self.params_dict["c_orth_4to23"]
			s_orth = self.params_dict["s_orth_4to23"]
			beta_P = self.params_dict["config_dict"]["W4to23_params"]["beta_P"]
			Wlim = self.params_dict["config_dict"]["W4to23_params"]["Wlim"]
			init_weights = self.params_dict["init_weights_4to23"]
			N = init_weights.shape[0]//2

			plasticity_rule = self.params_dict["config_dict"]["W4to23_params"]["plasticity_rule"]
			constraint_mode = self.params_dict["config_dict"]["W4to23_params"]["constraint_mode"]
			mult_norm = self.params_dict["config_dict"]["W4to23_params"]["mult_norm"]
			p_dict["p_4to23_e"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm,Wlim,init_weights[:N])
			p_dict["p_4to23_i"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm,Wlim,init_weights[N:])

		if self.params_dict["config_dict"]["W4to4_params"]["plasticity_rule"] is not None:
			c_orth = None
			s_orth = None
			beta_P = self.params_dict["config_dict"]["W4to4_params"]["beta_P"]
			plasticity_rule = self.params_dict["config_dict"]["W4to4_params"]["plasticity_rule"]
			constraint_mode = self.params_dict["config_dict"]["W4to4_params"]["constraint_mode"]
			mult_norm = self.params_dict["config_dict"]["W4to4_params"]["mult_norm"]
			p_dict["p_rec4_ei"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm)
			p_dict["p_rec4_ii"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm)

		if self.params_dict["config_dict"]["W23_params"]["plasticity_rule"] is not None:
			c_orth = None
			s_orth = None
			beta_P = self.params_dict["config_dict"]["W23_params"]["beta_P"]
			plasticity_rule = self.params_dict["config_dict"]["W23_params"]["plasticity_rule"]
			constraint_mode = self.params_dict["config_dict"]["W23_params"]["constraint_mode"]
			mult_norm = self.params_dict["config_dict"]["W23_params"]["mult_norm"]
			p_dict["p_rec23_ei"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm)
			p_dict["p_rec23_ii"] = plasticity_dynamics.Plasticity(dt,c_orth,s_orth,beta_P,\
								plasticity_rule,constraint_mode,mult_norm)

		self.p_dict = p_dict


	def update_developing_parameters(self,timestep):
		"""
		update before (!) each plasticity update step all parameters
		that are changing during development, 
		e.g. beta, off_bias_strength, off_bias_region_size,onoff_rel_weight
		arbor parameters, 
		"""

		for item in self.params_dict["config_dict"]["developing_params"]:

			if item=="learning_rate_l4":
				self.params_dict["config_dict"]["W4to4_params"]["learning_rate"] *=\
				 np.exp(-1./self.params_dict["config_dict"]["learning_rate_decay_const"])

			elif item=="learning_rate_l23":
				self.params_dict["config_dict"]["W23_params"]["learning_rate"] *=\
				 np.exp(-1./self.params_dict["config_dict"]["learning_rate_decay_const"])


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
		Wlim = self.params_dict["config_dict"]["Wlgn_to4_params"]["Wlim"]
		num_lgn_paths = self.params_dict["config_dict"]["num_lgn_paths"]

		T_pd = self.params_dict["config_dict"]["Inp_params"]["pattern_duration"]
		# mode = self.params_dict["normalisation_mode"]
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
		# print("init_sum",init_norm_alpha[:5],init_norm_x[0,:5])
		H = tf.convert_to_tensor(np.ones((2,N4**2)),dtype=tf.float32)


		## init activity
		y = y0
		yt = tf.convert_to_tensor([y0])
		l4t = tf.convert_to_tensor([y0[num_lgn_paths*lim:num_lgn_paths*lim+crt_size]])
		tf.random.set_seed(self.params_dict["config_dict"]["random_seed"]*113)
		
		print("Number of plasticity steps: {}".format(self.num_plasticity_steps))
		for istep in range(self.num_plasticity_steps):
			dW = defaultdict(lambda:0)
			for jinput in range(self.avg_no_inp):

				self.update_developing_parameters(istep)
				print("CHECK PARAM UPDATE",self.params_dict["config_dict"]["W4to4_params"]["learning_rate"])

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

				shift = tf.random.categorical(tf.math.log([[0.5,0.5]]),2) * 2 - 1
				shift_y,shift_x = shift[0,0],shift[0,1]
				for kexp in range(self.expanse_steps):
					print("")
					print("Plasticity step: {}, input: {}, exp: {}".format(istep,jinput,kexp))
					sys.stdout.flush()

					# shift input pattern
					lgn = tf.reshape(self.params_dict["lgn"],[-1,Nlgn,Nlgn])
					lgn = tf.roll(tf.roll(lgn,shift_y,axis=1),shift_x,axis=2)
					self.params_dict.update({"lgn" : tf.reshape(lgn,[-1,Nlgn**2])})

					y0 = self.yield_init_conditions(y)
					y = y0
					####### L4 activity dynamics #######
					y,l4_avg,theta_4 = l4_dynamics(scan_func,y,t,dt,self.params_dict,\
										rhs_l4=self.rhs_l4,rhs_l23=self.rhs_l23,\
										I_crt=self.I_crt,I_crt23=self.I_crt23)
					self.params_dict["config_dict"]["W4to4_params"]["theta_4"] = theta_4
					self.params_dict["config_dict"]["W4to4_params"]["l4_avg"] = l4_avg
					l4 = y[lim*num_lgn_paths:num_lgn_paths*lim+crt_size]
					print("L4",istep,np.nanmin(l4),np.nanmax(l4),np.sum(np.logical_not(np.isfinite(l4))))

					# compute running average of l4 activity
					self.running_avg(y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size])

					# save timestep of cortical activity
					if ((istep%self.saving_stepsize)==0 or istep==(self.num_plasticity_steps-1)):
						l4t = tf.concat([l4t,[y[lim*num_lgn_paths:num_lgn_paths*lim+crt_size]]], 0)

					# check whether cortical activity has diverged, if so break
					if np.nanmean(y[lim*num_lgn_paths:])>self.divergence_limit:
						dW += self.plastic.plasticity_wrapper(y*0,0)
						print("Activity diverged (={}), skip plasticity update step: {},\
							   input: {}".format(np.nanmean(y[lim*num_lgn_paths:]),istep,jinput))
						sys.stdout.flush()
					
					# compute target activity for homeostatic normalisation
					if self.l4_target is None:
						self.l4_target = self.avg_ampl*tf.reduce_mean(tf.reshape(\
										 y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size],[2,-1]),axis=1)

					# ff plasticity update
					# dW += self.plastic.plasticity_wrapper(y,0)
					# dW = tf.reshape(dW, [s])
					l4 = y[lim*num_lgn_paths:num_lgn_paths*lim+l4_size]
					l23 = y[lim*num_lgn_paths+l4_size:]
					Wlgn_to_4 = tf.reshape(y[:lim*num_lgn_paths],[num_lgn_paths,l4_size//2,-1])
					dW_dict = plasticity_dynamics.unconstrained_plasticity_wrapper(self.p_dict,\
								l4, l23, lgn, Wlgn_to_4, self.params_dict["W4to4"],\
								self.params_dict["W4to23"], self.params_dict["W23to23"], t)
					for key,value in dW_dict.items():
						dW[key] += value
						print("key",istep,np.nanmin(value),np.nanmax(value))
					
					
			if np.isnan(np.nanmedian(dW["dW_lgn_e"][dW["dW_lgn_e"]>0])):
				print("Median plasticity update is NaN at t={}, exit!".format(jinput))
				break

			
			# Plasticity update averaged over avg_no_inp number of input patterns
			s = num_lgn_paths*lim
			# Wlgn_to_4 = self.plastic.constraint_update_wrapper(y,dt,dW[:s],arbor2,c_orth,s_orth)

			Wlgn_to_4 = tf.reshape(y[:s],[num_lgn_paths,l4_size//2,Nlgn**2])
			Wlgn_to_4,W4to4,W4to23,W23to23 =\
			 plasticity_dynamics.constraint_update_wrapper(dW,self.p_dict,Wlgn_to_4,\
														self.params_dict["arbor2"],\
														self.params_dict["W4to4"],\
														self.params_dict["arbor4to4"],\
														self.params_dict["W4to23"],\
														self.params_dict["arbor4to23"],\
														self.params_dict["W23to23"],\
														self.params_dict["arbor23to23"],dt)
			self.params_dict["W4to4"] = W4to4
			self.params_dict["W4to23"] = W4to23
			self.params_dict["W23to23"] = W23to23

			# synaptic normalisation
			# Wlgn_to_4 = self.plastic.clip_weights(Wlgn_to_4,Wlim,arbor2)
			Wlgn_to_4,W4to4,W4to23,W23to23 =\
			 plasticity_dynamics.clip_weights_wrapper(self.p_dict,Wlgn_to_4,\
													self.params_dict["arbor2"],\
													self.params_dict["W4to4"],\
													self.params_dict["arbor4to4"],\
													self.params_dict["W4to23"],\
													self.params_dict["arbor4to23"],\
													self.params_dict["W23to23"],\
													self.params_dict["arbor23to23"])
			self.params_dict["W4to4"] = W4to4
			self.params_dict["W4to23"] = W4to23
			self.params_dict["W23to23"] = W23to23

			if True:#check_for_frozen_weights(Wlgn_to_4,Wlim,arbor2):
				# Wlgn_to_4,H = self.plastic.mult_normalization(Wlgn_to_4,arbor2,Wlim,\
				# 												init_weights,\
				# 												c_orth,H,self.running_l4_avg,\
				# 												self.l4_target)			  
				Wlgn_to_4,W4to4,W4to23,W23to23,H =\
				 plasticity_dynamics.mult_norm_wrapper(self.p_dict,Wlgn_to_4,\
													self.params_dict["arbor2"],\
													self.params_dict["W4to4"],\
													self.params_dict["arbor4to4"],\
													self.params_dict["W4to23"],\
													self.params_dict["arbor4to23"],\
													self.params_dict["W23to23"],\
													self.params_dict["arbor23to23"],\
													H,self.running_l4_avg,self.l4_target)
				self.params_dict["W4to4"] = W4to4
				self.params_dict["W4to23"] = W4to23
				self.params_dict["W23to23"] = W23to23

			# check fraction of synaptic ff weights are frozen
			frozen = np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)<=0) + \
					 np.sum(tf.boolean_mask(Wlgn_to_4,arbor2)>=Wlim*(arbor2[arbor2>0]))
			if frozen>(0.9*np.sum(arbor2.numpy())):
				self.frozen = True
				print("0.9 of ff weights are frozen at t={}, exit integration!".format(jinput))
				break
			Wlgn_to_4 = tf.reshape(Wlgn_to_4,[lim*num_lgn_paths])


			## if w4to23 is plastic do multiplicative normalisation
			# if self.params_dict["config_dict"]["W4to23_params"]["plastic"]:
			# 	l23_size = self.params_dict["N23"]**2*2
			# 	W4to23 = y[s+crt_size:]
			# 	arbor4to23_full = self.params_dict["arbor4to23_full"]
			# 	W4to23 = dynamics.constrain_plasticity_update_W4to23(y,dt,dW[s:],arbor4to23,\
			# 													arbor4to23_full,Wlim,\
			# 													self.params_dict,c_orth_4to23,\
			# 													s_orth_4to23)
			# 	W4to23 = self.plastic.clip_weights(W4to23,Wlim,arbor4to23_full)
			# 	init_weights_4to23 = self.params_dict["init_weights_4to23"]
			# 	W4to23,_ = synaptic_normalization(W4to23,arbor4to23_full,Wlim,init_weights_4to23,\
			# 									axis=0)
			# 	W4to23 = tf.reshape(W4to23,[l23_size*l4_size])
			# 	####### concatenate updated connectivity to y vector #######
			# 	y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:lim*num_lgn_paths+crt_size],\
			# 					W4to23],0)
			# else:
			y = tf.concat([Wlgn_to_4,y[lim*num_lgn_paths:]],0)


			if ((istep%self.saving_stepsize)==0 or istep==(self.num_plasticity_steps-1)):
				yt = tf.concat([yt,[y]], 0)
		return yt,l4t
		

	def _step_func(self, func, y, t, dt, **kwargs):
		if self.scheme=="Euler":
			dt_cast = tf.cast(dt, y.dtype)
			dy = dt_cast*func(y, t, **kwargs)
		elif self.scheme=="RK4":
			k1 = func(y, t, **kwargs)
			half_step = t + dt/2.
			dt_cast = tf.cast(dt, y.dtype)
			k2 = func(y + dt_cast * k1 / 2., half_step, **kwargs)
			k3 = func(y + dt_cast * k2 / 2., half_step, **kwargs)
			k4 = func(y + dt_cast * k3, t + dt, self.params_dict, **kwargs)
			dy = tf.add_n([k1, 2*k2, 2*k3, k4]) * (dt_cast / 6.)
		return dy


def odeint_new(func1, y0, t, dt, params_dict, mode="dynamic"):
	t = tf.convert_to_tensor(t, name='t', dtype=tf.float32)
	y0 = tf.convert_to_tensor(y0, name='y0', dtype=tf.float32)
	dt = tf.convert_to_tensor(dt,name="dt", dtype=tf.float32)
	tf_check_type(t, y0)
	if mode=="dynamic":
		return Tf_integrator_new(params_dict).integrate(func1,y0,t,dt)
	elif mode=="static":
		return Tf_integrator_new(params_dict).integrate(func1,y0,t,dt)
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

