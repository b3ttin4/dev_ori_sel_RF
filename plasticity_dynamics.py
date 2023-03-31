import tensorflow as tf
import numpy as np
# from collections import defaultdict

# plasticity rules
def activity_based(t,postsyn_act,presyn_act,W,beta_P,N=None):
	update = postsyn_act[:,None] * (beta_P * presyn_act[None,:] )
	return update

def activity_based_EI_input(t,postsyn_act,presyn_act,W,beta_P,N):
	E_update = postsyn_act[None,:N,None] * (beta_P * presyn_act[:2,None,:] )
	I_update = postsyn_act[None,N:,None] * (beta_P * presyn_act[2:,None,:] )
	update = tf.concat([E_update,I_update],0)
	## normalise learning rate such that learning rate is approximately indep of l4
	## activity amplitude
	# update = update/np.nanmean(postsyn_act) * 307.
	return update


def activity_corr_EI_input(t,corr,W,beta_P):
	"""
	input:
	t: timestep
	corr: activity correlation matrix N4 x N4 (assume same corr for E and I)
	W: ff weight matrix num_lgn_path x N4 x Nlgn
	"""
	W_D = tf.stack([W[0,:,:],W[2,:,:]]) - tf.stack([W[1,:,:],W[3,:,:]])
	
	dW_D = (tf.expand_dims(tf.expand_dims(tf.reduce_sum(corr,axis=1),axis=0),axis=2)*W_D -\
			tf.matmul(corr,W_D) )* beta_P


	dW = tf.stack([dW_D[0,:,:],-dW_D[0,:,:],dW_D[1,:,:],-dW_D[1,:,:]])
	return dW


# constraints for plasticity updates
def constrain_update_x(dW,W_old,mask,A,dt):
	# sum over x
	norm = tf.reduce_sum(A,axis=1)
	norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
	eps = 1.*tf.reduce_sum(dW,axis=1)/norm
	dW_constraint = (dW - eps[:,None,:] * A) * mask
	return dW_constraint*dt+W_old

def constrain_update_alpha(dW,W_old,mask,A,dt):
	# sum over alpha and on/off
	norm = tf.reduce_sum(A,axis=(0,2))
	norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
	eps = 1.*tf.reduce_sum(dW,axis=(0,2))/norm
	dW_constraint = (dW - eps[None,:,None]*A) * mask
	return dW_constraint*dt+W_old

def constrain_update_xalpha_approx(dW,W_old,mask,A,dt):
	## first sum over alpha and on/off, then over x
	norm = tf.reduce_sum(A,axis=(0,2))
	norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
	eps = 1.*tf.reduce_sum(dW,axis=(0,2))/norm
	dW_alpha = (dW - eps[None,:,None]*A) * mask

	norm = tf.reduce_sum(A,axis=1)
	norm = tf.where(tf.equal(norm, 0), tf.ones(tf.shape(norm),dtype=tf.float32), norm )
	eps = 1.*tf.reduce_sum(dW_alpha,axis=1)/norm
	dW_constraint = (dW_alpha - eps[:,None,:] * A) * mask
	return dW_constraint*dt+W_old

def constrain_update_xalpha(dW,W_old,mask,A,c_orth,s_orth,dt):
	dW_mask = dW[A>0] ## complete update incl multiplied by arbor
	mask_fl = mask[A>0]	## boolean mask as type float
	# delta_mask *= mask_fl	## inserted to see whether that incr conservation of weights

	dW_mask -= tf.reduce_sum(s_orth*tf.linalg.matvec(c_orth,dW_mask)[:,None],axis=0)
	dW_mask *= mask_fl
	dW_constraint = tf.scatter_nd(tf.where(A>0),dW_mask,A.shape)
	dW_constraint = tf.reshape(dW_constraint, tf.shape(dW))
	return dW_constraint*dt+W_old

def constrain_update_divisive(dW,W_old,A,dt):
	W_new = W_old + dW * A
	# print("W_new,W_old",np.nanmax(W_new),np.nanmax(W_old),np.nanmax(dW*A))
	if W_new.ndim==2:
		constraint = tf.reduce_sum(W_new,axis=-1,keepdims=True)
	elif W_new.ndim==3:
		if W_new.shape[0]==4:
			constraint1 = tf.reduce_sum(W_new[:2,:,:],axis=(0,-1),keepdims=True)
			constraint2 = tf.reduce_sum(W_new[2:,:,:],axis=(0,-1),keepdims=True)
			constraint = tf.concat([constraint1,constraint2]) * 0.5
		elif W_new.shape[0]==2:
			constraint = tf.reduce_sum(W_new,axis=(0,-1),keepdims=True) * 0.5
	W_new = W_new/np.abs(constraint)
	# print("constraint",W_new.ndim,constraint[:3],tf.reduce_sum(W_new,axis=-1)[:3],W_new.shape)
	return W_new

# multiplicative normlaisation
def synaptic_normalization(W_clipped,H,arbor,Wlim,init_W,c_orth=None,axis=1,mode="xalpha"):
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

	return W_new,H


def homeostatic_normalization(W_clipped,H,running_l4_avg,l4_target,Wlim,dt):
	"""
	homeostatic normalisation of weights depending on cortical activity relative
	to target activity
	"""

	if isinstance(W_clipped,tuple):
		W_clipped = W_clipped[0]

	# H = tf.reshape(H,[2,-1])
	# H_new = H + 0.01 * dt * (1 - l4_avg/tf.expand_dims(l4_target,axis=1))
	H_new = 1 + 0.1 * dt * (1 - running_l4_avg/l4_target)
	# num_lgn_paths,h,w = W_clipped.shape
	W_copy = W_clipped
	# W_new = tf.concat([W_clipped[:num_lgn_paths//2,:,:] *\
	# 				  tf.expand_dims(tf.expand_dims(H_new[0,:],axis=0),axis=2),\
	# 				  W_clipped[num_lgn_paths//2:,:,:] *\
	# 				  tf.expand_dims(tf.expand_dims(H_new[1,:],axis=0),axis=2)],axis=0)
	W_new = W_clipped * tf.expand_dims(tf.expand_dims(H_new,axis=0),axis=2)

	W_new = tf.where(tf.logical_or(W_copy<=0,W_copy>=Wlim),W_copy,W_new)

	return W_new, H_new



class Plasticity:
	def __init__(self, dt, c_orth, s_orth, beta_P, plasticity_rule,\
	 constraint_mode, mult_norm, clip_mode, weight_strength, Wlim=None, init_weights=None):
		self.dt = dt
		self.plasticity_rule = plasticity_rule
		self.constraint_mode = constraint_mode
		self.multiplicative_normalisation = mult_norm
		self.clip_mode = clip_mode
		self.connectivity_type = "E"
		self.c_orth = c_orth
		self.s_orth = s_orth
		self.beta_P = beta_P
		self.Wlim = Wlim
		self.init_weights = init_weights
		self.weight_strength = weight_strength
		print("self.weight_strength",self.weight_strength)

		self._init_plasticity_rule()
		self._init_plasticity_constraint()
		self._init_multiplicative_norm()
		self._init_clip_weights()


	def _init_plasticity_rule(self):
		""" defines synaptic plasticity rule """
		if self.plasticity_rule=="activity_based":
			if self.connectivity_type=="EI":
				self.unconstrained_update =\
				 lambda t,r,u,C,W,beta_P,N: activity_based_EI_input(t,r,u,W,beta_P,N)
			else:
				self.unconstrained_update =\
				 lambda t,r,u,C,W,beta_P,N: activity_based(t,r,u,W,beta_P,N)
		elif self.plasticity_rule=="activity_corr":
			self.unconstrained_update =\
			 lambda t,r,u,C,W,beta_P,N: activity_corr_EI_input(t,C,W,beta_P)

		else:
			raise Exception('_init_plasticity_rule.')

	def _init_plasticity_constraint(self):
		""" defines constraint applied to synaptic weights after plasticity update """
		if self.constraint_mode=="x":
			self.constrain_update =\
			 lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_x(dW*A,W_old,mask,A,dt)

		elif self.constraint_mode=="alpha":
			self.constrain_update =\
			 lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_alpha(dW*A,W_old,mask,A,dt)

		elif self.constraint_mode=="xalpha_approx":
			self.constrain_update =\
			 lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_xalpha_approx(dW*A,W_old,\
			 																		mask,A,dt)

		elif self.constraint_mode=="xalpha":
			self.constrain_update =\
			 lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_xalpha(dW*A,W_old,mask,\
			 																A,c_orth,s_orth,dt)

		elif self.constraint_mode=="None":
			self.constrain_update = lambda dW,W_old,mask,A,c_orth,s_orth,dt: dW*dt+W_old
		
		elif self.constraint_mode=="divisive":
			self.constrain_update =\
			 lambda dW,W_old,mask,A,c_orth,s_orth,dt: constrain_update_divisive(dW,\
			 	W_old/self.weight_strength,A,dt)*self.weight_strength

		else:
			raise Exception('constraint_mode.')

	def _init_clip_weights(self):
		if self.clip_mode:
			self.clip_weights = lambda W,A,Wlim: tf.clip_by_value(W,0,A*Wlim)
		else:
			self.clip_weights = lambda W,A,Wlim: W
		

	def _init_multiplicative_norm(self):
		""" optional multiplicative normalisation to account for loss of weight strength due
		to clipping """
		self.l4_target = None
		if self.multiplicative_normalisation=="x":
			self.mult_normalization =\
			 lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
																	self.init_weights,\
																	c_orth=None,axis=1)

		elif self.multiplicative_normalisation=="alpha":
			self.mult_normalization =\
			 lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,\
																	self.init_weights,\
																	c_orth=None,axis=(0,2))

		elif self.multiplicative_normalisation=="xalpha":
			self.mult_normalization =\
			 lambda Wnew,A,H,l4,l4_target: synaptic_normalization(Wnew,H,A,self.Wlim,self.init_weights,\
																	c_orth=self.c_orth,\
																	axis=None)

		elif self.multiplicative_normalisation=="homeostatic":
			self.mult_normalization =\
			 lambda Wnew,A,H,l4,l4_target: homeostatic_normalization(Wnew,H,l4,l4_target,\
			 														self.Wlim,self.dt)
		elif self.multiplicative_normalisation=="None":
			self.mult_normalization =\
			 lambda Wnew,A,H,l4,l4_target: (Wnew,H)

		else:
			raise Exception('multiplicative_normalisation not defined.\
							Choose either "x", "alpha", "xalpha" "homeostatic", "divisive".')


def unconstrained_plasticity_wrapper(p_dict, l4, l23, lgn, Wlgn_to_4, W4to4, W4to23, W23to23, t):
	""" apply plasticity update to all selected connections """
	dW_dict = {}

	if p_dict["p_lgn_e"] is not None:
		pop_size = Wlgn_to_4.shape[1]
		l4_e = l4[:pop_size]
		# Wlgn_to_4_e = tf.reshape(Wlgn_to_4[:2,:,:],[-1,Wlgn_to_4.shape[2]])
		# print("Wlgn_to_4_e",t,Wlgn_to_4_e.shape)
		dW = p_dict["p_lgn_e"].unconstrained_update(t,l4_e,tf.reshape(lgn[:2,:],[-1]),None,\
				None,p_dict["p_lgn_e"].beta_P,None)
		# print("Wlgn_to_4",np.sum(Wlgn_to_4[:2,:,:],axis=(0,2)))
		dW = tf.transpose(tf.reshape(dW,[dW.shape[0],2,-1]),perm=[1,0,2])
		dW_dict["dW_lgn_e"] = tf.reshape(dW, [-1])

	if p_dict["p_lgn_i"] is not None:
		pop_size = Wlgn_to_4.shape[1]
		l4_i = l4[pop_size:]
		# Wlgn_to_4_i = tf.reshape(Wlgn_to_4[2:,:,:],[-1,Wlgn_to_4.shape[2]])
		dW = p_dict["p_lgn_i"].unconstrained_update(t,l4_i,tf.reshape(lgn[2:,:],[-1]),None,\
				None,p_dict["p_lgn_i"].beta_P,None)
		dW = tf.transpose(tf.reshape(dW,[dW.shape[0],2,-1]),perm=[1,0,2])
		dW_dict["dW_lgn_i"] = tf.reshape(dW, [-1])

	if p_dict["p_4to23_e"] is not None:
		l4_e = l4[:tf.size(l4)//2]
		l23_e = l23[:tf.size(l23)//2]
		W4to23_e = W4to23[:tf.size(l23)//2,:tf.size(l23)//2]
		dW = p_dict["p_4to23_e"].unconstrained_update(t,l4_e,l23_e,None,W4to23_e,\
			 p_dict["p_4to23_e"].beta_P,None)
		dW_dict["dW_4to23_e"] = tf.reshape(dW, [-1])

	if p_dict["p_4to23_i"] is not None:
		l4_i = l4[:tf.size(l4)//2]
		l23_i = l23[:tf.size(l23)//2]
		W4to23_i = W4to23[tf.size(l23)//2:,:tf.size(l23)//2]
		dW = p_dict["p_4to23_i"].unconstrained_update(t,l4_i,l23_i,None,W4to23_i,\
			 p_dict["p_4to23_i"].beta_P,None)
		dW_dict["dW_4to23_i"] = tf.reshape(dW, [-1])

	if p_dict["p_rec4_ei"] is not None:
		l4_e = l4[:tf.size(l4)//2]
		l4_i = l4[tf.size(l4)//2:]
		W4to4_ei = W4to4[:tf.size(l4)//2,tf.size(l4)//2:]
		dW = p_dict["p_rec4_ei"].unconstrained_update(t,l4_i,l4_e,None,W4to4_ei,\
			 p_dict["p_rec4_ei"].beta_P,None)
		print("dW p_rec4_ei",np.nanmax(dW),np.nanmin(dW))
		dW_dict["dW_rec4_ei"] = -tf.reshape(dW, [-1])

	if p_dict["p_rec4_ii"] is not None:
		l4_i = l4[tf.size(l4)//2:]
		W4to4_ii = W4to4[tf.size(l4)//2:,tf.size(l4)//2:]
		dW = p_dict["p_rec4_ii"].unconstrained_update(t,l4_i,l4_i,None,W4to4_ii,\
			 p_dict["p_rec4_ii"].beta_P,None)
		dW_dict["dW_rec4_ii"] = -tf.reshape(dW, [-1])

	if p_dict["p_rec23_ei"] is not None:
		l23_e = l23[:tf.size(l23)//2]
		l23_i = l23[tf.size(l23)//2:]
		W23to23_ei = W23to23[:tf.size(l23)//2,tf.size(l23)//2:]
		# print("W23to23_ei",t,np.sum(W23to23_ei,axis=1),W23to23_ei.shape)
		dW = p_dict["p_rec23_ei"].unconstrained_update(t,l23_i,l23_e,None,W23to23_ei,\
			 p_dict["p_rec23_ei"].beta_P,None)
		dW_dict["dW_rec23_ei"] = -tf.reshape(dW, [-1])

	if p_dict["p_rec23_ii"] is not None:
		l23_i = l23[tf.size(l23)//2:]
		W23to23_ii = W23to23[tf.size(l23)//2:,tf.size(l23)//2:]
		dW = p_dict["p_rec23_ii"].unconstrained_update(t,l23_i,l23_i,None,W23to23_ii,\
			 p_dict["p_rec23_ii"].beta_P,None)
		dW_dict["dW_rec23_ii"] = -tf.reshape(dW, [-1])

	return dW_dict


def constraint_update_wrapper(dW_dict,p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
	W4to23,arbor4to23,W23to23,arbor23to23,dt,params_dict):

	if p_dict["p_lgn_e"] is not None:
		Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
		dW = tf.reshape(dW_dict["dW_lgn_e"],arbor_lgn[:2,:,:].shape)
		notfrozen = tf.math.logical_and(Wlgn_to_4_e>0, Wlgn_to_4_e<(p_dict["p_lgn_e"].Wlim*arbor_lgn[:2,:,:]))
		mask = tf.math.logical_and( notfrozen, arbor_lgn[:2,:,:]>0 )
		mask_fl = tf.cast(mask, tf.float32)	
		W_new = p_dict["p_lgn_e"].constrain_update(dW,Wlgn_to_4[:2,:,:],mask_fl,\
					arbor_lgn[:2,:,:],p_dict["p_lgn_e"].c_orth,p_dict["p_lgn_e"].s_orth,dt)
		# print("Wlgn_to_4 before",np.sum(Wlgn_to_4[2:,:,:],axis=(0,2)),Wlgn_to_4.shape)
		Wlgn_to_4 = tf.concat([W_new,Wlgn_to_4[2:,:,:]],0)
		# print("Wlgn_to_4 after",np.sum(Wlgn_to_4[2:,:,:],axis=(0,2)),Wlgn_to_4.shape)

	if p_dict["p_lgn_i"] is not None:
		Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
		dW = tf.reshape(dW_dict["dW_lgn_i"],arbor_lgn[2:,:,:].shape)
		notfrozen = tf.math.logical_and(Wlgn_to_4_i>0, Wlgn_to_4_i<(p_dict["p_lgn_i"].Wlim*arbor_lgn[2:,:,:]))
		mask = tf.math.logical_and( notfrozen, arbor_lgn[2:,:,:]>0 )
		mask_fl = tf.cast(mask, tf.float32)	
		W_new = p_dict["p_lgn_i"].constrain_update(dW,Wlgn_to_4[2:,:,:],\
			mask_fl,arbor_lgn[2:,:,:],p_dict["p_lgn_i"].c_orth,p_dict["p_lgn_i"].s_orth,dt)
		Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],W_new],0)

	if p_dict["p_4to23_e"] is not None:
		N = W4to23.shape[0]//2
		W4to23_e = W4to23[:N,:N]
		if arbor4to23 is None:
			A = 1
		else:
			A = arbor4to23[:N,:N]
		dW = tf.reshape(dW_dict["dW_4to23_e"],W4to23_e.shape)
		W_new = p_dict["p_4to23_e"].constrain_update(dW,W4to23_e,None,A,\
			p_dict["p_4to23_e"].c_orth,p_dict["p_4to23_e"].s_orth,dt)
		W4to23 = tf.concat([tf.concat([W_new,W4to23[N:,:N]],0),W4to23[:,N:]],1)
		params_dict["W4to23"] = W4to23

	if p_dict["p_4to23_i"] is not None:
		N = W4to23.shape[0]//2
		W4to23_i = W4to23[N:,:N]
		print("arbor4to23",arbor4to23.shape,W4to23_i.shape)
		if arbor4to23 is None:
			A = 1
		else:
			A = arbor4to23[N:,:N]
		dW = tf.reshape(dW_dict["dW_4to23_i"],W4to23_i.shape)
		print("dW",dW.shape,A.shape,N)
		W_new = p_dict["p_4to23_i"].constrain_update(dW,W4to23_i,None,A,\
				p_dict["p_4to23_i"].c_orth,p_dict["p_4to23_i"].s_orth,dt)
		W4to23 = tf.concat([tf.concat([W4to23[:N,:N],W_new],0),W4to23[:,N:]],1)
		params_dict["W4to23"] = W4to23

	if p_dict["p_rec4_ei"] is not None:
		N = W4to4.shape[0]//2
		W4to4_ei = W4to4[:N,N:]
		if arbor4to4 is None:
			A = 1
		else:
			A = arbor4to4[:N,N:]
		dW = tf.reshape(dW_dict["dW_rec4_ei"],W4to4_ei.shape)
		W_new = p_dict["p_rec4_ei"].constrain_update(dW,W4to4_ei,None,A,\
				p_dict["p_rec4_ei"].c_orth,p_dict["p_rec4_ei"].s_orth,dt)
		W4to4 = tf.concat([W4to4[:,:N],tf.concat([W_new,W4to4[N:,N:]],0)],1)
		params_dict["W4to4"] = W4to4

	if p_dict["p_rec4_ii"] is not None:
		N = W4to4.shape[0]//2
		W4to4_ii = W4to4[N:,N:]
		if arbor4to4 is None:
			A = 1
		else:
			A = arbor4to4[N:,N:]
		dW = tf.reshape(dW_dict["dW_rec4_ii"],W4to4_ii.shape)
		W_new = p_dict["p_rec4_ii"].constrain_update(dW,W4to4_ii,None,A,\
				p_dict["p_rec4_ii"].c_orth,p_dict["p_rec4_ii"].s_orth,dt)
		W4to4 = tf.concat([W4to4[:,:N],tf.concat([W4to4[:N,N:],W_new],0)],1)
		params_dict["W4to4"] = W4to4

	if p_dict["p_rec23_ei"] is not None:
		N = W23to23.shape[0]//2
		W23to23_ei = W23to23[:N,N:]
		if arbor23to23 is None:
			A = 1
		else:
			A = arbor23to23[:N,N:]
		dW = tf.reshape(dW_dict["dW_rec23_ei"],W23to23_ei.shape)
		W_new = p_dict["p_rec23_ei"].constrain_update(dW,W23to23_ei,None,A,\
				p_dict["p_rec23_ei"].c_orth,p_dict["p_rec23_ei"].s_orth,dt)
		# print("W23to23 after",np.sum(W23to23[:,N:],axis=1),W23to23.shape)
		W23to23 = tf.concat([W23to23[:,:N],tf.concat([W_new,W23to23[N:,N:]],0)],1)
		# print("W23to23 after",np.sum(W23to23[:,N:],axis=1),W23to23.shape)
		params_dict["W23to23"] = W23to23

	if p_dict["p_rec23_ii"] is not None:
		N = W23to23.shape[0]//2
		W23to23_ii = W23to23[N:,N:]
		if arbor23to23 is None:
			A = 1
		else:
			A = arbor23to23[N:,N:]
		dW = tf.reshape(dW_dict["dW_rec23_ii"],W23to23_ii.shape)
		W_new = p_dict["p_rec23_ii"].constrain_update(dW,W23to23_ii,None,A,\
				p_dict["p_rec23_ii"].c_orth,p_dict["p_rec23_ii"].s_orth,dt)
		W23to23 = tf.concat([W23to23[:,:N],tf.concat([W23to23[:N,N:],W_new],0)],1)
		params_dict["W23to23"] = W23to23

	return Wlgn_to_4,W4to4,W4to23,W23to23


def clip_weights_wrapper(p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
	W4to23,arbor4to23,W23to23,arbor23to23,params_dict):

	if (p_dict["p_lgn_e"] is not None and p_dict["p_lgn_e"].Wlim is not None):
		Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
		Wlgn_to_4_e = p_dict["p_lgn_e"].clip_weights(Wlgn_to_4_e,p_dict["p_lgn_e"].Wlim,arbor_lgn[:2,:,:])
		Wlgn_to_4 = tf.concat([Wlgn_to_4_e,Wlgn_to_4[2:,:,:]],0)

	if (p_dict["p_lgn_i"] is not None and p_dict["p_lgn_i"].Wlim is not None):
		Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
		Wlgn_to_4_i = p_dict["p_lgn_i"].clip_weights(Wlgn_to_4_e,p_dict["p_lgn_i"].Wlim,arbor_lgn[:2,:,:])

		Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],Wlgn_to_4_i],0)

	return Wlgn_to_4,W4to4,W4to23,W23to23


def mult_norm_wrapper(p_dict,Wlgn_to_4,arbor_lgn,W4to4,arbor4to4,\
	W4to23,arbor4to23,W23to23,arbor23to23,H,running_l4_avg,l4_target,params_dict):

	if p_dict["p_lgn_e"] is not None:
		Wlgn_to_4_e = Wlgn_to_4[:2,:,:]
		A = arbor_lgn[:2,:,:]

		Wlgn_to_4_e,H_new = p_dict["p_lgn_e"].mult_normalization(Wlgn_to_4_e,A,H[0,:],\
																running_l4_avg[0,:],\
																l4_target[0])
		if H.shape[0]==2:
			H = tf.stack([H_new,H[1,:]])
		else:
			H = tf.reshape(H_new,[1,H_new.shape[0]])
		print("H1",H.shape)
		Wlgn_to_4 = tf.concat([Wlgn_to_4_e,Wlgn_to_4[2:,:,:]],0)

	if p_dict["p_lgn_i"] is not None:
		Wlgn_to_4_i = Wlgn_to_4[2:,:,:]
		A = arbor_lgn[2:,:,:]

		Wlgn_to_4_i,H_new = p_dict["p_lgn_e"].mult_normalization(Wlgn_to_4_e,A,H[1,:],\
																running_l4_avg[1,:],\
																l4_target[1])
		H = tf.stack([H[0,:],H_new])
		Wlgn_to_4 = tf.concat([Wlgn_to_4[:2,:,:],Wlgn_to_4_i],0)


	return Wlgn_to_4,W4to4,W4to23,W23to23,H