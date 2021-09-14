#import tensorflow as tf
import numpy as np
from bettina.modeling.ori_dev_model import connectivity
from bettina.modeling.ori_dev_model.tools import gen_gaussian_random_field as ggrf

def distance(delta):
	''' assume periodic boundary conditions'''
	signs = np.sign(delta)
	return np.where(np.abs(delta) > 0.5, -signs*(1 - np.abs(delta)), delta)

class Inputs:
	'''
	input to retina
	'''
	def __init__(self, size, Version, random_seed):
		self.size = size
		self.Version = Version
		self.random_seed = random_seed
	
	def create_matrix(self, inp_params, profile):		
		self.profile = profile
		##TODO: choose time-varying or static input
		# if not inp_params:
		# 	from bettina.modeling.ori_dev_model import inp_params

		# rng = np.random.RandomState(self.Version*5101+202)
		rng = np.random.RandomState(self.random_seed*5101+202)
		if self.profile=="GRF":
			N = self.size[0]
			ndim = inp_params["ndim"]
			Nsur = inp_params["Nsur"]
			radius = inp_params["radius"]
			width = inp_params["width"]
			base_vectors = np.empty((N*N,ndim))*np.nan
			for idim in range(ndim):
				rng = np.random.RandomState(idim*10+self.Version*1001)
				cmap = ggrf.generate_topology_map(N, N, radius, width,\
		 				rng=rng, return_complex=True, symmetrise=True)
				base_vectors[:,idim] = np.real(cmap).reshape(N*N)
		
			# householder reflection (qr decomposition of mxn matrix, m>=n
			orth_base_vec,r = np.linalg.qr(base_vectors,mode='reduced')	#q: orthonormal, r: upper triangular
			orth_base_vec = orth_base_vec.T

			# normalization 
			#orth_base_vectors = base_vectors
			# orth_base_vec = orth_base_vec/np.sqrt(np.mean(orth_base_vec**2,axis=1))[:,None]
			orth_base_vec = (orth_base_vec - np.nanmean(orth_base_vec,axis=1)[:,None])/np.nanstd(orth_base_vec,axis=1)[:,None]

			# mu = 0.1*rng.randn(Nsur,ndim)
			mu = np.zeros((Nsur,ndim))
			idx_obl = rng.choice(np.arange(ndim),size=Nsur,replace=True)
			idx_car = rng.choice([-1.,1.],size=Nsur,replace=True)
			for isur in range(Nsur):
				mu[isur,idx_obl[isur]] = idx_car[isur]
			input_patterns = 1.*np.dot(mu,orth_base_vec)/ndim
			input_patterns = input_patterns.reshape(Nsur,N,N)

			## stim indices: -2, -1, 1, 2
			stim_idx = idx_car*(idx_obl+1)
			return input_patterns, orth_base_vec, stim_idx

		if self.profile=="white_noise":
			N = self.size[0]
			# T_pd = inp_params["pattern_duration"]
			T_exp = inp_params["expanse_time"]
			Nsur = (inp_params["Nsur"] + 1) * (T_exp + 1)
			pad_size = 0
			factor = inp_params["onoff_corr_factor"]
			print("factor",factor)

			grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
			x,y = np.meshgrid(grid,grid)

			whitenoise = rng.randn(Nsur,N,N,3)
			for isur in range(Nsur):
				if (isur+1)%(T_exp+1)==0:
					whitenoise[isur,...] = 0.
			whitenoise = np.pad(whitenoise,((0,0),(pad_size,pad_size),(pad_size,pad_size),\
						(0,0)),'constant')
			noise_on = 0 + 1.*whitenoise[...,0] - factor*whitenoise[...,-1]
			noise_of = 0 + 1.*whitenoise[...,1] + factor*whitenoise[...,-1]
			input_patterns = np.stack([noise_on,noise_of])
			print("noise_on",np.nanmin(noise_on),np.nanmax(noise_on))

			input_patterns = np.swapaxes(input_patterns.reshape(2,Nsur,N*N),1,2)
			return np.real(input_patterns)

		if self.profile=="white_noise_online":
			N = self.size[0]
			factor = inp_params["onoff_corr_factor"]
			Nsur = 1
			pad_size = 0

			grid = np.linspace(-0.5,0.5,N+2*pad_size,endpoint=False)
			x,y = np.meshgrid(grid,grid)

			whitenoise = rng.randn(N,N,3)
			whitenoise = np.pad(whitenoise,((pad_size,pad_size),(pad_size,pad_size),\
								(0,0)),'constant')
			noise_on = 0 + 1.*whitenoise[...,0] - factor*whitenoise[...,-1]
			noise_of = 0 + 1.*whitenoise[...,1] + factor*whitenoise[...,-1]
			input_patterns = np.stack([noise_on,noise_of])

			input_patterns = np.swapaxes(input_patterns.reshape(2,N*N),0,1)
			return np.real(input_patterns)

		elif self.profile=="ringlike":
			Nsur = inp_params["Nsur"]
			expansion_timesteps = inp_params["expanse_time"]
			init_radius = inp_params["init_radius"]
			mean_width = inp_params["mean_width"]
			h,w = self.size
			inputs = np.zeros((h*w,Nsur*(expansion_timesteps+1)))

			ctrs = []
			for input_pattern in range(Nsur):
				ctrx = rng.uniform(0,1,1)[0]
				ctry = rng.uniform(0,1,1)[0]
				gridx = np.linspace(-ctrx,1-ctrx,self.size[0],endpoint=False)
				gridy = np.linspace(-ctry,1-ctry,self.size[1],endpoint=False)
				x,y = np.meshgrid(gridx,gridy)
				# dist = np.sqrt(distance(x)**2 + distance(y)**2)
				dist = np.sqrt(x**2 + y**2)

				init_r = np.clip(rng.randn(1) * 0.2 + init_radius,0.05,1)
				# init_r = rng.uniform(0.01,0.5,1)
				vel = np.clip(rng.randn(1) * 0.05 + 0.1,0.01,0.5)
				# vel = rng.uniform(0.01,0.7,1)
				# width = rng.uniform(0.05,1.,1)
				width = np.clip(rng.randn(1) * 0.05 + mean_width,0.01,0.9)
				sign = rng.choice([-1.,1.],1)
				print("settings",init_r,vel,width,sign)

				for i in range(expansion_timesteps):
					circle = np.logical_and( dist > (init_r + vel*i),\
							dist < (init_r + vel*i + width) ).astype(float)\
							* sign * dist
					inputs[:,i+input_pattern*(expansion_timesteps+1)] = circle.flatten()
				ctrs.append(np.array([ctrx,ctry]))
				# inputs[:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps] /=\
				# max(np.nanstd(inputs[:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps],axis=0))

			return inputs, np.array(ctrs)

		elif self.profile=="ringlike_onoff":
			Nsur = inp_params["Nsur"]
			expansion_timesteps = inp_params["expanse_time"]
			init_radius = inp_params["init_radius"]
			h,w = self.size
			inputs = np.zeros((2,h*w,Nsur*(expansion_timesteps+1)))

			ctrs = []
			for input_pattern in range(Nsur):
				ctrx = rng.uniform(0,1,1)
				ctry = rng.uniform(0,1,1)
				gridx = np.linspace(-ctrx,1-ctrx,w,endpoint=False)
				gridy = np.linspace(-ctry,1-ctry,h,endpoint=False)
				x,y = np.meshgrid(gridx,gridy)
				# x = distance(x)
				# y = distance(y)
				dist_on = np.sqrt(x**2 + y**2)

				init_r = np.clip(rng.randn(1) * init_radius*0.9 + init_radius,0.01,1.)
				vel = np.clip(rng.randn(1) * 0.08 + 0.1,0.01,0.7)
				width = rng.uniform(0.05,1.,1)

				ctrx_of = rng.uniform(ctrx-0.3,ctrx+0.3,1)
				ctry_of = rng.uniform(ctry-0.3,ctry+0.3,1)
				gridx = np.linspace(-ctrx_of,1-ctrx_of,w,endpoint=False)
				gridy = np.linspace(-ctry_of,1-ctry_of,h,endpoint=False)
				width_of = rng.uniform(0.05,1.,1)
				x,y = np.meshgrid(gridx,gridy)
				# x = distance(x)
				# y = distance(y)
				dist_of = np.sqrt(x**2 + y**2)
				print("settings",init_r,vel,width)

				for i in range(expansion_timesteps):
					circle_on = np.logical_and( dist_on > (init_r + vel*i),\
						dist_on < (init_r + vel*i + width) ).astype(float) * dist_on
					inputs[0,:,i+input_pattern*(expansion_timesteps+1)] = circle_on.flatten()
					
					circle_of = np.logical_and( dist_of > (init_r + vel*i),\
						dist_of < (init_r + vel*i + width_of) ).astype(float) * dist_of
					inputs[1,:,i+input_pattern*(expansion_timesteps+1)] = circle_of.flatten()

				# inputs[:,:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps] /=\
				# max(np.nanstd(inputs[:,:,input_pattern*expansion_timesteps:(input_pattern+1)*expansion_timesteps],axis=0))
				ctrs.append(np.array([ctrx,ctry]))

			return inputs, np.array(ctrs)

		elif self.profile=="moving_grating":
			Nsur = inp_params["Nsur"]
			n = inp_params["spat_frequency"]
			theta = inp_params["orientation"]
			gridx = np.linspace(0,1,self.size[1],endpoint=False)
			gridy = np.linspace(0,1,self.size[0],endpoint=False)
			x,y = np.meshgrid(gridx,gridy)
			x = np.cos(theta) * x - np.sin(theta) * y
			y = np.sin(theta) * x + np.cos(theta) * y
			kx = 2*np.pi/self.size[1] * n
			ky = 2*np.pi/self.size[0] * n
			# Lambda = 2*np.pi/np.sqrt(kx**2+ky**2) #* np.sqrt(2)
			inputs = np.zeros((self.size+(Nsur,)))
			# print("check",theta,kx,ky,x[0,1],y[0,1])
			for isur in range(Nsur):
				phi = 1.*isur/Nsur * 2*np.pi
				inputs[:,:,isur] = np.sin(kx * x + 0*ky * y + phi)
			inputs = inputs.reshape(-1,Nsur)
			return inputs

class Inputs_lgn(Inputs):
	'''
	input to lgn which is inputs to retina convolved with connectivity
	from retina to lgn
	'''
	def __init__(self, size, Version, random_seed):
		super().__init__(size,Version,random_seed)
	
	def create_matrix(self, inp_params, profile):
		return super().create_matrix(inp_params, profile)

	def create_lgn_input(self, inp_params, profile, Wret_to_lgn):
		rng = np.random.RandomState(self.random_seed*5101+202)
		if "white_noise" in profile:
			inp_ret = self.create_matrix(inp_params, profile)

		elif "GRF" in profile:
			inp_ret, orth_base_vec, stim_idx = self.create_matrix(inp_params, "GRF")
			# lgn = np.dot(Wret_to_lgn,inp_ret)
			# lgn = np.swapaxes(lgn,0,1)

		elif "ringlike" in profile:
			inp_ret, ctrs = self.create_matrix(inp_params, "ringlike")
			inp_ret = np.stack([inp_ret,-inp_ret])
			inp_ret = np.swapaxes(inp_ret,0,1)
			# lgn = np.dot(Wret_to_lgn,inp_ret)
			### lgn = np.stack([dynamics.nl(lgn),dynamics.nl(-lgn)])
			# lgn = np.stack([lgn,-lgn])

		elif "ringlike_onoff" in profile:
			inp_ret, ctrs = self.create_matrix(inp_params, "ringlike_onoff")
		
		elif "moving_grating" in profile:
			inp_ret = self.create_matrix(inp_params, "moving_grating")

		## convolve input with Retina-to-LGN connections
		## and swap axes such that lgn.shape = 2 x dimensions
		if Wret_to_lgn.ndim==2:
			lgn = np.dot(Wret_to_lgn,inp_ret)
			lgn = np.swapaxes(lgn,0,1)
		else:
			lgn = np.stack([np.dot(Wret_to_lgn[0,...],inp_ret[:,0,...]),\
			 				np.dot(Wret_to_lgn[1,...],inp_ret[:,1,...])])

		## normalise lgn input to SD=1 and shift its mean to positive value
		if ("online" in profile):
			if "moving_grating" not in profile:
				# if True:
				# 	lgn /= np.nanstd(lgn,axis=1)[:,None]
				# 	## lgn += -np.nanmin(lgn,axis=(1))[:,None] + 1

				# 	lgn[0,:] *= 1.
				# 	lgn[1,:] *= inp_params["onoff_rel_weight"]
				# 	# lgn += -np.nanmin(lgn,axis=(1))[:,None] + 1
				# 	# ## old version
				# 	# print("compare (old)",0.1*np.random.randn(1),np.nanmin(lgn),lgn.shape)
				# 	lgn += -np.nanmin(lgn) + 1

				# # # try to keep mean of on/off inputs the same, but weight SD differently
				# else:
				lgn += -np.nanmean(lgn,axis=(1))[:,None]
				lgn /= np.nanstd(lgn,axis=1)[:,None]

				lgn[0,:] *= 1.
				lgn[1,:] *= inp_params["onoff_rel_weight"]
				## make input positive and slight diff in mean between on and off
				lgn += 10 + 0.5*rng.randn(2)[:,None]
				# lgn = np.clip(lgn,0,np.nanmax(lgn))

		else:
			T_exp = inp_params["expanse_time"]
			for isur in range(inp_params["Nsur"]+1):
				lgn_sd = np.nanstd(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1],axis=(1,2))
				lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] /= lgn_sd[:,None,None]
				lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] +=\
				 - np.nanmin(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1]) + 1
		
		# # ====================== add noise to events only ======================
		# for isur in range(inp_params["Nsur"]):
		# 	white_noise = 0.1*np.random.randn(lgn.shape[0],lgn.shape[1],T_exp)
		# 	lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1] += white_noise
		# 	lgn_sd = np.nanstd(lgn[:,:,isur*(T_exp+1):(isur+1)*(T_exp+1)-1],axis=(1,2))
		# 	# print(lgn_sd)
		# lgn = np.clip(lgn,0,np.nanmax(lgn))

		return lgn

	def apply_ONOFF_bias(self,lgn,inp_params):
		## OFF bias regions
		exp_off_dominant = 1.
		if "off_bias_strength" in inp_params.keys():
			if inp_params["off_bias_strength"]>0:
				N = int(self.size[0]) #cast to int, otherwise it's tf int
				rng_bias = np.random.RandomState(20210211)
				radius = N/10.
				width = N/30.
				cmap = ggrf.generate_topology_map(N, N, radius, width,rng=rng_bias,\
				        						  return_complex=True, symmetrise=True)
				off_bias = np.real(cmap).reshape(N*N)
				bias_thr = np.percentile(off_bias,100-inp_params["off_bias_region_size"]*100)
				off_dominant = off_bias - bias_thr
				off_dominant *= inp_params["off_bias_strength"]/np.nanstd(off_dominant)

				## multiplicative
				exp_off_dominant = np.exp(off_dominant)
				# lgn[0,:] /= exp_off_dominant
				# lgn[1,:] *= exp_off_dominant			

		return np.stack([lgn[0,:]/exp_off_dominant,lgn[1,:]*exp_off_dominant])



if __name__=="__main__":
	import matplotlib.pyplot as plt
	from bettina.modeling.ori_dev_model import Inp_params,connectivity,Wret_to_lgn_params
	from bettina.modeling.ori_dev_model.tools import analysis_tools
	
	rA = 0.25
	Wret_to_lgn_params["sigma"] = 0.4*rA
	N = 25
	random_seed = 19
	Inp_params.update({"Nsur" : 100})
	Wret_to_lgn_params.update({"mean_eccentricity"	:	0.9,
								"SD_eccentricity"	:	0.05,
								"SD_size"			:	0.05,
								"heterogeneity_type":	None#"independent"#
								})
	Wret_to_lgn = connectivity.Connectivity((N,N),(N,N),\
				  random_seed=random_seed).create_matrix(\
				  Wret_to_lgn_params,"Gaussian") * Wret_to_lgn_params["gamma_ret"]
	lgn = []

	Inp_params["onoff_rel_weight"] = 1.
	for i in range(Inp_params["Nsur"]):
		# rng_seed = Version*1000 + i
		rng_seed = random_seed*1000 + i
		one_stimulus = Inputs_lgn((N,N),1,rng_seed).create_lgn_input(\
						Inp_params, "white_noise_online", Wret_to_lgn)
		lgn.append( one_stimulus )
	lgn = np.swapaxes(np.swapaxes(np.array(lgn),0,1),1,2)
	print("lgn",lgn.shape)

	lgn_norm = (lgn-np.nanmean(lgn,axis=2)[:,:,None])/np.nanstd(lgn,axis=2)[:,:,None]
	cc = np.nanmean(lgn_norm[:,None,None,:,:] * lgn_norm[None,:,:,None,:],axis=4)
	print("cc.shape",cc.shape)
	
	print(r"Norm of LGN input: {}$\pm${} (mean pm SD)".format(\
		np.nanmean(lgn),np.nanstd(np.nanmean(lgn,axis=(0,1))),axis=0))

	fig = plt.figure()
	plt.suptitle("Example corr")
	ax = fig.add_subplot(141)
	ax.set_title("onon")
	im=ax.imshow(cc[0,0,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(142)
	ax.set_title("onoff")
	im=ax.imshow(cc[0,1,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(143)
	ax.set_title("offoff")
	im=ax.imshow(cc[1,1,:,:],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(144)
	ax.set_title("onon-onoff")
	diff = (0.5*cc[0,0,:,:]+0.5*cc[1,1,:,:]-cc[0,1,:,:])
	im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)


	cc = cc.reshape(2,2,N,N,N,N)
	fig = plt.figure()
	plt.suptitle("Example corr")
	ax = fig.add_subplot(141)
	ax.set_title("onon")
	im=ax.imshow(cc[0,0,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(142)
	ax.set_title("onoff")
	im=ax.imshow(cc[0,1,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(143)
	ax.set_title("offoff")
	im=ax.imshow(cc[1,1,N//2,N//2],interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(144)
	ax.set_title("onon-onoff")
	diff = (0.5*cc[0,0,N//2,N//2]+0.5*cc[1,1,N//2,N//2]-cc[0,1,N//2,N//2]).reshape(N,N)
	im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	print("Peak CC",cc[:,:,N//2,N//2,N//2,N//2])


	## average correlation pattern
	cc_avg = 0
	for i in range(N):
		for j in range(N):
			cc_avg += np.roll(np.roll(cc[:,:,i,j,:,:],N//2-i,axis=2),N//2-j,axis=3)
	cc_avg /= N*N

	fig = plt.figure()
	plt.suptitle("Avg corr")
	ax = fig.add_subplot(141)
	ax.set_title("onon")
	im=ax.imshow(cc_avg[0,0].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(142)
	ax.set_title("onoff")
	im=ax.imshow(cc_avg[0,1].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(143)
	ax.set_title("offoff")
	im=ax.imshow(cc_avg[1,1].reshape(N,N),interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(144)
	ax.set_title("onon-onoff")
	diff = (0.5*cc_avg[0,0]+0.5*cc_avg[1,1]-cc_avg[0,1]).reshape(N,N)
	im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	print("Avg Peak CC",cc_avg[:,:,N//2,N//2])
	
	fit_params,fitted_gauss,_ = analysis_tools.determine_LGN_input_correlations_scale(None,diff)
	fitted_gauss = fitted_gauss.reshape(diff.shape)
	print("fit_params",fit_params)
	fig = plt.figure()
	fig.suptitle("Gaussian fit to LGN input correlation difference")
	ax = fig.add_subplot(131)
	im=ax.imshow(diff,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax.contour(fitted_gauss,5,colors="b")
	ax = fig.add_subplot(132)
	im=ax.imshow(fitted_gauss,interpolation="nearest",cmap="RdBu_r",vmin=-0.75,vmax=0.75)
	plt.colorbar(im,ax=ax)
	ax.contour(diff,5,colors="b")
	ax = fig.add_subplot(133)
	ax.plot(diff[N//2,:]/np.nanmax(diff),'b',label="Data")
	ax.plot(fitted_gauss[N//2,:],'r',label="Fit")
	ax.plot(diff[:,N//2]/np.nanmax(diff),'b--',label="Data")
	ax.plot(fitted_gauss[:,N//2],'r--',label="Fit")
	ax.legend(loc="best")


	lgn = lgn.reshape(2,N,N,-1)
	fftabs = np.abs(np.fft.fftshift(np.fft.fft2(lgn[0,:,:,0])))
	fig = plt.figure(figsize=(3*3,2*2))
	plt.suptitle("Example convolved noise pattern")
	ax = fig.add_subplot(231)
	im=ax.imshow(lgn[0,:,:,0],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(232)
	im=ax.imshow(lgn[0,:,:,1],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(233)
	# ax.plot(lgn[0,N//2,N//2,:],'-')
	ax.plot(np.nanmean(lgn[1,:,:,:],axis=(0,1)),'--')
	ax.plot(np.nanmean(lgn[0,:,:,:],axis=(0,1)),'--')
	ax.set_ylim(bottom=0)
	ax = fig.add_subplot(234)
	ax.plot(lgn[1,N//2,N//2,:],'-')
	# ax.plot(np.nanmean(lgn[1,:,:,:],axis=(0,1)),'--')
	ax.plot(lgn[0,N//2,N//2,:],'-')
	# ax.plot(np.nanmean(lgn[0,:,:,:],axis=(0,1)),'--')
	ax.set_ylim(bottom=0)
	ax = fig.add_subplot(235)
	im=ax.imshow(lgn[1,:,:,0],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(236)
	im=ax.imshow(lgn[1,:,:,1],interpolation="nearest",cmap="binary")
	plt.colorbar(im,ax=ax)
	plt.show()