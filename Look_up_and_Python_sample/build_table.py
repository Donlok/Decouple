import numpy as np

def alph_S_N(S,N):							#generates a coupling in percent, given a signal and neighbor value
	A0=0.4									#matches eqn. 7 from PASP 2018 paper
	A1=0.4									#Hot pixel data matches when N=0
	k0=(1./20000.)							#Accounts for both relative depletion state (Difference between neighbors for field mixing)
	k1=(1./(20000.*np.sqrt(2)))					#and absolute depletion state (Field magnitude)
	a=0.65
	R=np.sqrt(S**2+N**2)
	return A0*np.exp(-1*k0*np.abs(S-N))+A1*np.exp(-1*k1*R)+a	#sum of difference and radius exponentials with different drop offs and amplitudes + constant


truex=np.arange(-5000,65001,50)	#S values
truey=np.arange(-5000,65001,50)	#N values
xx,yy=np.meshgrid(truex,truey)	#meshgrid
truecoup=alph_S_N(xx,yy)		#alpha at each IN PERCENT
print np.shape(truecoup)
f1=open('look_up.txt','w')
i=0
while(i<len(truex)):							#write in every S value separated by 
	f1.write{'\t')								#opening tab
	f1.write("{:2.4f}".format(truex[i]))		#4 decimal place precision
	i+=1
j=0
while(j<len(truey)):						
	f1.write('\n')								#terminate previous line
	f1.write("{:2.4f}".format(truey[j]))		#write that line's N value 4 decimal precision
	i=0
	while(i<len(truecoup[j])):
		f1.write('\t')							#separate data with tabs
		f1.write("{:2.4f}".format(truecoup[j][i]))	#write that point's alpha
		i+=1
	j+=1
f1.close()


