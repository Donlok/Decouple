import numpy as np
from skimage import restoration

def print_just_array(array,xsta,xra,ysta,yra):
	i=ysta
	while(i<ysta+yra):
		j=xsta
		while(j<xsta+xra):
			print "%.3f\t" %(array[i,j]),
			j+=1
		print
		i+=1

def print_array_clean(array):
	i=0
	while(i<len(array)):
		j=0
		while(j<len(array[0])):
			print "%4.1f\t" %(array[i,j]),
			try:
				print "%4.2f\t" %(alph_S_N(array[i,j],array[i,j+1])),
			except:
				True
			j+=1
		j=0
		print
		while(j<len(array[0])):
			try:
				print "%4.2f\t\t" %(alph_S_N(array[i,j],array[i+1,j])),
			except:
				True
			j+=1
		print
		i+=1
	
def alph_S_N(S,N):						#generates a coupling in percent, given a signal and neighbor value
	S=float(S)
	N=float(N)
	A0=0.4
	A1=0.4
	k0=(1./20000.)
	k1=(1./(20000.*np.sqrt(2)))
	a=0.65
	R=np.sqrt(S**2+N**2)
	return A0*np.exp(-1*k0*np.abs(S-N))+A1*np.exp(-1*k1*R)+a


def build_table(filename):
	f1=open(filename)
	i=0
	S=[]
	C=[]
	A=[]
	for line in f1:
		col=line.split('\t')
		col[-1]=col[-1][:-1]
		if(i==0):
			col=col[1:]
			S=col
		else:
			C.append(col[0])
			col=col[1:]
			A.append(col)
		i+=1
	S=np.asarray(S,dtype=np.float)
	C=np.asarray(C,dtype=np.float)
	A=np.asarray(A,dtype=np.float)
	f1.close()
	return S,C,A

def find_nearest(array,value):
	idx=np.searchsorted(array,value,side="left")
	return idx-1,idx

def ref_table(Sin,Cin,Stab,Ctab,Atab):
	x1,x2=find_nearest(Stab,Sin)
	try:
		xhw=(float(Sin-Stab[x1])/float(Stab[x2]-Stab[x1]))
		if(Stab[x1]>Stab[x2]):
			if(not(Stab[x2]==Sin)):
				print "Signal value below minimum: Minimum will be used"
				print Sin
				raw_input("wait... what!?")
				True
	except:
		print "Signal value above maximum: Maximum wll be used"
		xhw=0.0
		x2=x1
	xlw=1-xhw
	y1,y2=find_nearest(Ctab,Cin)
	try:
		yhw=(float(Cin-Ctab[y1])/float(Ctab[y2]-Ctab[y1]))
		if(Ctab[y1]>Ctab[y2]):
			if(not(Ctab[y2]==Cin)):
				print "Contrast value below minimum: Minimum will be used"
				print Cin
				True
			yhw=1.0
	except:
		print "Contrast value above maximum: Maximum will be used"
		yhw=0.0
		y2=y1
	ylw=1-yhw
	A_ret=(xlw*ylw)*Atab[y1][x1]+(xlw*yhw)*Atab[y2][x1]+(xhw*ylw)*Atab[y1][x2]+(xhw*yhw)*Atab[y2][x2]
	return A_ret

def pad_zeros(array):
	return np.pad(array,1,mode='constant')

def cut_edges(array):
	shape=np.shape(array)
	return array[1:shape[0]-1,1:shape[1]-1]

def decouple(array,S,C,A,measured):
	temp=np.zeros(np.shape(array))
	i=0
	while(i<np.shape(array)[0]):
		j=0
		while(j<np.shape(array)[1]):
			flag_1,flag_2,flag_3,flag_4=True,True,True,True
			if(i==0):
				coef_1_in=0.0
				coef_1_out=0.0
				flag_1=False
			else:
				coef_1_in=ref_table(array[i-1,j],array[i,j],S,C,A)
				coef_1_out=ref_table(array[i,j],array[i-1,j],S,C,A)
			if(i==np.shape(array)[0]-1):
				coef_2_in=0.0
				coef_2_out=0.0
				flag_2=False
			else:
				coef_2_in=ref_table(array[i+1,j],array[i,j],S,C,A)
				coef_2_out=ref_table(array[i,j],array[i+1,j],S,C,A)
			if(j==0):
				coef_3_in=0.0
				coef_3_out=0.0
				flag_3=False
			else:
				coef_3_in=ref_table(array[i,j-1],array[i,j],S,C,A)
				coef_3_out=ref_table(array[i,j],array[i,j-1],S,C,A)
			if(j==np.shape(array)[1]-1):
				coef_4_in=0.0
				coef_4_out=0.0
				flag_4=False
			else:
				coef_4_in=ref_table(array[i,j+1],array[i,j],S,C,A)
				coef_4_out=ref_table(array[i,j],array[i,j+1],S,C,A)
			coup_in=0.0
			if(flag_1):
				coup_in+=(coef_1_in*array[i-1,j]*0.01)
			if(flag_2):
				coup_in+=(coef_2_in*array[i+1,j]*0.01)
			if(flag_3):
				coup_in+=(coef_3_in*array[i,j-1]*0.01)
			if(flag_4):
				coup_in+=(coef_4_in*array[i,j+1]*0.01)
			coup_out=(coef_1_out+coef_2_out+coef_3_out+coef_4_out)*0.01*array[i,j]
			temp[i,j]=measured[i,j]-coup_in+coup_out
			j+=1
		i+=1
	return temp

def decouple_iterative(measured,S,C,A,iterations):
	i=0
	array=np.copy(measured)
	while(i<iterations):
		measured=pad_zeros(measured)
		array=pad_zeros(array)
		i+=1
	i=0
	while(i<iterations):
		previous=np.copy(array)
		array=decouple(array,S,C,A,measured)
		i+=1
	i=0
	while(i<iterations):
		array=cut_edges(array)
		i+=1
	return array

def rl_dec(array,coupling):
	kernal=np.zeros((3,3))
	coupling=coupling*.01
	kernal[0,1]=coupling
	kernal[1,0]=coupling
	kernal[1,2]=coupling
	kernal[2,1]=coupling
	kernal[1,1]=1-4*coupling
	psf=np.asarray(kernal)
	print psf
	return restoration.richardson_lucy(array,psf,iterations=30,clip=False)

def wh_dec(array,coupling):
	kernal=np.zeros((3,3))
	coupling=coupling*.01
	kernal[0,1]=coupling
	kernal[1,0]=coupling
	kernal[1,2]=coupling
	kernal[2,1]=coupling
	kernal[1,1]=1-4*coupling
	a,_=restoration.unsupervised_wiener(array,np.asarray(kernal),is_real=True,clip=False)
	return a


S,C,A=build_table('look_up.txt')

#build initial small array well padded with zeros

size=51

initial_array=np.zeros((size,size))

initial_array[25,25]=1000
initial_array[24,25]=400
initial_array[25,24]=400
initial_array[24,24]=200

init=np.copy(initial_array)
#couple initial small array


out=np.zeros((size,size))
i=1
while(i<size-1):
	j=1
	while(j<size-1):
		# so for each pixel, I evaluate the couple between cent and neighbor.  I then subtract off the 'away' from the central pixel
		# and I add in the neighbor value times the couple bewteen cent and neighbor
		#alright, let's revise this now.
		#what I need to measure is how much each neighbor couples in, and how much the center couples out to each neighbor
		b=alph_S_N(init[i,j],init[i+1,j])*0.01	#this is couple from cent to neighup
		c=alph_S_N(init[i,j],init[i-1,j])*0.01	#this is couple from cent to neighdown
		d=alph_S_N(init[i,j],init[i,j+1])*0.01	#this is couple from cent to neighright
		e=alph_S_N(init[i,j],init[i,j-1])*0.01	#this is couple from cent to neighleft
#		b=ref_table(init[i,j],init[i+1,j],S,C,A)*0.01	#this set of lines is used to reference SCA table instead of a build in function
#		c=ref_table(init[i,j],init[i-1,j],S,C,A)*0.01
#		d=ref_table(init[i,j],init[i,j+1],S,C,A)*0.01
#		e=ref_table(init[i,j],init[i,j+1],S,C,A)*0.01
		a=1.0-(b+c+d+e)			#this is the total couple away from neighbor
		

		out[i,j]=a*init[i,j]+b*init[i+1,j]+c*init[i-1,j]+d*init[i,j+1]+e*init[i,j-1]

		j+=1
	i+=1

#human readable output on this small 4 pixel array illustrating errors after so many decouplings (0,1,2,3,4,5,6,7,19(
print_just_array(out,23,4,23,4)
print
print_just_array(-1*(initial_array-out),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,1)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,2)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,3)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,4)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,5)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,6)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,7)),23,4,23,4)
print
print_just_array(-1*(initial_array-decouple_iterative(out,S,C,A,10)),23,4,23,4)
