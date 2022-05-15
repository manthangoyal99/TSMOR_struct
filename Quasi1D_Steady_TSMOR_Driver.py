# Driver program for Nair Balajewicz' (2019) Transported Snapshot Model Order
# Reduction method applied to the quasi 1D steady C-D nozzle flow problem

from cProfile import label
import numpy as np
import os
from optparse import OptionParser
import matplotlib.pylab as plt

from MyPythonCodes.tools.scipy_tools import scipy_slsqp

from MyPythonCodes.mesh import su2MeshData,UnstructuredMesh, getFileExtUSMD, meshCellFaceProps, polyMeshMetrics

import Quasi1D_Steady_TSMOR as tsmor


parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
parser.add_option('-p',dest='path',default='.',help='Output path')
parser.add_option('--nx',dest='nfx',default='3',help='No. of Fourier bases for x')
parser.add_option('--ny',dest='nfy',default='2',help='No. of Fourier bases for y')
parser.add_option('--ng',dest='ng',default='1',help='g basis function')
parser.add_option('-t',dest='train',default='0', \
    help='Training before validation (if supplied); else validation only')
(options, args) = parser.parse_args()
nfx = int(options.nfx) #No. of basis f-functions in x grid distortion
nfy = int(options.nfy) #No. of basis f-functions in y grid distortion
ng = int(options.ng)
train = int(options.train)

print(options.train)
print(train)
"""

Function that evaluates the velocity field in the direction of grid
As the grid is perpendicular to the bump we get the nomal component as 0 
and velocity only in the direction of bump
Now when this is mapped to the flat domain we get the velocities in the 
i and the j directions 
Now when the lower boundary is only slid the slip wal condition is implicitly
satisfied 

"""
def vel_along_grid(ux,uy,x,y):
    
    diff_x = np.diff(x,axis=0)
    diff_y = np.diff(y,axis=0)

    dis = (diff_x**2+diff_y**2)**.5

    u_i  = np.zeros_like(ux)
    u_j  = np.zeros_like(uy)

    u_j[:,:-1,:] = ux[:,:-1,:]*diff_x/dis + uy[:,:-1,:]*diff_y/dis
    u_j[:,-1,:] = ux[:,-1,:]*diff_x[-1,:]/dis[-1,:] + uy[:,-1,:]*diff_y[-1,:]/dis[-1,:]

    u_i[:,:-1,:] = ux[:,:-1,:]*diff_y/dis - uy[:,:-1,:]*diff_x/dis
    u_i[:,-1,:] = ux[:,-1,:]*diff_y[-1,:]/dis[-1,:] - uy[:,-1,:]*diff_x[-1,:]/dis[-1,:]

    return u_i,u_j

# Form the filename for the transport fields' database
transport_db_fn=os.path.join(options.path,'transport_fields_nfx'+str(nfx)+'_nfy'+str(nfy)+'struct.npz')

""" ===== Offline stage of TSMOR: load snapshots and compute transports ==== """

print('\n'+'-'*80+'\nOffline stage: Loading snapshots ...\n'+'-'*80)
# Load training database
# load the mesh in USM format
mshFn = "/home/manthangoyal/SU2_data/mesh_1x_bump" + getFileExtUSMD() #Mesh USM File
#mshFnFull = os.path.join(os.path.abspath(os.getcwd()), mshFn)
mesh_base = UnstructuredMesh(mshFn)
mesh_base._readMeshNodes()
Nodes = mesh_base.getNodes()
#plt.scatter(Nodes[:,0],Nodes[:,1])
#plt.show()      
# Load training database
train_db = np.load(os.path.join(options.path,'training_snaps.npz'))
Mus_db = train_db['mus']    #Parameters values of training snapshots
nMus = len(Mus_db)          #No. of training snapshots
print(nMus)
u_db = train_db['u_db']     #Training snapshot's flow variables 
dMus0 = Mus_db[1]-Mus_db[0] #Normalization constant for Mus_db
Mus_db_norm = Mus_db/dMus0  #Normalized parameter array of training database
print(np.shape(u_db))

u_db = u_db[:,:,:] #all
#u_db_max = np.amax(np.amax(u_db,axis=1,keepdims=True),axis=0,keepdims=True)
#print(u_db[0,:300,7]*287.9963)
print("max", np.shape(np.max(u_db,axis = 2)))
#u_db = u_db/np.amax(u_db,axis = 2)
x_flat = np.arange(256,dtype='float')
y_flat = np.arange(128,dtype='float')


Nodes_grid_x,Nodes_grid_y = np.reshape(Nodes[:,0],(128,256)),\
    np.reshape(Nodes[:,1],(128,256))

u_db = np.reshape(u_db,(nMus,128,256,-1))
print(u_db[0,0,:,5])
vel_i,vel_j = vel_along_grid(u_db[:,:,:,3],u_db[:,:,:,4],Nodes_grid_x,Nodes_grid_y)

u_db  = np.concatenate((u_db,np.stack((vel_i,vel_j,(vel_i**2+vel_j**2)**.5),axis=3)),axis=3)

#u_db_new = np.stack((u_db[:,:,:,2:3],vel_i,vel_j,u_db[:,:,:,5:6]),axis=3) #use this when the only variables 
# to be used are density, vel_i, vel_j, energy
print(np.shape(u_db))

u_db_normalised = u_db/np.amax(np.amax(np.amax(u_db,axis=2,keepdims=True),axis=1,keepdims=True),axis=0,keepdims=True)

plt.quiver(Nodes_grid_x,Nodes_grid_y,u_db_normalised[1,:,:,10],u_db_normalised[1,:,:,11])
plt.show()
#cs = plt.contour(Nodes_grid_x,Nodes_grid_y,(u_db[2,:,:,0]**2+u_db[2,:,:,1]**2)**.5,20,)
#cs = plt.contour(Nodes_grid_x,Nodes_grid_y,(u_db[1,:,:,3]**2+u_db[1,:,:,4]**2)**.5,50) 
plt.contourf(Nodes_grid_x,Nodes_grid_y,u_db_normalised[1,:,:,6],100)
#plt.clabel(cs)
plt.show()
plt.contour(Nodes_grid_x,Nodes_grid_y,u_db_normalised[1,:,:,3],100)
plt.contour(Nodes_grid_x,Nodes_grid_y,u_db_normalised[1,:,:,4],20,colors = 'red')
plt.contour(Nodes_grid_x,Nodes_grid_y,u_db_normalised[1,:,:,11],20,colors = 'black')
plt.show()
#for i in range(3):
#    plt.contour(Nodes_grid_x,Nodes_grid_y,u_db[i+1,:,:,4],20)
#plt.show()
for i in range(nMus):
    plt.plot(Nodes_grid_x[1,:],u_db_normalised[i,0,:,11],label = 'y')
    plt.plot(Nodes_grid_x[1,:],u_db_normalised[i,0,:,10],'r--',label = 'x')
    plt.plot(Nodes_grid_x[1,:],u_db_normalised[i,0,:,12],label = 'mag')
#exit()
plt.legend()
plt.show()

#u_db = np.concatenate((u_db[:,:,:,2:3],(u_db[:,:,:,3:4]**2+u_db[:,:,:,4:5]**2)**.5,\
#    u_db[:,:,:,5:6]),axis =3)
print(np.shape(u_db))
#exit()
#u_db = (u_db[:,:,:,3:4]**2+u_db[:,:,:,4:5]**2)**.5
u_db_new = u_db_normalised[:,:,:,11:12 ] 
if (train==1):
    # Pre-allocate array of grid-distortion coefficients for each snapshot
    trnsprt_Csx = np.zeros((nfx*ng,nMus))
    trnsprt_Csy = np.zeros((nfy*ng,nMus))

    # Loop over snapshots to calculate their grid-distortion field coefficients
    for iRef in range(nMus):
        print('\nOffline stage: Constructing transport field for sampled ' \
            +'snapshot #'+str(iRef+1)+' (mu = '+str(Mus_db[iRef])+') ...')
        # Indices of 2 nearest neighbours for the current reference snapshot in
        # the training database (by parameter value)
        iNbs = np.sort(np.argpartition(np.abs(Mus_db - Mus_db[iRef]),2)[1:3])
        # Create optimization project defining offline TSMOR problem; note that
        # a) We only focus on the first component of the snapshots, and
        # b) We normalize the parameter values for better performance
        project_off=tsmor.Project_TSMOR_Offline(u_db_new,Mus_db_norm,iRef,iNbs\
            ,x_flat,y_flat,nfx,nfy,ng=ng)
        #Form the initial guess of solution
        if iRef == 0:   #First snapshot
            coeffs0 = np.zeros((nfx+nfy)*ng)   #Nothing better than zeros
        else:
            coeffs0 = np.append(trnsprt_Csx[:,iRef-1],trnsprt_Csy[:,iRef-1])  #Use previous solution
        #coeffs0 = np.zeros((nf))
        # Run the optimization and obtain the output
        output = scipy_slsqp(project_off,x0=coeffs0,its=100,accu=1e-5, \
            eps=1e-6,disp=1)
        # Grid-distortion coefficients for current snapshot is the solution of
        # the optimization process that is returned as the first entry of
        # 'output'; store it in the overall array
        trnsprt_Csx[:,iRef] = output[0][:nfx*ng]
        trnsprt_Csy[:,iRef] = output[0][nfx*ng:]
        # Post-process the optimal solution (see it)
        #project_off.solnPost(output,Nodes_grid_x,Nodes_grid_y)
    #endfor iRef in range(nMus)   #Done addressing all training snapshots
    np.savez(transport_db_fn,mus=Mus_db,Cx=trnsprt_Csx,Cy=trnsprt_Csy,ng=ng)
else:
    print('\nOffline stage: Loading pre-computed transport fields ...')
    trnsprt_db = np.load(transport_db_fn)
    trnsprt_Csx = trnsprt_db['Cx']
    trnsprt_Csy = trnsprt_db['Cy']
print('\n'+'-'*80+'\nEnd of offline stage\n'+'-'*80+'\n')
print('exit statement at line 100 of tsmor driver')

print(np.shape(trnsprt_Csx))


""" = Online stage of TSMOR: predict unsampled snapshots from sampled data = """
print('\n'+'-'*80+'\nOnline stage: Loading testing snapshots ...\n'+'-'*80)
#xi = xarr[0];   xf = xarr[-1]
# Load testing database
test_db = np.load(os.path.join(options.path,'testing_snaps.npz'))
Mus_test = test_db['mus']   #Parameter values for testing snapshots
nMus_test = len(Mus_test)   #No. of testing snapshots
#xarr_test = test_db['xarr'] #x-grid on which testing snapshots are defined
u_test = test_db['u_db']    #Testing snapshot's flow variables
Mus_test_norm = Mus_test/dMus0  #Normalized parameter array of testing database 

u_test = np.concatenate((u_test[:,:,2:5],u_test[:,:,5:6]),axis = 2) # density, x-mom, y-mom, pressure
# Loop over testing database parameters to predict their flow fields
for iTest in range(len(Mus_test)):
    print('\nOnline stage: Prediction at testing snapshot #'+str(iTest+1) \
        +' of '+str(nMus_test)+' (mu = '+str(Mus_test[iTest])+') ...')
    # Indices of 2 nearest neighbours for the current test case (by parameter
    # value) that serve as the reference snapshots to be transported to form the
    # basis set for predicting the current case
    iRefs = np.sort(np.argpartition(np.abs(Mus_db - Mus_test[iTest]),1)[:2])
    assert np.all(np.diff(iRefs)==1),'iRefs array should be contiguous sequence'
    # Create optimization project defining online TSMOR problem; note that
    # normalized parameter of test case is being sent, to be compatible with
    # normalized parameters of training database that were used to derive the
    # grid distortion coefficients in the offline stage
    print(iTest)
    print(iRefs)

    #neghbouring snaps with density vel_i vel_j pressure
    uRefs = np.stack((u_db[iRefs[0]:iRefs[-1]+1,:,:,2],u_db[iRefs[0]:iRefs[-1]+1,:,:,10],\
        u_db[iRefs[0]:iRefs[-1]+1,:,:,11],u_db[iRefs[0]:iRefs[-1]+1,:,:,6]),axis = 3)
    print("urefs shape ",np.shape(uRefs))
    project_on = tsmor.Project_TSMOR_Online(u_test[iTest,:,:], uRefs, \
        Mus_db_norm[iRefs],trnsprt_Csx[:,iRefs[0]:iRefs[-1]+1],trnsprt_Csy\
            [:,iRefs[0]:iRefs[-1]+1],mesh_base,Mus_test_norm[iTest],x_flat, y_flat,ng=ng)
    # Run the optimization and obtain the output
    output = scipy_slsqp(project_on,x0=project_on.coords0,its=100, \
        accu=1e-6,eps=1e-10,disp=1)
    # Post-process the optimal solution (see it and compare with known solution)
    project_on.solnPost(output,u_test[:,:,iTest])
print('\n'+'-'*80+'\nEnd of online stage\n'+'-'*80+'\n')
plt.show()
