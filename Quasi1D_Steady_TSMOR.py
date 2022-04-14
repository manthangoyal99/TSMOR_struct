# This program implements the Transported Snapshot Model Order Reduction method
# of Nair and Balajewicz (2019), specifically for the Quasi-1D Steady C-D nozzle
# flow problem (problem #1 of their work)

from cProfile import label
import numpy as np
import matplotlib.pylab as plt
from math import pi as mathPi
from math import pow as mathPow
from scipy.interpolate import interp1d, interp2d

from MyPythonCodes.tools import Findiff_Taylor_uniform



def calc_grid_distortion_basis(arr,dMu,ib,nf,typ):
    """
    Calculate one particular basis function towards the grid distortion.
    Referring to eqn. 11 of Nair & Balajewicz (2019), this returns f_p*g_q for
    one p-q pair, towards the grid distortion c_s.
    
    INPUTS:
    xarr : Array of grid points
    dMu  : Parameter difference between reference and predicted snapshots
    ib   : Basis function index ('m' in eqn. 11 above but with 0-based indexing)
    nf   : Total number of 'f' basis functions ('N_p' in eqn. 11 above)
    
    OUTPUTS:
    fpgq : Product of f_p & g_q evaluated over the grid 'xarr'
    """
    ibf = ib % nf       #'f' basis function term's index corresponding to 'ib'
    ibg = ib//nf        #'g' basis function term's index corresponding to 'ib'
    arri = arr[0]        #Starting point of grid
    Larr = arr[-1] - arri  #Range of grid
    # Calculate the 'f_p' factor towards the basis, where 'p' is the index 'ibf'
    if typ=='x': #or typ=='y':

        if ibf == 0:
            grid_distortion_basis_term_f = np.ones_like(arr)
        else:
            grid_distortion_basis_term_f = np.sin(ibf*mathPi*(arr-arri)/Larr)
    
    else:

        grid_distortion_basis_term_f = np.sin((ibf+1)*mathPi*(arr-arri)/Larr)    
    
    # Calculate the 'g_q' factor towards the basis, where 'q' is the index 'ibg'
    grid_distortion_basis_term_g = mathPow(dMu,ibg+1)
    # Return the product of 'f_p' and 'g_q'
    return grid_distortion_basis_term_f*grid_distortion_basis_term_g
#enedef calc_grid_distortion_basis


def calc_grid_distortion(coeffs,arr,dMu,typ,ng=1):
    """
    Calculate the grid distortion.
    Referring to eqn. 11 of Nair & Balajewicz (2019), this returns c_s.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 above)
    
    OUTPUTS:
    cs : Sum-product of grid distortion basis functions and coefficients
    """
    nf = len(coeffs)/ng
    grid_distortion = np.zeros_like(arr)
    for ib, cc in enumerate(coeffs):
        grid_distortion += cc*calc_grid_distortion_basis(arr,dMu,ib,nf,typ)
    return grid_distortion
#enddef calc_grid_distortion


def calc_distorted_grid(coeffs,arr,dMu,typ,ng=1):
    """
    Calculate the distorted grid.
    Referring to eqn. 7 of Nair & Balajewicz (2019), this returns xd = x + c_s.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 above)
    
    OUTPUTS:
    xd : Distorted grid (supplied original grid 'xarr' plus the grid distortion
         'c_s')
    """
    return arr + calc_grid_distortion(coeffs,arr,dMu,typ,ng=ng)
#enddef calc_distorted_grid


def calc_transported_snap(coeffsx,coeffsy,xarr,yarr,u,dMu,ng=1):
    """
    Calculate the 'u' of LHS of eqn. 7 of Nair & Balajewicz (2019), by
    interpolating the given snapshot 'u' from the original grid to the distorted
    grid
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    u      : Reference snapshot; 2D array with rows corresponding to x-grid and
             columns corresponding to different components (flow variables)
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    ut : Transported version of given snapshot
    """
    # Distorted grid corresponding to given original grid, distortion
    # coefficients and parameter differential
    xd = calc_distorted_grid(coeffsx,xarr,dMu,'x',ng=ng)
    yd = calc_distorted_grid(coeffsy,yarr,dMu,'y',ng=ng)
    # We have to allow 'u' to be a 1D array corresponding to a single-component
    # (scalar) field; to make the subsequent steps agnostic to this, we reshape
    # it to a dummy 2D array if it is 1D in the following
    ushp = np.shape(u)
    ut = np.zeros_like(u) #For now, this is a 2D array (reshaped later)
    for ic in range(ushp[2]):    #Go thru each component
        # Generate the interpolation object by assuming that 'u' is specified
        # on the distorted grid
        f = interp2d(xd,yd,u[:,:,ic],kind='linear')
        # Evaluate the above interpolation object on the original grid
        ut[:,:,ic] = f(xarr,yarr)
    return np.reshape(ut,ushp)  #Make sure to return 1D array if input was so
#enddef calc_transported_snap


def calc_transported_snap_error(coeffsx,coeffy,xarr,yarr,uRef,uNbs,dMuNbs,\
    ref_error = None,ng=1):
    """
    Calculate the square of the 2-norm of error between the transported versions
    of a reference snapshot and other (neighbouring) snapshots
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    uRef   : Reference snapshot; 2D array with rows corresponding to x-grid and
             columns corresponding to different components (flow variables)
    uNbs   : List of neighbouring snapshots that should be predicted; each list
             list entry is a 2D array of the same shape as 'uRef'
    dMuNbs : Set of parameter differences between the reference snapshot and the
             neighbouring snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    error : Total error across all neighbouring snapshots
    """
    error = 0 #Initialize as 0 to calculate as running sum over all neighbours
    for inb, dMu in enumerate(dMuNbs): #Go thru all neighbours
        # Transport the reference snapshot 'uRef' by the parameter differential
        # 'dMu' using the grid distortion coefficients 'coeff' over the original
        # grid 'xarr'
        u0t = calc_transported_snap(coeffsx,coeffy,xarr,yarr,uRef,dMu,ng=ng)
        # Add the square of the 2-norm of the difference between the transported
        # snapshot and the neighbour to the running sum of error 
        error += np.linalg.norm(u0t - uNbs[inb])**2
    if ref_error==None:
        return error
    else:
        return error/ref_error
#enddef calc_transported_snap_error


def calc_grid_distortion_constraint(coeffsx,coeffsy,xarr,yarr,dMuNbs,ng=1):
    """
    Inequality constraint to be satisfied by the transport field coefficients
    
    Essentially, we do not want the distorted grid to collapse (i.e., have zero
    or negative spacing) anywhere for any neighbour. To this end, we impose a 
    minimum grid spacing (dx_min) on the transported grid 'xd' corresponding to
    each neighbour in 'dMuNbs'.
    The inequality constraint is
        xd_i - xd_(i-1) >= dx_min, for all i and all neighbours' distortions.
    The optimizer takes inequality constraints of the form
        cieq(coeffs) < 0,
    where 'cieq' takes the coefficients' array (array of quantities to be
    optimized) and returns a list of values.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMuNbs : Parameter differences between neighbouring and reference snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    ineq : List of inequality constraint values at all grid points and for all
           neighbours
    """
    # We arbitrarily specify the minimum allowed spacing of the distorted grid
    # as a small sub-multiple of the minimum grid spacing in the original
    # (undistorted) grid
    dx_min = np.min(np.diff(xarr))/50
    dy_min = np.min(np.diff(yarr))/50
    # Allocate array of inequality constraint values to be returned
    ineqx = np.zeros((len(xarr)-1,len(dMuNbs)))
    ineqy = np.zeros((len(yarr)-1,len(dMuNbs)))
    ineqx_dis = np.zeros((2,len(dMuNbs)))
    ineqy_dis = np.zeros((2,len(dMuNbs)))
    for iMu, dMu in enumerate(dMuNbs):
        xd = calc_distorted_grid(coeffsx,xarr,dMu,'x',ng=ng)
        yd = calc_distorted_grid(coeffsy,yarr,dMu,'y',ng=ng)
        ineqx[:,iMu] = dx_min - np.diff(xd)
        ineqy[:,iMu] = dy_min - np.diff(yd)
        ineqx_dis[0,iMu] = (xarr[0]-xd[0])-10
        ineqy_dis[0,iMu] = (yarr[0]-yd[0])-10
        ineqx_dis[1,iMu] = (-xarr[1]+xd[1])-10
        ineqy_dis[1,iMu] = (-yarr[1]+yd[1])-10
        
    # Reshape to 1D array with 1st index changing fastest (Fortran-style column
    # major order)
    ineq = np.concatenate((ineqx,ineqy))
    return np.reshape(ineq,(-1),'F')
#enddef calc_grid_distortion_constraint


class Project_TSMOR_Offline(object):
    """
    Starts an optimization project for offline part of TSMOR of the 1-D problem
    of Nair & Balajewicz (2019)
        
    ATTRIBUTES:
    uRef   : Reference snapshot that should be transported for predicting the
             following neighbouring snapshots; 2D array with rows corresponding
             to x-grid and columns corresponding to different components (flow
             variables)
    uNbs   : List of neighbouring snapshots that should be 'predicted'; each
             entry is a 2D array of the same shape as 'uRef'
    dMuNbs : Parameter differences between the reference snapshot and the above
             neighbouring snapshots
    xarr   : Array of grid points
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
         
    METHODS:
    Optimizer interface - The following methods take a design variable vector 
    (coefficients of grid distortion bases) for input as a list (shape n) or
    numpy array (shape n or nx1 or 1xn); values are returned as float or list
    or list of list
    obj_f     - objective function              : float
    obj_df    - objective function derivatives  : list
    con_ceq   - equality constraints            : list
    con_dceq  - equality constraint derivatives : list[list]
    con_cieq  - inequality constraints          : list
    con_dcieq - inequality constraint gradients : list[list]
    Auxiliary methods -
    solnPost - Post-process optimization solution
    """  
    
    def __init__(self,u_db,Mus_db,iRef,iNbs,x_flat,y_flat,nfx,nfy,ng=1):
        """
        Constructor of class
        
        INPUTS:
        u_db   : 3D array of snapshot database, with dimensions as:
                   1: grid points
                   2: components (flow variables)
                   3: snapshots
        Mus_db : Parameters of the snapshots in the database
        iRef   : Index of reference snapshot to be transported
        iNbs   : Indices of neighbouring snapshots to be predicted
        xarr   : Array of grid points
        ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair
                 and Balajewicz (2019))
        """
        self.uRef = u_db[iRef,:,:,:]    #Extract reference snapshot
        # Form list of neighbouring snapshots
        self.uNbs = [u_db[iNb,:,:,:] for iNb in iNbs]
        # Form list of corresponding parameter differentials from reference
        self.dMuNbs = [Mus_db[iNb]-Mus_db[iRef] for iNb in iNbs]
        self.xarr = x_flat #Store grid points
        self.yarr = y_flat
        self.ng = ng  #Store no. of 'g' basis functiona
        self.nfx = nfx
        self.nfy = nfy

        self.ref_error = calc_transported_snap_error(np.zeros(nfx),np.zeros(nfy),self.xarr,\
            self.yarr,self.uRef,self.uNbs,self.dMuNbs,ng=self.ng)
    
    # Evaluates the objective function based on the supplied set of independent
    # variables (coefficients towards grid distortion that are to be optimized)
    def obj_f(self,coeffs):
        # Evaluate the error in predicting the neighbouring snapshots by
        # transporting the reference snapshot, and return as a singleton list
        coeffsx = coeffs[:self.nfx]
        coeffsy = coeffs[self.nfx:]
        return [calc_transported_snap_error(coeffsx,coeffsy,self.xarr,self.yarr,\
            self.uRef,self.uNbs,self.dMuNbs,ref_error=self.ref_error,ng=self.ng)]

    # Evaluates inequality constraint vector based on supplied set of
    # independent variables (coefficients towards grid distortion that are to be
    # optimized)
    def con_cieq(self,coeffs):
        # Evaluate the set of inequality constraints to be satisfied by the grid
        # distortions required to predict each neighbour
        coeffsx = coeffs[:self.nfx]
        coeffsy = coeffs[self.nfx:]
        return calc_grid_distortion_constraint(coeffsx,coeffsy,self.xarr,self.yarr,\
        self.dMuNbs,ng=self.ng)

    """ All the remaining optimizer interfaces are left blank """
    def con_ceq(self,coeffs):
        return []
    
    def con_dceq(self,coeffs):
        return []
    
    def obj_df(self,coeffs):
        return []

    def con_dcieq(self,coeffs):
        return []
    
    # Post-process the optimization solution
    def solnPost(self,output,meshx,meshy):  
        coeffs = output[0]
        coeffsx = coeffs[:self.nfx]
        coeffsy = coeffs[self.nfx:]
        print('\t\tcoeffs = ['+', '.join(['%0.4f'%c for c in coeffs])+']')
        cieqs = np.array(self.con_cieq(coeffs))
        print('\t\tmax(con) = %0.4f, min(con) = %0.4f'%(max(cieqs),min(cieqs)))
        
        f_ij_to_x = interp2d(self.xarr,self.yarr,meshx)
        f_ij_to_y = interp2d(self.xarr,self.yarr,meshy)

        plt.figure()
        for iNb, dMu in enumerate(self.dMuNbs):
            
            dist_nb = calc_transported_snap(coeffsx,coeffsy,self.xarr,self.yarr,\
                self.uRef,dMu,ng=self.ng)

            
            plt.subplot(1,2,iNb+1)

            plt.contour(meshx,meshy,dist_nb[:,:,0],20)
            plt.contour(meshx,meshy,self.uNbs[iNb][:,:,0],20,colors='red')
            
            plt.title('contour plot') 
        plt.show()

        plt.figure()
        plt.subplot(1,2,1)
        i = 1
        for dMu in self.dMuNbs:
            plt.plot(self.xarr,calc_grid_distortion(coeffsx,self.xarr, \
                dMu,'x',ng=self.ng),label=str(i))
            plt.legend()
            i+=1
        plt.title('Grid distortions in x')

        plt.subplot(1,2,2)
        i = 1
        for dMu in self.dMuNbs:
            plt.plot(self.yarr,calc_grid_distortion(coeffsy,self.yarr, \
                dMu,'y',ng=self.ng),label=str(i))
            plt.legend()
        plt.title('Grid distortions in y')
        plt.show()

        for dMu in self.dMuNbs:
            
            xd = calc_distorted_grid(coeffsx,self.xarr, \
                dMu,'x',ng=self.ng)
            yd = calc_distorted_grid(coeffsy,self.yarr, \
                dMu,'y',ng=self.ng)
            
            #meshxd,meshyd = f_ij_to_x(xd,yd),f_ij_to_y(xd,yd)
            meshxd,meshyd = np.meshgrid(xd,yd)
            meshx_flat, meshy_flat = np.meshgrid(self.xarr,self.yarr)
            plt.scatter(meshx_flat,meshy_flat,c='blue',s=2)
            plt.scatter(meshxd,meshyd,c='red',s=2)
            plt.title('grid distortion')
            plt.show()
        

        #plt.subplot(2,2,4)
        for iNb, dMu in enumerate(self.dMuNbs):
            plt.scatter(meshx,meshy,c = abs(dist_nb[:,:,0]-self.uNbs[iNb][:,:,0]),s=5)
            plt.title('Error in predicting neighbours')
            plt.show()
#endclass Project_TSMOR_Offline


def calc_transported_basis(coeffsRefs,xarr,uRefs,dMuRefs,ng=1):
    """
    Calculate the basis set for a new case by transporting various reference
    snapshots using grid distortion coefficients precalculated in offline phase,
    and parameter differentials between the reference snapshots and the new case
    to be predicted
    
    INPUTS:
    coeffsRefs : 2D array of coefficients towards grid distortions, with
                 dimensions as:
                 1: Various grid distortion basis functions
                 2: Various snapshots to be transported
    xarr       : Array of grid points
    uRefs      : 3D array of reference snapshots to be 'transported', with
                 dimensions as:
                 1: data along 'xarr',
                 2: data for each component (flow variable),
                 3: data for each reference snapshot
    dMuRefs    : Parameter differences between the reference snapshots and the
                 case to be predicted
    ng         : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair
                 and Balajewicz (2019))
    
    OUTPUTS:
    phis : List of transported reference snapshots that form the basis set for
           the new case
    """
    phis = []
    for iRef in range(len(dMuRefs)):
        phis.append(calc_transported_snap(coeffsRefs[:,iRef],xarr, \
            uRefs[:,:,iRef],dMuRefs[iRef],ng=ng))
    return phis
#enddef calc_transported_basis


class Project_TSMOR_Online(object):
    """
    Starts an optimization project for online part of TSMOR of the 1-D problem
    of Nair & Balajewicz (2019)
        
    ATTRIBUTES:
    prblm_setup : Parameters specifying quasi 1D flow problem thru C-D nozzle
    xarr        : Array of x-grid points
    sigma       : Scale factor for enforcing boundary conditions
    phis        : Basis set obtained by transporting given reference snapshots
                  for predicting new case
    coords0     : Initial guess of generalized coordinates for above basis set
    Dx          : 1st-order finite difference operator on the 'xarr' grid
         
    METHODS:
    Optimizer interface - The following methods take a design variable vector 
    (generalized coordinates for basis set of transported reference snapshots)
    for input as a list (shape n) or numpy array (shape n or nx1 or 1xn); values
    are returned as float or list or list of lists
    obj_f     - objective function              : float
    obj_df    - objective function derivatives  : list
    con_ceq   - equality constraints            : list
    con_dceq  - equality constraint derivatives : list[list]
    con_cieq  - inequality constraints          : list
    con_dcieq - inequality constraint gradients : list[list]
    Auxiliary methods -
    compose_soln - Compose the solution from generalized coordinates
    solnPost     - Post-process optimization solution
    """  
    
    def __init__(self,uRefs,MuRefs,coeffsRefs,xarr,MuNew,prblm_setup, \
            sigma=100000):
        # Register some of the supplied variables directly as attributes of self
        self.prblm_setup = prblm_setup
        self.xarr = xarr
        self.sigma = sigma
        # Distances of new snapshot from reference ones in parameter space
        dMuRefs = MuNew - MuRefs
        # Basis calculation by transporting the supplied reference snapshots
        self.phis = calc_transported_basis(coeffsRefs,xarr,uRefs,dMuRefs)
        # Initial guess of generalized coordinates (for basis set of transported
        # snapshots), based on inverse-distance weights in parameter space
        self.coords0 = (1./abs(dMuRefs))/sum(1./abs(dMuRefs))
        # Calculate 1st-order finite difference operator with 4th-order accuracy
        self.Dx = Findiff_Taylor_uniform(len(xarr),1,4)/(xarr[1]-xarr[0])

    # Compose the snapshot solution using the supplied set of generalized
    # coordinates that comprise the weights of the basis set 'phis'
    def compose_soln(self,coords):
        # Initialize 2D snapshot array as 0's, to be subsequently composed as a
        # running sum
        u = np.zeros_like(self.phis[0])
        for iphi in range(len(coords)): #Go thru each generalized coordinate
            u += self.phis[iphi]*coords[iphi] #Add contribution of this basis
        return u
    
    # Evaluates the objective function (l_1 norm of residual of governing 
    # equations, augmented by scaled discrepancies in boundary conditions) based
    # on the supplied generalized coordinates
    def obj_f(self,coords):
        u = self.compose_soln(coords)
        ResAug = qs.quasi_1D_steady_soln_residual(u,self.prblm_setup, \
            self.xarr,Dx=self.Dx)
        # Retrieve the solution's values corresponding to the b.c.'s of the
        # problem
        u_rhoin, u_pin, u_pout = qs.quasi_1D_steady_soln_bcs(u,self.prblm_setup)
        # Supplant the inlet value of mass residual with the discrepancy in the
        # inlet density b.c., scaled by 'sigma'; the desired value of inlet
        # density is available as the 0th entry of the 'prblm_setup' array
        ResAug[0,0] = self.sigma*(u_rhoin - self.prblm_setup[0])
        # Supplant the inlet value of mom. residual with the discrepancy in the 
        # inlet pressure b.c., scaled by 'sigma'; the desired value of inlet
        # pressure is available as the 1st entry of the 'prblm_setup' array
        ResAug[0,1] = self.sigma*(u_pin - self.prblm_setup[1])
        # Supplant the outlet value of mom. residual with the discrepancy in the
        # outlet pressure b.c., scaled by 'sigma'; the desired value of outlet
        # pressure is available as the 2nd entry of the 'prblm_setup' array
        ResAug[-1,1] = self.sigma*(u_pout - self.prblm_setup[2])
        # Return singleton list of the l_1 norm of the augmented residuals
        # obtained above, reshaped into a 1D array
        return [np.linalg.norm(ResAug.reshape(-1),ord=1)]

    """ All the remaining optimizer interfaces are left blank """
    def con_cieq(self,coords):
        return []

    def con_ceq(self,coords):
        return []
    
    def con_dceq(self,coords):
        return []
    
    def obj_df(self,coords):
        return []

    def con_dcieq(self,coords):
        return []
    
    # Post-process the optimization solution
    def solnPost(self,output,q_vldt=None):
        coords = output[0]
        print('coords = ['+', '.join([str(c) for c in coords])+']')
        solns = [self.compose_soln(coords)]
        if q_vldt is not None:  solns.append(q_vldt)
        Res = qs.quasi_1D_steady_soln_residual(solns[0],self.prblm_setup, \
            self.xarr,Dx=self.Dx)
        qs.quasi_1D_steady_soln_plot(self.xarr,solns,legends=['ROM','True'])
        plt.subplot(2,2,4)
        for iv in range(3):
            plt.plot(self.xarr[1:-1],Res[1:-1,iv])
        plt.xlabel('x')
        plt.title('Res components')

#endclass Project_TSMOR_Online
