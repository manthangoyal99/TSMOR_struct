# This program implements the Transported Snapshot Model Order Reduction method
# of Nair and Balajewicz (2019), specifically for the Quasi-1D Steady C-D nozzle
# flow problem (problem #1 of their work)


import numpy as np
import matplotlib.pylab as plt
from math import pi as mathPi
from math import pow as mathPow
from scipy.interpolate import interp1d, interp2d

from MyPythonCodes.tools import Findiff_Taylor_uniform
from SteadyAeroDyn_PY.SteadyAeroDyn.DataHandling import IntegralHlprDomainGrad
#import SteadyAeroDyn_PY.SteadyAeroDyn


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
            grid_distortion_basis_term_f = np.sin(ibf*(mathPi)*(arr-arri)/Larr)
    
    else:

        grid_distortion_basis_term_f = np.sin((ibf+1)*(mathPi)*(arr-arri)/Larr)    
    
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
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]
        return [calc_transported_snap_error(coeffsx,coeffsy,self.xarr,self.yarr,\
            self.uRef,self.uNbs,self.dMuNbs,ref_error=self.ref_error,ng=self.ng)]

    # Evaluates inequality constraint vector based on supplied set of
    # independent variables (coefficients towards grid distortion that are to be
    # optimized)
    def con_cieq(self,coeffs):
        # Evaluate the set of inequality constraints to be satisfied by the grid
        # distortions required to predict each neighbour
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]
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
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]
        print('\t\tcoeffs = ['+', '.join(['%0.4f'%c for c in coeffs])+']')
        cieqs = np.array(self.con_cieq(coeffs))
        print('\t\tmax(con) = %0.4f, min(con) = %0.4f'%(max(cieqs),min(cieqs)))
        
        f_ij_to_x = interp2d(self.xarr,self.yarr,meshx)
        f_ij_to_y = interp2d(self.xarr,self.yarr,meshy)

        plt.figure()
        for iNb, dMu in enumerate(self.dMuNbs):
            
            dist_nb = calc_transported_snap(coeffsx,coeffsy,self.xarr,self.yarr,\
                self.uRef,dMu,ng=self.ng)

            
            plt.subplot(2,1,iNb+1)

            plt.contour(meshx,meshy,dist_nb[:,:,0],20)
            cs=plt.contour(meshx,meshy,self.uNbs[iNb][:,:,0],20,colors='red')
            #plt.clabel(cs)
            plt.title('contour plot') 
        plt.show()

        plt.figure()
        plt.subplot(2,1,1)
        i = 1
        for dMu in self.dMuNbs:
            plt.plot(self.xarr,calc_grid_distortion(coeffsx,self.xarr, \
                dMu,'x',ng=self.ng),label=str(i))
            plt.legend()
            i+=1
        plt.title('Grid distortions in x')

        plt.subplot(2,1,2)
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
            #plt.show()
        

        #plt.subplot(2,2,4)
        for iNb, dMu in enumerate(self.dMuNbs):
            plt.scatter(meshx,meshy,c = abs(dist_nb[:,:,0]-self.uNbs[iNb][:,:,0]),s=5)
            plt.title('Error in predicting neighbours')
            plt.show()
#endclass Project_TSMOR_Offline


def calc_transported_basis(coeffsRefsx,coeffsRefsy,xarr,yarr,uRefs,dMuRefs,ng=1):
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
        
        phis.append(calc_transported_snap(coeffsRefsx[:,iRef],coeffsRefsy[:,iRef],\
            xarr,yarr,uRefs[iRef,:,:,:],dMuRefs[iRef],ng=ng))
    return phis
#enddef calc_transported_basis

def vel_along_xy(ui,uj,x,y):
    
    diff_x = np.diff(x,axis=0)
    diff_y = np.diff(y,axis=0)

    dis = (diff_x**2+diff_y**2)**.5

    ux  = np.zeros_like(ui)
    uy  = np.zeros_like(uj)

    idotx = -(diff_y)/dis
    jdotx = diff_x/dis
    idoty = diff_x/dis
    jdoty = diff_y/dis

    ux[:-1,:] = ui[:-1,:]*idotx + uj[:-1,:]*jdotx
    ux[-1,:] = ui[-1,:]*idotx[-1,:] + uj[-1,:]*jdotx[-1,:]

    uy[:-1,:] = ui[:-1,:]*idoty + uj[:-1,:]*jdoty
    uy[-1,:] = ui[-1,:]*idoty[-1,:] + uj[-1,:]*jdoty[-1,:]

    ##returns [ux uy] a 4d array
    return np.stack((ux,uy),axis = 2)

def residual(mesh,u,uFOM,LNorm,sigma,refNorm=None):

        gamma_value = 1.4
        # Pre-allocate storage for normalized weighted residual vector (output);
        # it will be flattened prior to actual return
        ResN = 4
        NCellRes = mesh.getNCell()
        Res = np.zeros((ResN,NCellRes))
        dataType = 'N'
        NDime =2
        intHlprGrad = IntegralHlprDomainGrad(mesh,dataType)
        # We find the face flux values from (a) the flow variable values on the
        # faces, and (b) the pre-calculated face areas & their unit normals.
        # For node-based data, the average of the vertex values is deemed the
        # face value of the data.
        # Subsequently, the integrated cell divergences are calculated as the
        # dot product of the face fluxes with the face normal vectors, and the
        # results are added over all the faces.
        # To this, we add the pressure gradient for momentum residues.
        # N.B.: The algorithm doesn't need to make any special provision for the
        # various element types.
        ncrY = 0 #No. of Cells whose Residuals are found Yet (0 at start)
        for iTyp, FaceMeasuresTyp in enumerate(intHlprGrad.FaceMeasures):
            if FaceMeasuresTyp is None: #No cells of this type in mesh
                continue
            # Find No. of Cells whose Residuals are to be found for this Type
            ncrT = FaceMeasuresTyp.shape[0]
            nFacesInCellsOfTyp = FaceMeasuresTyp.shape[1]
            for ifc in range(nFacesInCellsOfTyp):#Go thru all faces of cell type
                # Extract the direction cosines and areas of this face in all
                # retained cells of this type
                faceDirCoss = FaceMeasuresTyp[:,ifc,:-1]
                faceAreas = FaceMeasuresTyp[:,ifc,-1]

                # In the following, we need density, pressure and temperature.
                # N.B. whereas density and temperature are normalized by their
                # respective ambient values (rho_inf & temp_inf), pressure is
                # normalized by rho_inf * a_inf**2 (for compatibility in the
                # momentum equation). Thus, the normalized ideal gas law is
                #   pressure = density * temperature / SpHt.
                
                # At this stage pressure, density and temperature are available;
                # these are all 1D arrays of same length
                dataLen = np.shape(u)[0]    #Get common length of data arrays
                faceContribs = intHlprGrad.faceContribs[iTyp][ifc]

                #Full-domain solution reconstructed in RAM
                # Get 2D array of node/face indices of mesh that contribute, with
                # rows being all cells of this type being considered, and columns
                # being the vertices of the face (for node-based data) or the
                # (singleton) face itself (for face-based data)
                iContribs = intHlprGrad.cellContribs[iTyp][:,faceContribs]
                print(iContribs)
                print(np.shape(u))
                # Retrieve momentum componets
                density = np.average(u[iContribs,0:1],axis=1)
                pressure = np.average(u[iContribs,3:4],axis=1)
                vels = np.average(u[iContribs,1:3],axis=1)
                print(np.shape(vels))
                print(np.shape(faceAreas))
                print(np.shape(pressure[:]))
                #vels = np.stack((vel_x,vel_y),axis=1)
                momenta = density*vels
                temperature = pressure / density * gamma_value
                
                print("hopefully values are correctly calculated")
                # Calculate mass flux on faces using momenta and face properties
                mf = np.sum(momenta*faceDirCoss,axis=1)*faceAreas
                # Compose the total enthalpy per unit mass
                #   enthalpyTot = Cp * T + (u**2+v**2+...)/2,
                # which is the sum of enthalpy + kinetic energies per unit mass.
                # Normalizing by a_inf**2, we have
                #print(temperature[:200,:])
                enthalpyTot = temperature.flatten()/(gamma_value-1)+np.sum(vels**2,axis=1)/2
                print("done")
                #exit()
                # Calculate contribution of the current face to the residues of
                # the various equations in each cell (integrated over the 
                # corresponding cell volumes), and add to running sum over all
                # faces
                # Residue of mass cons. eqn. (w/ U = velocity vector [u, v, w])
                #   div(density U)
                Res[0,ncrY:ncrY+ncrT] += mf
                # Residue of energy conservation equation
                #    div(Etot density U)
                Res[3,ncrY:ncrY+ncrT] += enthalpyTot*mf
                # Residue of x-momentum conservation equation
                #    div(density u U) + dp_dx.
                # Likewise for the other two components.
                print("done1")
                for iMom in range(NDime):
                    momContrib = pressure.flatten()*faceDirCoss[:,iMom]*faceAreas \
                        + mf*vels[:,iMom]
                    Res[1+iMom,ncrY:ncrY+ncrT] += momContrib
            #endfor ifc in range(nFacesInCellsOfTyp)
            # In case of L2-norm calculation, divide the residual in each cell
            # by the square root of its metric (volume in 3D; area in 2D; etc)
            # to get weighted residual, whose 2-norm will approximate L2-norm.
            # No such weighting is needed for 1-norm to approximate L1-norm.
            if LNorm == 2:
                for iRes in range(ResN):
                    Res[iRes,ncrY:ncrY+ncrT] \
                        /= np.sqrt(intHlprGrad.Metrics[iTyp])
            # Increment No. of Cells whose Residuals are found Yet with that of
            # Cells whose Residuals are found for this Type
            ncrY += ncrT
        #endfor iTyp, FaceMeasuresTyp in ...
        
        # Calculate the Lp^p norms of the individual (normalized) residuals
        ResNormsP = np.zeros((ResN))
        for iRes in range(ResN):
            ResNormsP[iRes] = np.linalg.norm(Res[iRes,:],ord=LNorm) \
                **LNorm        
        penalty = sigma*penalty_inlet(u,mesh,uFOM)
        print(penalty)
        ResNormsP_penalty = ResNormsP+penalty
        print(ResNormsP_penalty)
        if refNorm is None:
            return ResNormsP_penalty
        else:
            print(refNorm)
            print(ResNormsP_penalty)
            return np.linalg.norm(ResNormsP_penalty/refNorm,ord=1)
    #enddef evalObj
        
def penalty_inlet(u,mesh,uFOM):

    mesh._readMeshCellNodes()
    nodes_inlet = np.unique(mesh.getMarkCells('inlet')[0][0][0])
    u_inlet = u[nodes_inlet,:]
    uFOM_inlet = uFOM[nodes_inlet,:]

    penalNorm = np.zeros(np.shape(u)[1])
    for iRes in range(np.shape(u)[1]):
        penalNorm[iRes] = np.linalg.norm((u_inlet-uFOM_inlet)[:,iRes]\
            ,ord=2)**2        
    return penalNorm
    #u_error = np.linalg.norm(u_inlet-uFOM_inlet,2)**2

    #return u_error

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
    
    def __init__(self,uTest,uRefs,MuRefs,coeffsRefsx,coeffsRefsy,mesh,MuNew, \
            x_flat,y_flat,ng=1,sigma=100000):
        # Register some of the supplied variables directly as attributes of self
        self.uTest = uTest ## It has density velx vely pressure [2d]
        self.mesh = mesh
        self.sigma = sigma
        self.i = x_flat
        self.j = y_flat
        self.nodes = mesh.getNodes()
        self.x_grid,self.y_grid = np.reshape(self.nodes[:,0],(128,256)),\
                                    np.reshape(self.nodes[:,1],(128,256))
        
        # Distances of new snapshot from reference ones in parameter space
        dMuRefs = MuNew - MuRefs
        # Basis calculation by transporting the supplied reference snapshots
        self.phis = calc_transported_basis(coeffsRefsx,coeffsRefsy,self.i,self.j,\
            uRefs,dMuRefs)
        # Initial guess of generalized coordinates (for basis set of transported
        # snapshots), based on inverse-distance weights in parameter space
        self.coords0 = (1.0/abs(dMuRefs))/sum(1.0/abs(dMuRefs))
        # Calculate 1st-order finite difference operator with 4th-order accuracy
        #self.Di = Findiff_Taylor_uniform(len(self.iarr),1,3)/(self.iarr[1]-self.iarr[0])
        #self.Dj = Findiff_Taylor_uniform(len(self.jarr),1,3)/(self.jarr[1]-self.jarr[0])
        u = self.compose_soln(self.coords0)
        u[:,:,1:3] = vel_along_xy(u[:,:,1],u[:,:,2],self.x_grid,self.y_grid)
        u_shape = np.shape(u)
        u_flat = np.reshape(u,(-1,u_shape[2]))
        self.refNorm = residual(self.mesh,u_flat,uTest,2,self.sigma)
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
        print(np.shape(u))
        u[:,:,1:3] = vel_along_xy(u[:,:,1],u[:,:,2],self.x_grid,self.y_grid)
        u_shape = np.shape(u)
        u_flat = np.reshape(u,(-1,u_shape[2]))
        #print(np.shape(u_flat))
        res = residual(self.mesh,u_flat,self.uTest,2,self.sigma,refNorm = self.refNorm)
        
        return [res]

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
        u_computed = self.compose_soln(coords)
        u_computed[:,:,1:3] = vel_along_xy(u_computed[:,:,1],u_computed[:,:,2],self.x_grid,self.y_grid)
        u_shape = np.shape(u_computed)
        u_flat = np.reshape(u_computed,(-1,u_shape[2]))
        uTest_grid = np.reshape(self.uTest,(u_shape))
        #print(np.shape(u_flat))
        res = residual(self.mesh,u_flat,self.uTest,2,self.sigma,refNorm = self.refNorm)
        
        plt.figure()
        for iNb in range(u_shape[2]):
            
            plt.subplot(2,2,iNb+1)

            plt.contour(self.x_grid,self.y_grid,u_computed[:,:,iNb],20)
            #cs=plt.contour(self.x_grid,self.y_grid,uTest_grid[:,:,iNb],20,colors='red')
            #plt.clabel(cs)
            plt.title('contour plots') 
        plt.show()

#endclass Project_TSMOR_Online
