from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy as np
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt
from dolfinx import default_real_type, geometry
from tqdm import tqdm
import os
import shutil

# if os.path.exists('results'):
#     shutil.rmtree('results')
# os.mkdir('results')
class initial_condition():
    def __init__(self,coef):
        '''
        coef: 9,3
        '''
        self.coef = coef
        loc = np.zeros((9,2))
        ind = 0
        for ii in range(3):
            for jj in range(3):
                loc[ind,0] = 0.3*ii+0.2-0.5
                loc[ind,1] = 0.3*jj+0.2-0.5
        self.loc = loc       
    def __call__(self, x):
        val = 0
        for i in range(9):
            val += self.coef[i,2]*np.exp(-self.coef[i,0]*(x[0]-self.loc[i,0])**2)*np.exp(-self.coef[i,1]*(x[1]-self.loc[i,1])**2)
        return val


class ax():
    def __init__(self):
        pass
    def __call__(self, x):
        return np.sin(x[0]) + np.sin(x[1])- 1/2*(x[0]**6+x[1]**6)

class fx():
    def __init__(self):
        pass
    def __call__(self, x):
        return np.cos(x[0]) ,np.cos(x[1])
    
class DataGen():
    """"A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, T = 0.1, 
                 num_step = 50, 
                 R = 3, 
                 h =0.1, 
                 N_sample = 100,
                 al = 20,
                 au = 30,
                 gl = -1,
                 gu = 1):
        super().__init__()
        self.T = T
        self.num_step = num_step
        self.R = R
        self.h = h
        self.N_sample = N_sample
        self.al = al
        self.au = au
        self.gl = gl
        self.gu = gu
    def gen_coef_gauss9(self,N_sample, al, au, gl, gu):
        coef = np.zeros((N_sample, 9, 3))
        for n in range(N_sample):
            coef[n,0:9,0] = np.random.rand(9)*(au-al)+al
            coef[n,0:9,1] = np.random.rand(9)*(au-al)+al
            coef[n,0:9,2] = np.random.rand(9)*(gu-gl)+gl
        return coef
    def sample(self):
        self.coef = self.gen_coef_gauss9(N_sample=self.N_sample,al = self.al, au = self.au, gl = self.gl, gu = self.gu)
    
    def get_value(self,u,domain,nx,ny,R):
        bb_tree = geometry.bb_tree(domain, 2)
        x = np.linspace(-R,R,nx+1)
        y = np.linspace(-R,R,ny+1)
        X,Y = np.meshgrid(x,y)
        p = np.concatenate((X[:,:,np.newaxis],Y[:,:,np.newaxis],np.zeros((nx+1,ny+1,1))),axis = -1)
        value = np.zeros((nx+1,ny+1))
        u.x.scatter_forward()
        for i in range(0,nx+1):
            for j in range(0,ny+1):
                cell_candidates = geometry.compute_collisions_points(bb_tree, p[i,j,:])
                cells = geometry.compute_colliding_cells(domain, cell_candidates, p[i,j,:])
                value[i,j] = u.eval(p[i,j,:], cells[0])
        return value

    def generate(self,iter):
        t = 0
        T = self.T
        num_steps = self.num_step
        dt = (T - t) / num_steps  # Time step size\
        R = self.R
        h = self.h
        nx, ny = int(2*R/h), int(2*R/h)
        domain = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((-R, -R), (R, R)), n=(nx, ny),
                            cell_type=mesh.CellType.triangle)
        V = fem.FunctionSpace(domain, ("Lagrange", 1))
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=boundary_facets)
        bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
        u_init = initial_condition(self.coef[iter,:,:])
        u_n = fem.Function(V)
        u_n.interpolate(u_init)
        a_fun = ax()
        a = fem.Function(V)
        a.interpolate(a_fun)
        f_fun = fx()
        v_2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
        V_2 = fem.FunctionSpace(domain, v_2)
        f = fem.Function(V_2)
        f.interpolate(f_fun)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        F = u * v * ufl.dx + 1/2*dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + dt* ufl.dot(f,ufl.grad(u))*v* ufl.dx- dt*a*u*v*ufl.dx - u_n * v * ufl.dx
        lhs = fem.form(ufl.lhs(F))
        rhs = fem.form(ufl.rhs(F))
        A = assemble_matrix(lhs, bcs=[bc])
        A.assemble()
        b = create_vector(rhs)
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        uh = fem.Function(V)
        u_data = np.zeros((num_steps+1,nx+1,nx+1))
        for n in range(num_steps):
            # Update Diriclet boundary condition
            # Update the right hand side reusing the initial vector
            u_data[n,:,:] = self.get_value(u=u_n,domain = domain, nx = nx, ny = ny, R= R)
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, rhs)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [lhs], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, uh.vector)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array
            if n%10 == 0:
                print('time step',n)
        u_data[-1,:,:]  =  self.get_value(u=u_n,domain = domain, nx = nx, ny = ny, R= R)
        np.save('data/sol_'+str(iter)+'.npy',u_data)
        


if __name__ == '__main__':
    
    gen = DataGen(N_sample=300)
    gen.sample()
    for i in tqdm(range(300)):
        gen.generate(i)