from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy as np
from petsc4py.PETSc import ScalarType  # type: ignore
import matplotlib.pyplot as plt
from dolfinx import default_real_type, geometry

import os
import shutil

if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')





t = 0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
dt = (T - t) / num_steps  # Time step size
R = 5
h = 0.1
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

def gen_coef_gauss9(N_sample, al, au, gl, gu):
    coef = np.zeros((N_sample, 9, 3))
    for n in range(N_sample):
        coef[n,0:9,0] = np.random.rand(9)*(au-al)+al
        coef[n,0:9,1] = np.random.rand(9)*(au-al)+al
        coef[n,0:9,2] = np.random.rand(9)*(gu-gl)+gl
    return coef
N_sample = 1
coef_f = gen_coef_gauss9(N_sample, al=10, au=20, gl=-1, gu=1)[0]
print(coef_f)


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
u_init = initial_condition(coef_f)
u_n = fem.Function(V)
u_n.interpolate(u_init)

class ax():
    def __init__(self):
        pass
    def __call__(self, x):
        return np.sin(x[0]) + np.sin(x[1])- 1/2*(x[0]**6+x[1]**6)
a_fun = ax()
a = fem.Function(V)
a.interpolate(a_fun)

class fx():
    def __init__(self):
        pass
    def __call__(self, x):
        return np.cos(x[0]) ,np.cos(x[1])
f_fun = fx()

v_2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
V_2 = fem.FunctionSpace(domain, v_2)
f = fem.Function(V_2)
f.interpolate(f_fun)

def visual(u,iter):
    bb_tree = geometry.bb_tree(domain, 2)
    nx = 100
    ny = 100
    x = np.linspace(-1,1,nx+1)
    y = x
    X,Y = np.meshgrid(x,y)
    p = np.concatenate((X[:,:,np.newaxis],Y[:,:,np.newaxis],np.zeros((nx+1,ny+1,1))),axis = -1)
    # Check against standard table value
    #p = np.array([2.,2.,0.], dtype=np.float64)
    value = np.zeros((nx+1,ny+1))
    u.x.scatter_forward()
    for i in range(0,nx+1):
        for j in range(0,ny+1):
            cell_candidates = geometry.compute_collisions_points(bb_tree, p[i,j,:])
            cells = geometry.compute_colliding_cells(domain, cell_candidates, p[i,j,:])
            value[i,j] = u.eval(p[i,j,:], cells[0])
    plt.figure()
    plt.imshow(value)
    plt.colorbar()
    plt.clim(-0.5,0.5)
    plt.savefig('results/'+str(iter)+'.png')
    plt.close()

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
for n in range(num_steps):
    # Update Diriclet boundary condition
    # Update the right hand side reusing the initial vector
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
    visual(u_n,n)

from PIL import Image, ImageSequence

png_images = ['results/'+str(i)+'.png' for i in range(num_steps)]
# 输出GIF文件名
output_gif = "output.gif"
 
# 打开第一张图片来创建动画
first_image = Image.open(png_images[0])
 
# 创建一个动画的帧列表
frames = [Image.open(img) for img in png_images]
 
# 设置GIF的参数，如延迟等
duration = 0.05  # 每帧的持续时间（秒）
 
# 保存为GIF
first_image.save(output_gif, save_all=True, append_images=frames, duration=duration, loop=0)
shutil.rmtree('results')