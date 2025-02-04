from jet_bcs import JetBCValue
from dolfin import *
import numpy as np
from generate_msh import generate_mesh_slit
import os, subprocess
from msh_convert import convert


class FlowSolver(object):
    '''IPCS scheme with explicit treatment of nonlinearity.'''
    def __init__(self, flow_params, geometry_params, solver_params):
        # Using very simple IPCS solver
        self.mu = Constant(flow_params['mu'])              # dynamic viscosity
        self.rho = Constant(flow_params['rho'])            # density

        mu, rho = self.mu, self.rho

        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params

        # Update mesh
        #generate_mesh_slit(geometry_params, template=self.geometry_params['template'])
        #subprocess.call(['python3 generate_msh.py -output mesh/geometry_2d.geo -slit_angle %f -slit_width %f -clscale %f' % (geometry_params['slit_angle'], geometry_params['slit_width'], geometry_params['clscale'])], shell = True)
        subprocess.call(['python3 generate_msh.py -output mesh/geometry_2d.geo -slit_angle %f -slit_width %f -clscale %f' % (geometry_params['slit_angle'], geometry_params['slit_width'], geometry_params['clscale'])], shell = True)
        convert('mesh/geometry_2d.msh', 'mesh/geometry_2d.h5')

        mesh_file = geometry_params['mesh']

        # Load mesh with markers
        mesh = Mesh()
        comm = mesh.mpi_comm()
        mesh_file = "mesh/geometry_2d.h5"
        h5 = HDF5File(comm, mesh_file, 'r')

        h5.read(mesh, 'mesh', False)

        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        h5.read(surfaces, 'facet')

        # These tags should be hardcoded by gmsh during generation
        self.inlet_tag = 3
        self.outlet_tag = 2
        self.wall_tag = 1
        self.cylinder_noslip_tag = 4
        inlet_tag = 3
        outlet_tag = 2
        wall_tag = 1
        cylinder_noslip_tag = 4
        # Tags 5 and higher are jets

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 1)
        Q = FunctionSpace(mesh, 'CG', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        u_n, p_n = Function(V), Function(Q)
        #u_nodal_values = u_n.vector()
        #u_array = np.array(u_nodal_values)
        #coor = mesh.coordinates()
        #if mesh.num_vertices() == len(u_array):
        #    for i in range(mesh.num_vertices()):
        #        print('u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i]))
        # Starting from rest or are we given the initial state
        # FIXME: Where does the initial state come from? 
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):
            # The following condition is False
            if path in flow_params:
                comm = mesh.mpi_comm()

                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)
                # assert func.vector().norm('l2') > 0

        u_, p_ = Function(V), Function(Q)  # Solve into these

        #file = File("u_n.pvd")
        #file << u_n
        #file = File("u.pvd")
        #file << u_
        #file = File("p_n.pvd")
        #file << p_n
        #file = File("p.pvd")
        #file << p_
        dt = Constant(solver_params['dt'])
        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        # Define variational problem for step 1
        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        inflow_profile = flow_params['inflow_profile'](mesh, degree=2)
        #self.inflow_profile = inflow_profile
        # Define boundary conditions, first those that are constant in time
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
        # No slip
        bcu_wall = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag)
        bcu_cyl_wall = DirichletBC(V, Constant((0, 0)), surfaces, cylinder_noslip_tag)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)

        # Now the expression for the jets
        # NOTE: they start with Q=0
        radius = geometry_params['jet_radius']
        width = geometry_params['jet_width']
        positions = geometry_params['jet_positions']

        # Now the expression for the slit
        slit_width = geometry_params['slit_width']
        slit_angle = geometry_params['slit_angle']

        bcu_jet = []
        jet_tags = list(range(cylinder_noslip_tag+1, cylinder_noslip_tag+1+len(positions)))

        jets = [JetBCValue(radius, width, theta0, Q=0, degree=1) for theta0 in positions]

        #for tag, jet in zip(jet_tags, jets):
        #    bc = DirichletBC(V, jet, surfaces, tag)
        #    bcu_jet.append(bc)

        # All bcs objects togets
        #bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall] + bcu_jet
        bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall]
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get('la_solve', 'lu')
        assert solver_type in ('lu', 'la_solve')

        if solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver(), range(3)))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)

            if not solver_type == 'lu':
                s.parameters['relative_tolerance'] = 1E-8
                s.parameters['monitor_convergence'] = True

        gtime = 0.  # External clock

        # Things to remeber for evolution
        self.jets = jets
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n
        #print('flow, ', np.array(self.u_), np.array(self.p_))

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.cylinder_surface_tags = [cylinder_noslip_tag]


    def bc_enforcement(self, flow_params, geometry_params, solver_params):
        '''Boundary enforcement'''

        self.mu = Constant(flow_params['mu'])              # dynamic viscosity
        self.rho = Constant(flow_params['rho'])            # density

        mu, rho = self.mu, self.rho

        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params

        mesh_file = geometry_params['mesh']

        # Load mesh with markers
        mesh = Mesh()
        comm = mesh.mpi_comm()
        mesh_file = "/home/fenics/host/Cylinder2DFlowControlDRLParallel/Cylinder2DFlowControlWithRL/mesh/geometry_2d.h5" #FIXME: Now use the absolute path
        h5 = HDF5File(comm, mesh_file, 'r')

        h5.read(mesh, 'mesh', False)

        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        h5.read(surfaces, 'facet')

        # These tags should be hardcoded by gmsh during generation
        self.inlet_tag = 3
        self.outlet_tag = 2
        self.wall_tag = 1
        self.cylinder_noslip_tag = 4
        inlet_tag = 3
        outlet_tag = 2
        wall_tag = 1
        cylinder_noslip_tag = 4
        # Tags 5 and higher are jets

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        # FIXME: recheck the following stuff
        u_n, p_n = Function(V), Function(Q)
        u_nodal_values = u_n.vector()
        u_array = np.array(u_nodal_values)
        coor = mesh.coordinates()
        print('init ', mesh.num_vertices(), len(u_array))
        if mesh.num_vertices() == len(u_array):
            for i in range(mesh.num_vertices()):
                print('u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i]))
        # Starting from rest or are we given the initial state
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):
            if path in flow_params:
                comm = mesh.mpi_comm()

                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)
                # assert func.vector().norm('l2') > 0

        u_, p_ = Function(V), Function(Q)  # Solve into these

        #file = File("u_n.pvd")
        #file << u_n
        #file = File("u.pvd")
        #file << u_
        #file = File("p_n.pvd")
        #file << p_n
        #file = File("p.pvd")
        #file << p_
        dt = Constant(solver_params['dt'])
        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        # Define variational problem for step 1
        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        inflow_profile = flow_params['inflow_profile'](mesh, degree=2)
        self.inflow_profile = inflow_profile
        # Define boundary conditions, first those that are constant in time
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
        # No slip
        bcu_wall = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag)
        bcu_cyl_wall = DirichletBC(V, Constant((0, 0)), surfaces, cylinder_noslip_tag)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)

        # Now the expression for the jets
        # NOTE: they start with Q=0
        radius = geometry_params['jet_radius']
        width = geometry_params['jet_width']
        positions = geometry_params['jet_positions']

        # Now the expression for the slit
        slit_width = geometry_params['slit_width']
        slit_angle = geometry_params['slit_angle']

        bcu_jet = []
        jet_tags = list(range(cylinder_noslip_tag+1, cylinder_noslip_tag+1+len(positions)))

        jets = [JetBCValue(radius, width, theta0, Q=0, degree=1) for theta0 in positions]

        #for tag, jet in zip(jet_tags, jets):
        #    bc = DirichletBC(V, jet, surfaces, tag)
        #    bcu_jet.append(bc)

        # All bcs objects togets
        #bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall] + bcu_jet
        bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall]
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get('la_solve', 'lu')
        assert solver_type in ('lu', 'la_solve')

        if solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver(), range(3)))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)

            if not solver_type == 'lu':
                s.parameters['relative_tolerance'] = 1E-8
                s.parameters['monitor_convergence'] = True
  
        gtime = 0.  # External clock

        # Things to remeber for evolution
        self.jets = jets
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n
        #print('flow, ', np.array(self.u_), np.array(self.p_))

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.cylinder_surface_tags = [cylinder_noslip_tag]


    def evolve(self, jet_bc_values, slit_width, slit_angle):
        '''Make one time step with the given values of jet boundary conditions'''
        assert len(jet_bc_values) == len(self.jets)

        # Update bc expressions
        for Q, jet in zip(jet_bc_values, self.jets): jet.Q = Q

        geometry_params = self.geometry_params
        flow_params = self.flow_params
        solver_params = self.solver_params

        # Update mesh and bcs
        #generate_mesh_slit(geometry_params, template=self.geometry_params['template'])

        #self.bc_enforcement(flow_params, geometry_params, solver_params)

        # Make a step
        self.gtime += self.dt(0)

        inflow = self.inflow_profile
        if hasattr(inflow, 'time'):
            inflow.time = self.gtime

        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        #print('test ', np.array(bs))
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n


        #file = File("u_n1.pvd")
        #file << u_n
        #file = File("u1.pvd")
        #file << u_n
        #file = File("p_n1.pvd")
        #file << p_n
        #file = File("p1.pvd")
        #file << p_


        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            solver.solve(uh.vector(), b)

        u_n.assign(u_)
        p_n.assign(p_)

        # Share with the world
        return u_, p_


def profile(mesh, degree):
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]
    H = top - bot
    
    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                       '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)

# --------------------------------------------------------------------

if __name__ == '__main__':
    geometry_params = {'jet_radius': 0.05,
                       'jet_width': 0.41,
                       'jet_positions': [0, 270],
                       'slit_width': 0.1,
                       'slit_angle': 10,
                       'mesh': './mesh/geometry_2d.h5',
                       'clscale': 0.25}

    flow_params = {'mu': 1E-3,
                   'rho': 1,
                   'inflow_profile': profile}
    
    solver_params = {'dt': 5E-4}


    solver = FlowSolver(flow_params, geometry_params, solver_params)

    while True:
        u, p = solver.evolve([0, 0], 0.1, 10)

        print('|u|= %g' % u.vector().norm('l2'), '|p|= %g' % p.vector().norm('l2'))
