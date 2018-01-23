"""
Optimal control setup for an acrocart system.
This requires cyipopt, a Python wrapper for IPOPT (https://github.com/matthias-k/cyipopt).
Of course, IPOPT itself is necessary too (https://github.com/coin-or/Ipopt).
See Optimizer class docstring for more details.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
import ipopt


class Optimizer(object):
    """
    Class for the optimal controller of an acrocart system.
    Transcription method is trapezoidal direct collocation.
    IPOPT's interior-point method is used to solve the nonlinear program.

    dyn:       acrocart.Dynamics object
    tol:       float nonlinear-program solver convergence tolerance
    max_iter:  integer maximum number of IPOPT iterations per optimization
    verbosity: integer (-1 to 12) for print level of IPOPT (-1 silences Optimizer too)
    
    """
    def __init__(self, dyn, tol=1E-6, max_iter=500, verbosity=0):
        self.dyn = dyn
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbosity = int(np.clip(verbosity, 0, 12))
        if verbosity < 0: self.opt_prints = False
        else: self.opt_prints = True

        # For converting a trajectory (U, Q) to and from a solution vector x
        self.traj_from_x = lambda x, N: (x[:N].reshape(N, self.dyn.n_u), x[N:].reshape(N, self.dyn.n_q))
        self.x_from_traj = lambda U, Q: np.concatenate((U.ravel(), Q.ravel()))

        # IPOPT's numerical infinity
        self.inf = 2E19

    def make_controller(self, T, U, kp=(0, 20, 20), kd=(0, 20, 20), band=np.deg2rad(20)): #??? THIS IS UNDER CONSTRUCTION
        """
        Returns a function of array state, scalar time, and scalar positional-goal
        that first linearly interpolates an open-loop input timeseries and then
        implements a cascade balancing controller that hones the goal position.
        If goal is None, the controller will just balance anywhere.

        T:    array N-element time-grid in ascending order
        U:    array N-element input timeseries
        kp:   tuple of proportional gains for position and angle freedoms
        kd:   tuple of derivative gains for position and angle freedoms
        band: scalar angular band in radians about upright in which balancer takes over

        """
        kp = np.array(kp, dtype=np.float64)
        kd = np.array(kd, dtype=np.float64)
        openloop = interp1d(T, U, kind="linear", assume_sorted=True, axis=0)
        def control(q, t, goal=None):
            if (np.abs(np.pi-q[1]) > band) and (np.abs(np.pi-q[2]) > band) and \
               (t >= T[0]) and (t <= T[-1]):
                return openloop(t)
            if goal is None:
                ref = np.pi
            else:
                ref = np.pi + kp[0]*(q[0] - goal) + kd[0]*q[3]
            err1 = ref - np.mod(q[1], 2*np.pi)
            err2 = ref - np.mod(q[2], 2*np.pi)
            return kp[1]*err1 + kp[2]*err2 - kd[1]*q[4] - kd[2]*q[5]
        return control

    def make_trajectory(self, q0, qN, tN, H, rush=0.0, plot=False):
        """
        Returns a time-grid, input timeseries, and state timeseries that
        execute an energy-optimal acrocart trajectory between two states.
        Also returns optimizer success bool.

        q0:   array starting state
        qN:   array ending state at time tN
        tN:   scalar final time
        H:    tuple of discretization step-sizes in descending order
        rush: scalar (0 to 1) that scales-up error weight compared to effort weight
        plot: bool for if intermediate solutions should be plotted (requires MatPlotLib)

        """
        rush = np.clip(rush, 0.0, 1.0)
        if plot: from matplotlib import pyplot

        # Solve transcribed optimization problem for increasingly refined time grids
        for grid_number, h in enumerate(H):
            T = np.arange(0, tN+h, h, dtype=np.float64)
            N = len(T)

            # Generate initial trajectory guess or interpolate previous result
            if grid_number == 0:
                U = np.zeros((N, self.dyn.n_u), dtype=np.float64)
                Q = np.zeros((N, self.dyn.n_q), dtype=np.float64)
                L = np.zeros((N-1, self.dyn.n_q), dtype=np.float64)  # costates
                for i in xrange(self.dyn.n_q):
                    Q[:, i] = np.linspace(q0[i], qN[i], N)
            else:
                U = interp1d(T_prev, U, kind="linear", assume_sorted=True, axis=0)(T)
                Q = interp1d(T_prev, Q, kind="quadratic", assume_sorted=True, axis=0)(T) # ??? should use Qdot for accuracy
                L = interp1d(T_prev[:-1], L, kind="quadratic", assume_sorted=True, axis=0,
                             bounds_error=False, fill_value="extrapolate")(T[:-1])
            if grid_number < len(H)-1: T_prev = T
            x = self.x_from_traj(U, Q)

            # Generate constraint and solution bounds
            c_eq = np.zeros((N-1)*self.dyn.n_q)
            x_lower = np.concatenate(([self.dyn.force_lims[0]]*N, q0,
                                      ([self.dyn.rail_lims[0]]+(self.dyn.n_q-1)*[-self.inf])*(N-2), qN))
            x_upper = np.concatenate(([self.dyn.force_lims[1]]*N, q0,
                                      ([self.dyn.rail_lims[1]]+(self.dyn.n_q-1)*[self.inf])*(N-2), qN))

            # Configure and call IPOPT
            nlp = ipopt.problem(n=len(x), m=len(c_eq),
                                problem_obj=self._Problem(self.dyn, N, h, qN, rush, self.dyn.n_u, self.dyn.n_q),
                                lb=x_lower, ub=x_upper, cl=c_eq, cu=c_eq)
            nlp.addOption("tol", self.tol)
            nlp.addOption("max_iter", self.max_iter)
            nlp.addOption("sb", "yes")
            nlp.addOption("print_level", self.verbosity)
            nlp.addOption("print_frequency_iter", self.max_iter)
            nlp.addOption("warm_start_init_point", "yes")
            nlp.addOption("warm_start_bound_push", 100*self.tol)
            nlp.addOption("warm_start_mult_bound_push", 100*self.tol)
            nlp.addOption("mu_init", 100*self.tol)
            nlp.addOption("mu_strategy", "adaptive")
            nlp.addOption("nlp_scaling_method", "user-scaling")
            nlp.setProblemScaling(obj_scaling=h)
            if self.opt_prints: print "Making optimal trajectory with dt = {}...".format(h)
            # nlp.addOption("derivative_test", "first-order")  # for debug only
            x[:], info = nlp.solve(x, lagrange=L.ravel())
            if self.opt_prints: print "--------------------"

            # Extract trajectory from solution
            U[:], Q[:] = self.traj_from_x(x, N)
            L[:] = np.reshape(info["mult_g"], (N-1, self.dyn.n_q))
            if plot: self._plot(T, Q, U, L, h, pyplot)

        # Return grid, trajectory, and success bool
        return T, U, Q, (not bool(info["status"]))

    class _Problem(object):
        """
        Special class for configuring a cyipopt optimization problem.
        See: http://pythonhosted.org/ipopt/reference.html
        The decision variable here x is [U.ravel(), Q.ravel()],
        subject to dynamics collocation constraints.

        """
        def __init__(self, dyn, N, h, qN, rush, n_u, n_q):
            self.dyn = dyn
            self.N = int(N)
            self.h = np.float64(h)
            self.qN = np.array(qN, dtype=np.float64)
            self.rush = np.float64(rush)
            self.n_u = int(n_u)
            self.n_q = int(n_q)

            # Initialize static memory for large arrays
            len_x = self.N*(1+self.n_q)
            self.E = np.zeros((self.N, self.n_q), dtype=np.float64)
            self.U2 = np.zeros(self.N, dtype=np.float64)
            self.E2 = np.zeros(self.N, dtype=np.float64)
            self.Qdot = np.zeros((self.N, self.n_q), dtype=np.float64)
            self.AA = np.zeros((self.N, self.n_q, self.n_q), dtype=np.float64)
            self.BB = np.zeros((self.N, self.n_q, self.n_u), dtype=np.float64)
            self.dFdx = np.zeros((self.N*self.n_q, len_x), dtype=np.float64)
            self.dcdx = np.zeros(((self.N-1)*self.n_q, len_x), dtype=np.float64)

            # Analyze jacobian sparsity
            xtest = 20*(np.random.sample(len_x)-0.5)
            self.AA[:], self.BB[:] = self.dyn.linearize(xtest[self.N:].reshape(self.N, self.n_q), xtest[:self.N].reshape(self.N, self.n_u))
            self.BB_idx = self.BB.nonzero()
            self.AA_idx = self.AA.nonzero()
            self.dFdx[:] = np.hstack((block_diag(*self.BB), block_diag(*self.AA)))
            self.dFdx_Bidx = self.dFdx[:, :self.N].nonzero()
            self.dFdx_Aidx = self.dFdx[:, self.N:].nonzero()
            self.dFdx_lidx = self.dFdx[:-self.n_q].nonzero()
            self.dFdx_ridx = self.dFdx[self.n_q:].nonzero()
            Q_extractor = np.eye(len_x, dtype=np.int32)[self.N:]
            Q_diffmat = Q_extractor[:-self.n_q] - Q_extractor[self.n_q:]
            Q_diffmat_idx = Q_diffmat.nonzero()
            self.Q_diffarr_l = Q_diffmat[self.dFdx_lidx].astype(np.float64)
            self.Q_diffarr_r = Q_diffmat[self.dFdx_ridx].astype(np.float64)
            self.dcdx[self.dFdx_lidx] = self.dFdx[:-self.n_q][self.dFdx_lidx]
            self.dcdx[self.dFdx_ridx] = self.dFdx[self.n_q:][self.dFdx_ridx]
            self.dcdx[Q_diffmat_idx] += Q_diffmat[Q_diffmat_idx]  # important: this actually sets certain nonzero constants in dcdx
            self.jac_rows, self.jac_cols = self.dcdx.nonzero()

        def objective(self, x):
            """
            Simple sum-of-squared input force functional.

            """
            self.U2[:] = x[:self.N]**2
            self.E2[:] = np.sum((x[self.N:].reshape(self.N, self.n_q) - self.qN)**2, axis=1)
            return (self.h/2) * ((1-self.rush)*(self.U2[0] + 2*np.sum(self.U2[1:-1]) + self.U2[-1]) +
                                 self.rush*(self.E2[0] + 2*np.sum(self.E2[1:-1]) + self.E2[-1]))

        def gradient(self, x):
            """
            Gradient of the objective function.

            """
            self.E[:] = x[self.N:].reshape(self.N, self.n_q) - self.qN
            return self.h * np.concatenate(([(1-self.rush)*x[0]], 2*(1-self.rush)*x[1:self.N-1], [(1-self.rush)*x[self.N-1]],
                                            self.rush*self.E[0], 2*self.rush*self.E[1:-1].ravel(), self.rush*self.E[-1]))

        def constraints(self, x):
            """
            Trapezoidal collocation constraints. Should equal zero.

            """
            U = x[:self.N].reshape(self.N, self.n_u)
            Q = x[self.N:].reshape(self.N, self.n_q)
            self.Qdot[:] = self.dyn.F(Q, U)
            return ((self.h/2)*(self.Qdot[:-1] + self.Qdot[1:]) + (Q[:-1] - Q[1:])).ravel()

        def jacobian(self, x):
            """
            Returns the jacobian of the constraints as a row-major-flattened
            array of its nonzero values.

            """
            self.AA[:], self.BB[:] = self.dyn.linearize(x[self.N:].reshape(self.N, self.n_q), x[:self.N].reshape(self.N, 1))
            self.dFdx[self.dFdx_Bidx] = self.BB[self.BB_idx]
            self.dFdx[:, self.N:][self.dFdx_Aidx] = self.AA[self.AA_idx]
            self.dcdx[self.dFdx_lidx] = (self.h/2)*self.dFdx[:-self.n_q][self.dFdx_lidx] + self.Q_diffarr_l
            self.dcdx[self.dFdx_ridx] = (self.h/2)*self.dFdx[self.n_q:][self.dFdx_ridx] + self.Q_diffarr_r
            return self.dcdx[self.jac_rows, self.jac_cols]

        def jacobianstructure(self):
            """
            Returns the (rows, columns) indices where the full jacobian of
            the constraints is nonzero.

            """
            return (self.jac_rows, self.jac_cols)

        # def hessian(self, x, lam, factor):
        #     """
        #     Optional, returns hessian of the problem lagrangian.

        #     """
        #     pass

        # def hessianstructure(self):
        #     """
        #     Optional, returns (rows, columns) indices where the lagrangian hessian is nonzero.

        #     """
        #     pass

        # def intermediate(self, alg_mod, iter_count, obj_value,
        #                  inf_pr, inf_du, mu, d_norm, regularization_size,
        #                  alpha_du, alpha_pr, ls_trials):
        #     """
        #     Optional, callback for each iteration IPOPT takes.

        #     """
        #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

    def _plot(self, T, Q, U, L, h, pyplot):
        fig = pyplot.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.set_ylabel("Solution", fontsize=16)
        ax.plot(T, Q[:, 0], "k", label="pos")
        ax.plot(T, Q[:, 1], "g", label="ang1")
        ax.plot(T, Q[:, 2], "b", label="ang2")
        ax.plot(T, Q[:, 3], "k--", label="vel")
        ax.plot(T, Q[:, 4], "g--", label="angvel1")
        ax.plot(T, Q[:, 5], "b--", label="angvel2")
        ax.plot(T, U[:, 0], "r", label="input")
        ax.set_xlim([T[0], T[-1]])
        ax.legend(fontsize=16)
        ax.grid(True)
        ax = fig.add_subplot(2, 1, 2)
        ax.set_ylabel("Costates", fontsize=16)
        ax.set_xlabel("Time (dt = {})".format(h), fontsize=16)
        ax.plot(T[:-1], L[:, 0], "k", label="l_pos")
        ax.plot(T[:-1], L[:, 1], "g", label="l_ang1")
        ax.plot(T[:-1], L[:, 2], "b", label="l_ang2")
        ax.plot(T[:-1], L[:, 3], "k--", label="l_vel")
        ax.plot(T[:-1], L[:, 4], "g--", label="l_angvel1")
        ax.plot(T[:-1], L[:, 5], "b--", label="l_angvel2")
        ax.set_xlim([T[0], T[-1]])
        ax.grid(True)
        print "Showing intermediate optimization result..."
        print "(close plot to continue)"
        pyplot.show()  # blocking
        print "--------------------"
