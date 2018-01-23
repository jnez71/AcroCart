"""
Simulation setup of an acrocart system.
See Simulator class docstring for more details.

"""
from __future__ import division
import numpy as np; npl = np.linalg
import time


class Simulator(object):
    """
    Class for simulation setup of an acrocart system.
    Faster-than-realtime timeseries simulation requires MatPlotLib for plotting.
    Realtime simulation requires Mayavi for graphics. (http://docs.enthought.com/mayavi/mayavi/installation.html)

    dyn: acrocart.Dynamics object

    """
    def __init__(self, dyn):
        self.dyn = dyn

    def simulate(self, q0, control, tN=0, dt=0.005, goal=None, override=None):
        """
        Runs the simulation and displays / plots the results.

        q0:       array initial state condition [pos, ang1, ang2, vel, angvel1, angvel2]
        control:  function of array state, scalar time, and scalar goal that returns a scalar input force
        tN:       scalar duration in time units, set to 0 for realtime simulation
        dt:       scalar timestep for simulation
        goal:     optional scalar position to be passed to control and marked as "goal" in visualizations
        override: optional function of scalar time that returns an array state that overrides
                  the actual simulation (allows you to visualize an arbitrary trajectory)

        """
        q0 = np.array(q0, dtype=np.float64)

        # Positive tN means timeseries simulation
        if tN > 0:
            from matplotlib import pyplot

            # Record real start time
            print "----"
            print "Simulating acrocart for a horizon of {}...".format(tN)
            start_time = time.time()

            # Run timeseries simulation and store results
            T = np.arange(0, tN+dt, dt)
            Q = np.zeros((len(T), self.dyn.n_q), dtype=np.float64)
            U = np.zeros((len(T), self.dyn.n_u), dtype=np.float64)
            Q[0] = np.copy(q0)
            for i, t in enumerate(T[:-1]):
                U[i] = control(Q[i], t, goal)
                if override is None:
                    Q[i+1] = self.dyn.step(Q[i], U[i], dt)
                else:
                    Q[i+1] = override(t)
            print "Simulation finished in {} realtime seconds.".format(np.round(time.time()-start_time, 3))
            print "Final state: {}".format(np.round(Q[-1], 3))
            print "Sum of squared input forces: {}".format(np.round(np.sum(U[:, 0]**2), 3))

            # Plot results
            print "Plotting results... (close plots to continue)"
            print "----"
            if goal is not None: pyplot.plot(T, [goal]*len(T), "k--", label="goal")
            pyplot.plot(T, Q[:, 0], "k", label="pos")
            pyplot.plot(T, Q[:, 1], "g", label="ang1")
            pyplot.plot(T, Q[:, 2], "b", label="ang2")
            pyplot.plot(T, Q[:, 3], "k--", label="vel")
            pyplot.plot(T, Q[:, 4], "g--", label="angvel1")
            pyplot.plot(T, Q[:, 5], "b--", label="angvel2")
            pyplot.plot(T, U[:, 0], "r", label="input")
            pyplot.xlim([T[0], T[-1]])
            pyplot.legend(fontsize=16)
            pyplot.xlabel("Time", fontsize=16)
            pyplot.title("AcroCart Simulation", fontsize=16)
            pyplot.grid(True)
            pyplot.show()  # blocking

        # Nonpositive tN implies realtime simulation
        else:
            print "----"
            print "Starting realtime acrocart simulation..."
            print "Setting-up Mayavi graphics..."
            from mayavi import mlab
            import os, vtk
            if os.path.exists("/dev/null"): shadow_realm = "/dev/null"
            else: shadow_realm = "c:\\nul"
            mlab_warning_output = vtk.vtkFileOutputWindow()
            mlab_warning_output.SetFileName(shadow_realm)
            vtk.vtkOutputWindow().SetInstance(mlab_warning_output)

            # Generate visualization objects and initial figure view
            fig = mlab.figure(size=(500, 500), bgcolor=(0.25, 0.25, 0.25))
            rail = mlab.plot3d(self.dyn.rail_lims, (0, 0), (0, 0), line_width=1, color=(1, 1, 1))
            cart = mlab.points3d(q0[0], 0, 0, scale_factor=0.2, mode="cube", color=(0, 0, 1))
            joint1 = mlab.points3d(q0[0], -0.125, 0, scale_factor=0.12, color=(0, 1, 1))
            x1, y1 = q0[0]+self.dyn.l1*np.sin(q0[1]), -self.dyn.l1*np.cos(q0[1])
            pole1 = mlab.plot3d((q0[0], x1), (-0.15, -0.15), (0, y1), line_width=1, color=(0, 1, 0))
            joint2 = mlab.points3d(x1, -0.175, y1, scale_factor=0.12, color=(0, 1, 1))
            pole2 = mlab.plot3d((x1, x1+self.dyn.l2*np.sin(q0[2])), (-0.2, -0.2), (y1, y1-self.dyn.l2*np.cos(q0[2])), line_width=1, color=(1, 0, 0))
            disp = mlab.text3d(-0.6, 0, 1.2*(self.dyn.l1+self.dyn.l2), "t = 0.0", scale=0.45)
            if goal is not None: mlab.points3d(goal, 0, -1.2*(self.dyn.l1+self.dyn.l2), scale_factor=0.2, mode="axes", color=(1, 0, 0))
            recenter = lambda: mlab.view(azimuth=-90, elevation=90, focalpoint=(np.mean(self.dyn.rail_lims), 0, 0), distance=1.8*np.sum(np.abs(self.dyn.rail_lims)))
            recenter()

            # Setup user keyboard interactions
            disturb = [0.0]
            reset = [False]
            def keyPressEvent(event):
                k = str(event.text())
                if k == '.': disturb[0] += 0.5
                elif k == ',': disturb[0] -= 0.5
                elif k == ' ': disturb[0] = 0.0
                elif k == 'v': recenter()
                elif k == 'r':
                    t[0] = 0.0
                    q[0] = np.copy(q0)
                    disturb[0] = 0.0
                    print "User triggered reset!"
                    reset[0] = True
                    start_time[0] = time.time()
            fig.scene._vtk_control.keyPressEvent = keyPressEvent
            print "--\nUSER KEYBOARD CONTROLS:"
            if override is None:
                print "Increment / decrement disturbance cart-force with '>' / '<' and cancel disturbance with ' ' (spacebar)."
            print "Reset view with 'v'. Reset simulation with 'r'.\n--"
            print "(close all Mayavi windows to continue...)"

            # Wrap simulation in animation
            @mlab.animate(delay=50)  # 20 FPS is best Mayavi can do
            def realtime_loop():
                while True:

                    # Simulate physics up to realtime
                    while (t[0] < time.time()-start_time[0]) and not reset[0]:
                        if override is None:
                            q[0] = self.dyn.step(q[0], control(q[0], t[0], goal), dt, disturb[0])
                        else:
                            q[0] = override(t[0])
                        t[0] += dt

                    # Update animation
                    reset[0] = False
                    cart.mlab_source.set(x=q[0][0])
                    joint1.mlab_source.set(x=q[0][0])
                    x1, y1 = q[0][0]+self.dyn.l1*np.sin(q[0][1]), -self.dyn.l1*np.cos(q[0][1])
                    pole1.mlab_source.set(x=(q[0][0], x1), z=(0, y1))
                    joint2.mlab_source.set(x=x1, z=y1)
                    pole2.mlab_source.set(x=(x1, x1+self.dyn.l2*np.sin(q[0][2])), z=(y1, y1-self.dyn.l2*np.cos(q[0][2])))
                    disp.text = "t = " + str(np.round(t[0], 1))
                    yield

            # Begin simulation and visualization
            t = [0.0]
            q = [np.copy(q0)]
            start_time = [time.time()]
            realtime_loop()
            mlab.show()  # blocking
            print "----"
