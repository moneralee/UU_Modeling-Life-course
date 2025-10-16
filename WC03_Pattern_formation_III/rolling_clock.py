import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Parameters
dt=1
totaltime=3*24*3600
starting_time_step_visualizer=1000
time_steps = int(totaltime / dt)
alpha = 1.0/60
beta = 1.0/60
K = 1.0
n=4
tau=20*60
mu=0.06/60
v=0.03/60

class Clock:
    def __init__(self, alpha, beta, K, n, tau, mu, v, m0, p0, t=0):
        """Initialize the clock with parameters and initial conditions.

        Parameters:
        alpha (float): Growth rate of m.
        beta (float): Growth rate of p.
        K (float): Carrying capacity.
        n (float): Hill coefficient.
        tau (float): Delay time.
        mu (float): Degradation rate of m.
        v (float): Degradation rate of p.
        m0 (float): Initial value of m.
        p0 (float): Initial value of p.
        t (float): Initial time.
        """
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.n = n
        self.tau = tau
        self.max_tau = 1.5*tau
        self.index_tau = None
        self.mu = mu
        self.v = v
        self.m_values = deque([m0])
        self.p_values = deque([p0])
        self.t_values = deque([t])  # Initialize with the initial time
        self.t_values_hours = np.array([t / 3600])  # Convert to hours

    def __copy__(self):
        """Create a copy of the Clock instance."""
        new_clock = Clock(self.alpha, self.beta, self.K, self.n, self.tau, self.mu, self.v, self.m_values[0], self.p_values[0], self.t_values[0])
        new_clock.max_tau = self.max_tau
        new_clock.index_tau = self.index_tau
        new_clock.m_values = deque(self.m_values)
        new_clock.p_values = deque(self.p_values)
        new_clock.t_values = deque(self.t_values)
        return new_clock

    def set_tau(self, tau):
        """Set the delay time."""
        if tau > self.max_tau:
            raise ValueError(f"tau must be less than {self.max_tau}, otherwise no memory of time delay")
        # update tau and max_tau
        self.tau = tau
        self.index_tau = - int(self.tau / dt)

        if tau > 0.75*self.max_tau:
            self.max_tau = int(1.5*tau)
        # drop values older than new max_tau
        if len(self.t_values) > self.max_tau / dt:
            # print(len(self.t_values), -int(self.max_tau / dt),self.t_values[-int(self.max_tau / dt):])
            self.t_values = deque([self.t_values[i] for i in range(-int(self.max_tau / dt), 0)])
        if len(self.p_values) > self.max_tau / dt:
            self.p_values = deque([self.p_values[i] for i in range(-int(self.max_tau / dt), 0)])
        if len(self.m_values) > self.max_tau / dt:
            self.m_values = deque([self.m_values[i] for i in range(-int(self.max_tau / dt), 0)])


    def simulate(self, t, dt):
        # Update time values, limit memory to max_tau time
        if len(self.t_values) < self.max_tau /dt:
            self.t_values.append(t + dt)
        else:
            self.t_values.popleft()
            self.t_values.append(t + dt)  # Keep the last tau/dt values

        # Convert time to hours
        # self.t_values_hours.append((t + dt)/3600)

        # Determine indices for delayed times
        # index_tau = max(0, int((t - self.tau) / dt))  # Ensure index is not negative

        # Get delayed values
        if self.index_tau is None:
            self.index_tau = - int(self.tau / dt)
        try:
            p_delayed = self.p_values[self.index_tau]
        except IndexError:
            p_delayed = self.p_values[0]  # Use the last known value if index is out of range

        # Compute derivatives
        dtm = self.alpha / (1 + (p_delayed / self.K)**self.n) - self.mu * self.m_values[-1]
        dtp = self.beta * self.m_values[-1] - self.v * self.p_values[-1]

        # Update values, limit memory to max_tau time
        if len(self.m_values) < self.max_tau / dt:
            self.m_values.append(self.m_values[-1] + dt * dtm)
        else:
            self.m_values.popleft()
            self.m_values.append(self.m_values[-1] + dt * dtm)
        if len(self.p_values) < self.max_tau / dt:
            self.p_values.append(self.p_values[-1] + dt * dtp)
        else:
            self.p_values.popleft()
            self.p_values.append(self.p_values[-1] + dt * dtp)
        return self.m_values[-1], self.p_values[-1]

    def init_plot(self):
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        self.t_values_hours = np.array(self.t_values)/ 3600
        #Create plot of m and p over time
        graph_m = ax[0].plot(self.t_values_hours, self.m_values, label="m")[0]
        graph_p = ax[0].plot(self.t_values_hours, self.p_values, label="p")[0]
        ax[0].set_xlim(0, totaltime / 3600)
        ax[0].set_ylim(0, 1.01 * max(max(self.m_values), max(self.p_values)))
        ax[0].set_xlabel("Time (hours)")
        ax[0].set_ylabel("Values")
        ax[0].legend(loc='center right')
        ax[0].set_title('m and p over time')

        # Create plot of m and p in phase space
        graph_phase_space = ax[1].plot(self.m_values[0], self.p_values[0], color='k', zorder=1)[0]
        pos_phase_space = ax[1].quiver(self.m_values[0], self.p_values[0], [0], [0], scale_units='inches', angles='xy', scale=3, facecolor='grey', ec='white', headlength=15, headwidth=10, headaxislength=15, pivot='tip', zorder=2)
        ax[1].set_xlabel("m")
        ax[1].set_ylabel("p")
        ax[1].set_title('Trajectory in m,p space')
        return fig, ax, graph_m, graph_p, graph_phase_space, pos_phase_space

    def update_plot(self, fig, ax, graph_m, graph_p, graph_phase_space, pos_phase_space):
        # Update plot m and p over time
        self.t_values_hours = np.array(self.t_values)/ 3600
        graph_m.set_data(self.t_values_hours, self.m_values)
        graph_p.set_data(self.t_values_hours, self.p_values)

        # Update trajectory in phase space
        graph_phase_space.set_data(self.m_values, self.p_values)
        pos_phase_space.set_offsets(np.c_[self.m_values[-1], self.p_values[-1]])
        pos_phase_space.set_UVC((self.m_values[-1] - self.m_values[-2]) / np.sqrt((self.m_values[-1] - self.m_values[-2])**2 + (self.p_values[-1] - self.p_values[-2])**2),
                                (self.p_values[-1] - self.p_values[-2]) / np.sqrt((self.m_values[-1] - self.m_values[-2])**2 + (self.p_values[-1] - self.p_values[-2])**2))

        # Perform update on canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

    def update_plot_limits(self, ax):
        # Update y-axis limits for m and p over time
        if max(self.m_values) > ax[0].get_ylim()[1]:
            ax[0].set_ylim(0, 1.01 * max(self.m_values))
        if max(self.p_values) > ax[0].get_ylim()[1]:
            ax[0].set_ylim(0, 1.01 * max(self.p_values))

        # Update x-axis limits for m and p over time
        if max(self.t_values_hours) > ax[0].get_xlim()[1]:
            ax[0].set_xlim(min(self.t_values_hours), 1.01 * max(self.t_values_hours))

        # Update x-axis and y-axis limits for trajectory in phase space
        if max(self.m_values)*1.01 > ax[1].get_xlim()[1]:
            ax[1].set_xlim(0.99 * min(self.m_values), max(self.m_values) * 1.01)
        if max(self.p_values) > ax[1].get_ylim()[1]:
            ax[1].set_ylim(0.99 * min(self.p_values), max(self.p_values) * 1.01)


# Time loop
def time_loop():
    clock=Clock(alpha, beta, K, n, tau, mu, v, 0.1, 3.86) # Create an instance of the Clock class
    time_step_visualizer = starting_time_step_visualizer

    max_times_shown=10000

    fig, ax, graph_m, graph_p, graph_phase_space, pos_phase_space=clock.init_plot()


    for t in range(time_steps):
        clock.simulate(t,dt)
        if t!=0 and t % time_step_visualizer == 0:
            # print(t)
            clock.update_plot(fig, ax, graph_m, graph_p, graph_phase_space, pos_phase_space)
            clock.update_plot_limits(ax)
            # if t*dt >= max_times_shown:
                # time_step_visualizer=int(time_step_visualizer*3/np.sqrt(2))
                # max_times_shown*=3
                # ax[0].set_xlim(0,min(max_times_shown/3600,totaltime/3600))
        if t == time_steps//2:
            clock.set_tau(1.5*tau)
            print("Increased tau twofold")


    # Report end of simulation
    print(f"Number of time steps taken: {time_steps}")
    total_time_hours =totaltime # (time_steps * dt) / 3600
    print(f"Total simulation time: {total_time_hours:.2f} hours")
    # Save the final plot, close the interactive mode
    fig.savefig("clock_simulation.png", dpi=300)
    plt.ioff()


if __name__ == "__main__":
    # Call the time loop
    time_loop()
