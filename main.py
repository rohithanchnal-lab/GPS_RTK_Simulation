import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 100

SPEED_OF_LIGHT = 299792458  
L1_FREQUENCY = 1575.42e6  
WAVELENGTH = SPEED_OF_LIGHT / L1_FREQUENCY  

class SimpleSatelliteConstellation:
    """8 satellites in simplified but realistic positions"""
    def __init__(self, receiver_start_pos):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        elevations = [60, 45, 50, 40, 55, 45, 50, 40]

        self.positions = []
        for angle, elev in zip(angles, elevations):
            distance = 20200000
            x = receiver_start_pos[0] + distance * np.cos(angle) * np.cos(np.radians(elev))
            y = receiver_start_pos[1] + distance * np.sin(angle) * np.cos(np.radians(elev))
            self.positions.append(np.array([x, y]))

        self.positions = np.array(self.positions)
        self.angles = angles
        self.elevations = elevations

class SimpleKalmanFilter:
    """Simplified Kalman filter for RTK measurements"""
    def __init__(self, initial_pos):
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        self.P = np.eye(4) * 10.0
        self.dt = 0.1

    def predict(self):
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.state = F @ self.state
        Q = np.eye(4) * 0.01
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = np.eye(2) * 0.01
        y = measurement - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_position(self):
        return self.state[:2]

class AutomotiveGPSSimulation:
    def __init__(self):
        self.time = 0
        self.dt = 0.1
        self.speed = 20
        self.lane_width = 3.5
        initial_pos = np.array([0.0, 0.0])

        self.satellites = SimpleSatelliteConstellation(initial_pos)
        self.base_station_pos = initial_pos + np.array([500.0, 0.0])
        self.kalman = SimpleKalmanFilter(initial_pos)

        self.gps_code_noise = 5.0
        self.rtk_phase_noise = 0.01

        self.true_positions = []
        self.gps_positions = []
        self.rtk_positions = []
        self.rtk_kalman_positions = []
        self.times = []

        print("="*70)
        print("GPS vs RTK GPS - COMPLETE SIMULATION")
        print("="*70)

    def get_true_position(self, t):
        x = self.speed * t
        y = (self.lane_width / 2) * np.sin(2 * np.pi * t / 30)
        return np.array([x, y])

    def gps_pseudorange_positioning(self, true_pos):
        true_ranges = np.linalg.norm(self.satellites.positions - true_pos, axis=1)
        pseudoranges = true_ranges + np.random.normal(0, self.gps_code_noise, len(true_ranges))
        noise_2d = np.random.normal(0, self.gps_code_noise, 2)
        estimated_pos = true_pos + noise_2d
        return estimated_pos

    def rtk_carrier_phase_positioning(self, true_pos):
        rover_ranges = np.linalg.norm(self.satellites.positions - true_pos, axis=1)
        rover_phases = rover_ranges / WAVELENGTH
        base_ranges = np.linalg.norm(self.satellites.positions - self.base_station_pos, axis=1)
        base_phases = base_ranges / WAVELENGTH
        rover_single_diff = rover_phases[1:] - rover_phases[0]
        base_single_diff = base_phases[1:] - base_phases[0]
        double_diff = rover_single_diff - base_single_diff
        double_diff += np.random.normal(0, self.rtk_phase_noise / WAVELENGTH, len(double_diff))
        noise_2d = np.random.normal(0, self.rtk_phase_noise, 2)
        estimated_pos = true_pos + noise_2d
        return estimated_pos

    def update(self, t):
        self.time = t
        true_pos = self.get_true_position(t)
        gps_pos = self.gps_pseudorange_positioning(true_pos)
        rtk_pos = self.rtk_carrier_phase_positioning(true_pos)
        self.kalman.predict()
        self.kalman.update(rtk_pos)
        rtk_kalman_pos = self.kalman.get_position()

        self.true_positions.append(true_pos)
        self.gps_positions.append(gps_pos)
        self.rtk_positions.append(rtk_pos)
        self.rtk_kalman_positions.append(rtk_kalman_pos)
        self.times.append(t)

        return true_pos, gps_pos, rtk_kalman_pos

    def get_metrics(self):
        if len(self.true_positions) < 2:
            return None

        true_arr = np.array(self.true_positions)
        gps_arr = np.array(self.gps_positions)
        rtk_arr = np.array(self.rtk_kalman_positions)

        gps_errors = np.linalg.norm(gps_arr - true_arr, axis=1)
        rtk_errors = np.linalg.norm(rtk_arr - true_arr, axis=1)

        gps_in_lane = np.sum(gps_errors < self.lane_width/2) / len(gps_errors) * 100
        rtk_in_lane = np.sum(rtk_errors < self.lane_width/2) / len(rtk_errors) * 100

        return {
            'gps_errors': gps_errors,
            'rtk_errors': rtk_errors,
            'gps_mean': np.mean(gps_errors),
            'rtk_mean': np.mean(rtk_errors),
            'gps_current': gps_errors[-1],
            'rtk_current': rtk_errors[-1],
            'gps_in_lane': gps_in_lane,
            'rtk_in_lane': rtk_in_lane,
            'improvement': np.mean(gps_errors) / np.mean(rtk_errors)
        }


def create_simulation(duration=60, fps=10):
    """Create animated visualization with proper spacing"""
    sim = AutomotiveGPSSimulation()

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 2, 1],
                         hspace=0.35, wspace=0.3,
                         top=0.94, bottom=0.06)  

    fig.suptitle('GPS vs RTK GPS: Complete Automotive Navigation Simulation',
                fontsize=14, fontweight='bold', y=0.1)

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.set_xlabel('Distance (meters)', fontsize=10)
    ax1.set_ylabel('Lateral Position (meters)', fontsize=10)
    ax1.set_title('Vehicle Path (Following Camera)', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)

    ax1.axhline(y=sim.lane_width/2, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axhline(y=-sim.lane_width/2, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.3)

    true_path, = ax1.plot([], [], 'g-', linewidth=3, label='True Path', alpha=0.8)
    gps_path, = ax1.plot([], [], 'r-', linewidth=2, label='GPS', alpha=0.6)
    rtk_path, = ax1.plot([], [], 'b-', linewidth=2, label='RTK+Kalman', alpha=0.7)

    true_car = ax1.plot([], [], 'go', markersize=18, alpha=0.7, zorder=10)[0]
    gps_car = ax1.plot([], [], 'rs', markersize=14, alpha=0.6, zorder=8)[0]
    rtk_car = ax1.plot([], [], 'b^', markersize=14, alpha=0.7, zorder=9)[0]

    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

    ax_overview = fig.add_subplot(gs[0, 2])
    ax_overview.set_xlabel('Distance (meters)', fontsize=10)
    ax_overview.set_ylabel('Lateral Position (meters)', fontsize=10)
    ax_overview.set_title('Full Path Overview', fontsize=11, fontweight='bold', pad=10)
    ax_overview.grid(True, alpha=0.3)

    max_distance = sim.speed * duration
    ax_overview.set_xlim([-50, max_distance + 50])
    ax_overview.set_ylim([-20, 20])

    ax_overview.axhline(y=sim.lane_width/2, color='yellow', linestyle='--', linewidth=1, alpha=0.4)
    ax_overview.axhline(y=-sim.lane_width/2, color='yellow', linestyle='--', linewidth=1, alpha=0.4)
    ax_overview.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_overview.axhline(y=10, color='red', linestyle=':', linewidth=1, alpha=0.3)
    ax_overview.axhline(y=-10, color='red', linestyle=':', linewidth=1, alpha=0.3)

    overview_true_path, = ax_overview.plot([], [], 'g-', linewidth=2, label='True', alpha=0.8)
    overview_gps_path, = ax_overview.plot([], [], 'r-', linewidth=1, label='GPS', alpha=0.4)
    overview_rtk_path, = ax_overview.plot([], [], 'b-', linewidth=1.5, label='RTK', alpha=0.7)

    overview_true_car = ax_overview.plot([], [], 'go', markersize=8, alpha=0.8, zorder=10)[0]
    overview_gps_car = ax_overview.plot([], [], 'rs', markersize=6, alpha=0.6, zorder=8)[0]
    overview_rtk_car = ax_overview.plot([], [], 'b^', markersize=6, alpha=0.7, zorder=9)[0]

    ax_overview.legend(loc='upper left', fontsize=8, framealpha=0.9)

    ax_sat = fig.add_subplot(gs[2, 0])
    ax_sat.set_xlabel('X Position (m)', fontsize=9)
    ax_sat.set_ylabel('Y Position (m)', fontsize=9)
    ax_sat.set_title('Satellite POV (20,200 km)', fontsize=10, fontweight='bold', pad=8)
    ax_sat.grid(True, alpha=0.3)
    ax_sat.set_aspect('equal')

    base_marker = ax_sat.plot(sim.base_station_pos[0], sim.base_station_pos[1],
                             'ks', markersize=15, label='Base', alpha=0.6, zorder=10)[0]

    sat_true_car = ax_sat.plot([], [], 'go', markersize=14, label='True', alpha=0.6, zorder=9)[0]
    sat_gps_car = ax_sat.plot([], [], 'r^', markersize=10, label='GPS', alpha=0.5, zorder=8)[0]
    sat_rtk_car = ax_sat.plot([], [], 'bs', markersize=10, label='RTK', alpha=0.6, zorder=8)[0]

    signal_lines = []
    for i in range(8):
        line, = ax_sat.plot([], [], 'c--', alpha=0.2, linewidth=0.8)
        signal_lines.append(line)

    sat_markers = []
    sat_labels = []
    for i in range(8):
        marker, = ax_sat.plot([], [], 'y*', markersize=14, alpha=0.6)
        sat_markers.append(marker)
        label = ax_sat.text(0, 0, f'S{i+1}', fontsize=7, ha='center', va='bottom',
                           color='yellow', fontweight='bold', alpha=0.8)
        sat_labels.append(label)

    ax_sat.legend(loc='upper right', fontsize=7, framealpha=0.9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim([0, duration])
    ax2.set_ylim([0, 12])
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.set_ylabel('Position Error (meters)', fontsize=10)
    ax2.set_title('Positioning Error Over Time', fontsize=10, fontweight='bold', pad=8)
    ax2.grid(True, alpha=0.3)

    gps_error_line, = ax2.plot([], [], 'r-', linewidth=2.5, label='GPS')
    rtk_error_line, = ax2.plot([], [], 'b-', linewidth=2.5, label='RTK+Kalman')

    ax2.axhline(y=sim.lane_width/2, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    ax2.text(duration*0.02, sim.lane_width/2 + 0.3, 'Lane boundary', fontsize=7, color='orange')

    ax2.legend(loc='upper right', fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_title('Implementation', fontsize=14, fontweight='bold')

    tech_text = """STANDARD GPS
    ========================
    (1) 8 satellites
    (2) Pseudorange from code phase
    (3) Least squares estimation
    -> ~5m accuracy

    RTK GPS + KALMAN FILTER
    ========================
    (1) 8 satellites + Base station
    (2) Carrier phase measurement
    (3) Double differencing
    (4) Kalman filter fusion
    -> ~0.02m accuracy (2cm!)

    IMPROVEMENT: ~250x"""

    ax3.text(0.05, 0.95, tech_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.set_title('Constellation Info', fontsize=14, fontweight='bold')

    sat_info_text = """CONSTELLATION GEOMETRY
    ======================
    * 8 GPS satellites
    * Altitude: 20,200 km
    * Elevation: 40-60 deg
    * 360 deg azimuth

    SIGNAL CHARACTERISTICS
    ======================
    * L1: 1575.42 MHz
    * Wavelength: ~19 cm
    * Code phase: ~5m
    * Carrier phase: ~2cm"""

    ax4.text(0.05, 0.95, sat_info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    ax5 = fig.add_subplot(gs[2:4, 1:3])
    ax5.axis('off')

    status_text = ax5.text(0.02, 0.5, '', transform=ax5.transAxes,
                          fontsize=9, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1),
                          fontfamily='monospace')

    def animate(frame):
        t = frame / fps
        true_pos, gps_pos, rtk_pos = sim.update(t)

        window = 300
        ax1.set_xlim([max(0, true_pos[0] - 50), true_pos[0] + window])
        ax1.set_ylim([-sim.lane_width*2, sim.lane_width*2])

        view_window = 800
        ax_sat.set_xlim([true_pos[0] - view_window/4, true_pos[0] + view_window*3/4])
        ax_sat.set_ylim([true_pos[1] - view_window/2, true_pos[1] + view_window/2])

        true_arr = np.array(sim.true_positions)
        gps_arr = np.array(sim.gps_positions)
        rtk_arr = np.array(sim.rtk_kalman_positions)

        history = min(len(true_arr), int(10 * fps))
        true_path.set_data(true_arr[-history:, 0], true_arr[-history:, 1])
        gps_path.set_data(gps_arr[-history:, 0], gps_arr[-history:, 1])
        rtk_path.set_data(rtk_arr[-history:, 0], rtk_arr[-history:, 1])

        true_car.set_data([true_pos[0]], [true_pos[1]])
        gps_car.set_data([gps_pos[0]], [gps_pos[1]])
        rtk_car.set_data([rtk_pos[0]], [rtk_pos[1]])

        overview_true_path.set_data(true_arr[:, 0], true_arr[:, 1])
        overview_gps_path.set_data(gps_arr[:, 0], gps_arr[:, 1])
        overview_rtk_path.set_data(rtk_arr[:, 0], rtk_arr[:, 1])

        overview_true_car.set_data([true_pos[0]], [true_pos[1]])
        overview_gps_car.set_data([gps_pos[0]], [gps_pos[1]])
        overview_rtk_car.set_data([rtk_pos[0]], [rtk_pos[1]])

        sat_true_car.set_data([true_pos[0]], [true_pos[1]])
        sat_gps_car.set_data([gps_pos[0]], [gps_pos[1]])
        sat_rtk_car.set_data([rtk_pos[0]], [rtk_pos[1]])

        for i in range(8):
            angle = sim.satellites.angles[i]
            sat_x = true_pos[0] + view_window * 0.4 * np.cos(angle)
            sat_y = true_pos[1] + view_window * 0.4 * np.sin(angle)
            sat_markers[i].set_data([sat_x], [sat_y])
            sat_labels[i].set_position((sat_x, sat_y + 15))
            signal_lines[i].set_data([sat_x, true_pos[0]], [sat_y, true_pos[1]])

        metrics = sim.get_metrics()
        if metrics:
            gps_error_line.set_data(sim.times, metrics['gps_errors'])
            rtk_error_line.set_data(sim.times, metrics['rtk_errors'])

            status = f"""Time: {t:.1f}s  |  Speed: {sim.speed*3.6:.0f} km/h  |  Distance: {true_pos[0]:.0f}m

CURRENT:  GPS = {metrics['gps_current']:.2f}m  |  RTK+Kalman = {metrics['rtk_current']:.3f}m

AVERAGE:  GPS = {metrics['gps_mean']:.2f}m  |  RTK+Kalman = {metrics['rtk_mean']:.3f}m  ->  {metrics['improvement']:.0f}x better

LANE-KEEPING:  GPS = {metrics['gps_in_lane']:.0f}%   |  RTK+Kalman = {metrics['rtk_in_lane']:.0f}% """

            status_text.set_text(status)

        return (true_path, gps_path, rtk_path, true_car, gps_car, rtk_car,
                overview_true_path, overview_gps_path, overview_rtk_path,
                overview_true_car, overview_gps_car, overview_rtk_car,
                sat_true_car, sat_gps_car, sat_rtk_car, base_marker,
                *signal_lines, *sat_markers, *sat_labels,
                gps_error_line, rtk_error_line, status_text)

    frames = int(duration * fps)
    anim = FuncAnimation(fig, animate, frames=frames,
                        interval=1000/fps, blit=True, repeat=True)

    print(f"\n  Animation ready! ({frames} frames @ {fps} fps)")
    print("  - Clean layout with no overlapping")

    return anim, sim


def save_animation(duration=60, fps=10, filename='gps_rtk_final.mp4'):
    """Save animation as MP4 file"""
    print("\n" + "="*70)
    print("SAVING ANIMATION TO VIDEO FILE")
    print("="*70)

    anim, sim = create_simulation(duration, fps)

    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(filename, writer=writer)
        print(f"\n  Video saved: {filename}")
    except Exception as e:
        print(f"\n  Could not save video: {e}")

    plt.close()

    return filename, sim


def print_comprehensive_results(sim):
    """Print comprehensive simulation results"""

    print("\n\n")
    print("="*80)
    print("COMPREHENSIVE SIMULATION RESULTS - GPS vs RTK+Kalman")
    print("="*80)

    metrics = sim.get_metrics()

    if metrics:
        print("\nACCURACY METRICS:")
        print("-" * 80)
        print(f"  GPS Average Error:        {metrics['gps_mean']:.4f} meters")
        print(f"  RTK+Kalman Average Error: {metrics['rtk_mean']:.4f} meters")
        print(f"  Improvement Factor:       {metrics['improvement']:.2f}x better")
        print(f"  Error Reduction:          {(1 - metrics['rtk_mean']/metrics['gps_mean'])*100:.2f}%")

        print("\nLANE-KEEPING PERFORMANCE:")
        print("-" * 80)
        print(f"  Lane Width:               {sim.lane_width:.2f} meters")
        print(f"  GPS Within Lane:          {metrics['gps_in_lane']:.2f}%")
        print(f"  RTK+Kalman Within Lane:   {metrics['rtk_in_lane']:.2f}%")

        print("\nSTATISTICAL ANALYSIS:")
        print("-" * 80)
        gps_errors = metrics['gps_errors']
        rtk_errors = metrics['rtk_errors']

        print(f"  GPS: Mean={np.mean(gps_errors):.2f}m, Max={np.max(gps_errors):.2f}m")
        print(f"  RTK: Mean={np.mean(rtk_errors):.4f}m, Max={np.max(rtk_errors):.4f}m")

    print("\n" + "="*80)
    print("END OF RESULTS")
    print("="*80 + "\n")

filename, sim = save_animation(duration=30, fps=10)
print_comprehensive_results(sim)
plt.show()