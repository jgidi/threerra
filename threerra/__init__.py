#!/usr/bin/env python3

import numpy as np
import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.pulse.library import Waveform
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, transpile, schedule as build_schedule
from scipy.optimize import curve_fit

from threerra.discriminators import LDA_discriminator
from threerra.tools import closest_multiple
from threerra.units import ns, MHz, GHz


class QuantumCircuit3:
    """Create a new circuit for a three-level system."""

    def __init__(self, backend):
        # Select quantum and clasical memory slots
        self.qubit = 0
        self.mem_slot = 0

        # Backend
        self.backend = backend
        self.backend_props    = self.backend.properties()
        self.backend_config   = self.backend.configuration()
        self.backend_defaults = self.backend.defaults()

        # Drive pulse parameters
        self.drive_sigma_s = 40 * ns           # Width (duration) of gaussian pulses in microseconds # TODO use 80 ns (0.080 us)
        self.drive_samples_s = self.drive_sigma_s*4 # Truncate gaussian duration

        self.dt = self.backend_config.dt # Device sampling period
        self.drive_sigma = closest_multiple(self.drive_sigma_s / self.dt, 16)
        self.drive_samples = closest_multiple(self.drive_samples_s / self.dt, 16)

        # Frequencies and amplitudes for pi-pulses
        # 01
        self.qubit_freq_est_01 = self.backend_defaults.qubit_freq_est[self.qubit]
        self.pi_amp_01 = 0.1556930479027419
        # 12
        self.qubit_freq_est_12 = self.qubit_freq_est_01 + self.backend_props.qubit_property(self.qubit)['anharmonicity'][0]
        self.pi_amp_12 = 0.2797548240848574
        
        # data discriminator
        self.data_disc = np.loadtxt("data_disc.txt")
        
        # Channels
        self.drive_chan = pulse.DriveChannel(self.qubit)
        self.meas_chan = pulse.MeasureChannel(self.qubit)
        self.acq_chan = pulse.AcquireChannel(self.qubit)

        # Circuit schedule accumulator
        self.list_schedule = []

        
    def apply_sideband(self, pulse, freq, name=None):
        """
        Apply a modulation for a signal 'pulse' according to a frequency 'freq'
        """
        if name is None:
            name = 'Sideband'
        t = np.linspace(0, self.dt * self.drive_samples, self.drive_samples)
        sine = np.sin(2*np.pi * (freq - self.qubit_freq_est_01) * t)

        sideband_pulse = Waveform(np.real(pulse.samples) * sine, name)
        return sideband_pulse

    
    def sx_01(self):
        """
        Apply a pi/2 pulse on levels 01
        """
#         pi_half_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
#                                          amp=self.pi_amp_01/2,
#                                          sigma=self.drive_sigma,
#                                          name='sx_01')
#         self.list_schedule.append(pulse.Play(pi_half_pulse_01, self.drive_chan))
        circ = QuantumCircuit(1)
        circ.sx(self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))

        
    def sx_12(self):
        """
        Apply a pi/2 pulse on levels 12
        """
        pi_half_pulse_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                              amp=self.pi_amp_12/2,
                                              sigma=self.drive_sigma,
                                              name='sx_12')
        # make sure this pulse is sidebanded
        pi_half_pulse_12 = self.apply_sideband(pi_half_pulse_12,
                                               self.qubit_freq_est_12,
                                               name='sx_12')
        self.list_schedule.append(pulse.Play(pi_half_pulse_12, self.drive_chan))

        
    def x_01(self):
        """
        Apply a pi pulse on levels 01
        """
#         pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
#                                          amp=self.pi_amp_01,
#                                          sigma=self.drive_sigma,
#                                          name='x_01')
#         self.list_schedule.append(pulse.Play(pi_pulse_01, self.drive_chan))
        circ = QuantumCircuit(1)
        circ.x(self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))

        
    def rx_01(self, angle):
        """
        Apply a rx gate at levels 01
                input: it has to be in randians
        """
#         pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
#                                          amp=self.pi_amp_01*angle/np.pi,
#                                          sigma=self.drive_sigma,
#                                          name='rx_01')
#         self.list_schedule.append(pulse.Play(pi_pulse_01, self.drive_chan))
        circ = QuantumCircuit(1)
        circ.rx(angle, self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))

        
    def y_01(self):
        """
        Apply a y gate on levels 01
        """
#         phase_pi = pulse.ShiftPhase(np.pi, self.drive_chan)
#         self.list_schedule.append(phase_pi)
#         y_01 = pulse_lib.gaussian(duration=self.drive_samples,
#                                          amp=self.pi_amp_01,
#                                          sigma=self.drive_sigma,
#                                          name='y_01')
#         pulse_y_01 = pulse.Play(y_01, self.drive_chan)
#         self.list_schedule.append(pulse_y_01)
        circ = QuantumCircuit(1)
        circ.y(self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))

        
    def ry_01(self, angle):
        """
        Apply a ry gate on levels 01
                input: it has to be in randians
        """
#         phase_pi = pulse.ShiftPhase(np.pi, self.drive_chan)
#         self.list_schedule.append(phase_pi)
#         ry_01 = pulse_lib.gaussian(duration=self.drive_samples,
#                                          amp=self.pi_amp_01*angle/np.pi,
#                                          sigma=self.drive_sigma,
#                                          name='ry_01')
#         pulse_ry_01 = pulse.Play(ry_01, self.drive_chan)
#         self.list_schedule.append(pulse_ry_01)
        circ = QuantumCircuit(1)
        circ.ry(angle, self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))

        
    def x_12(self):
        """
        Apply a pi pulse on levels 12
        """
        pi_pulse_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_12,
                                         sigma=self.drive_sigma,
                                         name='x_12')
        # make sure this pulse is sidebanded
        pi_pulse_12 = self.apply_sideband(pi_pulse_12, self.qubit_freq_est_12,
                                          name="x_12")
        self.list_schedule.append(pulse.Play(pi_pulse_12, self.drive_chan))


    def ry_12(self, angle):
        """
        Apply a ry gate on levels 12
                input: it has to be in randians
        """
        phase_pi = pulse.ShiftPhase(np.pi, self.drive_chan)
        self.list_schedule.append(phase_pi)
        ry_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                   amp=self.pi_amp_12*angle/np.pi,
                                   sigma=self.drive_sigma,
                                   name='ry_12')
        # make sure this pulse is sidebanded
        ry_12 = self.apply_sideband(ry_12,
                                    self.qubit_freq_est_12,
                                    name="ry_12")
        self.list_schedule.append(pulse.Play(ry_12, self.drive_chan))
       
    
    def rx_12(self, angle):
        """
        Apply a ry gate on levels 12
                input: it has to be in randians
        """
        
        rx_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                   amp=self.pi_amp_12*angle/np.pi,
                                   sigma=self.drive_sigma,
                                   name='rx_12')
        # make sure this pulse is sidebanded
        rx_12 = self.apply_sideband(rx_12,
                                    self.qubit_freq_est_12,
                                    name="rx_12")
        self.list_schedule.append(pulse.Play(rx_12, self.drive_chan))

    def rz(self, phase):
        self.list_schedule.append(pulse.ShiftPhase(phase, self.drive_chan))


    def calibrate_freq_01(self, freqs=None):

        f0 = self.qubit_freq_est_01
        if freqs is None:
            freqs = f0 + np.linspace(-20*MHz, 20*MHz, 75)

        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)

        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )

        gaussian_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                            sigma=self.drive_sigma,
                                            amp=0.1)

        # Define schedule
        schedule = pulse.Schedule()
        schedule |= pulse.Play(gaussian_pulse, self.drive_chan)
        schedule |= measure_pulse << schedule.duration

        job = self.backend.run(schedule,
                               meas_level=1,
                               meas_return='avg',
                               shots=1024,
                               schedule_los=[{self.drive_chan: freq} for freq in freqs])

        # Make notice about the on-going job
        print("Calibrating qubit_freq_est_01...")
        job_monitor(job)

        results = job.result(timeout=120)
        results_data = [results.get_memory(i)[self.qubit] for i in range(len(results.results))]

        # Fit Lorentzian response
        fun = lambda x, qf, a, b, c: a * b / ((x-qf)**2 + b**2 ) + c
        ydata = np.real(results_data)
        ydata /= ydata.max()-ydata.min() # Normalize data height
        b = (freqs.max()-freqs.min())/10 # Half-width
        fit_params, *_ = curve_fit(fun, freqs, ydata, [f0, b, b, ydata.mean()])

        # Update frequency estimate
        self.qubit_freq_est_01 = fit_params[0]

        print(f'qubit_freq_est_01 updated from {f0/GHz}GHz to {fit_params[0]/GHz}GHz.')

        
    def calibrate_pi_amp_01(self, amps=None):

        amp0 = self.pi_amp_01
        if amps is None:
            amps = np.linspace(-0.95, 0.95, 51)

        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)

        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )

        schedules = []
        for amp in amps:
            gaussian_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                                sigma=self.drive_sigma,
                                                amp=amp)

            # Define schedule
            schedule = pulse.Schedule()
            schedule |= pulse.Play(gaussian_pulse, self.drive_chan)
            schedule |= measure_pulse << schedule.duration

            # Accumulate
            schedules.append(schedule)

        job = self.backend.run(schedules,
                               meas_level=1,
                               meas_return='avg',
                               shots=1024,
                               schedule_los=[{self.drive_chan: self.qubit_freq_est_01}] * len(amps))

        # Make notice about the on-going job
        print("Calibrating pi_amp_01...")
        job_monitor(job)

        results = job.result(timeout=120)
        results_data = [results.get_memory(i)[self.qubit] for i in range(len(results.results))]

        # Fit response
        fun = lambda x, period, a, b, c: a * np.cos( 2*np.pi*x/period - b ) + c
        ydata = np.real(results_data)
        # Ansatz
        period = amps[-1]/2
        a = ydata.max()-ydata.min()
        b = 0.0
        c = ydata.mean()
        fit_params, *_ = curve_fit(fun, amps, ydata, [period, a, b, c])

        # Update amplitude estimate
        self.pi_amp_01 = abs(fit_params[0] / 2)

        print(f'pi_amp_01 updated from {amp0} to {self.pi_amp_01}.')

    def calibrate_freq_12(self, freqs=None):

        f0 = self.qubit_freq_est_12
        if freqs is None:
            freqs = f0 + np.linspace(-20*MHz, 20*MHz, 75)

        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)

        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )

        gaussian_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                            sigma=self.drive_sigma,
                                            amp=0.3)

        pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01,
                                         sigma=self.drive_sigma,
                                         name='pi_pulse_01')

        schedules = []          # Accumulator
        for freq in freqs:
            sidebanded_pulse = self.apply_sideband(gaussian_pulse, freq)

            schedule = pulse.Schedule()
            schedule |= pulse.Play(pi_pulse_01, self.drive_chan)
            schedule |= pulse.Play(sidebanded_pulse, self.drive_chan) << schedule.duration
            schedule |= measure_pulse << schedule.duration

            schedules.append(schedule)

        job = self.backend.run(schedules,
                               meas_level=1,
                               meas_return='avg',
                               shots=1024,
                               schedule_los=[{self.drive_chan: self.qubit_freq_est_01}] * len(freqs))

        # Make notice about the on-going job
        print("Calibrating qubit_freq_est_12...")
        job_monitor(job)

        results = job.result(timeout=120)
        results_data = [results.get_memory(i)[self.qubit] for i in range(len(results.results))]

        # Fit Lorentzian response
        fun = lambda x, qf, a, b, c: a * b / ((x-qf)**2 + b**2 ) + c
        ydata = np.real(results_data)
        ydata /= ydata.max()-ydata.min() # Normalize data height
        b = (freqs.max()-freqs.min())/10 # Half-width
        fit_params, *_ = curve_fit(fun, freqs, ydata, [f0, b, b, ydata.mean()])

        # Update frequency estimate
        self.qubit_freq_est_12 = fit_params[0]

        print(f'qubit_freq_est_12 updated from {f0/GHz}GHz to {fit_params[0]/GHz}GHz.')

        
    def calibrate_pi_amp_12(self, amps=None):

        amp0 = self.pi_amp_12
        if amps is None:
            amps = np.linspace(0, 0.75, 75)

        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)

        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )

        pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01,
                                         sigma=self.drive_sigma,
                                         name='pi_pulse_01')

        schedules = []
        for amp in amps:
            base_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                            sigma=self.drive_sigma,
                                            amp=amp)

            sidebanded_pulse = self.apply_sideband(base_pulse, self.qubit_freq_est_12)

            # Define schedule
            schedule = pulse.Schedule()
            schedule |= pulse.Play(pi_pulse_01, self.drive_chan)
            schedule |= pulse.Play(sidebanded_pulse, self.drive_chan) << schedule.duration
            schedule |= measure_pulse << schedule.duration

            # Accumulate
            schedules.append(schedule)

        job = self.backend.run(schedules,
                               meas_level=1,
                               meas_return='avg',
                               shots=1024,
                               schedule_los=[{self.drive_chan: self.qubit_freq_est_01}] * len(amps))

        # Make notice about the on-going job
        print("Calibrating pi_amp_12...")
        job_monitor(job)

        results = job.result(timeout=120)
        results_data = [results.get_memory(i)[self.qubit] for i in range(len(results.results))]

        # Fit response
        fun = lambda x, period, a, b, c: a * np.cos( 2*np.pi*x/period - b ) + c
        ydata = np.real(results_data)
        # Ansatz
        period = amps[-1]/2
        a = ydata.max()-ydata.min()
        b = 0.0
        c = ydata.mean()
        fit_params, *_ = curve_fit(fun, amps, ydata, [period, a, b, c])

        # Update amplitude estimate
        self.pi_amp_12 = abs(fit_params[0] / 2)

        print(f'pi_amp_12 updated from {amp0} to {self.pi_amp_12}.')

        
    def draw(self, backend=None, *args, **kwargs):
        """
        Join all pulses and draw
        """

        # Default backend
        if backend is None:
            backend = self.backend

        # Join pulses
        schedule = pulse.Schedule()
        for s in self.list_schedule:
            schedule |= s << schedule.duration

        return schedule.draw(backend=backend,
                             *args,
                             **kwargs)

    
    def run(self,
            shots=1024,
            meas_level=1,
            meas_return='single',
            disc012=False,
            *args, **kwargs):
        """
        Run circuit on backend
        """

        schedule = pulse.Schedule()
        for s in self.list_schedule:
            schedule |= s << schedule.duration
        counter = 0
        while counter < 3:
            try:
                job = self.backend.run(schedule,
                                       shots=shots,
                                       meas_level=meas_level,
                                       meas_return=meas_return,
                                       *args,
                                       **kwargs)

                # Make notice about the on-going job
                job_monitor(job)
                break
            except:
                counter = counter + 1
        results = job.result(timeout=120)
        if disc012:
            lul = []
            for i in range(len(results.results)):
                lul.append(results.get_memory(i)[:, 0])
            lul_reshaped = LDA_discriminator.reshape_complex_vec(lul[0])
            counts012 = LDA_discriminator.LDA_dis(self.data_disc, lul_reshaped, acc=True)
            return counts012
        else:
            return results

    
    def measure(self):
        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)
        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )
        self.list_schedule.append(measure_pulse)

        
    def discriminator012(self, shots=1024):
        
        pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01,
                                         sigma=self.drive_sigma,
                                         name='x_01')
    
        pi_pulse_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_12,
                                         sigma=self.drive_sigma,
                                         name='x_12')
        # make sure this pulse is sidebanded
        pi_pulse_12 = self.apply_sideband(pi_pulse_12, self.qubit_freq_est_12,
                                          name="x_12")
        
        meas_idx = [self.qubit in group
                        for group in self.backend_config.meas_map].index(True)

        measure_pulse = self.backend_defaults.instruction_schedule_map.get(
            'measure',
            qubits = self.backend_config.meas_map[meas_idx],
            )

        # Create the three schedules

        # Ground state schedule
        zero_schedule = pulse.Schedule(name="zero schedule")
        zero_schedule |= measure_pulse

        # Excited state schedule
        one_schedule = pulse.Schedule(name="one schedule")
        one_schedule |= pulse.Play(pi_pulse_01, self.drive_chan)
        one_schedule |= measure_pulse << one_schedule.duration

        # Excited state schedule
        two_schedule = pulse.Schedule(name="two schedule")
        two_schedule |= pulse.Play(pi_pulse_01, self.drive_chan)
        two_schedule |= pulse.Play(pi_pulse_12, self.drive_chan) << two_schedule.duration
        two_schedule |= measure_pulse << two_schedule.duration
        
        IQ_012_job = self.backend.run([zero_schedule, one_schedule, two_schedule],
                           meas_level=1,
                           meas_return='single',
                           shots=shots,
                           schedule_los=[{self.drive_chan: self.qubit_freq_est_01}] * 3)

        job_monitor(IQ_012_job)
        
        # Get job data (single); split for zero, one and two

        job_results = IQ_012_job.result(timeout=120)
        
        IQ_012_data = []
        for i in range(len(job_results.results)):
            IQ_012_data.append(job_results.get_memory(i)[:, self.qubit])  
        
        zero_data = IQ_012_data[0]
        one_data = IQ_012_data[1]
        two_data = IQ_012_data[2]
        
        # Create IQ vector (split real, imag parts)
        zero_data_reshaped = LDA_discriminator.reshape_complex_vec(zero_data)
        one_data_reshaped = LDA_discriminator.reshape_complex_vec(one_data)
        two_data_reshaped = LDA_discriminator.reshape_complex_vec(two_data)

        IQ_012_data_reshaped = np.concatenate((zero_data_reshaped, one_data_reshaped, two_data_reshaped))
        
        self.data_disc =  IQ_012_data_reshaped
