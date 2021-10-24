#!/usr/bin/env python
# coding: utf-8

import numpy as np
import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.pulse.library import Waveform
from qiskit.tools.monitor import job_monitor

from scipy.optimize import curve_fit


# Out-of-class helpers
def closest_multiple(N, base : int = 16):
    """
    Return the closest multiple of 'base' to 'N'
    """
    return base * round( N / base )

class QuantumCircuit3:
    """Create a new circuit for a three-level system."""

    def __init__(self, backend):
        # Select quantum and clasical memory slots
        self.qubit = 0
        self.mem_slot = 0

        # Conversion from SI units
        self.ns = 1.0e-9 # Nanoseconds
        self.us = 1.0e-6 # Microseconds
        self.MHz = 1.0e6 # Megahertz
        self.GHz = 1.0e9 # Gigahertz

        # Backend
        self.backend = backend
        self.backend_props    = self.backend.properties()
        self.backend_config   = self.backend.configuration()
        self.backend_defaults = self.backend.defaults()

        # Drive pulse parameters
        self.drive_sigma_s = 75 * self.ns           # Width (duration) of gaussian pulses in microseconds # TODO use 80 ns (0.080 us)
        self.drive_samples_s = self.drive_sigma_s*8 # Truncate gaussian duration

        self.dt = self.backend_config.dt # Device sampling period
        self.drive_sigma = closest_multiple(self.drive_sigma_s / self.dt, 16)
        self.drive_samples = closest_multiple(self.drive_samples_s / self.dt, 16)

        # Frequencies and amplitudes for pi-pulses
        # 01
        self.qubit_freq_est_01 = self.backend_defaults.qubit_freq_est[self.qubit]
        self.pi_amp_01 = 0.297457728644259 # TODO Estimate accessible from backend. 0.9032... for sigma=80ns
        # 12
        self.qubit_freq_est_12 = self.qubit_freq_est_01 + self.backend_props.qubit_property(self.qubit)['anharmonicity'][0]
        self.pi_amp_12 = 0.37256049920143336

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
        pi_half_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01/2,
                                         sigma=self.drive_sigma,
                                         name='sx_01')
        self.list_schedule.append(pulse.Play(pi_half_pulse_01, self.drive_chan))

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
        pi_pulse_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01,
                                         sigma=self.drive_sigma,
                                         name='x_01')
        self.list_schedule.append(pulse.Play(pi_pulse_01, self.drive_chan))


    def y_01(self):
        """
        Apply a y gate on levels 01
        """
        phase_pi = pulse.ShiftPhase(np.pi, self.drive_chan)
        self.list_schedule.append(phase_pi)
        y_01 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_01,
                                         sigma=self.drive_sigma,
                                         name='y_01')
        pulse_y_01 = pulse.Play(y_01, self.drive_chan)
        self.list_schedule.append(pulse_y_01)


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


    def y_12(self):
        """
        Apply a pi pulse on levels 12
        """
        phase_pi = pulse.ShiftPhase(np.pi, self.drive_chan)
        self.list_schedule.append(phase_pi)
        pi_pulse_12 = pulse_lib.gaussian(duration=self.drive_samples,
                                         amp=self.pi_amp_12,
                                         sigma=self.drive_sigma,
                                         name='y_12')
        # make sure this pulse is sidebanded
        pi_pulse_12 = self.apply_sideband(pi_pulse_12, self.qubit_freq_est_12,
                                          name="y_12")
        self.list_schedule.append(pulse.Play(pi_pulse_12, self.drive_chan))


    def rz(self, phase):
        self.list_schedule.append(pulse.ShiftPhase(phase, self.drive_chan))


    def calibrate_freq_01(self, freqs=None):

        f0 = self.qubit_freq_est_01
        if freqs is None:
            freqs = f0 + np.linspace(-20*self.MHz, 20*self.MHz, 75)

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

        print(f'qubit_freq_est_01 updated from {f0/self.GHz}GHz to {fit_params[0]/self.GHz}GHz.')

    def calibrate_pi_amp_01(self, amps=None):

        amp0 = self.pi_amp_01
        if amps is None:
            amps = np.linspace(0, 0.75, 75)

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

        # Update frequency estimate
        self.pi_amp_01 = abs( (np.pi + fit_params[2]) * fit_params[0]/(2*np.pi) )

        print(f'pi_amp_01 updated from {amp0} to {self.pi_amp_01}.')

    def calibrate_freq_12(self, freqs=None):

        f0 = self.qubit_freq_est_12
        if freqs is None:
            freqs = f0 + np.linspace(-20*self.MHz, 20*self.MHz, 75)

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

        print(f'qubit_freq_est_12 updated from {f0/self.GHz}GHz to {fit_params[0]/self.GHz}GHz.')

    def compile(self):
        """
        Join all pulses and draw
        """
        schedule = pulse.Schedule(name='')
        for s in self.list_schedule:
            schedule |= s << schedule.duration
        return schedule.draw(backend=self.backend)
