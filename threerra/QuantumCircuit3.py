#!/usr/bin/env python3

import numpy as np
import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.pulse.library import Waveform
from qiskit import QuantumCircuit, transpile, schedule as build_schedule

from threerra import pulses
from threerra.units import ns
from threerra.tools import closest_multiple

from threerra import calibrations


class QuantumCircuit3:
    """Create a new circuit for a three-level system."""

    def __init__(self, backend):
        """
        description

            Args:
                backend: backend provider
        """
        # Select quantum and clasical memory slots
        self.qubit = 0
        self.mem_slot = 0

        # Backend
        self.backend = backend
        self.backend_props    = self.backend.properties()
        self.backend_config   = self.backend.configuration()
        self.backend_defaults = self.backend.defaults()

        # Drive pulse parameters
        self.drive_sigma_s = 75 * ns           # Width (duration) of gaussian pulses in microseconds
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

        # Channels
        self.drive_chan = pulse.DriveChannel(self.qubit)
        self.meas_chan = pulse.MeasureChannel(self.qubit)
        self.acq_chan = pulse.AcquireChannel(self.qubit)

        # Circuit schedule accumulator
        self.list_schedule = []


    def apply_sideband(self, pulse, freq, name=None):
        """
        Apply a modulation for a signal 'pulse' according to a frequency 'freq'

            Args:
                pulse: The pulse of interest
                freq: Local Oscillator frecuency for which we want to apply the modulation
                name: Name of the sideband

            Returns:
                Pulse with a sideband applied (oscillates at difference between freq and self.qubit_freq_est_01)

        """
        if name is None:
            name = 'Sideband'
        t = np.linspace(0, self.dt * self.drive_samples, self.drive_samples)
        sine = np.sin(2*np.pi * (freq - self.qubit_freq_est_01) * t)

        sideband_pulse = Waveform(np.real(pulse.samples) * sine, name)
        return sideband_pulse


    def sx_01(self):
        """
        Apply a SX-gate on the 01 subspace
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


    def x_01(self):
        """
        Apply a X-gate on the 01 subspace
        """
        circ = QuantumCircuit(1)
        circ.x(self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))


    def rx_01(self, angle):
        """
        Apply a RX-gate on the 01 subspace
        The angle parameter must be in radians i.e (pi/s), with s in [0, 2pi].

                Args:
                    angle: angle desired for the rotation.
        """
        circ = QuantumCircuit(1)
        circ.rx(angle, self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))


    def y_01(self):
        """
        Apply a Y-gate on the 01 subspace
        """
        circ = QuantumCircuit(1)
        circ.y(self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))


    def ry_01(self, angle):
        """
        Apply a Y-gate on the 01 subspace
        The angle parameter must be in radians i.e (pi/s), with s in [0, 2pi].

                Args:
                    angle: angle desired for the rotation.
        """
        circ = QuantumCircuit(1)
        circ.ry(angle, self.qubit)
        transpiled_circ = transpile(circ, self.backend)
        self.list_schedule.append(build_schedule(transpiled_circ, self.backend))


    def x_12(self):
        """
        Apply a X-gate on the 12 subspace
        """
        self.list_schedule.append(pulse.Play(pulses.pi_pulse_12(self),
                                             self.drive_chan))

    def sx_12(self):
        """
        Apply a SX-gate on the 12 subspace
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
        # self.list_schedule.append(pulses.gen_pulse(self,
        #                                     angle=np.pi/2,
        #                                     sideband_freq=self.qubit_freq_est_12,
        #                                     name="sx_12"))

    def ry_12(self, angle):
        """
        Apply a RY-gate on the 12 subspace
        The angle parameter must be in radians i.e (pi/s), with s in [0, 2pi].

                Args:
                    angle: angle desired for the rotation.
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
        Apply a RX-gate on the 12 subspace
        The angle parameter must be in radians i.e (pi/s), with s in [0, 2pi].

                Args:
                    angle: angle desired for the rotation.
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
        """
        Apply a Z-gate
        """
        self.list_schedule.append(pulse.ShiftPhase(phase, self.drive_chan))


    # Calibrations
    def calibrate_freq_01(self, freqs=None):
        """
        description

            Args:
                freqs:

        """

        calibrations.calibrate_freq_01(self, freqs)

    def calibrate_pi_amp_01(self, freqs=None):
        """
        description

            Args:
                freqs:
        """
        calibrations.calibrate_pi_amp_01(self, freqs)

    def calibrate_freq_12(self, freqs=None):
        """
        description

            Args:
                freqs:
        """
        calibrations.calibrate_freq_12(self, freqs)

    def calibrate_pi_amp_12(self, freqs=None):
        """
        description

            Args:
                freqs:
        """
        calibrations.calibrate_pi_amp_12(self, freqs)

    def draw(self, backend=None, *args, **kwargs):
        """
        Join all pulses and draw

            Arg:
                backend: backend provider
                *args:
                **kwargs:
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
            meas_return='single',
            *args, **kwargs):
        """
        Run circuit on backend

            Arg:
                shots: number of shots
                meas_return:
                *args:
                **kwargs:
        """
        schedule = pulse.Schedule()
        for s in self.list_schedule:
            schedule |= s << schedule.duration

        job = self.backend.run(schedule,
                               shots=shots,
                               meas_level=1,
                               meas_return=meas_return,
                               *args,
                               **kwargs)

        return job

    def measure(self):
        """
        Add a measurement to the circuit
        """
        self.list_schedule.append(pulses.measure(self))
