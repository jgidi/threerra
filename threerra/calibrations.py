#!/usr/bin/env python3

import numpy as np
import qiskit.pulse as pulse
import qiskit.pulse.library as pulse_lib
from qiskit.tools.monitor import job_monitor
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile, schedule as build_schedule
from threerra.units import MHz, GHz
from threerra import pulses

def calibrate_freq_01(qc3, freqs=None):
    """
    Does a frequency sweep to calibrate the 0 --> 1 transition frequency.

        Args:
            qc3: QuantumCircuit3 circuit.
            freqs: Numpy array of frequencies to sweep 
            
    """

    f0 = qc3.qubit_freq_est_01
    if freqs is None:
        freqs = f0 + np.linspace(-20*MHz, 20*MHz, 75)

    meas_idx = [qc3.qubit in group
                    for group in qc3.backend_config.meas_map].index(True)

    measure_pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
    )

    gaussian_pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                                        sigma=qc3.drive_sigma,
                                        amp=0.1)

    # Define schedule
    schedule = pulse.Schedule()
    schedule |= pulse.Play(gaussian_pulse, qc3.drive_chan)
    schedule |= measure_pulse << schedule.duration

    job = qc3.backend.run(schedule,
                           meas_level=1,
                           meas_return='avg',
                           shots=1024,
                           schedule_los=[{qc3.drive_chan: freq} for freq in freqs])

    # Make notice about the on-going job
    print("Calibrating qubit_freq_est_01...")
    job_monitor(job)

    results = job.result(timeout=120)
    results_data = [results.get_memory(i)[qc3.qubit] for i in range(len(results.results))]

    # Fit Lorentzian response
    fun = lambda x, qf, a, b, c: a * b / ((x-qf)**2 + b**2 ) + c
    ydata = np.real(results_data)
    ydata /= ydata.max()-ydata.min() # Normalize data height
    b = (freqs.max()-freqs.min())/10 # Half-width
    fit_params, *_ = curve_fit(fun, freqs, ydata, [f0, b, b, ydata.mean()])

    # Update frequency estimate
    qc3.qubit_freq_est_01 = fit_params[0]

    print(f'qubit_freq_est_01 updated from {f0/GHz}GHz to {fit_params[0]/GHz}GHz.')


def calibrate_pi_amp_01(qc3, amps=None):
    """
    Does a Rabi experiment to compute the 0 --> 1 pi pulse amplitude.

        Args:
            qc3: QuantumCircuit3 circuit.
            amps: Numpy array of amplitudes to to iterate over 
            
    """

    amp0 = qc3.pi_amp_01
    if amps is None:
        amps = np.linspace(0, 0.75, 75)

    meas_idx = [qc3.qubit in group
                    for group in qc3.backend_config.meas_map].index(True)

    measure_pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
        )

    schedules = []
    for amp in amps:
        gaussian_pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                                            sigma=qc3.drive_sigma,
                                            amp=amp)

        # Define schedule
        schedule = pulse.Schedule()
        schedule |= pulse.Play(gaussian_pulse, qc3.drive_chan)
        schedule |= measure_pulse << schedule.duration

        # Accumulate
        schedules.append(schedule)

    job = qc3.backend.run(schedules,
                           meas_level=1,
                           meas_return='avg',
                           shots=1024,
                           schedule_los=[{qc3.drive_chan: qc3.qubit_freq_est_01}] * len(amps))

    # Make notice about the on-going job
    print("Calibrating pi_amp_01...")
    job_monitor(job)

    results = job.result(timeout=120)
    results_data = [results.get_memory(i)[qc3.qubit] for i in range(len(results.results))]

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
    qc3.pi_amp_01 = abs(fit_params[0] / 2)

    print(f'pi_amp_01 updated from {amp0} to {qc3.pi_amp_01}.')

def calibrate_freq_12(qc3, freqs=None):
    """
    Does a frequency sweep to calibrate the 1 --> 2 transition frequency.

        Args:
            qc3: QuantumCircuit3 circuit.
            freqs: Numpy array of frequencies to sweep 
            
    """

    f0 = qc3.qubit_freq_est_12
    if freqs is None:
        freqs = f0 + np.linspace(-20*MHz, 20*MHz, 75)

    meas_idx = [qc3.qubit in group
                    for group in qc3.backend_config.meas_map].index(True)

    measure_pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
        )

    gaussian_pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                                        sigma=qc3.drive_sigma,
                                        amp=0.3)

    pi_pulse_01 = pulses.pi_pulse_01_sched(qc3)
    schedules = []          # Accumulator
    for freq in freqs:
        sidebanded_pulse = qc3.apply_sideband(gaussian_pulse, freq)

        schedule = pulse.Schedule()
        schedule |= pi_pulse_01
        schedule |= pulse.Play(sidebanded_pulse, qc3.drive_chan) << schedule.duration
        schedule |= measure_pulse << schedule.duration

        schedules.append(schedule)

    job = qc3.backend.run(schedules,
                           meas_level=1,
                           meas_return='avg',
                           shots=1024,
                           schedule_los=[{qc3.drive_chan: qc3.qubit_freq_est_01}] * len(freqs))

    # Make notice about the on-going job
    print("Calibrating qubit_freq_est_12...")
    job_monitor(job)

    results = job.result(timeout=120)
    results_data = [results.get_memory(i)[qc3.qubit] for i in range(len(results.results))]

    # Fit Lorentzian response
    fun = lambda x, qf, a, b, c: a * b / ((x-qf)**2 + b**2 ) + c
    ydata = np.real(results_data)
    ydata /= ydata.max()-ydata.min() # Normalize data height
    b = (freqs.max()-freqs.min())/10 # Half-width
    fit_params, *_ = curve_fit(fun, freqs, ydata, [f0, b, b, ydata.mean()])

    # Update frequency estimate
    qc3.qubit_freq_est_12 = fit_params[0]

    print(f'qubit_freq_est_12 updated from {f0/GHz}GHz to {fit_params[0]/GHz}GHz.')


def calibrate_pi_amp_12(qc3, amps=None):
    """
    Does a Rabi experiment to compute the 1 --> 2 pi pulse amplitude using the sideband method.

        Args:
            qc3: QuantumCircuit3 circuit.
            amps: Numpy array of amplitudes to to iterate over 
            
    """

    amp0 = qc3.pi_amp_12
    if amps is None:
        amps = np.linspace(0, 0.75, 75)

    meas_idx = [qc3.qubit in group
                    for group in qc3.backend_config.meas_map].index(True)

    measure_pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
        )

    pi_pulse_01 = pulses.pi_pulse_01_sched(qc3)

    schedules = []
    for amp in amps:
        base_pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                                        sigma=qc3.drive_sigma,
                                        amp=amp)

        sidebanded_pulse = qc3.apply_sideband(base_pulse, qc3.qubit_freq_est_12)

        # Define schedule
        schedule = pulse.Schedule()
        schedule |= pi_pulse_01
        schedule |= pulse.Play(sidebanded_pulse, qc3.drive_chan) << schedule.duration
        schedule |= measure_pulse << schedule.duration

        # Accumulate
        schedules.append(schedule)

    job = qc3.backend.run(schedules,
                           meas_level=1,
                           meas_return='avg',
                           shots=1024,
                           schedule_los=[{qc3.drive_chan: qc3.qubit_freq_est_01}] * len(amps))

    # Make notice about the on-going job
    print("Calibrating pi_amp_12...")
    job_monitor(job)

    results = job.result(timeout=120)
    results_data = [results.get_memory(i)[qc3.qubit] for i in range(len(results.results))]

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
    qc3.pi_amp_12 = abs(fit_params[0] / 2)

    print(f'pi_amp_12 updated from {amp0} to {qc3.pi_amp_12}.')
