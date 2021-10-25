#!/usr/bin/env python3

import qiskit.pulse.library as pulse_lib

from threerra import QuantumCircuit3

def measure(qc : QuantumCircuit3):
    meas_idx = [qc.qubit in group
                       for group in qc.backend_config.meas_map].index(True)

    pulse = qc.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc.backend_config.meas_map[meas_idx],
    )

    return pulse

def pi_pulse_01(qc : QuantumCircuit3):

    pulse = pulse_lib.gaussian(duration=qc.drive_samples,
                               amp=qc.pi_amp_01,
                               sigma=qc.drive_sigma,
                               name='x_01')
    return pulse

def pi_pulse_12(qc : QuantumCircuit3):

    pulse = pulse_lib.gaussian(duration=qc.drive_samples,
                               amp=qc.pi_amp_12,
                               sigma=qc.drive_sigma,
                               name='x_12')

    # Shift frequency by sidebanding
    pulse = qc.apply_sideband(pulse, qc.qubit_freq_est_12,
                              name="x_12")

    return pulse
