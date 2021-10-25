#!/usr/bin/env python3

import qiskit.pulse.library as pulse_lib

def measure(qc3):
    meas_idx = [qc3.qubit in group
                       for group in qc3.backend_config.meas_map].index(True)

    pulse = qc3.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc3.backend_config.meas_map[meas_idx],
    )

    return pulse

def pi_pulse_01(qc3):

    pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                               amp=qc3.pi_amp_01,
                               sigma=qc3.drive_sigma,
                               name='x_01')
    return pulse

def pi_pulse_12(qc3):

    pulse = pulse_lib.gaussian(duration=qc3.drive_samples,
                               amp=qc3.pi_amp_12,
                               sigma=qc3.drive_sigma,
                               name='x_12')

    # Shift frequency by sidebanding
    pulse = qc3.apply_sideband(pulse, qc3.qubit_freq_est_12,
                               name="x_12")

    return pulse
