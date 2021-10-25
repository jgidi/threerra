#!/usr/bin/env python3

from threerra import QuantumCircuit3

def measure(qc : QuantumCircuit3):
    meas_idx = [qc.qubit in group
                       for group in qc.backend_config.meas_map].index(True)

    measure_pulse = qc.backend_defaults.instruction_schedule_map.get(
        'measure',
        qubits = qc.backend_config.meas_map[meas_idx],
    )

    return measure_pulse
