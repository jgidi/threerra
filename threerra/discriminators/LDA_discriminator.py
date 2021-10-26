import numpy as np
import qiskit.pulse as pulse
from qiskit.tools.monitor import job_monitor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

import numpy as np
from threerra import pulses

LDA_discriminator_data = np.loadtxt('data_disc.txt')

def discriminator(data, shots=1024, print_acc=False):

    points = np.array([[np.real(dat), np.imag(dat)] for dat in data])

    # construct vector w/ 0's, 1's and 2's (for testing)
    state_012 = np.zeros(shots) # shots gives number of experiments
    state_012 = np.concatenate((state_012, np.ones(shots)))
    state_012 = np.concatenate((state_012, 2*np.ones(shots)))

    # Shuffle and split data into training and test sets
    IQ_012_train, IQ_012_test, state_012_train, state_012_test = train_test_split(LDA_discriminator_data, state_012, test_size=0.5)

    # Set up the LDA
    LDA_012 = LinearDiscriminantAnalysis()
    LDA_012.fit(IQ_012_train, state_012_train)

    if print_acc==True:
        score_012 = LDA_012.score(IQ_012_test, state_012_test)
        print(score_012)

    counts = np.round(LDA_012.predict(points)).astype(int)

    return counts

def train_discriminator(qc3, shots=1024):

    # Pulses
    pi_pulse_01 = pulses.pi_pulse_01(qc3)
    pi_pulse_12 = pulses.pi_pulse_12(qc3)
    measure_pulse = pulses.measure(qc3)

    # Create the three schedules

    # Ground state schedule
    zero_schedule = pulse.Schedule(name="zero schedule")
    zero_schedule |= measure_pulse

    # Excited state schedule
    one_schedule = pulse.Schedule(name="one schedule")
    one_schedule |= pulse.Play(pi_pulse_01, qc3.drive_chan)
    one_schedule |= measure_pulse << one_schedule.duration

    # Excited state schedule
    two_schedule = pulse.Schedule(name="two schedule")
    two_schedule |= pulse.Play(pi_pulse_01, qc3.drive_chan)
    two_schedule |= pulse.Play(pi_pulse_12, qc3.drive_chan) << two_schedule.duration
    two_schedule |= measure_pulse << two_schedule.duration

    IQ_012_job = qc3.backend.run([zero_schedule, one_schedule, two_schedule],
                       meas_level=1,
                       meas_return='single',
                       shots=shots,
                       schedule_los=[{qc3.drive_chan: qc3.qubit_freq_est_01}] * 3)

    job_monitor(IQ_012_job)

    # Get job data (single); split for zero, one and two
    result = IQ_012_job.result(timeout=120)

    IQ_012_data = np.concatenate([result.get_memory(i)[:, qc3.qubit]
                                  for i in range(len(result.results))])

    # Reorder as real pairs
    IQ_012_data = np.array([np.real(dat), np.imag(dat)]
                           for dat in IQ_012_data)

    global LDA_discriminator_data
    LDA_discriminator_data = IQ_012_data
