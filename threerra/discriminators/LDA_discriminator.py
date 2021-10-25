import numpy as np
import qiskit.pulse as pulse
from qiskit.tools.monitor import job_monitor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from threerra import QuantumCircuit3
from threerra import pulses

def reshape_complex_vec(vec):
    """Take in complex vector vec and return 2d array w/ real, imag entries. This is needed for the learning.
    Args:
        vec (list): complex vector of data
    Returns:
        list: vector w/ entries given by (real(vec], imag(vec))
    """
    length = len(vec)
    vec_reshaped = np.zeros((length, 2))
    for i in range(len(vec)):
        vec_reshaped[i]=[np.real(vec[i]), np.imag(vec[i])]
    return vec_reshaped


def discriminator(IQ_012_data, points, shots=1024, acc=False):

        # construct vector w/ 0's, 1's and 2's (for testing)
        state_012 = np.zeros(shots) # shots gives number of experiments
        state_012 = np.concatenate((state_012, np.ones(shots)))
        state_012 = np.concatenate((state_012, 2*np.ones(shots)))

        # Shuffle and split data into training and test sets
        IQ_012_train, IQ_012_test, state_012_train, state_012_test = train_test_split(IQ_012_data, state_012, test_size=0.5)

        # Set up the LDA
        LDA_012 = LinearDiscriminantAnalysis()
        LDA_012.fit(IQ_012_train, state_012_train)

        if acc==True:
            score_012 = LDA_012.score(IQ_012_test, state_012_test)
            print(score_012)

        counts = LDA_012.predict(points)

        return counts

def train_discriminator012(self : QuantumCircuit3, shots=1024):

    # Pulses
    pi_pulse_01 = pulses.pi_pulse_01(self)
    pi_pulse_12 = pulses.pi_pulse_12(self)
    measure_pulse = pulses.measure(self)

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
    zero_data_reshaped = reshape_complex_vec(zero_data)
    one_data_reshaped = reshape_complex_vec(one_data)
    two_data_reshaped = reshape_complex_vec(two_data)

    IQ_012_data_reshaped = np.concatenate((zero_data_reshaped, one_data_reshaped, two_data_reshaped))

    self.data_disc =  IQ_012_data_reshaped
