from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time

#measurements = np.asarray([(399,293),(403,299),(489,308),(416,315),(418,318),(320,323),(429,326),(423,328),(229,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(310,313),(406,306),(402,299),(397,291),(391,294),(376,270),(272,272),(351,248),(336,244),(327,236),(317,220)])

measurements = np.asarray([(399,293),(403,299)])#,(403,299),(489,308),(416,315),(418,318),(320,323),(429,326),(423,328),(229,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(310,313),(406,306),(402,299),(397,291),(391,294),(376,270),(272,272),(351,248),(336,244),(327,236),(317,220)])
print("shape: ", measurements.shape)
print("Measurements: ", measurements)
def apply_kalman_filter(measurements, show=False):
    initial_state_mean = [measurements[0, 0],
                        0,
                        measurements[0, 1],
                        0]

    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

    kf = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean, n_dim_obs=2)

    kf1 = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)


    if show:
        plt.figure(1)
        times = range(measurements.shape[0])
        plt.plot(times, measurements[:, 0], 'bo',
                times, measurements[:, 1], 'ro',
                times, smoothed_state_means[:, 0], 'b--',
                times, smoothed_state_means[:, 2], 'r--',)
        plt.show()

apply_kalman_filter(measurements)

