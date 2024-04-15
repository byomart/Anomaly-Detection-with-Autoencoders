import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

class AutomaticThresholdDetector:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def detect_threshold(self, losses_normal, losses_anomalies):
        kde_normal = sns.kdeplot(losses_normal, color='b', legend=False)
        kde_anomalies = sns.kdeplot(losses_anomalies, color='r', legend=False)
        logging.info(f'kde_normal (x, y, weight, height of the subfigure): {kde_normal}')

        # to extract data points from KDE plots
        val1 = 0
        val2 = 1
        x_normal = kde_normal.get_lines()[val1].get_xdata()
        y_normal = kde_normal.get_lines()[val1].get_ydata()
        x_anomalies = kde_anomalies.get_lines()[val2].get_xdata()
        y_anomalies = kde_anomalies.get_lines()[val2].get_ydata()

        # data points as tuples (x, y)
        list_points_normal = [(x, y) for x, y in zip(x_normal, y_normal)]
        list_points_anomalies = [(x, y) for x, y in zip(x_anomalies, y_anomalies)]
        logging.info(list_points_normal)

        # Euclidean Distance between points
        def distance(A, B):
            return np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

        # intersecting points
        closest_point = []
        min_dist = float('inf')
        for point_normal in list_points_normal:
            for point_anomalie in list_points_anomalies:
                if point_normal[1] >= 0 and point_anomalie[1] >= 0:  # verify both positive values
                    dist = distance(point_normal, point_anomalie)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point.append(point_normal)

        # last intersecting point
        intersect_point = closest_point[-1] 
        threshold = intersect_point[0]
        logging.info(f'Intersection Point: {intersect_point}')
        logging.info(f'Threshold: {threshold}')

        plt.figure()
        plt.plot(x_normal, y_normal, color='b', label='Normal')
        plt.plot(x_anomalies, y_anomalies, color='r', label='Anomalies')
        plt.ylabel('Density')
        plt.title('Automatic Threshold Detection')
        plt.axvline(x=threshold, color='k', linestyle='--')
        plt.legend()
        plt.savefig(os.path.join('images', 'th.png'))
        #plt.show()
        plt.close()

        return threshold
