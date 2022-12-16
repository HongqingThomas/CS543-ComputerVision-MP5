# import lxr as love
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

def load_data():
    # In total, there are 215 3D points visible in all 101 views.
    # measurement_matrix's shape = (202, 215)
    measurement_matrix = np.loadtxt('factorization_data/measurement_matrix.txt')
    return measurement_matrix

def normailize_data(measurement_matrix):
    # normalize data for each view: x_ij = x_ij - mean(x_ik), k=(1,n)
    mean_matrix = np.mean(measurement_matrix, axis=1).reshape(202,1)
    registered_measurement_matrix = measurement_matrix - mean_matrix
    return registered_measurement_matrix

def singular_value_decompostion(matrix):
    U, S, V = la.svd(matrix)
    U = U[:,0:3]
    S = np.diag(S)[0:3,0:3]
    V = V[0:3,:]
    Q_init = sqrtm(S)
    A_init = U.dot(Q_init)
    X_init = Q_init.dot(V)
    # A_init = U
    # X_init = S @ V
    x_estimate = A_init @ X_init
    return A_init, X_init, x_estimate

def calculate_Q(A_init):
    # A_init.shape = (202,3)
    row = A_init.shape[0]
    Identity_matrix = np.identity(row)
    L = np.matrix(A_init).I @ Identity_matrix @ np.matrix(A_init.T).I
    Q = np.linalg.cholesky(L)
    return Q

def visualize_3d_points_in_3d_spaces(points_3d):
    points_3d = points_3d.T
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points_3d[:,0], points_3d[:,1], points_3d[:,2], c="r")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3d Scatter plot')
    plt.show()

def compute_residual(observed_coordinate, estimate_coordinate):
    row, col = observed_coordinate.shape[0], observed_coordinate.shape[1]
    per_frame_residual = np.zeros(shape=(int(row/2),1))
    for i in range(int(row/2)):
        for j in range(col):
            per_frame_residual[i][0] += np.sqrt((observed_coordinate[2*i][j] - estimate_coordinate[2*i][j])**2 + (observed_coordinate[2*i+1][j] - estimate_coordinate[2*i+1][j])**2)
    return per_frame_residual

def visualize_residual_plot(per_frame_residual):
    row = per_frame_residual.shape[0]
    x = np.arange(1,row+1,1)
    plt.title("Residual (per frame)")
    plt.plot(x,per_frame_residual.reshape(row))
    plt.xlabel('frame')
    plt.ylabel('residual')
    plt.show()

def plot_img(Img, measurement_matrix, estimate_coordinate, index):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(Img)
    ax.plot(measurement_matrix[2*index-2,:], measurement_matrix[2*index-1,:], '+r')
    mean_matrix = np.mean(measurement_matrix, axis=1).reshape(202, 1)
    ax.plot(estimate_coordinate[2*index-2,:]+mean_matrix[2*index-2,0], estimate_coordinate[2*index-1,:]+mean_matrix[2*index-1,0], '+g')
    plt.show()

def display_frame(measurement_matrix, estimate_coordinate):
    selected_frame = [1, 50, 100]
    for i in range(len(selected_frame)):
        index = selected_frame[i]
        if index<10:
            strindex = "00%d" % index
        elif index <100:
            strindex = "0%d" % index
        else:
            strindex = str(index)
        file_name = 'factorization_data/frame00000' + strindex + '.jpg'
        Img = mpimg.imread(file_name)
        plot_img(Img, measurement_matrix, estimate_coordinate, index)

if __name__ == '__main__':
    # step1 load data
    measurement_matrix = load_data()
    registered_measurement_matrix= normailize_data(measurement_matrix)

    # step2 calculting W~
    # A_init.shape = (202,3); X_init.shape = (3,215)
    A_init, X_init, x_estimate = singular_value_decompostion(registered_measurement_matrix)

    # step3 compute matrix Q
    Q = calculate_Q(A_init)
    print('Q:', Q)
    # visualize
    visualize_3d_points_in_3d_spaces(Q.I @ X_init)

    # step4 compute residul
    per_frame_residual = compute_residual(registered_measurement_matrix, x_estimate)
    print('total residual:', np.sum(per_frame_residual))
    visualize_residual_plot(per_frame_residual)

    # step5 display three frames
    display_frame(measurement_matrix, x_estimate)
