import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from utils import load_pose

def camera_parameters(structured_point_cloud, learning_rate=0.01, batch=32, max_iters=-1, decay=0.5):
    def params_to_intrinsic_matrix(fx, fy, cx, cy, dtype):
        matrix = Variable(torch.zeros(3, 4).type(dtype), requires_grad=False)
        matrix[0, 0] = fx
        matrix[1, 1] = fy
        matrix[2, 2] = 1.0
        matrix[0, 2] = cx
        matrix[1, 2] = cy
        return matrix
    
    def params_to_extrinsic_matrix(rotation, translation, dtype):
        matrix = Variable(torch.zeros(4, 4).type(dtype), requires_grad=False)
        matrix[0 : 3, 0 : 3] = rotation
        matrix[0 : 3, 3] = translation[:, 0]
        matrix[3, 3] = 1.0        
        return matrix
    
    x = structured_point_cloud[:, :, 0]
    y = structured_point_cloud[:, :, 1]
    z = structured_point_cloud[:, :, 2]
    mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))

    h, w, _ = structured_point_cloud.shape
    vu = np.mgrid[0 : h, 0 : w]
    py = vu[0][mask]
    px = vu[1][mask]
    uv = np.stack((px, py, np.ones(len(px)))).T

    points = structured_point_cloud[mask]
    points = np.hstack((points, np.ones((len(points), 1))))

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    fx = Variable(dtype([1.0]), requires_grad=True)
    fy = Variable(dtype([1.0]), requires_grad=True)
    cx = Variable(dtype([0.0]), requires_grad=True)
    cy = Variable(dtype([0.0]), requires_grad=True)
    rotation_matrix = Variable(torch.eye(3).type(dtype), requires_grad=True)
    translation_vector = Variable(torch.zeros(3, 1).type(dtype), requires_grad=True)

    iters = 0
    intrinsic_parameters = np.zeros((3, 4))
    extrinsic_parameters = np.zeros((4, 4))
    while True:
        if iters == max_iters:
            break
        
        idx = np.random.choice(len(points), len(points), False)
        uv = uv[idx]
        points = points[idx]

        for i in range(0, len(uv), batch):
            batch_points = torch.from_numpy(points[i : i + 32, :].T).type(dtype)
            batch_uv = torch.from_numpy(uv[i : i + 32, :].T).type(dtype)

            if torch.cuda.is_available():
                batch_points = batch_points.cuda()
                batch_uv = batch_uv.cuda()

            params1 = params_to_intrinsic_matrix(fx, fy, cx, cy, dtype)
            params2 = params_to_extrinsic_matrix(rotation_matrix, translation_vector, dtype)
            prediction = params1 @ params2 @ batch_points
            ws = prediction[2, :]
            prediction = prediction / ws

            loss = torch.sum((prediction[0 : 3, :] - batch_uv[0 : 3, :]) ** 2)
            loss.backward()
            fx.data -= learning_rate * fx.grad.data
            fy.data -= learning_rate * fy.grad.data
            cx.data -= learning_rate * cx.grad.data
            cy.data -= learning_rate * cy.grad.data
            rotation_matrix.data -= learning_rate * rotation_matrix.grad.data
            translation_vector.data -= learning_rate * translation_vector.grad.data

            fx.grad.data.zero_()
            fy.grad.data.zero_()
            cx.grad.data.zero_()
            cy.grad.data.zero_()
            rotation_matrix.grad.data.zero_()
            translation_vector.grad.data.zero_()
   
        if torch.cuda.is_available():
            intrinsic_parameters[0, 0] = fx.data.detach().cpu().numpy()
            intrinsic_parameters[1, 1] = fy.data.detach().cpu().numpy()
            intrinsic_parameters[2, 2] = 1.0
            intrinsic_parameters[0, 2] = cx.data.detach().cpu().numpy()
            intrinsic_parameters[1, 2] = cy.data.detach().cpu().numpy()
        else:
            intrinsic_parameters[0, 0] = fx.data.numpy()
            intrinsic_parameters[1, 1] = fy.data.numpy()
            intrinsic_parameters[2, 2] = 1.0
            intrinsic_parameters[0, 2] = cx.data.numpy()
            intrinsic_parameters[1, 2] = cy.data.numpy()
        
        if torch.cuda.is_available():
            extrinsic_parameters[0 : 3, 0 : 3] = rotation_matrix.data.detach().cpu().numpy()
            extrinsic_parameters[0 : 3, 3] = translation_vector.data.detach().cpu().numpy().flatten()
            extrinsic_parameters[3, 3] = 1.0
        else:
            extrinsic_parameters[0 : 3, 0 : 3] = rotation_matrix.data.numpy()
            extrinsic_parameters[0 : 3, 3] = translation_vector.data.numpy().flatten()
            extrinsic_parameters[3, 3] = 1.0

        projection = (intrinsic_parameters @ extrinsic_parameters @ points.T).T
        projection = projection / projection[:, -1][:, None]

        x = np.round(projection[:, 0]).astype(int)
        u = uv[:, 0]

        y = np.round(projection[:, 1]).astype(int)
        v = uv[:, 0]

        acc = np.mean(np.logical_and(x == u, y == v))
        print(f"Iteration: {iters}, accuracy: {acc}")

        #print(projection)
        #print(uv)
        if 1.0 >= acc > 0.99:
            break
        iters += 1
        learning_rate *= decay

    return intrinsic_parameters, extrinsic_parameters


def camera_parameters(structured_point_cloud, learning_rate=0.01, batch=32, max_iters=-1, decay=0.5):
    x = structured_point_cloud[:, :, 0]
    y = structured_point_cloud[:, :, 1]
    z = structured_point_cloud[:, :, 2]
    mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))

    h, w, _ = structured_point_cloud.shape
    vu = np.mgrid[0 : h, 0 : w]
    py = vu[0][mask]
    px = vu[1][mask]
    uv = np.stack((px, py, np.ones(len(px)))).T

    points = structured_point_cloud[mask]
    points = np.hstack((points, np.ones((len(points), 1))))

    fx = 1.0
    fy = 1.0
    cx = 0.0
    cy = 0.0
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros(3)

    iters = 0
    intrinsic_parameters = np.zeros((3, 4))
    extrinsic_parameters = np.zeros((4, 4))
    while True:
        if iters == max_iters:
            break
        
        idx = np.random.choice(len(points), len(points), False)
        uv = uv[idx]
        points = points[idx]

        for i in range(0, len(uv), batch):
            print(fx)
            batch_points = points[i : i + batch, :].T
            batch_uv = uv[i : i + batch, :].T

            params1 = np.zeros((3, 4))
            params1[0, 0] = fx
            params1[1, 1] = fy
            params1[0, 2] = cx
            params1[1, 2] = cy
            params1[2, 2] = 1.0

            params2 = np.zeros((4, 4))
            params2[0 : 3, 0 : 3] = rotation_matrix
            params2[0 : 3, 3] = translation_vector
            params2[3, 3] = 1.0

            f = params2 @ batch_points
            g = params1 @ f

            ws = g[2, :]
            prediction = g / ws

            loss = np.mean((prediction - batch_uv) ** 2)
            dloss = (2 / ws) * (prediction - batch_uv) / batch_points.shape[1]
            
            dparams1 = dloss @ f.T
            dg = params1.T @ dloss
            #print(dgx.shape)
            dparams2 = dg @ batch_points.T
            print(dparams1)
            print(dparams2)
            params1 -= learning_rate * dparams1
            params2 -= learning_rate * dparams2

            fx = params1[0, 0]
            fy = params1[1, 1]
            cx = params1[0, 2]
            cy = params1[1, 2]
            rotation_matrix = params2[0 : 3, 0 : 3]
            translation_vector = params2[0 : 3, 3]

        intrinsic_parameters[0, 0] = fx
        intrinsic_parameters[1, 1] = fy
        intrinsic_parameters[2, 2] = 1.0
        intrinsic_parameters[0, 2] = cx
        intrinsic_parameters[1, 2] = cy

        extrinsic_parameters[0 : 3, 0 : 3] = rotation_matrix
        extrinsic_parameters[0 : 3, 3] = translation_vector
        extrinsic_parameters[3, 3] = 1.0

        projection = (intrinsic_parameters @ extrinsic_parameters @ points.T).T
        projection = projection / projection[:, -1][:, None]

        x = np.round(projection[:, 0]).astype(int)
        u = uv[:, 0]

        y = np.round(projection[:, 1]).astype(int)
        v = uv[:, 0]

        acc = np.mean(np.logical_and(x == u, y == v))
        print(f"Iteration: {iters}, accuracy: {acc}")

        print(projection)
        print(uv)
        if 1.0 >= acc > 0.99:
            break
        iters += 1
        learning_rate *= decay

    return intrinsic_parameters, extrinsic_parameters

if __name__ == '__main__':
    pose = load_pose('../codes/Dataset', 'CMU', 1, 1, 1)
    camera_parameters(pose, learning_rate=0.001, batch=128, max_iters=20, decay=0.9)
