from enum import Enum

import numpy as np
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

from motpy.core import Box, Vector
from loguru import logger
import collections
""" The list of model presets below is not complete, more reasonable
options will be added in the future """
import itertools


np.set_printoptions(threshold=np.inf)

class ModelPreset(Enum):
    constant_velocity_and_static_box_size_2d = {'order_pos': 1, 'dim_pos': 2,
                                                'order_size': 0, 'dim_size': 2}

    constant_acceleration_and_static_box_size_2d = {'order_pos': 2, 'dim_pos': 2,
                                                    'order_size': 0, 'dim_size': 2}


def _base_dim_block(dt: float, order: int = 1):
    block = np.array([[1, dt, (dt**2) / 2],
                      [0, 1, dt],
                      [0, 0, 1]])
    cutoff = order + 1
    return block[:cutoff, :cutoff]

def _base_dim_block2(dt: float, order: int = 1):

    block = np.array([[1, dt, (dt**2) / 2],
                      [0, 1, dt],
                      [0, 0, 1]])
    cutoff = order + 1
    return block[:order,:order]

def _zero_pad(arr, length: int):
    ret = np.zeros((length,))
    ret[:arr.shape[0]] = arr
    return ret


class Model:
    ndim = 4
    def __init__(
            self,
            dt: float,
            order_pos: int = 1,
            dim_pos: int = 2,
            order_size: int = 1,
            dim_size: int = 2,
            q_var_pos: float = 70.0,
            q_var_size: float = 10.0,
            r_var_pos: float = 1,
            r_var_size: float = 1,
            p_cov_p0: float = 1000.):



#        dt = 1
        self.dt = dt

        #logger.info(self.dt)
        self.order_pos = order_pos
        self.dim_pos = dim_pos
        self.order_size = order_size
        self.dim_size = dim_size

        self.q_var_pos = q_var_pos
        self.q_var_size = q_var_size
        self.r_var_pos = r_var_pos
        self.r_var_size = r_var_size
        self.p_cov_p0 = p_cov_p0

        if self.order_pos > 2 or self.order_size > 2:
            raise ValueError('Currently only system orders <= 2 are supported')

        # the expected input/output box length
        self.dim_box = 2 * max(self.dim_pos, self.dim_size)

        # precalculate utility indexes
        self.pos_idxs, self.size_idxs, self.z_in_x_idxs, self.offset_idx = self._calc_idxs()

        self.flow_idxs = [8,9]
        self.flow_idxs2 = [10,11]
        self.flow_idxs3 = [12,13]
        self.flow_idxs4 = [14,15]


        # number of variables in model state
        # self.state_length = (self.dim_pos * (self.order_pos + 1) + \
        #     self.dim_size * (self.order_size + 1))#+ (self.dim_pos )
        self.state_length = (self.dim_pos * (self.order_pos + 1) + \
            self.dim_size * (self.order_size + 1))

#        logger.info(self.state_length)
        #self.state_length = self.dim_pos * self.ndim
        # length of z (observation) vector
        self.measurement_length = self.dim_pos + self.dim_size + 2
 #       logger.info(self.measurement_length)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.ndim = 4
        self.F = np.eye(self.ndim * 2)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = dt
#        self.ndim = 4


    def _calc_idxs(self):
        offset_idx = max(self.dim_pos, self.dim_size)

        pos_idxs = [pidx * (self.order_pos + 1)
                    for pidx in range(self.dim_pos)]

        size_idxs = [self.dim_pos * (self.order_pos + 1) + sidx * (self.order_size + 1)
                     for sidx in range(self.dim_size)]
        #print(pos_idxs)

        flow_idxs = [8,9]

        # indexes of measured quantities (z) in state (x) vector
        z_in_x_idxs = np.concatenate((pos_idxs, size_idxs))
        # z_in_x_idxs = np.concatenate((pos_idxs, size_idxs, flow_idxs))
        # print(z_in_x_idxs)

        return np.array(pos_idxs), np.array(size_idxs), z_in_x_idxs, offset_idx

    def build_F(self, box,x0):
        """ returns constructed F matrix with specified positional
            e.g. (x,y,z) and size e.g. (w,h) dimensions """
        block_pos = _base_dim_block(self.dt, self.order_pos)
        block_size = _base_dim_block(self.dt, self.order_size)
        flow_block = _base_dim_block2(self.dt, self.order_size)
        print(flow_block )
        diag_components = ([block_pos] * self.dim_pos)*2 + [flow_block]*2
        print(self.flow_size)
        diag_components = ([block_pos] * self.dim_pos)*2 + [flow_block]*self.flow_size*2
        print(block_diag(*diag_components))
        return block_diag(*diag_components)
        dt = self.dt
        # print((8+len(box[4])*2))
        F = np.zeros((8+len(box[4])*2,8+len(box[4])*2 ))
        # print(x0)
        # self.w = box[2]-box[0]
        # u= box[4][0][0]-box[0]
        # v = box[4][0][1]-box[1]
        # u1 = box[4][1][0]-box[0]
        # v1 = box[4][1][1]-box[1]
        # u2 = box[4][2][0]-box[0]
        # v2 = box[4][2][1]-box[1]
        # u3 = box[4][3][0]-box[0]
        # v3 = box[4][3][1]-box[1]
        # print("t_box")
        # print(box)

        self.w = x0[4]
        self.h = x0[6]

        self.dw =  box[2]-x0[4]
        self.dh = box[3]-x0[6]
        u= x0[8]
        v = x0[9]
        u1 = x0[10]
        v1 = x0[11]
        u2 = x0[12]
        v2 = x0[13]
        u3 = x0[14]
        v3 = x0[15]



        F[0, 0] = 1
        F[0, 1] = 0.3
        F[1, 1] = 1
        F[2, 2] = 1
        F[2, 3] = 0.3
        F[3, 3] = 1
        F[4, 4] = 1
        F[4, 5] = 0.3
        F[5, 5] = 1
        F[6, 6] = 1
        F[6, 7] = 0.3
        F[7, 7] = 1


        for i in range((len(box[4])*2)):
            F[8+i, 8+i] = 1


            if (8+i) % 2 == 0:
                u= x0[8+i]
                
                F[8+i, 5] = -(u * dt) / (self.w ** 2)
                # F[8+i, 4] = -(u * self.dw * dt) / (self.w ** 2)
            else:
                v = x0[8+i]
                F[8+i, 7] = -(v* dt) / (self.h ** 2)
                # F[8+i, 6] = -(v * self.dh * dt )/ (self.h ** 2)





        # F[8, 8] = 1
        # F[9, 9] = 1
        # F[10, 10] = 1
        # F[11, 11] = 1
        # F[12, 12] = 1
        # F[13, 13] = 1
        # F[14, 14] = 1
        # F[15, 15] = 1
        # # Time-based elements
        # F[8, 5] = u * dt / self.w
        # F[9, 7] = v * dt / self.h
        # F[10, 5] = u1 * dt / self.w
        # F[11, 7] = v1 * dt / self.h
        # F[12, 5] = u2 * dt / self.w
        # F[13, 7] = v2 * dt / self.h
        # F[14, 5] = u3 * dt / self.w
        # F[15, 7] = v3 * dt / self.h
        # # Non-linear elements
        # F[8, 4] = -u * self.w * dt / (self.w ** 2)
        # F[9, 6] = -v * self.h * dt / (self.h ** 2)
        # F[10, 4] = -u1 * self.w * dt / (self.w ** 2)
        # F[11, 6] = -v1 * self.h * dt / (self.h ** 2)
        # F[12, 4] = -u2 * self.w * dt / (self.w ** 2)
        # F[13, 6] = -v2 * self.h * dt / (self.h ** 2)
        # F[14, 4] = -u3 * self.w * dt / (self.w** 2)
        # F[15, 6] = -v3 * self.h * dt / (self.h ** 2)

        # print(F)




        return F


    def build_Q(self, box):
        """ process noise """
        var_pos = self.q_var_pos
        var_size = self.q_var_size

        #logger.info(self.dt)
        q_pos = var_pos if self.order_pos == 0 else Q_discrete_white_noise(
            dim=self.order_pos + 1, dt=self.dt, var=var_pos)

        q_size = var_size if self.order_size == 0 else Q_discrete_white_noise(
            dim=self.order_size + 1, dt=self.dt, var=var_size)

        print("dffffffff")
        print(q_pos)

        f_q = [[0,   0  ],
        [0,   0]]

        # diag_components = [q_pos] * self.dim_pos + [q_size] * self.dim_size + [q_pos]
        diag_components = [q_pos] * self.dim_pos + [q_size] * self.dim_size + [f_q]*(len(box[4]))
        #print(block_diag(*diag_components))




        return block_diag(*diag_components)

        # std_pos = [
        #     self._std_weight_position * mean[0],
        #     self._std_weight_position * mean[1],
        #     1 * mean[2],
        #     self._std_weight_position * mean[3],
        # ]
        # std_vel = [
        #     self._std_weight_velocity * mean[0],
        #     self._std_weight_velocity * mean[1],
        #     0.1 * mean[2],
        #     self._std_weight_velocity * mean[3],
        # ]

        # motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        # print(motion_cov)

        # return motion_cov 

    def build_R(self, box,mean, confidence=0.0):
        """ measurement noise, expected order is positon first, then size """
        block_pos = np.eye(self.dim_pos) * self.r_var_pos
        block_size = np.eye(self.dim_size) * self.r_var_size
        flow_pos = np.eye((len(box[4])*2))* self.r_var_pos

        #logger.info(block_diag(block_pos, block_size))
        print(block_diag(block_pos, block_size, flow_pos))
        return block_diag(block_pos, block_size, flow_pos)

        #logger.info("s")
        # self.x = mean
        # self.P = confidence

        # std = [
        #     self._std_weight_position * mean[3],
        #     self._std_weight_position * mean[3],
        #     1e-1,
        #     self._std_weight_position * mean[3],
        # ]
        # #logger.info(confidence)
        # std = [(1 - confidence) * x for x in std]

        # innovation_cov = np.diag(np.square(std))
        # #logger.info(innovation_cov)
        # return innovation_cov

    def build_H(self,box):
        """ measurement matrix """
        # we only measure the first variable in each dimension
        def _base_block(order): return np.array([1] + [0] * order)
        flow_block = _base_dim_block2(self.dt, self.order_size)
        # print(self.dim_pos*4)
        diag_components = \
            [_base_block(self.order_pos)] * self.dim_pos +\
            [_base_block(self.order_size)] * self.dim_size +\
            [flow_block] * (len(box[4])*2)
        print( block_diag(*diag_components))
        return block_diag(*diag_components)
        # return np.eye(self.ndim, 2 * self.ndim)

    def build_P(self, box):
        # std = [
        #     2 * self._std_weight_position * measurement[0],  # the center point x
        #     2 * self._std_weight_position * measurement[1],  # the center point y
        #     1 * measurement[2],  # the ratio of width/height
        #     2 * self._std_weight_position * measurement[3],  # the height
        #     10 * self._std_weight_velocity * measurement[0],
        #     10 * self._std_weight_velocity * measurement[1],
        #     0.1 * measurement[2],
        #     10 * self._std_weight_velocity * measurement[3],
        # ]
        # self.P = np.diag(np.square(std))

        # #logger.info(self.P)
        # return self.P
        print(np.eye(self.state_length+(len(box[4])*2)) * self.p_cov_p0)
        print(self.state_length+(len(box[4])*2))
        return np.eye(self.state_length+(len(box[4])*2)) * self.p_cov_p0

    def box_to_z(self, box: Box) -> Vector:
        #print(self.dim_box)
        #print(len(box[4]))
        self.flow_size = len(box[4])
        assert self.dim_box+2 == len(box)+1
        #print((len(box[4])+2))
        #print(np.array(box))
        box[4] = np.array(box[4])
        # print(box)
        # box = [box[0],box[1],box[0]+box[2],box[1]+box[3]]
        # box = np.array(box).reshape(3, (int((self.dim_box)/ 2)))
        n_box = np.array(box[0:4]).reshape(2, (int((self.dim_box)/ 2)))
        box =  np.vstack((n_box, box[4]))
        # print(box)
        # print( self.box_result)

            # self.box_result = self.box_result

        # flow = [box[2][0]- box[0][0], box[2][1]- box[0][1] ]


        self.box_result = (box[2:] - box[0])
        # print("sassasssss")
        # print(self.box_result)
        self.box_result = self.box_result.flatten().tolist()
        # print(len(box_result) )
        box = box[0:2]

        center = (np.sum(box, axis=0) / 2.0)[:self.dim_pos]
        length = (box[1, :] - box[0, :])[:self.dim_size]


        # self.box_result = box_result.flatten().tolist()
        # print(len(box_result))

        #center = (np.sum(box, axis=0) / 2.0)[:self.dim_pos]
        # center = (np.sum(box, axis=0) / 2.0)[:self.dim_pos]
        # print(box[1, :])
        # print(box[0, :])
        # length = (box[1, :] - box[0, :])[:self.dim_size]
        #length = (box[1, :])[:self.dim_size]
        # print("asas")
        # print(center)
        # print(length)
        # print("aere")
        #return np.concatenate((center, length))
        # x = np.r_[center, length, flow[0], flow[1] ]
        x = np.r_[center, length]
        # x = np.vstack((x, box_result))
        # print(x)
        return x , self.box_result


    def box_to_x(self, box: Box) -> Vector:
        """ box is expected to be in [xmin, ymin, zmin, ..., xmax, ymax, zmax, ...] format
        for 2d-1ord+2d-0ord case returns np.array([cx, 0, 0, cy, 0, 0, w, h]) """
        print((box))
        x = np.zeros((self.state_length+(len(box[4])*2),))
        # print(self.state_length+(len(box[4])*2))
        # print(self.z_in_x_idxs)
        indices_to_remove = [1, 3, 5, 7]
        # x = np.delete(self.box_to_z(box), indices_to_remove)
        x[self.z_in_x_idxs], flow = self.box_to_z(box)
        # print(len(x[4:]))
        x[8:] = flow


        # print("z_X")

        # print(x)




        #mean_pos = box
        # mean_pos = self.box_to_z(box)

        # mean_vel = np.zeros_like(mean_pos)


        # x = np.r_[mean_pos, mean_vel]
        #print(x)
        #logger.info(x)
        return x

    def x_to_box(self, x):
        size = max(self.dim_pos, self.dim_size)
        # print(x)
        # print(self.size_idxs)
        center = _zero_pad(x[self.pos_idxs], size)
        length = _zero_pad(x[self.size_idxs], size)
        flow = _zero_pad(x[self.flow_idxs], size)
        flow2 = _zero_pad(x[self.flow_idxs2], size)
        flow3 = _zero_pad(x[self.flow_idxs3], size)
        flow4 = _zero_pad(x[self.flow_idxs4], size)
        # print(flow)
        # print(center)
        # print("wwwe")
        # print(np.concatenate((center - (length) / 2, center + (length) / 2, (center - (length) / 2) + flow,(center - (length) / 2) + flow,(center - (length) / 2) + flow,(center - (length) / 2) + flow)))
        # return np.concatenate((center - (length) / 2, center + (length) / 2, (center - (length) / 2) + flow))
        return np.concatenate((center - (length) / 2, center + (length) / 2,  flow, flow2,flow3,flow4))
        # return np.concatenate((center - (length) / 2, center + (length) / 2, (center - (length) / 2) + flow,(center - (length) / 2) + flow,(center - (length) / 2) + flow,(center - (length) / 2) + flow))


    def flatten_sequences(self,sequences) -> list:
        sequences = [i if type(i) == list else [i] for i in sequences]
        flattened = list(itertools.chain.from_iterable(sequences))
        return flattened
