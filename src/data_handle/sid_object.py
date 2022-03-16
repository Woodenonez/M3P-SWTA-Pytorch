import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# SID - Single-target Interaction Dataset

def return_map(map=1, block=False):
    '''10m x 10m area'''
    boundary_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    if map == 1:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
            [(0.0, 6.0), (0.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
            [(6.0, 6.0), (6.0, 10.0), (10.0, 10.0), (10.0, 6.0)],
            [(6.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
        ]
    elif map == 2:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
            [(0.0, 6.0), (0.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
            [(6.0, 6.0), (6.0, 10.0), (10.0, 10.0)],
            [(6.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
        ]
    else:
        raise ModuleNotFoundError("Map not found!")
    if block:
        obstacle_list.append([(4.0, 4.1), (4.0, 5.9), (3.0, 5.9), (3.0, 4.1)])

    return boundary_coords, obstacle_list

def return_path(path=1, map=1, block=False):
    if path == 1: # forward
        start = (5.5, 0)
        turning = (5.5, 5)
        end = (5.5, 10)
    elif path == 2: # left
        start = (5.5, 0)
        turning = (5.5, 5.5)
        end = (0, 5.5)
    elif path == 3: # right
        if map == 1:
            start = (5.5, 0)
            turning = (5.5, 4.5)
            end = (10, 4.5)
        else:
            start = (5.5, 0)
            turning = (5.5, 4.5)
            end = (10, random.randint(45, 85)/10)
    else:
        raise ModuleNotFoundError("Path not found!")
    return [start, turning, end]

def return_dyn_obs_path(ts, start=10):
    return [(4.5, start-x*ts) for x in range(int(10/ts)+1)]

class Graph:
    def __init__(self, map, block):
        self.map = map
        self.block = block
        self.boundary_coords, self.obstacle_list = return_map(map, block)

    def get_obs_path(self, ts, start=10):
        return return_dyn_obs_path(ts, start=start)

    def get_path(self, path):
        self.path = return_path(path, self.map, self.block)
        return self.path

    def plot_map(self, ax, clean=False, empty=False):
        boundary = self.boundary_coords + [self.boundary_coords[0]]
        boundary = np.array(boundary)
        if empty:
            ax.plot(boundary[:,0], boundary[:,1], 'white')
        else:
            for obs in self.obstacle_list:
                obs = np.array(obs)
                poly = patches.Polygon(obs, color='gray')
                ax.add_patch(poly)
            if not clean:
                ax.plot(boundary[:,0], boundary[:,1], 'k')
        ax.set_xlim([min(boundary[:,0]), max(boundary[:,0])])
        ax.set_ylim([min(boundary[:,1]), max(boundary[:,1])])

    def plot_path(self, ax, color='k--', path=None):
        if path is None:
            this_path = self.path
        else:
            this_path = path
        for i in range(len(this_path)-1):
            ax.plot([this_path[i][0], this_path[i+1][0]], [this_path[i][1], this_path[i+1][1]], color)

class MovingObject():
    def __init__(self, current_position, stagger=0):
        self.stagger = stagger
        self.traj = [current_position]

    @staticmethod
    def motion_model(ts, state, action):
        x,y = state[0], state[1]
        vx, vy = action[0], action[1]
        x += ts*vx
        y += ts*vy
        return (x,y)

    def one_step(self, ts, action):
        self.traj.append(self.motion_model(ts, self.traj[-1], action))

    def run(self, path, ts=.2, vmax=0.5, dyn_obs_path=[(0,0)]):
        coming_path = path[1:]
        cnt = 0
        while(len(coming_path)>0):
            cnt += 1
            try:
                dyn_obs_pos = dyn_obs_path[cnt]
            except:
                dyn_obs_pos = dyn_obs_path[-1]

            stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
            x, y = self.traj[-1][0], self.traj[-1][1]
            if ((y>4) & (dyn_obs_pos[1]>4.5)): # hard constraint from the dynamic obstacle
                self.one_step(ts, (0,0))
                continue
            dist_to_next_goal = math.hypot(coming_path[0][0]-x, coming_path[0][1]-y)
            if dist_to_next_goal < (vmax*ts):
                coming_path.pop(0)
                continue
            else:
                dire = ((coming_path[0][0]-x)/dist_to_next_goal, (coming_path[0][1]-y)/dist_to_next_goal)
                action = (dire[0]*math.sqrt(vmax)+stagger, dire[1]*math.sqrt(vmax)+stagger)
                self.one_step(ts, action)


if __name__ == '__main__':

    ts = 0.2

    map_idx  = 2
    path_idx = 3
    block = True
    interact = False

    plot_dyn = False

    stagger = 0.2
    vmax = 1

    graph = Graph(map=map_idx, block=block)
    path  = graph.get_path(path_idx)
    if interact:
        dyn_obs_path = graph.get_obs_path(ts)
    else:
        dyn_obs_path = [(-1,-1)]

    obj = MovingObject(path[0], stagger)
    obj.run(path, ts, vmax, dyn_obs_path=dyn_obs_path)
    traj = obj.traj

    if plot_dyn:
        fig, ax = plt.subplots()
        for i, pos in enumerate(traj):
            ax.cla()
            graph.plot_map(ax, clean=False, empty=False)
            plt.plot(pos[0], pos[1], 'rx')
            try:
                plt.plot(dyn_obs_path[i][0], dyn_obs_path[i][1], 'go')
            except:
                plt.plot(dyn_obs_path[-1][0], dyn_obs_path[-1][1], 'go')
            ax.axis('off')
            ax.set_aspect('equal', 'box')
            ax.set_ylim([0,10])
            plt.pause(0.1)
        plt.show()

    else:
        fig, ax = plt.subplots()
        # ------------------------
        ax.axis('off')
        graph.plot_map(ax, clean=True, empty=False)
        graph.plot_path(ax, color='rx--')
        graph.plot_path(ax, color='go--', path=dyn_obs_path)
        ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'k.')
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        # ------------------------
        plt.show()