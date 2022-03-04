#!/usr/bin/env python
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.draw import circle
from scipy.linalg import block_diag

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_settings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_settings_filename)

        #Get the metric bounds of the map, in real space (inertial_frame)
        #Origin is the location of the botton left corner of the image in real space
        #The robot is located at [0,0] in real space, so [420, 985] in pixels (image space)
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[1] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.26 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 5.0 #s Note: was intially 1 s
        self.num_substeps = 10 #Note: was intially 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        self.duplicates = [np.array([[421],[985]])]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 1

        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        #The real bounds are at [[-21.  ,  59.  ], [-49.25,  30.75]]
        # The bounds of the house are at pixels length:  380 to 1300 height: 400 to 1570, or 30 to 1200
        #In meters this is 18 to 65 and 1.5 to 60
        # Origin is at -21.0, -49.25,
        #So this translates to -3.5 to 43.5 and -47.5 to 10.5 in the real frame
        new_bounds = np.array([[-3.5, 43.5],[-49.25, 10.5]])
        #new_bounds = np.copy(self.bounds)

        pt = np.random.rand(1,2) #Random numbers from 0, 1
        pt = np.multiply(pt, np.abs(new_bounds[:, 0] - new_bounds[:, 1])) + new_bounds[:,0]
        return pt.T

    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        for n in self.duplicates:
            if np.array_equal(n, point):
                return True
        return False

    def closest_node(self, point):
        #Returns the index of the closest node
        x = lambda node : np.linalg.norm(point-node.point[0:2,:])
        dist = np.array(map(x, self.nodes))
        idx = np.argmin(dist)
        min_dist = dist[idx]

        return idx, min_dist

    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)

        #Convert from robot frame to inertial frame
        R = np.array([[np.cos(node_i[2,0]), -np.sin(node_i[2,0]), 0],
                      [np.sin(node_i[2,0]), np.cos(node_i[2,0]), 0],
                      [0, 0, 1]])
        #robot_traj = np.vstack((robot_traj, np.ones((1, robot_traj.shape[1])))) #Homogeneous coordinates
        inert_traj = node_i + np.matmul(R, robot_traj)

        return inert_traj

    def robot_controller(self, node_i, point_s, lin_vel = 0):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        #Assume max linear velocity by default
        if lin_vel == 0:
            lin_vel = self.vel_max
        # angle between heading and connecting line segment (between start and goal)
        y =point_s[1] - node_i[1]
        x =point_s[0] - node_i[0]
        phi = (np.arctan2(y, x) - node_i[2])
        phi = (phi + np.pi) % (2 * np.pi) - np.pi #Convert angle to range -pi to pi
        #phi = np.pi / 2 - np.arctan2(point_s[1]-node_i[1], node_i[0]-point_s[0]) - node_i[2]

        #If goal is directly ahead or behind the robot
        if phi == 0:
            return lin_vel, 0
        if phi == np.pi:
            return -lin_vel, 0

        if phi > np.pi/2 or phi < -np.pi/2:
            lin_vel = -lin_vel

        # distance between starting and goal position
        r = np.sqrt(np.square(node_i[0]-point_s[0]) + np.square(node_i[1]-point_s[1]))

        # arc length
        L = (r * phi / np.sin(phi))[0]
        # Time to get to destination
        t = L / lin_vel

        rot_vel = 2 * phi / t

        # Limiting the rotational velocity - might need to be changed
        if rot_vel > self.rot_vel_max:
            rot_vel = self.rot_vel_max
        if rot_vel < - self.rot_vel_max:
            rot_vel = - self.rot_vel_max

        return lin_vel, rot_vel

    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        trajectory = np.zeros((3, self.num_substeps+1))
        sub_step = self.timestep / self.num_substeps
        p = np.array([[vel, rot_vel]]).T
        for t in range(0, self.num_substeps):
            G = np.array([[np.cos(trajectory[2, t]), 0],
                         [np.sin(trajectory[2, t]), 0],
                         [0, 1]])
            q_dot = np.matmul(G,p)
            trajectory[:, t+1] = trajectory[:, t] + (q_dot*sub_step).T
        return trajectory

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell (i.e. pixel) in the occupancy map
        #point is a 2 by N matrix of points of interest

        # Origin is at -21.0, -49.25
        #Get the location of the botton left corner of the image in real space
        origin = np.array([self.map_settings_dict['origin'][0:2]]).T
        #Convert map point to pixel
        cell = (point - origin)/self.map_settings_dict['resolution']
        cell = np.rint(cell).astype(int)
        return cell

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The circle function is included to help you with this function
        # Points is a 2xN vector
        rob_locations = self.point_to_cell(points).T
        rob_radius = (self.robot_radius)/self.map_settings_dict['resolution'] #Get the robot radius in pixel form
        rob_radius = np.int(np.ceil(rob_radius))
        rr = [[] for x in rob_locations]
        cc = [[] for x in rob_locations]
        for idx, center in enumerate(rob_locations):
            rr[idx], cc[idx] = circle(center[0], center[1], rob_radius)
        return rr, cc
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)

    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #node is a 3 by 1 node
        #point is a 2 by 1 point

        #Get the vel and rot vel to get between the two points
        lin_vels = [0.25,0.2,0.15,0.1,0.05] #Try a variety of linear velocities
        for i, lin_vel in enumerate(lin_vels):
            #Calculate linear and angular velocity
            phi = (np.arctan2(point_f[1] - node_i[1], point_f[0] - node_i[0]) - node_i[2])
            phi = (phi + np.pi) % (2 * np.pi) - np.pi #Convert angle to range -pi to pi

            if phi > np.pi/2 or phi < -np.pi/2:
                lin_vel = -lin_vel
            if phi == 0 or phi == np.pi:
                rot_vel = 0

            # distance between starting and goal position
            r = np.sqrt(np.square(node_i[0]-point_f[0]) + np.square(node_i[1]-point_f[1]))
            # arc length
            L = (r * phi / np.sin(phi))[0]
            # Time to get to destination
            t = L / lin_vel

            rot_vel = 2 * phi / t
            if abs(rot_vel) > self.rot_vel_max:
                continue

            #Generate the trajectory
            trajectory = np.zeros((3, self.num_substeps+1))
            trajectory[:,0] = node_i[0:3].T
            sub_step = t / self.num_substeps
            p = np.array([[lin_vel, rot_vel]]).T
            for t in range(0, self.num_substeps):
                G = np.array([[np.cos(trajectory[2, t]), 0],
                            [np.sin(trajectory[2, t]), 0],
                            [0, 1]])
            q_dot = np.matmul(G,p)
            trajectory[:, t+1] = trajectory[:, t] + (q_dot*sub_step).T

            #Check for collisions
            rr, cc = self.points_to_robot_circle(trajectory[0:2,:]) #Get list of pixels the robot will pass through
            path = np.vstack([np.concatenate(rr), np.concatenate(cc)]) # Merging rr and cc into 2D array (columns are pixel positions)
            path = np.unique(path, axis=1) #Remove duplicate pixels
            path = np.flip(path, axis=0) #Convert to index the occupancy map (reverse x & y for rows & col)
            path[0,:] = 1600 - path[0,:] #Index from top left as opposed to bottom left corner
            path = np.clip(path, 0, 1599)
            not_collision = self.occupancy_map[tuple(path)] #Index occupany map at this position
            if np.sum(not_collision) < path.shape[1]:
                continue
            return trajectory

        #No collison free or valid rotational velocities were found
        return np.ones((3, self.num_substeps+1))*np.NaN

    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        dist = np.linalg.norm(trajectory_o[0] - trajectory_o[-1])
        # print("TO DO: Implement a cost to come metric")
        return dist

    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        children = self.nodes[node_id].children_ids
        for child in children:
            traj = self.connect_node_to_point(self.nodes[node_id].point, self.nodes[child].point[0:2])
            self.nodes[child].cost = self.nodes[node_id].cost + self.cost_to_come(traj)
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        # actual_points = [np.zeros((2,1)),]
        sim_trajs = []
        max_dist = self.vel_max*self.timestep
        min_dist = self.vel_max*self.timestep*0.2
        n = 20000
        for i in range(n):
            #Sample map space
            point = self.sample_map_space()

            if i >= n-1:
                print('Goal not reached, sampling at goal')
                point = self.goal_point

            #Get the closest point
            closest_node_id, dist = self.closest_node(point)
            closest_node = self.nodes[closest_node_id].point

            #Modify the distance
            if dist < min_dist:
                continue
            if dist > max_dist:
                v = point - closest_node[0:2,:]
                point = closest_node[0:2,:] + (v/np.linalg.norm(v))*max_dist

            #Simulate driving the robot towards the closest point (note: you don't actualy have to reach point s)
            traj = self.simulate_trajectory(closest_node, point)

            #Check if duplicate
            final_pix = self.point_to_cell(traj[0:2,-1][np.newaxis].T)
            if self.check_if_duplicate(final_pix):
                continue

            #Check for collisions
            rr, cc = self.points_to_robot_circle(traj[0:2,:]) #Get list of pixels the robot will pass through
            path = np.vstack([np.concatenate(rr), np.concatenate(cc)]) # Merging rr and cc into 2D array (columns are pixel positions)
            path = np.unique(path, axis=1) #Remove duplicate pixels
            path = np.flip(path, axis=0) #Convert to index the occupancy map (reverse x & y for rows & col)
            path[0,:] = 1600 - path[0,:] #Index from top left as opposed to bottom left corner
            path = np.clip(path, 0, 1599)
            is_collision = self.occupancy_map[tuple(path)] #Index occupany map at this position
            if np.sum(is_collision) < path.shape[1]:
                continue

            #Add to list of nodes
            # actual_points += [point]
            self.duplicates += [final_pix]
            self.nodes += [Node(traj[:,-1][np.newaxis].T, closest_node_id, 0)] #May need to change cost later
            self.nodes[closest_node_id].children_ids += [len(self.nodes)-1]
            sim_trajs += [self.point_to_cell(traj[0:2,:])]

            #Check if goal has been reached
            goal_dist = np.linalg.norm(traj[0:2,-1] - self.goal_point.T)
            if goal_dist < self.stopping_dist:
                print('Goal Reached')
                break


        #Plotting to check
        plt.imshow(self.occupancy_map, cmap = 'Greys_r', interpolation = 'nearest')
        goal_cell = self.point_to_cell(self.goal_point)
        plt.plot(goal_cell[0], 1600 - goal_cell[1], marker='o', color = 'green')
        for i in range(0, len(self.nodes)):
            node = self.nodes[i]
            pt = self.point_to_cell(node.point[0:2])
            if i != 0:
                p_pt = self.point_to_cell(self.nodes[node.parent_id].point[0:2])
                plt.plot(sim_trajs[i-1][0,:],1600 - sim_trajs[i-1][1,:], color = 'blue')

            plt.plot(pt[0], 1600-pt[1], marker = 'x', color = 'red')
        #Plot path to goal
        path = self.recover_path()
        for i in range(1,len(path)):
            prev_pix = self.point_to_cell(path[i-1][0:2])
            curr_pix = self.point_to_cell(path[i][0:2])
            plt.plot([prev_pix[0], curr_pix[0]],[1600 - prev_pix[1],1600 - curr_pix[1]], color = 'green')
        plt.show()

        return self.nodes

    def collision_check(self, traj):
        rr, cc = self.points_to_robot_circle(traj[0:2,:]) #Get list of pixels the robot will pass through
        path = np.vstack([np.concatenate(rr), np.concatenate(cc)]) # Merging rr and cc into 2D array (columns are pixel positions)
        path = np.unique(path, axis=1) #Remove duplicate pixels
        path = np.flip(path, axis=0) #Convert to index the occupancy map (reverse x & y for rows & col)
        path[0,:] = 1600 - path[0,:] #Index from top left as opposed to bottom left corner
        path = np.clip(path, 0, 1599)
        is_collision = self.occupancy_map[tuple(path)] #Index occupany map at this position
        if np.sum(is_collision) < path.shape[1]:
            return True
        return False


    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        sim_trajs = []
        max_dist = self.vel_max*self.timestep
        min_dist = self.vel_max*self.timestep*0.2
        n = 20000
        for i in range(n): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            if i >= n-1:
                print('Goal not reached, sampling at goal')
                point = self.goal_point

            #Get the closest point
            closest_node_id, dist = self.closest_node(point)
            closest_node = self.nodes[closest_node_id].point

            #Modify the distance
            if dist < min_dist:
                continue
            if dist > max_dist:
                v = point - closest_node[0:2,:]
                point = closest_node[0:2,:] + (v/np.linalg.norm(v))*max_dist

            #Simulate driving the robot towards the closest point (note: you don't actualy have to reach point s)
            traj = self.simulate_trajectory(closest_node, point)

            #Check if duplicate
            final_pix = self.point_to_cell(traj[0:2,-1][np.newaxis].T)
            if self.check_if_duplicate(final_pix):
                continue

            #Check for collisions
            if self.collision_check(traj):
                continue

            #Add to list of nodes
            self.duplicates += [final_pix]
            final_point = traj[0:2,-1][np.newaxis].T
            cost =  self.cost_to_come(traj) + self.nodes[closest_node_id].cost
            self.nodes += [Node(traj[:,-1][np.newaxis].T, closest_node_id, cost)]
            self.nodes[closest_node_id].children_ids += [len(self.nodes)-1]
            sim_trajs += [self.point_to_cell(traj[0:2,:])]

            #Last (most recently sampled) node rewire
            x = lambda node : np.linalg.norm(final_point-node.point[0:2,:])
            all_dist = np.array(map(x, self.nodes[:-1])) #Make sure not to compare with last node added
            close_idx = np.where(all_dist < self.epsilon)[0]
            best_ctc = np.inf
            best_id = closest_node_id
            for id in close_idx:
                new_traj = self.connect_node_to_point(self.nodes[id].point, final_point)
                if np.isnan(new_traj).any():
                    continue
                curr_ctc = self.cost_to_come(new_traj) + self.nodes[id].cost
                if curr_ctc < best_ctc:
                    best_ctc = curr_ctc
                    best_id = id

            #Change parent/children/cost of most recent node if shorter path found
            if best_id != closest_node_id:
                self.nodes[-1].parent_id = best_id #Change parent id
                self.nodes[-1].cost = best_ctc #change cost
                self.nodes[best_id].children_ids += [len(self.nodes)-1] #Add to children of new parent
                self.nodes[closest_node_id].children_ids.remove(len(self.nodes)-1) #Remove from children of old parent

            #Close node rewire (rewire other nodes that are close to the sampled node)
            for id in close_idx:
                new_traj = self.connect_node_to_point(self.nodes[-1].point, self.nodes[id].point[0:2])
                if np.isnan(new_traj).any():
                    continue
                new_poss_ctc = self.cost_to_come(new_traj) + self.nodes[-1].cost
                if new_poss_ctc  < self.nodes[id].cost: #Rewire
                    self.nodes[self.nodes[id].parent_id].children_ids.remove(id) #Note: this line must be first. Remove from children of old parent
                    self.nodes[id].parent_id = len(self.nodes)-1 #Change parent id
                    self.nodes[id].cost = new_poss_ctc #change cost
                    self.nodes[-1].children_ids += [id] #Add to children of recently sampled node
                    self.update_children(id)

            #Check for early end
            goal_dist = np.linalg.norm(final_point - self.goal_point.T)
            if goal_dist < self.stopping_dist:
                print('Goal Reached')
                break

        #Plotting to check
        plt.imshow(self.occupancy_map, cmap = 'Greys_r', interpolation = 'nearest')
        goal_cell = self.point_to_cell(self.goal_point)
        plt.plot(goal_cell[0], 1600 - goal_cell[1], marker='o', color = 'green')
        for i in range(0, len(self.nodes)):
            node = self.nodes[i]
            pt = self.point_to_cell(node.point[0:2])
            if i != 0:
                p_pt = self.point_to_cell(self.nodes[node.parent_id].point[0:2])
                plt.plot(sim_trajs[i-1][0,:],1600 - sim_trajs[i-1][1,:], color = 'blue')

            plt.plot(pt[0], 1600-pt[1], marker = 'x', color = 'red')
        #Plot path to goal
        path = self.recover_path()
        print('recovered path')
        for i in range(1,len(path)):
            prev_pix = self.point_to_cell(path[i-1][0:2])
            curr_pix = self.point_to_cell(path[i][0:2])
            plt.plot([prev_pix[0], curr_pix[0]],[1600 - prev_pix[1],1600 - curr_pix[1]], color = 'green')
        plt.show()
        return self.nodes

    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        path_ids = []
        print('starting recover path')
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
            temp = any(np.array_equal(self.nodes[current_node_id].point, x) for x in path)
            if temp:
                print('current node', self.nodes[current_node_id].point)
                print('parent_node', self.nodes[self.nodes[current_node_id].parent_id].point)
                break
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
