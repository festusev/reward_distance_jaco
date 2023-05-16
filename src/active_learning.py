#! /usr/bin/env python
"""
This node moves the Jaco using a specified controller, tracking a trajectory
given by a specified planner.

Given a start, a goal, and specific planner and controller parameters, the
planner plans a path from start to goal, while the controller moves the Jaco
manipulator along the path.

Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu)
Based on: https://w3.cs.jmu.edu/spragunr/CS354_S15/labs/pid_lab/pid_lab.shtml
"""

import roslib; roslib.load_manifest('kinova_demo')

import rospy
import math
import sys, select, os
import time

from utils import ros_utils, generate_traj_set
from utils.environment import Environment
from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner

import kinova_msgs.msg
from kinova_msgs.srv import *

import numpy as np

class ActiveLearner(object):
	"""
	This class represents a node that computes an optimal path and moves the Jaco along.

	Subscribes to:
		/$prefix$/out/joint_angles	- Jaco sensed joint angles

	Publishes to:
		/$prefix$/in/joint_velocity	- Jaco commanded joint velocities
	"""

	def __init__(self):
		# Create ROS node.
		rospy.init_node("active_learner")

		np.random.seed(0)

		# Load parameters and set up subscribers/publishers.
		self.load_parameters()
		self.register_callbacks()

		# Publish to ROS at 100hz.
		r = rospy.Rate(100)

		results = []
		past_queries = np.zeros((0, 2, len(self.feat_list)))  # 2 is because K=2
		for query_no in range(self.num_questions):

			print "----------------------------------"
			print "Moving robot, press ENTER to quit:"

			print("Query:", query_no)
			samples = self.sample(past_queries)
			results[-1].append(self.metrics(samples, self.weights, self.eval_query_set))
			vals = self.mutual_information(self.query_set, samples)
			best_query_id = np.argmax(vals)

			for i, traj in enumerate(self.query_set[best_query_id]):
				print "----------------------------------"
				print "Query", i, ": Moving robot, press ENTER to quit:"
				self.controller.set_trajectory(traj)
				while not rospy.is_shutdown():

					if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
						line = raw_input()
						break

					self.vel_pub.publish(ros_utils.cmd_to_JointVelocityMsg(self.cmd))
					r.sleep()
				print "----------------------------------"

			pref = int(input("Which trajectory do you prefer (0 or 1)?"))

			if pref == 0:
				past_queries = np.vstack((past_queries, np.expand_dims(self.query_set[best_query_id], 0)))
			else:
				past_queries = np.vstack((past_queries, np.expand_dims(self.query_set[best_query_id], 0)))

			samples = self.sample(past_queries)
			results[-1].append(self.metrics(samples, self.weights, self.eval_query_set))

	def load_parameters(self):
		"""
		Loading parameters and setting up variables from the ROS environment.
		"""

		# ----- General Setup ----- #
		self.prefix = rospy.get_param("setup/prefix")
		self.T = rospy.get_param("setup/T")
		self.timestep = rospy.get_param("setup/timestep")
		self.feat_list = rospy.get_param("setup/feat_list")
		self.weights = rospy.get_param("setup/feat_weights")
		self.num_questions = rospy.get_param("setup/num_questions")
		self.MH_BURNIN = rospy.get_param("setup/mh_burnin")
		self.MH_THIN = rospy.get_param("setup/mh_thin")
		self.NUM_SAMPLES = rospy.get_param("setup/num_samples")
		self.NUM_TRAJECTORIES = rospy.get_param("setup/num_trajectories")

		# Openrave parameters for the environment.
		model_filename = rospy.get_param("setup/model_filename")
		object_centers = rospy.get_param("setup/object_centers")
		self.environment = Environment(model_filename, object_centers)

		# ----- Planner Setup ----- #
		# Retrieve the planner specific parameters.
		planner_type = rospy.get_param("planner/type")
		if planner_type == "trajopt":
			max_iter = rospy.get_param("planner/max_iter")
			num_waypts = rospy.get_param("planner/num_waypts")

			# Initialize planner and compute trajectory to track.
			self.planner = TrajoptPlanner(self.feat_list, max_iter, num_waypts, self.environment)
		else:
			raise Exception('Planner {} not implemented.'.format(planner_type))
		
		# self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.weights, self.T, self.timestep)
		#
		# # Save the intermediate target configuration.
		# self.curr_pos = None

		# ----- Controller Setup ----- #
		# Retrieve controller specific parameters.
		controller_type = rospy.get_param("controller/type")
		if controller_type == "pid":
			# P, I, D gains.
			P = rospy.get_param("controller/p_gain") * np.eye(7)
			I = rospy.get_param("controller/i_gain") * np.eye(7)
			D = rospy.get_param("controller/d_gain") * np.eye(7)

			# Stores proximity threshold.
			epsilon = rospy.get_param("controller/epsilon")

			# Stores maximum COMMANDED joint torques.
			MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)

			self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
		else:
			raise Exception('Controller {} not implemented.'.format(controller_type))

		full_traj_set = np.load('data/traj_sets/traj_rand.p')
		query_set = []
		for i in range(self.NUM_TRAJECTORIES):
			for j in range(i + 1, self.NUM_TRAJECTORIES):
				query_set.append(np.vstack((full_traj_set[i], full_traj_set[j])))
		self.query_set = np.array(query_set)

		eval_traj_set = np.load('data/traj_sets/traj_rand.p')
		eval_query_set = []
		for i in range(self.NUM_TRAJECTORIES):
			for j in range(i + 1, self.NUM_TRAJECTORIES):
				eval_query_set.append(np.vstack((eval_traj_set[i], eval_traj_set[j])))
		self.eval_query_set = np.array(eval_query_set)

		# Planner tells controller what plan to follow.
		# self.controller.set_trajectory(self.traj)

		# Stores current COMMANDED joint torques.
		self.cmd = np.eye(7)

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""

		# Create joint-velocity publisher.
		self.vel_pub = rospy.Publisher(self.prefix + '/in/joint_velocity', kinova_msgs.msg.JointVelocity, queue_size=1)

		# Create subscriber to joint_angles.
		rospy.Subscriber(self.prefix + '/out/joint_angles', kinova_msgs.msg.JointAngles, self.joint_angles_callback, queue_size=1)
	
	def joint_angles_callback(self, msg):
		"""
		Reads the latest position of the robot and publishes an
		appropriate torque command to move the robot to the target.
		"""

		# Read the current joint angles from the robot.
		self.curr_pos = np.array([msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6,msg.joint7]).reshape((7,1))

		# Convert to radians.
		self.curr_pos = self.curr_pos*(math.pi/180.0)

		# Update cmd from PID based on current position.
		self.cmd = self.controller.get_command(self.curr_pos)

	def mutual_information(self, query_set, samples):
		if len(query_set.shape) == 2:
			query_set = np.expand_dims(query_set, 0)
		# query_set is NUM_QUERIES x K x d
		probs = ssp.softmax(query_set @ samples.T, axis=1)
		return np.sum(probs * np.log2(self.NUM_SAMPLES * probs / np.sum(probs, axis=2, keepdims=True)),
					  axis=(1, 2)) / self.NUM_SAMPLES

	def logposterior(self, w, past_queries):
		if len(past_queries.shape) == 2:
			past_queries = np.expand_dims(past_queries, 0)
		# past_queries is NUM_QUERIES x K x d

		# logprior = sst.multivariate_normal.logpdf(w, mean=np.zeros_like(w)) # we assume a normal distribution as the prior
		logprior = -np.inf if np.linalg.norm(w) >= 15 else 0.
		if past_queries.shape[0] == 0:
			return logprior
		loglikelihood = np.sum(ssp.log_softmax(past_queries @ w, axis=1)[:, 0])
		return logprior + loglikelihood

	def sample(self, past_queries):
		# samples = [np.random.randn(OBSERVED_D)]
		samples = [np.zeros(len(self.feat_list))]
		logposteriors = [self.logposterior(samples[0], past_queries)]
		for _ in range(1, self.MH_BURNIN + self.MH_THIN * self.NUM_SAMPLES):
			new_sample = samples[-1] + 0.05 * np.random.randn(len(self.feat_list))
			new_logposterior = self.logposterior(new_sample, past_queries)
			if np.log(np.random.rand()) >= new_logposterior - logposteriors[-1]:
				new_sample = samples[-1]
				new_logposterior = logposteriors[-1]
			samples.append(new_sample)
			logposteriors.append(new_logposterior)
		return np.array(samples[self.MH_BURNIN::self.MH_THIN])

	def metrics(self, samples, true_reward, eval_query_set):
		eval_query_set = np.array(eval_query_set)
		rews = oe.contract("ijk,k->ij", eval_query_set, true_reward)
		argmax_rews = np.argmax(rews, axis=1)
		pred_rews = oe.contract("ijk,fk->fij", eval_query_set, samples)
		pred_rews = ssp.softmax(pred_rews, axis=-1)
		return np.mean(np.log(pred_rews[:, np.arange(pred_rews.shape[1]), argmax_rews]))


if __name__ == '__main__':
	PathFollower()
