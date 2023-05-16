import gym
from tqdm import tqdm

import time

import Environments
# from Kendalls import Kendalls
from RewardMetricAquisitionFunction import posterior_reward_distance, epic_dist, Epic
import scipy

from Rho import Rho
from Spearmans import Spearmans
import numpy as np
import itertools

import pickle
import os
import threading

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from datetime import datetime

import pdb
import threading
import copy

import scipy.spatial.distance as ssd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

GYM_ENV = 'Reacher-v4'
STATE_SIZE = 13
ACTION_SIZE = 2
HORIZON = 10
BETA = 100/HORIZON # human rationality
PPO_TIMESTEPS = 100000
NUM_QUESTIONS = 50
NUM_RUNS = 5
SEED_OFFSETS = np.arange(0, 100, NUM_RUNS)#[0 ]#[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95] #, 80] #np.arange(70, 100, NUM_RUNS)

METHODS = ["euclidean", "spearmans", "rho", "epic", "mutual_information", "random"]
RUNNING = [0, 1, 2, 3, 4, 5] # the methods that we are running

NMEAN = np.array([ 9.19900276e-01,  9.18585219e-01,  2.54401121e-03,  2.30060253e-03,
  6.21498053e-04, -1.34873415e-03,  2.46193256e-02,  1.83058350e-02,
  1.85396766e-01,  2.11823317e-03,  0.00000000e+00, -2.10466828e-01,
 -6.07258522e-01,  5.46042969e-04, -2.32254724e-04])

NSTD = np.array([0.09355685, 0.09489204, 0.27907083, 0.2809547,  0.09966712, 0.09999249,
 3.81455532, 3.84586811, 0.10253977, 0.1172587,  0,         0.08290544,
 0.12130407, 0.16656524, 0.16829246])

TRUE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 1, 1])
# NREWARD = np.array([0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0.08290544, 0.12130407, 0, 0])
NREWARD = TRUE_WEIGHTS*NSTD

# NMEAN = np.array([0]*15)
# NSTD = np.array([1]*15)
# NREWARD = np.array([0]*11 + [1, 1, 0, 0])

import time
class BasicActiveLearningExperiment:
    def __init__(self, env, aprel_env, feature_dim, aq_func, bs, traj_set=None):
        """aquisition lambda """
        self.env = env
        self.aprel_env = aprel_env
        self.aq_func = aq_func
        self.bs = bs

        # print("TRAJ SET FEATURES:", self.traj_set.features_matrix)
        # print("SELF TRAJ_SET FEATURE MATRIX:", self.traj_set.features_matrix)
        self.traj_set = traj_set

        self.feature_dim = feature_dim
        self.query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(self.traj_set)
        # print("QUERY SLATE:", self.query_optimizer.slate)
        params = {"weights":[0]*feature_dim}
        self.belief = aprel.SamplingBasedBelief(aprel.SoftmaxUser(params), [], params)
        print("START WEIGHTS:", self.belief.mean["weights"])
        self.dummy_query = aprel.PreferenceQuery(self.traj_set[:2])

    def get_query(self):


    def run_rounds_active_learning(self, num_rounds, eval_functions=None, verbose=False):
        metrics = {"traj_set": self.traj_set, "user_correct":[], "queries": []} # user_correct is a list of 1 or 0, 1 being the user correctly responded, 0 otherwise

        if eval_functions:
            for key, func in eval_functions.items():
                metrics[key] = []

        t1 = time.time()
        if eval_functions:
            for key, func in eval_functions.items():
                metrics[key].append(func(self.belief))
        # print("Eval Function time:", time.time() - t1)

        for round in range(num_rounds):
            if verbose:
                print("---ROUND", round, "---")

            # use_our_greedy_batch = False
            if isinstance(self.aq_func, str):
                if self.aq_func != "random":
                    aq_func = self.query_optimizer.acquisition_functions[self.aq_func]
                else:
                    aq_func = "random" #self.aq_func
            else:
                t1 = time.time()
                samples = []
                for sample in self.belief.samples:
                    samples.append(sample["weights"])

                dist = self.aq_func(np.array(samples))
                # print("Calculating distances time:", time.time() - t1, "Distance Function:", self.aq_func)

                aq_func = lambda belief,query: -posterior_reward_distance(belief, query, dist)
            query, obj = self.greedy_batch(self.query_optimizer, aq_func,
                                           self.belief, self.dummy_query,
                                           batch_size=1)  # self.query_optimizer.optimize(lambda x,y:posterior_reward_distance(x,y,self.aq_func,None),self.belief,self.dummy_query,self.bs)
            # print("Posterior:", obj)
            resp = boltzman_choice(query[0].slate[0].reward, query[0].slate[1].reward)
            if query[0].slate[resp[0]].reward >= query[0].slate[1 - resp[0]].reward:
                metrics["user_correct"].append(1)
            else:
                metrics["user_correct"].append(0)

            metrics["queries"].append([query[0].slate[0], query[0].slate[1]])

            #print(">>>>>>>> ROUND", round)
            #print("Reward 0:", query[0].slate[0].reward, "Reward 1:", query[0].slate[1].reward, "Response:", resp)
            #print("Belief before:", self.belief.mean)

            #self.belief.update([aprel.Preference(q, response)for q, response in zip(query, resp)])
            initial_sampling_param = {"weights": [0] * self.feature_dim}
            self.belief.update([aprel.Preference(q, response) for q, response in zip(query, resp)],
                               initial_point=initial_sampling_param)

            # print("Belief after:", self.belief.samples[0])
            # print("query:", query.slate)
            if verbose:
                print("Rewards:", query[0].slate[0].reward, query[0].slate[1].reward)
                print("resp:", resp)
                print("belief:", self.belief.mean)

            if eval_functions:
                for key, func in eval_functions.items():
                    metric = func(self.belief)

                    if verbose:
                        print(key, ":", metric)

                    metrics[key].append(metric)

        return metrics

    def greedy_batch(_, self,
                     acquisition_func,
                     belief,
                     initial_query,
                     batch_size,
                     **kwargs):
        """
        Uses the greedy method to find a batch of queries by selecting the :py:attr:`batch_size` individually most optimal queries.

        Args:
            acquisition_func (Callable): the acquisition function to be maximized by each individual query.
            belief (Belief): the current belief distribution over the user.
            initial_query (Query): an initial query such that the output query will have the same type.
            batch_size (int): the batch size of the output.
            **kwargs: extra arguments needed for specific acquisition functions.

        Returns:
                2-tuple:

                    - List[Query]: The optimized batch of queries as a list.
                    - numpy.array: An array of floats that keep the acquisition function values corresponding to the output queries.
        """
        if acquisition_func == "random":
            slate = np.random.choice(np.arange(self.trajectory_set.size), size=2, replace=False)
            query = initial_query.copy()
            query.slate = self.trajectory_set[slate]

            return [query], None

        subsets = np.array(
            [list(tup) for tup in itertools.combinations(np.arange(self.trajectory_set.size), initial_query.K)])
        if len(subsets) < batch_size:
            batch_size = len(subsets)
            print(
                'The number of possible queries is smaller than the batch size. Automatically reducing the batch size.')
        vals = []

        t1 = time.time()
        for ids in subsets:
            curr_query = initial_query.copy()
            curr_query.slate = self.trajectory_set[ids]
            vals.append(acquisition_func(belief, curr_query, **kwargs))
        # print("Calculating vals time:", time.time() - t1)
        vals = np.array(vals)
        inds = np.argpartition(vals, -batch_size)[-batch_size:]

        best_batch = [initial_query.copy() for _ in range(batch_size)]
        for i in range(batch_size):
            best_batch[i].slate = self.trajectory_set[subsets[inds[i]]]
        return best_batch, vals[inds]

class RhoWrapper:
    def __init__(self, rho):
        self.rho = rho
        # self.exp_rew = exp_rew
        # self.uni_rew = uni_rew
        # self.true_rho = rho.rew_dist(exp_rew, uni_rew)

    def eval(self, belief):
        mean_linalg = []
        t1 = time.time()
        for sample in belief.samples:
            weights = sample["weights"]
            rho_sample = self.rho.pdist(np.array([10*NREWARD, weights]))

            mean_linalg.append(rho_sample[0][1]) #np.linalg.norm(rho_sample - self.true_rho))
        # print("RHO SAMPLE:", rho_sample, "Between weights:", np.array([NREWARD, weights]))
        # print("Rho eval time:", time.time() - t1)
        return np.mean(mean_linalg)

def boltzman_choice(rew0, rew1):
    p1 = np.exp(BETA*rew1)/(np.exp(BETA*rew0) + np.exp(BETA*rew1))
    #print("Rew 0:", rew0, "rew1:", rew1, "p1:", p1)
    return [int(np.random.rand() < p1)]

def generate_on_policy_trajs(env, model, n=10):
    trajs = TrajectorySet([])
    for i in range(n):
        obs = env.reset()
        traj = []

        done = False
        j = 0
        while not done:
            action, _states = model.predict(obs)
            traj.append((obs, action))
            obs, rewards, done, info = env.step(action)

            if done:
                traj.append((obs, action))
            j += 1

        trajs.append(Trajectory(env, traj))
    return trajs

def generate_on_policy_trajs_reward(env, model, n=10):
    rewards = []
    for i in range(n):
        obs = env.reset()

        rew = 0
        done = False
        j = 0
        while not done:
            action, _states = model.predict(obs)
            obs, r, done, info = env.step(action)
            # print("Pos:", env.pos, "Obs:", obs, "ACTION:", action, "REW:", r)

            rew += r
            j += 1

        rewards.append(rew)
    return rewards

def generate_random_trajs(env, n=10):
    trajs = TrajectorySet([])
    rewards = [[]]
    for i in range(n):
        env.reset()
        obs = env.reset_model()
        traj = []

        done = False
        rew = 0
        j = 0
        while not done:
            action = env.action_space.sample()
            traj.append((obs, action))
            obs, r, done, info = env.step(action)

            rew += r
            if done:
                traj.append((obs, None))
            j += 1

        trajs.append(Trajectory(env, traj))
        rewards[0].append(rew)
    return trajs, np.array(rewards)


'''
        self.traj_set, traj_rewards = generate_random_trajs(env, num_traj) #aprel.generate_trajectories_randomly(aprel_env, num_traj, HORIZON,
                        #                                 file_name="random_traj")

        # Add a "reward" item to each trajectory object in traj_set
        for i in range(self.traj_set.size):
            self.traj_set[i].reward = traj_rewards[0][i]'''
# def thread_experiment():
def euclidean(reward_models):
    # print("REWARD MODELS SHAPE", reward_models.shape)
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(reward_models))

def start_experiment(run, seed, verbose, metric_queue, rho_eval, aq_name, env, aprel_env, epic_eval, loglike_eval, traj_set,
                     uniform_trajectories_features, expert_trajectories_features):
    np.random.seed(seed)
    print("--- Run:", run)

    # uniform_trajectories, _ = generate_random_trajs(env, 100)
    # uniform_trajectories_features = np.vstack([traj.features for traj in uniform_trajectories])
    #
    # expert_trajectories, _ = generate_random_trajs(env, 50)
    # expert_trajectories_features = np.vstack([traj.features for traj in expert_trajectories])

    aq_function = aq_name
    if aq_name == 'rho':
        aq_function = Rho(expert_trajectories_features, uniform_trajectories_features).pdist
    elif aq_name == 'spearmans':
        aq_function = Spearmans(uniform_trajectories_features).dist
    elif aq_name == 'kendalls':
        aq_function = Kendalls(uniform_trajectories_features).fdist
    elif aq_name == 'epic':
        aq_function = Epic(env).dist
    elif aq_name == 'euclidean':
        aq_function = euclidean

    eval_function = {"rho": rho_eval, "weights": lambda belief: copy.deepcopy(belief), "ll": loglike_eval} #), "epic": epic_eval}  # , "ppo": ppo_eval}
    experiment = BasicActiveLearningExperiment(env, aprel_env, STATE_SIZE + ACTION_SIZE, aq_function,
                                               15, traj_set)

    print("Starting experiment <", run, ">")
    ts = time.time()
    metrics = experiment.run_rounds_active_learning(NUM_QUESTIONS, eval_functions=eval_function,
                                                    verbose=verbose)  ###Check size of gridworld
    metric_queue.put(metrics)

    time_taken = time.time() - ts
    print("<", run, " - ", time_taken) #, "s> LEARNED METRICS:", metrics)

    # return helper

def ppo_eval(learned_belief):
    """
    Train a PPO policy on the learned belief.
    """
    learned_weights = learned_belief.mean["weights"]
    if verbose:
        print("Learned Weights:", learned_weights)

    learned_env = Environments.ReacherWrapper(learned_weights = learned_weights, add_kernel=True, horizon=HORIZON)

    learned_policy = PPO("MlpPolicy", learned_env)
    learned_policy.learn(total_timesteps=PPO_TIMESTEPS)
    # learned_policy.save("learned_policy")
    learned_reward = np.average(generate_on_policy_trajs_reward(env, learned_policy, n=1000))

    if verbose:
        print("Learned Reward:", learned_reward)
    return learned_reward


def run_experiment(dir_name, rho_eval, aq_name, env, aprel_env, epic_eval, loglike_eval, seed_offset, trajs = {}, name=None, num_runs=1, verbose=False):

    # aq_function = scipy.spatial.distance.pdist
    # aq_function = spearmans.dist
    # aq_function = rho.pdist
    # aq_function = kendalls.dist

    metric_queue = mp.Queue()
    procs = []
    for run in range(num_runs):
        # thread_experiment(experiment, eval_function, verbose, cumulative_metrics, run)()
        proc = mp.Process(target=start_experiment, args=(run, seed_offset + run, verbose, metric_queue, rho_eval, aq_name,
                                                         env, aprel_env, epic_eval, loglike_eval,
                                                         trajs[run]["query_trajs"], trajs[run]["uniform"], trajs[run]["expert"]))
        proc.start()
        procs.append(proc)

    cumulative_metrics = []
    for _ in procs:
        cumulative_metrics.append(metric_queue.get())


    print("METHOD:", name, "\n",[metric["ll"] for metric in cumulative_metrics])
    run_descriptor = [cumulative_metrics, {"STATE_SIZE": STATE_SIZE, "ACTION_SIZE": ACTION_SIZE,
                                           "num_runs": num_runs, "HORIZON": HORIZON, "BETA": BETA, "GYM_ENV": GYM_ENV,
                                           "PPO_TIMESTEPS": PPO_TIMESTEPS, "START_SEED": seed_offset}]
    with open(dir_name + "/" + name + "_experiment.pickle", "wb") as fp:  # Pickling
        pickle.dump(run_descriptor, fp)

def get_trajs(start_seed, runs, env, num_query_trajs=100, num_expert=50, num_uniform=100):
    trajs = []

    t1 = time.time()
    for i in range(runs):
        np.random.seed(i + start_seed) # Seed the generation
        query_trajs, rewards = generate_random_trajs(env, num_query_trajs)
        for i in range(query_trajs.size):
            query_trajs[i].reward = rewards[0][i]

        uniform_trajectories, _ = generate_random_trajs(env, num_uniform)
        uniform_trajectories_features = np.vstack([traj.features for traj in uniform_trajectories])

        expert_trajectories, _ = generate_random_trajs(env, num_expert)
        expert_trajectories_features = np.vstack([traj.features for traj in expert_trajectories])
        # t2 = time.time()
        # expert_superset, _ = generate_random_trajs(env, num_expert*1000)
        # expert_trajectories = LLEval.pds_memory_efficient(expert_superset, num_expert)
        # expert_trajectories_features = np.vstack([traj.features for traj in expert_trajectories])
        # print("PDS Sampling time:", time.time() - t2)

        trajs.append({"query_trajs": query_trajs, "uniform": uniform_trajectories_features, "expert": expert_trajectories_features})
    print("Get trajs time:", time.time() - t1)
    return trajs

class LLEval:
    def __init__(self, val):
        self.val = val

    def ll_eval(self, belief):
        local_loglike = []  # List of log likelihoods, one for each pair in "val", using the current run's weights
        local_accuracy = []

        for sample in belief.samples:
            for j in range(self.val.size - 1):
                query = sorted([self.val[j], self.val[j + 1]], key=lambda x: x.reward)  # Sort the queries based on their rewards

                rew0 = np.dot(query[0].features, sample["weights"])
                rew1 = np.dot(query[1].features, sample["weights"])

                numerator = np.exp(rew1)
                denominator = np.exp(rew0) + np.exp(rew1)
                like = np.log(numerator / denominator)
                local_loglike.append(like)
                local_accuracy.append(int(rew1 > rew0))

        # print("LOGLIKE:", np.mean(local_loglike))
        return np.mean(local_loglike), np.mean(local_accuracy)

    def pds_memory_efficient(trajectories: TrajectorySet, num_samples: int):
        distances = ssd.squareform(
            ssd.pdist(trajectories.features_matrix[:1000], 'euclidean'))  # only for max_threshold estimation

        min_threshold = 0
        max_threshold = 2 * np.max(
            distances)  # TODO: this is a very rough approximation. Is there a more efficient way?
        sensitivity = np.min([distance for distance in distances.flatten() if distance > 0.])
        while max_threshold - min_threshold > sensitivity:
            threshold = (max_threshold + min_threshold) / 2
            not_eliminated = np.arange(trajectories.size)
            sampled = []
            while not_eliminated.size > 0:
                sampled.append(np.random.choice(not_eliminated))
                dists = ssd.cdist(trajectories[int(sampled[-1])].features.reshape((1, -1)),
                                  trajectories.features_matrix[not_eliminated], 'euclidean')
                not_eliminated = not_eliminated[dists[0] >= threshold]
            if len(sampled) < num_samples:
                max_threshold = threshold
            elif len(sampled) > num_samples:
                min_threshold = threshold
            else:
                break
        return TrajectorySet([trajectories[int(i)] for i in sampled])

def main(seed_offset):
    #pdb.set_trace()
    env = Environments.ReacherWrapper(add_kernel = True, horizon=HORIZON, nmean=NMEAN, nstd=NSTD,
                                      learned_weights=TRUE_WEIGHTS)
    # env = Environments.RareFeaturesGridworldNavigationEnvironment(gridworld_size, n_colors, weights, horizon, map)

    # ntrajs, _ = generate_random_trajs(env, 10000)
    # nfeats = np.vstack([traj.features for traj in ntrajs])
    # print("NMEAN:", np.mean(nfeats, axis=0), "STD:", np.std(nfeats, axis=0))
    # (expert_trajectories_metric_features - np.mean(expert_trajectories_metric_features, axis=0)) / np.std(
    #     expert_trajectories_metric_features, axis=0)

    aprel_env = aprel.Environment(env, env.features)

    print("Random Reward:", np.average(generate_random_trajs(env, n=1000)[1]))

    expert_trajectories_metric, exp_metric_rewards = generate_random_trajs(env, 100)
    expert_trajectories_metric_features = np.vstack([traj.features for traj in expert_trajectories_metric])

    uniform_trajectories_metric, uni_metric_rewards = generate_random_trajs(env, 300)
    uniform_trajectories_metric_features = np.vstack([traj.features for traj in uniform_trajectories_metric])

    user_correct = 0
    user_incorrect = 0
    for i in np.arange(0, len(uni_metric_rewards[0]) - 1, 2):
        correct_choice = int(uni_metric_rewards[0][i] < uni_metric_rewards[0][i+1])
        user_choice = boltzman_choice(uni_metric_rewards[0][i], uni_metric_rewards[0][i+1])[0]
        #print("Correct choice:", correct_choice, "User choice:", user_choice)
        if correct_choice == user_choice:
            user_correct += 1
        else:
            user_incorrect += 1

    print("USER CORRECT:", user_correct, "USER INCORRECT:", user_incorrect, "User Accuracy:", user_correct/(user_correct + user_incorrect))

    rho_metric = Rho(expert_trajectories_metric_features, uniform_trajectories_metric_features)
    rho_eval = RhoWrapper(rho_metric).eval #, exp_metric_rewards, uni_metric_rewards).eval

    epic_metric = Epic(env, true_weights=np.array([HORIZON*NREWARD]))
    ee = epic_metric.eval


    val, val_rew = generate_random_trajs(env, 100)
    for i in range(val.size):
        val[i].reward = val_rew[0][i]
    # with open("val_trajs_norm_pds.pickle", 'rb') as fb:
    #     val = pickle.load(fb)
    #     fb.close()

    ll_metric = LLEval(val)
    loglike_eval = ll_metric.ll_eval

    # print("\n\n\nKENDALLS:")
    # run_experiment(kendalls.fdist, name="kendalls")

    trajs = get_trajs(seed_offset, NUM_RUNS, env)

    verbose = True
    print("TRAJS LENGTH:", len(trajs))
    startt = time.time()

    dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    dir_name = "BasicALExperiments/" + GYM_ENV + "_dt_" + dt_string
    os.mkdir(dir_name)

    procs = []
    for i in RUNNING:
        method = METHODS[i]
        print("\n\n", method, ":")
        proc = mp.Process(target = run_experiment,
                            args=(dir_name, rho_eval, method, env, aprel_env, ee, loglike_eval, seed_offset),
                            kwargs={"num_runs":NUM_RUNS, "name": method + GYM_ENV, "trajs": trajs,
                                    "verbose":verbose})
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    print("< Finished experiments -- time taken:", time.time() - startt, " SEED_OFFSET: ", seed_offset, " -- directory:", dir_name)