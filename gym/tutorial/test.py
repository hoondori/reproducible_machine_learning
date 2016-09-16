
import gym
env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1')
for i_epispde in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timestamps".format(t+1))
            break
env.monitor.close()


print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from gym import spaces
space = spaces.Discrete(8)
x = space.sample()
assert space.contains(x)
assert space.n == 8

from gym import envs
print(envs.registry.all())
