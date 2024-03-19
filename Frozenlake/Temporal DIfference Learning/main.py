env = gym.make('FrozenLake-v1')
agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.9999995, n_actions=4, n_states=16)
scores = []
win_pct_list = []
n_games =500000

for i in range(n_games):
    done=False
    observation = env.reset()
    observation = observation[0]
    score=0
    while not done:
        #print(observation)
        action = agent.choose_action(observation)
        observation_, reward, done,  trunc, info =env.step(action)
        #print(observation_, action, reward, observation)
        agent.learn(observation, action, reward, observation_)
        score+= reward
        observation = observation_
    scores.append(score)
    if i%100 ==0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)
        if i%1000 == 0:
            print("episode : {}, win_pct : {}, epsilon : {}".format(i, win_pct, agent.epsilon))
plt.plot(win_pct_list)
plt.show()
