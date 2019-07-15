from wrappers import make_env
import torch
import torch.nn as nn
import numpy as np
import tensorboardX

writer = tensorboardX.SummaryWriter('./logs/')


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class Baseline(nn.Module):

    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        shape = 1
        for dim in input_shape:
            shape *= dim
        self.fc = nn.Sequential(
            nn.Linear(shape, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)


class Agent:

    def __init__(self, policy_net, baseline_net):
        self.policy_net = policy_net
        self.baseline = baseline_net

    def train(self, env, num_traj, iterations, gamma, base_epochs):
        for iter in range(iterations):
            trajectories = []
            ITER_REW = 0
            for _ in range(num_traj):
                rewards = []
                log_probs = []
                s = env.reset()
                done = False
                while not done:
                    s = torch.FloatTensor([s]).cuda()
                    a = self.policy_net(s)
                    del s
                    a2 = a.detach().cpu().numpy()
                    vec = [0, 1, 2, 3, 4, 5]
                    u = np.random.choice(vec, 1, replace=False, p=a2[0])
                    log_probs.append(a[0][u])
                    del a
                    sp, r, done, _ = env.step(u[0])
                    ITER_REW += r
                    rewards.append(r)
                    # env.render()
                    s = sp
                trajectories.append({'log_probs': log_probs, 'rewards': rewards})
            # self.update_baseline(base_epochs, trajectories, gamma)
            self.update_policy(trajectories, gamma)
            print("ITERATION:", iter+1, "AVG REWARD:", ITER_REW/num_traj)
            writer.add_scalar("avg reward", ITER_REW/num_traj, iter)
            torch.save(self.policy_net.state_dict(), './policynet')
            # torch.save(slbase_net.state_dict(), './basenet')

    def update_baseline(self, epochs, trajectories, gamma):
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.baseline.parameters())
        for epoch in range(epochs):
            loss = torch.tensor(0).float().cuda()
            for trajectory in trajectories:
                for t in range(len(trajectory)):
                    r_t = 0
                    for t_d in range(t, len(trajectory)):
                        r_t += gamma**(t_d - t) * trajectory[t_d]['r']
                    pred = self.baseline(trajectory[t]['s'])
                    loss += criterion(pred, torch.FloatTensor([r_t]).cuda())
            print(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()

    def update_policy(self, trajectories, gamma):
        loss = torch.tensor([0]).float().cuda()
        optim = torch.optim.Adam(self.policy_net.parameters(), lr=0.1)
        for trajectory in trajectories:
            for t in range(len(trajectory['rewards'])):
                r_t = 0
                log_prob = trajectory['log_probs'][t]
                temp = trajectory['rewards'][t:]
                for i, reward in enumerate(temp):
                    r_t += gamma**i * reward
                # for t_d in range(t, len(trajectory)):
                #     r_t += gamma ** (t_d - t) * trajectory['rewards'][t_d]
                # advantage += torch.FloatTensor([r_t]).cuda() - self.baseline(trajectory[t]['s'])[0]
                advantage = torch.FloatTensor([r_t]).cuda()
                loss += -log_prob * advantage
                # loss.backward()
            # loss += -log_probs * advantage
        loss = loss/len(trajectories)
        loss.backward()
        # print("\nBefore zerograd\n")
        # for name, param in self.policy_net.named_parameters():
        #     print(name, param.grad.data.sum())
        optim.step()
        optim.zero_grad()
        # print("\nAfter zerograd\n")
        # for name, param in self.policy_net.named_parameters():
        #     print(name, param.grad.data.sum())


def main():
    env = make_env('PongNoFrameskip-v4')
    policy_net = PolicyNet(env.observation_space.shape, env.action_space.n).to(torch.device('cuda'))
    base_net = Baseline(env.observation_space.shape).to(torch.device('cuda'))
    policy_net.load_state_dict(torch.load('./policynet'))
    base_net.load_state_dict(torch.load('./basenet'))
    agent = Agent(policy_net, base_net)
    agent.train(env, 16, 20000, 0.98, 5)


main()