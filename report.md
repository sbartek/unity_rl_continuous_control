# Continuous learing: Reacher (version 1)

Here we have adopted Deep Deterministic Policy Gradient (DDPG) algoritm from Actor-Critic Methods lessons. In order to be able to apply it here we have extensivelly followed ideas form Slack forum. We have tested many of them, but finally, what was working was very close to ideas proposed there. In particular:

* adding batch normalization after first layer in both: Actor and Critic Net.
* We have cliped gradient norm for Critic before updating its weights.
* NNs with hidden leyers of 128 nodes worked best.

## Actor/Critic Architecture

### Actor

```
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
```

### Critic

```
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
```

## Other parameter

* replay buffer size 1e5
* minibatch size 128       
* discount factor gamma 0.99            
* tau from soft update of target parameters  1e-3             
* learning rate of the actor 2e-4         
* learning rate of the critic 2e-4  
* sigma from noise 0.03  

## Result

Evirionment was solved in 242 episodes (gettig more then 30 points on averege among 100 episodes).

![Alt text](https://github.com/sbartek/unity_rl_continuous_control/blob/master/imgs/scores1.png?raw=true "Optional Title")

One can observe trained agent running: `ContinuousControl-RunTrainedAgent.ipynb` notebook and compare it with random agent `ContinuousControl-RunRandomAgent.ipynb`.

## Future improvements

We should investigate different configurations of NN. Iw particular we could try to add dropout.