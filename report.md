# Continuous learing: Reacher

Here we have adopted Deep Deterministic Policy Gradient (DDPG) algoritm from Actor-Critic Methods lessons. In order to be able to apply it here we have extensivelly followed ideas form Slack forum. We have tested many of them, but finally, what was working was very close to ideas proposed there. In particular:

* adding batch normalization after first layer in both: Actor and Critic Net.
* 

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

