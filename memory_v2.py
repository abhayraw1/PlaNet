import torch
import numpy as np
from np.random import choice 


"""
The need for MemoryV2 is due to the following reasons:
1 > The original experience replay did not account for the termination of the
    episodes. This kind of creates a poblem in our case! Where the goal state 
    might be quite different, based on the initial state of the env.

2 > .. ?

The changes that this structure brings is:
1 > Memory is a list of episodes and episodes is a list of experiences.
2 > During sampling, to make it easier for the integration part, make sure to
    sample sequences of len 'chunk_size' only. Nothing less!

v1 contructor inputs:
    size, symbolic_env, observation_size, action_size, bit_depth, device
"""


def postprocess_img(image, depth):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2 ** depth)
    return np.clip(image * 2**(8 - depth), 0, 2**8 - 1).astype(np.uint8)


def preprocess_img(image, depth):
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.rand_like(image).div_(2 ** depth))


class Memory:
  def __init__(self, size, _, observation_size, action_size, bit_depth, device):
    self.device = device
    self.action_size = action_size
    self.observation_size = observation_size
    self.data = deque(maxlen=size)
    self.episode = None

  @property
  def size(self):
    return len(self.data)

  def start_episode(self, obs):
    if self.episode is not None and isinstance(self.episode, Episode):
      self.data.append(self.episode)
    self.episode = Episode(device, bit_depth)
    self.episode.append_just_obs(obs)
  
  def append(self, obs, u, r, d):
    self.episode.append(obs, u, r, d)

  def sample(self, batch_size, tracelen):
    """
    Make sure to include only episodes with length >= tracelen
    """
    episode_idx = choice(self.size, batch_size)
    init_st_idx = [choice(self.data[i].size - tracelen) for i in episode_idx]
    R = torch.zeros((batch_size, tracelen)).to(device)
    D = torch.zeros((batch_size, tracelen)).to(device)
    U = torch.zeros((batch_size, tracelen, *self.action_size)).to(device)
    X = torch.zeros((batch_size, tracelen, *self.observation_size)).to(device)
    for n, (i, s) in enumerate(zip(episode_idx, init_st_idx)):
      X[n], U[n], R[n], D[n] = self.data[i].prepare(s, s + tracelen)
    return data


class Episode:
  def __init__(self, device, bit_depth):
    self.device = device
    self.bit_depth = bit_depth
    self.clear()

  @property
  def size(self):
    return self._size

  def clear(self):
    self.x = []
    self.u = []
    self.d = []
    self.r = []
    self._size = 0

  def append(self, x, u, r, d):
    self._size += 1
    self.x.append(postprocess_img(x.numpy(), self.bit_depth))
    self.u.append(u.numpy())
    self.r.append(r)
    self.d.append(d)

  def append_just_obs(self, x):
    self.x.append(postprocess_img(x.numpy(), self.bit_depth))

  def prepare(self, s=0, e=None):
    e = e or self.size
    prossx = torch.tensor(self.x[s:e+1], dtype=F32, device=self.device)
    preprocess_img(prossx, self.bit_depth),
    return (
      prossx,
      torch.tensor(self.u[s:e], dtype=F32, device=self.device),
      torch.tensor(self.r[s:e], dtype=F32, device=self.device),
      torch.tensor(self.d[s:e], dtype=F32, device=self.device),
    )
