# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from random import choice, random
from time import sleep, time
import itertools as it
from vizdoom import *

from constants_vizdoom import CONFIG
from constants_vizdoom import ACTION_SIZE

# VizDoom Config
frame_repeat = 4
episodes = 10
timelimit = 1 # minutes
players = 8 # number of players
skip = 4
mode = Mode.PLAYER # or Mode.ASYNC_PLAYER
random_sleep = True
const_sleep_time = 0.005
window = False
resolution = ScreenResolution.RES_256X144
scaled = (84,84)
args = ""
console = False

# All actions
noop = [0,0,0,0,0,0,0,0,0]

class GameState(object):
  def __init__(self, display=False, no_op_max=7):
    self.vizdoom = DoomGame()
    self.vizdoom.load_config(CONFIG)
    self.vizdoom.set_mode(mode)
    self.vizdoom.add_game_args(args)
    self.vizdoom.set_screen_resolution(resolution)
    self.vizdoom.set_console_enabled(console)
    self.vizdoom.set_screen_format(ScreenFormat.GRAY8)
    self.vizdoom.set_window_visible(display)
    self.vizdoom.init()
    print("Doom initialized.")

    self._no_op_max = no_op_max

    # Action = which buttons are pressed
    #n = self.vizdoom.get_available_buttons_size()
    self._screen = np.empty((scaled[0], scaled[1], 1), dtype=np.uint8)

    # collect minimal action set
    #n = self.vizdoom.get_available_buttons_size()
    #actions = [list(a) for a in it.product([0, 1], repeat=n)]
    #self.real_actions = [list(a) for a in it.product([0, 1], repeat=ACTION_SIZE)]
    self.real_actions = []
    for i in range(ACTION_SIZE):
      self.real_actions.append([0] * ACTION_SIZE)
      self.real_actions[i][i] = 1

    # height=210, width=160
    self._screen = np.empty((256, 144, 1), dtype=np.uint8)

    self.reset()

  def _process_frame(self, action, reshape):
    self.vizdoom.make_action(self.real_actions[action], frame_repeat)
    reward = self.vizdoom.get_total_reward()
    terminal = self.vizdoom.is_episode_finished()

    # screen shape is (144, 256)
    if terminal:
      self._screen = np.zeros((144, 256))
    else:
      self._screen = self.vizdoom.get_state().screen_buffer
      #self.ale.getScreenGrayscale(self._screen)
    
    # reshape it into (210, 160)
    #reshaped_screen = np.reshape(self._screen, (256, 144))
    
    # resize to height=84, width=84
    resized_screen = cv2.resize(self._screen, scaled)

    x_t = resized_screen[0:,:]
    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))

    x_t = x_t.astype(np.float32)
    x_t *= (1.0/255.0)
    return reward, terminal, x_t
    
    
  def _setup_display(self):
    print("Nothing to do")

  def reset(self):
    self.vizdoom.new_episode()
    
    # randomize initial state
    '''
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
        self.vizdoom.make_action(self.real_actions[noop], 1)
    '''

    _, _, x_t = self._process_frame(0, False)
    
    self.reward = 0
    self.terminal = False
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
  def process(self, action):
    # convert original 18 action index to minimal action set index
    #real_action = self.real_actions[action]
    
    r, t, x_t1 = self._process_frame(action, True)

    self.reward = r
    self.terminal = t
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    

  def update(self):
    self.s_t = self.s_t1
