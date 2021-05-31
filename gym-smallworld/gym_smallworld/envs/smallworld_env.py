import gym
import numpy as np
from collections import deque

# import pygame
from gym import error, spaces, utils
from gym.utils import seeding
import torch


class SmallWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 60}

    def __init__(self):
        self.xsize = 84
        self.ysize = 84
        self.xpos = self.xsize // 2
        self.ypos = self.ysize // 2
        self.resource = 100.0
        self.foodmap = np.zeros((self.xsize, self.ysize, 3))
        self.maxfood = 100.0
        self.sight_radius = 42
        self.sight_diam = self.sight_radius * 2 + 1
        self.egomap = np.zeros((self.sight_diam, self.sight_diam, 3))
        self.viewer = None
        self.action_space = spaces.Discrete(5)
        egohigh = np.ones_like(self.egomap) * self.maxfood
        # self.observation_space = spaces.Box(self.egomap, egohigh)
        self.observation_space = spaces.Box(
            np.zeros((self.xsize, self.ysize), dtype=np.float32),
            np.ones((self.xsize, self.ysize), dtype=np.float32),
        )
        self.window = 4
        self.state_buffer = deque([], maxlen=self.window)

        self.steps = 0

    def action_space(self):
        return self.action_space.n

    def step(self, action):
        # action can be the following
        # 0: stay, 1:move north, 2: move east, 3: move south, 4: move west
        if action == 1:
            self.ypos += 1
            self.ypos = self.ypos % self.ysize
        elif action == 2:
            self.xpos += 1
            self.xpos = self.xpos % self.xsize
        elif action == 3:
            self.ypos -= 1
            self.ypos = self.ypos % self.ysize
        elif action == 4:
            self.xpos -= 1
            self.xpos = self.xpos % self.xsize

        if action == 0:
            # self.resource -= 0.5
            self.resource -= 1.0  # tentatively turning it off for training
        else:
            self.resource -= 1.0

        # if the action make the agent to overlap with food, take it.
        if self.foodmap[self.xpos, self.ypos, 0] != 0:
            reward = self.foodmap[self.xpos, self.ypos, 0] / 100
            self.resource += reward * 100
            self.foodmap[self.xpos, self.ypos, 0] = 0.0
            self.eraseloc = (self.xpos, self.ypos)
        else:
            reward = 0.0

        # upda the self position in the map
        self.foodmap[:, :, 1] /= 2
        self.foodmap[self.xpos, self.ypos, 1] = 100

        # randomly, add food to the field.
        # with 2% probability, it adds rand(0-100) value at one location.
        if np.random.rand() < 0.02:
            emx = np.random.randint(self.xsize)
            emy = np.random.randint(self.ysize)
            self.foodmap[emx, emy, 0] = np.random.rand() * self.maxfood
            self.emergeloc = (emx, emy)

        # if the resource is zero or below, it dies.
        done = bool((self.resource <= 0) or (self.steps > 2000))
        if done:
            if self.resource <= 0:  # punish only if it did not complete
                reward = -3.0

        # matching the observation to the rainbow
        # obs = self.makeegomap()
        self.state_buffer.append(self._get_state())
        obs = self.state_buffer

        self.steps += 1

        return torch.stack(list(obs), 0), reward, done, {}

    def _get_state(self):
        return torch.tensor(
            self.foodmap[:, :, 0] + self.foodmap[:, :, 1], dtype=torch.float32
        )

    def makeegomap(self):
        # make an egocentric map
        # Use np.roll!!
        radius = self.sight_radius
        diam = self.sight_diam
        for i in range(diam):
            for j in range(diam):
                self.egomap[i, j, 0] = self.foodmap[
                    self.wrapx(i - radius + self.xpos),
                    self.wrapy(j - radius + self.ypos),
                    0,
                ]
        return self.egomap

    def wrapx(self, xval):
        return xval % self.xsize

    def wrapy(self, yval):
        return yval % self.ysize

    def reset(self):
        self.xpos = self.xsize // 2
        self.ypos = self.ysize // 2
        self.resource = 100
        self.steps = 0
        self.foodreset()
        # return self.makeegomap()
        for i in range(self.window):
            self.state_buffer.append(self._get_state())

        return torch.stack(list(self.state_buffer), 0)

    def foodreset(self):
        # randomly locate food resource in the map
        self.foodmap[:, :, 0] = (
            (np.random.random((self.xsize, self.ysize)) < 0.05)
            * np.random.random((self.xsize, self.ysize))
            * self.maxfood
        )
        self.drawfull = True
        self.eraseloc = None
        self.emergeloc = None

    def render_boo(self, mode="human"):
        return 0

    # def render_pygame(self, mode='human'):
    #     # this is a version with pygame
    #     if self.viewer is None:
    #         # setup pygame window
    #         displaysize = (self.xsize * 6 + 200, self.ysize * 6)
    #         pygame.init()
    #         pygame.display.set_caption("World 1")
    #         self.viewer = pygame.display.set_mode(displaysize)
    #         #self.clock = pygame.time.Clock()
    #         #self.fps = 60
    #
    #     #self.clock.tick(self.fps)
    #     self.viewer.fill((144, 206, 198))
    #     pygame.draw.rect(self.viewer, (255, 0, 0), [self.xpos * 6, self.ypos * 6, 6, 6])
    #     pygame.display.update()
    #     return True

    def render(self, mode="human"):
        screen_width = self.xsize * 6
        screen_height = self.ysize * 6
        unitleng = 3
        scale = unitleng * 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            # self.viewer = rendering.Viewer(screen_width, screen_height)
            l = unitleng
            # square = rendering.FilledPolygon([(-l, -l), (-l, l), (l, l), (l, -l)])
            # square.set_color(0.0, 1.0, 1.0)
            # self.squaretrans = rendering.Transform()
            # square.add_attr(self.squaretrans)
            # self.viewer.add_geom(square)
            self.viewer = rendering.SimpleImageViewer()

        # self.squaretrans.set_translation(self.xpos * scale, self.ypos * scale)
        self.viewer.imshow(
            np.uint8(
                np.kron(
                    np.transpose(np.flip(self.foodmap, 1) * 3, (1, 0, 2)),
                    np.ones((scale, scale, 1)),
                )
            )
        )

        return True  # self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
