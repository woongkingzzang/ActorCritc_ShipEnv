import math
from this import d

from turtle import position
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw

# Standard Module
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from random import randrange # random.randrange(a,b) : a이상 b미만




class ShipEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    '''
    단계별 개발
     step 1. Goal만 있는 환경 => 진행중
     step 2. Goal + Static Obstacle
     step 3. Goal + Static + Dynamic Obstacle
    

    from 우주현 교수님 논문
    
    1. observation 
    - 무인 수상선이 추종하는 경로 정보
    - 동적 장애물 정보
    - 정적 장애물 정보
    이 부분에 대해 grid 좌표로써 나타냄
    => 각, 포지션 (자선), 
    
    2. action
    from Velocity Obstacle + COLREGs Rule
    - 무인 수상선의 경로 추정
    - 좌현변침을 통한 장애물 회피
    - 우현변침을 통한 장애물 회피
    
    경로 추정의 경우, 목표 침로각 수식을 활용
    
    3. reward
    - 경로 추정에 따른 양의 보상
    - 충돌 회피 성공에 따른 양의 보상
    - 충돌에 따른 음의 보상

    reward 수식을 활용
    
    4. Transition Dynamics
    논문 5.1 부분을 확인해보면 됨
    자선: WAM-V 16
    *** Dynamics 수식 ***
    u_dot = -1.091 * u + 0.0028 * T_x (x방향 가속도)
    v_dot = 0.0161 * v - 0.0052 * r + 0.0002 * T_n
    r_dot = 8.2861 * v - 0.9860 * r + 0.0307 * T_n
    
    차후 원하는 실험 선박에 맞춰 바꿔야함
    '''
    
    def __init__(self, velocity = 1.5):
        # screen
        self.screen_width = 1300
        self.screen_height = 800
        # observation = [자선의 정보, static_obstacle, dynamic_obstacle]
        # state = [자선의 속도, 각도, 포지션]
        
        # 자선의 포지션
        self.min_position_x = 0
        self.max_position_x = self.screen_width
        self.min_position_y = 0
        self.max_position_y = self.screen_height
        self.position_x = self.screen_width/2
        self.position_y = self.screen_height - 30
        # 자선의 속도
        self.min_speed = 0
        self.max_speed = 2
        self.velocity = velocity
        
        # 자선의 각도
        # 오른쪽 angle = 270 ~ 360
        # 왼쪽 angle = 0 ~ 90
        # 직진 angle = 0
        self.angle = 0
        self.min_angle = -90
        self.max_angle = 90
        
        # action 및 step function에서 쓰일 각도, 속도 변환을 위한 Dynamics
        self.dt = 1
        
        # self.T_x = 100 # 자선의 추력
        # self.T_n = 50 # 자선의 추력
        
        self.u = 0 # 속도
        self.v = 0
        self.r = 0  # d_deg/dt
        
        # self.u_dot = -1.091 * self.u + 0.0028 * self.T_x
        # self.v_dot = 0.0161 * self.v - 0.0052 * self.r + 0.0002 * self.T_n
        # self.r_dot = 8.2681 * self.v - 0.9860 * self.r + 0.0307 * self.T_n
        
        self.u_dot = 0
        self.v_dot = 0
        self.r_dot = 0
        
        
        self.low = np.array([self.min_position_x, self.min_position_y, self.min_speed, self.min_angle], dtype=np.float32)
        self.high = np.array([self.max_position_x, self.max_position_y, self.max_speed, self.max_angle], dtype=np.float32)
        
        # Goal
        self.goal_x = self.screen_width /2
        self.goal_y = 770
        
        # Path
        #self.path 
        self.screen = None
        self.clock  = None
        self.isopen = True
    
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        
    def step(self, action):
        
        '''
        물리 모델애 대한 정의를 해놓은 부분 from 희수형님
        
        질문사항
        1. action에 대한 처리 (input으로 action이 정의되어져 있지만, action을 어떻게 처리하는지)
        답변: 당장은 action은 고려 x, 다이나믹스 먼저 정의
        2. position부분 정의
        답변: 선체고정좌표계와 지구고정좌표계를 통해 정의
        
        
        필요 수정 사항
        - rotate_matrix를 만들어서 아래 한줄 한줄 적은 코드들을 깔끔하게 고칠 수 있음
        - action에 따른 Tx, Tn 값 정하기
        - 동훈이형이 KASS 운동 모델 , Velocity Obstacle 반영
        
        '''
        
        # 추후 action에 따라 Tx, Tn 정의
        action_Tx = 0
        action_Tn = 50

        self.r_dot = 8.2681 * self.v - 0.9860 * self.r + 0.0307 * action_Tn
        self.v_dot = 0.0161 * self.v - 0.0052 * self.r + 0.0002 * action_Tn
        self.u_dot = -1.091 * self.u + 0.0028 * action_Tx

        self.r += self.r_dot * self.dt
        self.v += self.v_dot * self.dt
        self.u += self.u_dot * self.dt

        # local
        angle = self.r * self.dt
        y = self.v * self.dt
        x = self.u * self.dt

        # global
        self.angle += angle
        self.position_x += x * math.cos(self.angle) - y * math.sin(self.angle)
        self.position_y += x * math.sin(self.angle) + y * math.cos(self.angle)

        self.velocity = math.sqrt(math.pow(self.u, 2) + math.pow(self.v, 2))

        done = bool(self.position_x == self.goal_x and self.position_y == self.goal_y)
        reward = - 1.0  # mountain car에서 일단 가져옴

        self.state = (self.position_x, self.position_y, self.velocity, self.angle)

        return np. array(self.state, dtype=np.float32), reward, done, {}
     
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=650, high=660), self.np_random.uniform(low=770, high=780), 1.5, 0])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
        
        
    def render(self, mode= 'human'):
        
        '''
        pygame좌표계를 우주현 교수님 좌표계와 통일
        
        '''
        screen_width = self.screen_width
        screen_height = self.screen_height
        
        
        # world_width = self.max_position_x - self.min_position_x
        # scale = screen_width/world_width
    
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        
         # Path 그리기
        pygame.draw.line(self.surf, (0,0,255),[screen_width/2,0],[screen_width/2,screen_height],3)
        
        # Goal 그리기
        pygame.draw.circle(self.surf, (255,0,0), (self.goal_x, self.goal_y), 15)
        
        # 자선 그리기
        '''
        
        *** 이 부분이 pygame과 우주현 교수님 논문 좌표를 통일한 부분 ***
        중요한 점은 따로 함수를 통한 회전이 아닌 center에서 좌표를 변환하여 주었다는 점
        
        '''
        center = (self.state[1] - 120, -self.state[0] + 1200)
        scale =8
        self.os_img: pygame.Surface = pygame.image.load("./self_ship.png")
        self.ship_size = [i//scale for i in self.os_img.get_size()]
        self.os_img: pygame.Surface = pygame.transform.scale(self.os_img, self.ship_size)
        # rotate
        self.os_img = pygame.transform.rotate(self.os_img, - self.state[3])
        # rotate된 이미지를 덮어씌우기
        self.img: pygame.Surface = pygame.transform.scale(self.os_img, self.ship_size)
        self.rect = self.img.get_rect()
        self.rect.center = center

        # 창 뒤집기
        self.surf = pygame.transform.flip(self.surf, False, True)
        # blit : 이미지 덮어씌우기 => 업데이트
        self.screen.blit(self.surf,(0,0))
        self.screen.blit(self.os_img, self.rect)

        print(center)
        
        if mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.isopen = False
    
    # def rotate_angle(self, angle):
    #     if 0 <= angle <= 90:
    #         angle1 = angle + 270
    #     else:
    #         angle1 = angle - 90
        
        # return angle1
    def rad2deg(self,rad):
        deg = 180 / math.pi * rad
        return deg
    
    def deg2rad(self,deg):
        rad = math.pi/180 * deg
        return rad
    