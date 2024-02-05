'''
    ShipEnv-ActorCritic

    ** 단계별 개발 **
     step 1. Goal만 있는 환경 => 진행중
     step 2. Goal + Static Obstacle
     step 3. Goal + Static + Dynamic Obstacle
    

    ** 환경 설계 from 우주현 교수님 논문 **
    
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
        u_dot = -1.091 * u + 0.0028 * T_x 
        v_dot = 0.0161 * v - 0.0052 * r + 0.0002 * T_n
        r_dot = 8.2861 * v - 0.9860 * r + 0.0307 * T_n
        
        차후 원하는 실험 선박에 맞춰 바꿔야함 
    
    ** 현재 단계 **
        - step 1
        - Dynamics : WAM-V 16
        - state = [x, y, velocity, angle(rad)]
    
    ** 해결된 문제 **
        - 선박의 운동 모델 반영
        - 선체 고정 좌표계 + 회전 좌표계 -> 지구 고정 좌표계
        - math.sin, cos에서 deg가 아닌 rad를 사용
    
    
    ** 해결할 문제 **  
        - rotate_matrix를 만들어서 아래 한줄 한줄 적은 코드들을 깔끔하게 고칠 수 있t음
        - action에 따른 Tx, Tn 값 정하기
        - 동훈이형이 KASS 운동 모델 , COLREGs 반영 (최소 2~3단계)
    
'''

import math
from math import atan2
# from this import d

# from turtle import position
from typing import Optional
from defusedxml import DTDForbidden

import numpy as np
import pygame
from pygame import gfxdraw

# Standard Module
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randrange # random.randrange(a,b) : a이상 b미만
from gym_master.gym.utils.renderer import Renderer
from gym_master.gym.envs.classic_control import utils
from CRI import CRI
class ShipEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    
    def __init__(self, render_mode: Optional[str] = None):
        # screen
        self.screen_width = 900
        self.screen_height = 800
        
        # observation = [자선의 정보, static_obstacle, dynamic_obstacle]
        # state = [자선의 속도, 각도, 포지션]
        
        # 자선의 포지션
        self.min_position_x = 0
        self.max_position_x = self.screen_width
        self.min_position_y = 0
        self.max_position_y = self.screen_height
        
        ### Own Ship
        # position
        self.position_x = 100
        self.position_y = 450
    
        self.psi = self.deg2rad(0)    # rad
        
        # 자선의 속도
        self.min_speed = 0
        self.max_speed = 2
        self.velocity = 0
        
        # 자선의 각도
        self.min_angle = -90
        self.max_angle = 90
        
        # action 및 step function에서 쓰일 각도, 속도 변환을 위한 Dynamics
        self.dt = 0.1
        
        # 자선의 Beam
        self.beam = 2.5

        ### Target Ship
        self.ts_pos_x = 700
        self.ts_pos_y = 500
        self.ts_psi = self.deg2rad(200)
        
        '''
        이렇게 변수는 보기 좋게 모아두고 정의할 필요가 있음
        ''' 
        
        # 가속도와 속도 초기화
        # Local coordinate
        self.x, self.y, self.angle = 0, 0, 0
        self.u, self.v, self.r = 0, 0, 0 # self.r: d_deg/dt
        self.u_dot, self.v_dot, self.r_dot = 0, 0, 0

        ## TS
        self.ts_x, self.ts_y, self.ts_angle = 0, 0, 0
        self.ts_u, self.ts_v, self.ts_r = 0, 0, 0
        self.ts_u_dot, self.ts_v_dot, self.ts_r_dot = 0, 0, 0
        self.ts_vel = 0

        # Rotated coordinate
        self.X, self.Y = 0, 0
        self.ts_X, self.ts_Y = 0,0 
        self.low = np.array([self.min_position_x, self.min_position_y, self.min_speed, self.min_angle], dtype=np.float32)
        self.high = np.array([self.max_position_x, self.max_position_y, self.max_speed, self.max_angle], dtype=np.float32)
        
        # Goal(Real)
        # self.goal_x = self.screen_width - 100 # goal_x = 800
        # self.goal_y = self.screen_height/2 # goal_y = 400

        self.goal_x = 800
        self.goal_y = 400
        
        # Path
        self.screen = None
        self.clock  = None
        self.isopen = True
    
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

        # simulation save
        self.action_list = []
        self.simul_test = False
        self.deg = 0
        self.rd = 0

    def get_init(self):
        ## OS ##
        self.position_x = 100
        self.position_y = 400
        self.psi = self.deg2rad(0)    # rad

        self.x, self.y, self.angle = 0, 0, 0
        self.u, self.v, self.r = 0, 0, 0 # self.r: d_deg/dt
        self.u_dot, self.v_dot, self.r_dot = 0, 0, 0

        # Rotated coordinate
        self.X, self.Y = 0, 0

        ## TS ##
        self.ts_pos_x = 700
        self.ts_pos_y = 500
        self.ts_psi = self.deg2rad(200)

        self.ts_x, self.tx_y, self.ts_angle = 0, 0, 0
        self.ts_u, self.ts_v, self.ts_r = 0, 0, 0
        self.ts_u_dot, self.ts_v_dot, self.ts_r_dot = 0, 0, 0

        self.ts_X, self.ts_Y = 0,0 
        self.simul_test = False

    def step(self, action):
        # print(action)
        # Thrust_port, Thrust_starboard
        T_l = 10
        T_r = 10
        self.action_list.append(action)

        if action == 0 :
            T_l = 10
            T_r = 10
            # self.angle = self.rad2
            
        elif action == 1: # 우현
            T_l = 9
            T_r = 10
            
        
        elif action == 2: # 좌현
            T_l =10
            T_r =9
        
        elif action == 3:
            T_l = 0
            T_r = 10
            

        elif action == 4:
            T_l = 10
            T_r = 0

        self.action = action
        # print(self.action_list)
        # print(self.action)
        action_Tx = T_l +  T_r
        action_Tn = (T_l - T_r) * self.beam / 2
        
        self.r_dot = 8.2681 * self.v - 0.9860 * self.r + 0.0307 * action_Tn
        self.v_dot = 0.0161 * self.v - 0.0052 * self.r + 0.0002 * action_Tn
        self.u_dot = -1.0191 * self.u + 0.0028 * action_Tx

        # 선체고정좌표계
        self.r += self.r_dot * self.dt
        self.v += self.v_dot * self.dt
        self.u += self.u_dot * self.dt
        self.velocity = math.sqrt(self.u**2 + self.v**2)
        self.angle += self.r * self.dt  # deg
        
        # self.angle = ang
        # print(self.angle)
        self.y += self.v * self.dt
        self.x += self.u * self.dt
        
        '''   
        중요한점!! math.cos, math.sin은 rad을 사용!!
        '''
        
        # Rotated coordinate
        
        # self.psi += self.angle * 180 / math.pi # deg to rad
        self.psi = self.angle
        # print(self.rad2deg(self.psi)%360)
        # self.psi += self.angle * math.pi/180
        self.X = self.x * math.cos(self.psi) - self.y * math.sin(self.psi) 
        self.Y = self.x * math.sin(self.psi) + self.y * math.cos(self.psi)

        # Global position
        self.position_x += self.X
        self.position_y += self.Y
        
        ## TS ##
        
        ts_T_r , ts_T_l = 7,7
        ts_Tx = ts_T_r + ts_T_l
        ts_Tn = (ts_T_l - ts_T_r) * self.beam / 2

        self.ts_r_dot = 8.2681 * self.ts_v - 0.9860 * self.ts_r + 0.0307 * ts_Tn
        self.ts_v_dot = 0.0161 * self.ts_v - 0.0052 * self.ts_r + 0.0002 * ts_Tn
        self.ts_u_dot = -1.0191 * self.ts_u + 0.0028 * ts_Tx

        self.ts_r += self.ts_r_dot * self.dt
        self.ts_v += self.ts_v_dot * self.dt
        self.ts_u += self.ts_u_dot * self.dt
        self.ts_vel = math.sqrt(self.ts_u**2 + self.ts_v**2)

        self.ts_angle += self.ts_r * self.dt  # deg
        self.ts_y += self.ts_v * self.dt
        self.ts_x += self.ts_u * self.dt
        
        self.ts_psi += self.ts_angle * 180 / math.pi # deg to rad
        self.ts_X = self.ts_x * math.cos(self.ts_psi) - self.ts_y * math.sin(self.ts_psi) 
        self.ts_Y = self.ts_x * math.sin(self.ts_psi) + self.ts_y * math.cos(self.ts_psi)

        self.ts_pos_x += self.ts_X
        self.ts_pos_y += self.ts_Y

        # reward
        pos_x = self.position_x
        # print(pos_x)
        pos_y = self.position_y
        psi = self.rad2deg(self.psi) % 360
        done = self.done
        # print("os_psi:", self.psi)
        # cri = CRI(50, 0.6, pos_x, pos_y, self.ts_pos_x, self.ts_pos_y, self.psi, self.ts_psi, \
        #     self.ms2kn(self.velocity), self.ms2kn(self.ts_vel))
        # cri_idx = cri.CRI()

        # print("UDCPA", cri.UDCPA())
        # print("UTCPA", cri.UTCPA())
        # print("UD", cri.UD())
        # print("UB", cri.UB())
        # print("UK", cri.UK())
        ## done ##
        if pos_x == self.goal_x and pos_y == self.goal_y:
            done = True
        elif self.ts_pos_x - 30 < pos_x < self.ts_pos_x +30\
             and self.ts_pos_y - 30 < pos_y < self.ts_pos_y + 30:
            done = True
        else:
            done = bool(pos_x  >= self.screen_width
                        or pos_x <= 0
                        or pos_y  >= self.screen_height
                        or pos_y <= 0
                        or pos_x == self.ts_pos_x and pos_y == self.ts_pos_y
                        )
        ## reward ##
        if not done:
            reward = 0.0
            dist = self.distance(pos_x, pos_y, self.ts_pos_x, self.ts_pos_y)
            

            self.deg = psi
            self.rd = dist
            
            # print(cri_idx)
            opt_rad= atan2(self.goal_x - pos_x, self.goal_y - pos_y) 
            deg = self.rad2deg(opt_rad)
            opt_deg = self.opt_deg(deg)
            if opt_deg <0:
                opt_deg += 360
            # print(opt_deg, psi)
            if self.goal_x - 50 <= pos_x <= self.goal_x + 50 and self.goal_y -50 <= pos_y <= self.goal_y:
                reward = 1
                self.simul_test = True
                print("####reward#####")
            
            ## TS 
            if pos_y <= self.ts_pos_y:
                # if cri_idx < 0.66:
                if dist < 400:
                    if pos_x < self.ts_pos_x:
                        if 0 < psi < 60:
                            reward = 1
                        else:
                            reward = 0
                    else:
                        if opt_deg + 20 > 360:
                            if psi > opt_deg -20 or psi < opt_deg + 20 -360:
                                reward = 1
                            else:
                                reward = 0
                        else:
                            if opt_deg - 20 < psi < opt_deg +20:
                                reward = 1
                            else:
                                reward = 0
                else:
                    if opt_deg + 20 > 360:
                        if psi > opt_deg -20 or psi < opt_deg + 20 -360:
                            reward = 1
                        else:
                            reward = 0
                    else:
                        if opt_deg - 20 < psi < opt_deg +20:
                            reward = 1
                        else:
                            reward = 0
            else:
                if opt_deg + 20 > 360:
                    if psi > opt_deg -20 or psi < opt_deg + 20 -360:
                        reward = 1
                    else:
                        reward = 0
                else:
                    if opt_deg - 20 < psi < opt_deg +20:
                        reward = 1
                    else:
                        reward = 0

            if self.ts_pos_x -50 < pos_x < self.ts_pos_x + 50 and self.ts_pos_y -50 < pos_y < self.ts_pos_y + 50:
                reward = 0
                    
        else:
            if pos_x == self.goal_x and pos_y== self.goal_y:
                reward = 1
                print("########Reward!######")
            elif self.ts_pos_x -30 < pos_x < self.ts_pos_x + 30 and self.ts_pos_y -30 < pos_y < self.ts_pos_y + 30:
                reward = 0
            else:
                reward = 0

        self.state = (T_l, T_r, self.deg, self.ts_pos_x, self.ts_pos_y, self.rd)
        # self.state = (self.deg, self.rd)
        self.renderer.render_step()
        
        return np.array(self.state, dtype=np.float32), reward, done, {}
     
    def reset(
        self, 
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.get_init()
        self.state = np.array([self.np_random.uniform(low=90, high=110), self.np_random.uniform(low=390, high=410), 0, 650, 700, -90])
        self.action_list= []
        self.simul_test = False
        # low, high = utils.maybe_parse_reset_bounds(options, 660, 660)
        # self.state = np.array([self.np_random.uniform(low=low, high=high), self.np_random.uniform(low=low, high=high), 0, 0])
        # self.get_init()
        # self.renderer.reset()
        # self.renderer.render_step()
        self.done = False
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else: 
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode = 'human'):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode= 'human'):
        assert mode in self.metadata["render_modes"]
        
        '''
        pygame좌표계를 우주현 교수님 좌표계와 통일
        x축: 오른쪽으로 +y
        y축: 아래쪽으로 +x 
        '''
        
        screen_width = self.screen_width
        screen_height = self.screen_height
        
        # height 와 width의 위치를 바꿔준 이유 = 선박 이미지 좌표와 pygame 이미지 좌표를 맞춰주기 위해
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_height, screen_width))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((screen_height, screen_width))
        self.surf.fill((255, 255, 255))
        
        
         # Path 그리기
        pygame.draw.line(self.surf, (0,0,255),[screen_height/2,0],[screen_height/2,screen_width],3)

       
        # Goal 그리기 
        visual_goal_x, visual_goal_y = self.render_cordinate(self.goal_x, self.goal_y)
        pygame.draw.circle(self.surf, (255,0,0), (visual_goal_x, visual_goal_y), 15)

        # 자선 그리기
        
        '''
        
        *** 이 부분이 pygame과 우주현 교수님 논문 좌표를 통일한 부분 ***
        중요한 점은 따로 함수를 통한 회전이 아닌 center에서 좌표를 변환하여 주었다는 점
        '''

        # print(self.state)
        # visual_os_x, visual_os_y = self.render_cordinate(self.state[0], self.state[1])
        # visual_ts_x, visual_ts_y = self.render_cordinate(self.state[3], self.state[4])
        visual_os_x, visual_os_y = self.render_cordinate(self.position_x, self.position_y)
        visual_ts_x, visual_ts_y = self.render_cordinate(self.ts_pos_x, self.ts_pos_y)
        # center = (self.state[1] - 370 , -self.state[0]+ 1000 )
        center = (visual_os_x, visual_os_y)
        center_ts = (visual_ts_x, visual_ts_y)
        scale = 8
        scale_ts = 7
        self.os_img: pygame.Surface = pygame.image.load("./self_ship.png")
        self.ts_img: pygame.Surface = pygame.image.load("./self_ts copy.png")

        self.ship_size = [i//scale for i in self.os_img.get_size()]
        self.ts_size = [i//scale_ts for i in self.ts_img.get_size()]
        self.os_img: pygame.Surface = pygame.transform.scale(self.os_img, self.ship_size)
        self.ts_img: pygame.Surface = pygame.transform.scale(self.ts_img, self.ts_size)

        # rotate
        # self.os_img = pygame.transform.rotate(self.os_img, -self.state[2] * 180 / math.pi)
        self.os_img = pygame.transform.rotate(self.os_img, -self.psi * 180 / math.pi)
        self.ts_img = pygame.transform.rotate(self.ts_img, -self.ts_psi * 180 / math.pi)
        # print(-self.state[5] * 180 / math.pi)   
        #      
        # rotate된 이미지를 덮어씌우기
        self.rect = self.os_img.get_rect()
        self.rect.center = center

        self.rect_ts = self.ts_img.get_rect()
        self.rect_ts.center = center_ts

        # 창 뒤집기
        # self.surf = pygame.transform.flip(self.surf, False, True) # 좌표계를 뒤집는데, 보여지는 것만 뒤집어짐 
        pygame.draw.circle(self.surf, (255, 50,50), center, 5) # 뒤집고 나서 그려야됨
        pygame.draw.circle(self.surf, (255, 50,50), center_ts, 5) 
        # blit : 이미지 덮어씌우기 => 업데이트
        self.screen.blit(self.surf,(0,0))
        self.screen.blit(self.os_img, self.rect)
        self.screen.blit(self.ts_img, self.rect_ts)

        # print("center, self.state[3]: ", center, self.state[3])
        # print(self.action)
        
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
    
   
    def rad2deg(self,rad):
        deg = 180 / math.pi * rad
        return deg
    
    def deg2rad(self,deg):
        rad = math.pi/180 * deg
        return rad

    def render_cordinate(self, real_x, real_y):

        vis_x = real_y
        vis_y = -real_x + self.screen_width
        
        return vis_x, vis_y

    def distance(self, x1, y1, x2, y2):
        result = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return result
    
    def ms2kn(self, v):
        return v*1.943844
    
    def opt_deg(self, deg):
        if deg < 0:
            deg += 360
        opt = (-deg + 450) % 360
        return opt

    