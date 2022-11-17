import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np 
from math import cos, sin, atan, sqrt, pi, exp
from KASS_information import KASS_info

kass = KASS_info()
Model = kass.Model
Coefficient = kass.Coefficient

model_scale = Model['scale']
model_LPP = Model['LBP']

class KASS_MMG:
    """  자선의 Dynamics를 바탕으로 t+1의 udot, vdot, rdot 추정
    
    Params
        ship_scale: mmg 파일을 호출할 때 대상선의 scale 대입을 통해서 `scale=3.8`에서 수행된 `Resistacne test`의 결과를 상사하는데 사용
           (예시) `scale = 11.0`이라면, `Realtime_U = Realtime_U * sqrt(11.0) / sqrt(3.8)`임
    """
    def __init__(self, ship_scale):
        self.ship_scale = ship_scale
        self.Model = kass.Model # 다른 class 및 module에서 파라미터 호출하기 위함

        self.m_ = Model['Mass']/(0.5 * Model['rho_calm'] * (Model['LBP'])**3)
        self.Izz_ = Model['Izz']/(0.5 * Model['rho_calm'] * (Model['LBP'])**5)              
        self.xG_ = Model['LCB'] / Model['LBP']                          
        
        self.m11_ = self.m_ - Coefficient['Xudot']
        self.m22_ = self.m_ - Coefficient['Yvdot']
        self.m33_ = self.Izz_ - Coefficient['Nrdot']
        self.B_ = self.m_ * self.xG_ - Coefficient['Yrdot']
        self.C_ = self.m_ * self.xG_ - Coefficient['Nvdot']

    def resistance_test(self, Realtime_U): 
        """ 실선속도 U에 대응하는 모형선 저항, 반류계수, 추력계수, 자항점 추정

        Returns:
            R_, w, tP, n
        """   
        Realtime_U = Realtime_U * sqrt(self.ship_scale) / sqrt(model_scale)  
        ## 기존 모형테스트는 5m 급 모형선으로 진행되었으므로 `ship scale` 급의 모형선으로 스케일을 맞춰주기 위함임
        LBP_38 = model_LPP    # model_scale의 선박의 LBP

        R = 5.4573652477 * exp(1.5281010960 * Realtime_U)   # 모형선 저항
        w = 0.0218369303 * Realtime_U + 0.0570360587   # 모형선 반류계수
        tP = 0.0265 * Realtime_U + 0.093   # 모형선 추력감소계수
        n = 0.8502443714 * Realtime_U**2 + 3.2507343246 * Realtime_U + 1.1038657300   # 모형선 자항점
        
        R_ = R / (0.5 * Model['rho_calm'] * LBP_38**2 * Realtime_U**2)
        R_ = round(R_, 4)
        w = round(w, 3)
        tP = round(tP, 3)
        n = round(n, 2)
        return R_, w, tP, n        
    
    def KT_curve(self, JP_port, JP_star):   
        """ 전진비 Jp에 대응하는 추력계수 KT 계산 
        
        Returns:
            KT_port, KT_star """         
        KT_port = Coefficient['KT_k2'] * JP_port**2 + Coefficient['KT_k1'] * JP_port + Coefficient['KT_k0']
        KT_star = Coefficient['KT_k2'] * JP_star**2 + Coefficient['KT_k1'] * JP_star + Coefficient['KT_k0']

        return KT_port, KT_star

    def JP_and_FN(self, u, v, r, Realtime_U, delta, n):
        """ 프로펠러의 전진비 및 타 직압력 계산

        Returns:
            Jp_port, Jp_star, FN_port, FN_star """
        ########## ship scale에 맞게 모형선의 제원 값을 수정 #######
        LBP = (Model['LBP'] * model_scale) / self.ship_scale
        xP = (Model['xP'] * model_scale) / self.ship_scale
        Propeller_Diameter = (Model['Propeller_Diameter'] * model_scale) / self.ship_scale
        Rudder_Area = Model['Rudder_Area'] * (model_scale**2) / (self.ship_scale **2)

        ################### for Propeller force `Kt ###################
        _, w, _, _ = self.resistance_test(Realtime_U)
        v_ = v/Realtime_U
        r_ = r/(Realtime_U/LBP)       
        xP_ = xP/LBP

        beta_P = -(v_ + xP_*r_) # Inflow angle of Propeller
        vP_ = -beta_P
        omega_P0 = w
        
        if beta_P > 0:
            omega_port = omega_P0 * exp(-Coefficient['cP_minus']*vP_**2)
            omega_star = omega_P0 * exp(-Coefficient['cP_plus']*vP_**2)
        elif beta_P < 0:
            omega_port = omega_P0 * exp(-Coefficient['cP_plus']*vP_**2)
            omega_star = omega_P0 * exp(-Coefficient['cP_minus']*vP_**2)
        else:
            omega_port = omega_P0
            omega_star = omega_P0

        # Propeller induced velocity    
        uP_port = (1 - omega_port) * u
        uP_star = (1 - omega_star) * u
        
        # Advanced ratio
        JP_port = uP_port / (n * Propeller_Diameter)
        JP_star = uP_star / (n * Propeller_Diameter)

        KT_port, KT_star = self.KT_curve(JP_port, JP_star)

        ################### for Rudder force `FN` ###################        
        lR_ = -1.2
        lR = lR_ * LBP # Effective inflow angle to rudder
        beta_R = -(v_ + lR_ * r_)
        vR_app = v + lR * r   # 타 유입류 겉보기 횡속도

        # Longitudinal inflow velocity of Rudder            
        uR_port = Coefficient['epsilon'] * uP_port * sqrt(Coefficient['eta'] * (1 + Coefficient['kappa'] *(sqrt(1 + 8*KT_port/(pi*JP_port**2))-1))**2 + (1 - Coefficient['eta']))
        uR_star = Coefficient['epsilon'] * uP_star * sqrt(Coefficient['eta'] * (1 + Coefficient['kappa'] *(sqrt(1 + 8*KT_star/(pi*JP_star**2))-1))**2 + (1 - Coefficient['eta']))

        # Transverse inflow velocity of Rudder        
        if beta_R > 0 :
            vR_port = Coefficient['gammaR_minus'] * vR_app
            vR_star = Coefficient['gammaR_plus'] * vR_app
        elif beta_R < 0 :
            vR_port = Coefficient['gammaR_plus'] * vR_app
            vR_star = Coefficient['gammaR_minus'] * vR_app
        else :
            vR_port = vR_app
            vR_star = vR_app
        
        # 타 유입각 alpha
        alphaR_port = delta - atan(vR_port / uR_port)
        alphaR_star = delta - atan(vR_star / uR_star)
        
        UR_port = sqrt(uR_port**2 + vR_port**2)
        UR_star = sqrt(uR_star**2 + vR_star**2)
                

        FN_port = 0.5 * Model['rho_calm'] * Rudder_Area * UR_port**2 * Model['Rudder_f_alpha'] * sin(alphaR_port)
        FN_star = 0.5 * Model['rho_calm'] * Rudder_Area * UR_star**2 * Model['Rudder_f_alpha'] * sin(alphaR_star)
        
        return JP_port, JP_star, FN_port, FN_star
    
    def uvrdot(self, u, v, r, Realtime_U, delta, n):
        """ 
            `t`에서의 자선의 속도와 타각을 바탕으로 `t+1`에서의 자선의 가속도 계산
            
            Units:
                - r: rad./sec.
                - delta: rad.
        
        """        
        # NOTE: Avoid zero `n` becuase it brings error at 
        #       `def JP_and_FN()` -> `JP_port = uP_port / (n * Propeller_Diameter)`
        if n == 0:
            n = 0.01

        ########## ship scale에 맞게 모형선의 제원 값을 수정 #######   
        LBP = (Model['LBP'] * model_scale) / self.ship_scale
        yP = (Model['yP'] * model_scale) / self.ship_scale
        Propeller_Diameter = (Model['Propeller_Diameter'] * model_scale) / self.ship_scale
        
        ########## MMG 조종운동 방정식 적용 ##########
        R_, _, tP, _ = self.resistance_test(Realtime_U)
        JP_port, JP_star, FN_port, FN_star = self.JP_and_FN(u, v, r, Realtime_U, delta, n)
        KT_port, KT_star = self.KT_curve(JP_port, JP_star)

        u_ = u/Realtime_U
        v_ = v/Realtime_U
        r_ = r/(Realtime_U/LBP)
        
        FN_port_ = FN_port / (0.5 * Model['rho_calm'] * LBP**2 * Realtime_U**2)
        FN_star_ = FN_star / (0.5 * Model['rho_calm'] * LBP**2 * Realtime_U**2)

        m_ = self.m_
        Izz_ = self.Izz_
        xG_ = self.xG_
        
        m11_ = self.m11_
        m22_ = self.m22_
        m33_ = self.m33_
        B_ = self.B_
        C_ = self.C_

        X_H_ = (Coefficient['Xvv']*(v_)**2) + (Coefficient['Xvvvv']*v_**4) + ((m_ + Coefficient['Xvr'])*v_*r_) + ((m_*xG_ + Coefficient['Xrr'])*(r_)**2) - R_ 
        Y_H_ = Coefficient['Yv']*v_ + Coefficient['Yr']*r_ + Coefficient['Yvvv']*v_**3 + Coefficient['Yrrr']*r_**3 + Coefficient['Yvvr']*(v_)**2*r_ + Coefficient['Yvrr']*v_*(r_)**2 - m_*u_*r_
        N_H_ = Coefficient['Nv']*v_ + Coefficient['Nr']*r_ + Coefficient['Nvvv']*v_**3 + Coefficient['Nrrr']*r_**3 + Coefficient['Nvvr']*(v_)**2*r_ + Coefficient['Nvrr']*v_*(r_)**2 - m_*xG_*u_*r_


        if n >= 0:
            X_P = (1-tP) * Model['rho_calm'] * (Propeller_Diameter**4) * (n**2 * KT_port + n**2 * KT_star)      # `n_P = n_S` 으로 가정 
            X_P_ = X_P / (0.5 * Model['rho_calm'] * LBP**2 * Realtime_U**2)
            N_P = yP * (1-tP) * Model['rho_calm'] * (Propeller_Diameter**4) * (n**2 * KT_port - n**2 * KT_star)
            N_P_ = N_P / (0.5 * Model['rho_calm'] * LBP**3 * Realtime_U**2)
        elif n < 0:
            X_P = (1-tP) * Model['rho_calm'] * (Propeller_Diameter**4) * (-n**2 * KT_port - n**2 * KT_star)      # `n_P = n_S` 으로 가정 
            X_P_ = X_P / (0.5 * Model['rho_calm'] * LBP**2 * Realtime_U**2)
            N_P = yP * (1-tP) * Model['rho_calm'] * (Propeller_Diameter**4) * (-n**2 * KT_port + n**2 * KT_star)
            N_P_ = N_P / (0.5 * Model['rho_calm'] * LBP**3 * Realtime_U**2)

        X_R_ = -Coefficient['one_minus_tR'] * (FN_port_ + FN_star_) * sin(delta)
        Y_R_ = Coefficient['one_plus_aH'] * (FN_port_ + FN_star_) * cos(delta)
        N_R_ = Coefficient['xR_plus_aHxH'] * (FN_port_ + FN_star_) * cos(delta) - (Model['yR']/Model['LBP']) * Coefficient['one_minus_tR'] * (FN_port_ - FN_star_) * sin(delta)
        
        X_ = X_H_ + X_P_ + X_R_
        Y_ = Y_H_ + Y_R_
        N_ = N_H_ + N_P_ + N_R_

        udot = (X_/m11_)*(Realtime_U**2/LBP)
        vdot = ((m33_*Y_ - B_*N_)/(m22_*m33_ - B_*C_))*(Realtime_U**2/LBP)
        rdot = ((m22_*N_ - C_*Y_)/(m22_*m33_ - B_*C_))*(Realtime_U**2/LBP**2)
        
        acceleration_matrix = np.array([[udot], [vdot], [rdot]])  
        return acceleration_matrix