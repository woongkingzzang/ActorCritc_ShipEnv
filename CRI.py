
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from math import *
import numpy as np


class CRI:
    def __init__(self, L, B, Xo, Yo, Xt, Yt, Co, Ct, Vo, Vt):
        self.L = L      #타선의 길이 [m] from pram
        self.B = B      # [m]
        self.Xo = Xo    #자선 x좌표  [m]
        self.Yo = Yo    #자선 y좌표  [m] 
        self.Xt = Xt    #타선 x좌표  [m]
        self.Yt = Yt    #타선 y좌표  [m]
        self.Co = Co    #자선 Heading angle [rad]
        self.Ct = Ct    #타선 Heading angle [rad]
        self.Vo = Vo    #자선 속도   [knots]
        self.Vt = Vt    #타선 속도   [knots]
        self.ratio = 0.6

    def RD(self):
        '''Relative Distance, 자선과 타선 사이의 상대 거리'''
        result = sqrt(((self.Xt - self.Xo) ** 2) + ((self.Yt - self.Yo) ** 2)) + 0.0001
        return result

    def TB(self):
        '''True Bearing, 자선의 위치 기준 타선의 절대 방위'''
        Xot = self.Xt - self.Xo
        Yot = self.Yt - self.Yo
        result = atan2(Yot, Xot) % (2*pi)
        return result

    def RB(self):
        '''Relative Bearing, 자선의 Heading angle에 대한 타선의 방위'''
        if self.TB() - self.Co >= 0:
            result = self.TB() - self.Co
        elif self.TB() - self.Co < 0:
            result = self.TB() - self.Co + (2 * pi)
        return result

    def HAD(self):
        '''Heading angle difference'''
        result = self.Ct - self.Co
        if result < 0 :
            result += 2*pi
        return result

    def Vox(self):
        '''자선 x방향 속도'''
        result = self.Vo * cos(self.Co)
        return result

    def Voy(self):
        '''자선 y방향 속도'''
        result = self.Vo * sin(self.Co)
        return result    

    def Vtx(self):
        '''타선 x방향 속도'''
        result = self.Vt * cos(self.Ct)
        return result

    def Vty(self):
        '''타선 y방향 속도'''
        result = self.Vt * sin(self.Ct)
        return result

    def RV(self):
        '''Relative Velocity, 자선에 대한 타선의 상대속도'''
        Vrx = self.Vtx() - self.Vox()
        Vry = self.Vty() - self.Voy()
        result = sqrt(pow(Vrx, 2) + pow(Vry, 2)) + 0.001
        return result

    def RC(self):
        '''Relative speed heading direction, 상대속도(RV)의 방향'''
        Vrx = self.Vtx() - self.Vox()
        Vry = self.Vty() - self.Voy()
        result = atan2(Vry, Vrx) % (2*pi)
        return result

    def tcpa(self):
        pB_pA = (self.Xt - self.Xo, self.Yt - self.Yo)
        vA_vB = (self.Vox() - self.Vtx(), self.Voy()- self.Vty())

        numerator_tcpa = np.dot(pB_pA, vA_vB)
        denominator_tcpa = np.sqrt(np.dot(vA_vB, vA_vB) + 0.0001)

        result = numerator_tcpa / denominator_tcpa

        # result1 = self.RD() * cos()
        return result

    def dcpa(self):
        # result = self.RD() * sin(self.RC() - self.TB() - pi)
        # result = sin(pi/180)
        vector_DCPA = ((self.Xo + self.Vox() * self.tcpa()) - (self.Xt + self.Vtx() * self.tcpa()), (self.Yo + self.Voy() * self.tcpa()) - (self.Yt + self.Vty() * self.tcpa()))
        result1 = np.sqrt(np.dot(vector_DCPA, vector_DCPA) + 0.0001)
        # print (result, result1)      
        return result1

    def d1(self):
        '''minimum safety distance between encountered vessels, 단위 Nautical mile, Goodwin observation data'''
        if 0 <= self.RB() < np.deg2rad(112.5):
            result = 1.1 - 0.2 * (self.RB()/pi)
        elif np.deg2rad(112.5) <= self.RB() < np.deg2rad(180):
            result = 1.0 - 0.4 * (self.RB()/pi)
        elif np.deg2rad(180) <= self.RB() < np.deg2rad(247.5):
            result = 1.0 - 0.4 * ((2 * pi - self.RB())/pi)
        elif np.deg2rad(247.5) <= self.RB() <= np.deg2rad(360):
            result = 1.1 - 0.2 * ((2 * pi - self.RB())/pi)
        return result * self.ratio

    def d2(self):
        '''absolute safety distance, twice the value of d1'''
        result = 2 * self.d1()
        return result

    def UDCPA(self):
        '''#d1, d2의 범위에 따른 DCPA의 계수'''
        if abs(self.dcpa()) <= self.d1():
            result = 1
        elif self.d1() < abs(self.dcpa()) <= self.d2():
            result = 0.5 - 0.5 * sin((pi/(self.d2() - self.d1())) * (abs(self.dcpa()) - ((self.d1() + self.d2())/2)))
        elif self.d2() < abs(self.dcpa()):
            result = 0
        return result

    def t1(self):
        '''the time remaining until collision'''
        if abs(self.dcpa()) <= self.D1():
            result = sqrt(pow(self.D1(), 2) - pow(self.dcpa(), 2))/self.RV()
        elif abs(self.dcpa()) > self.D1():
            result = (self.D1() - abs(self.dcpa())) / self.RV()
        return result

    def t2(self):
        '''the collision avoidance time'''

        # result = sqrt(pow(12 * self.ratio, 2) - pow(self.dcpa(), 2))/self.RV()
        result = self.t1()*2

        return result

    def UTCPA(self):
        '''t1, t2의 범위에 따른 TCPA의 계수'''
        if 0 <= abs(self.tcpa()) <= self.t1():
            result = 1
        elif self.t1() < abs(self.tcpa()) <= self.t2():
            result = pow(((self.t2() - abs(self.tcpa()))/(self.t2() - self.t1())), 2)
        elif self.t2() < abs(self.tcpa()):
            result = 0
        return result

    def D1(self):
        '''critical safety distance, 12 times the length of the own vessel, 미터단위 해리로 변경'''
        result = 12 * self.L
        return result

    def D2(self):
        '''the distance at which the final action of collision avoidance can be taken,
        equal to R which is the radius of marine power model obtained by Davis'''
        result = (1.7 * cos(self.RB() - np.deg2rad(19))) + sqrt(4.4 + 2.89 * pow(cos(self.RB() - np.deg2rad(19)), 2))
        return result * self.ratio

    def UD(self):
        '''D1, D2의 범위에 따른 Relative distance의 계수'''
        if 0 <= self.RD() < self.D1():
            result = 1
        elif self.D1() <= self.RD() <= self.D2():
            result = pow((self.D2() - self.RD())/(self.D2() - self.D1()), 2)
        elif self.D2() < self.RD():
            result = 0
        return result

    def UB(self):
        '''Relative bearing에 대한 계수 UB'''
        result = 0.5 * (cos(self.RB() - np.deg2rad(19)) + sqrt((440/289) + pow(cos(self.RB() - np.deg2rad(19)), 2))) - (5/17)
        return result

    def K(self):
        '''Speed factor'''
        if self.Vt == 0 or self.Vo == 0:
            result = 1
        else:
            result = self.Vt / self.Vo
        return result

    def sinC(self):
        '''Collision angle, UK의 계산에 사용'''
        result = abs(sin(abs(self.Ct - self.Co)))
        return result

    def UK(self):
        '''Speed factor에 대한 계수 UK'''
        result = 1 / (1 + (2 / (self.K() * sqrt(pow(self.K(), 2) + 1 + (2 * self.K() * self.sinC())))))
        return result

    def CRI(self):
        '''충돌위험도지수, UDCPA, UTCPA, UD, UB, UK 5개의 파라미터에 가중치를 곱하여 계산'''
        result = 0.4457 * self.UDCPA() + 0.2258 * self.UTCPA() + 0.1408 * self.UD() + 0.1321 * self.UB() + 0.0556 * self.UK()
        return result

    def VCD(self):
        result = self.RB() - self.RB()
        # time step

        return result

    def encounter_classification(self):
        HAD = self.HAD() % (2 * pi)
        RB = self.RB() % (2 * pi)
        if HAD > (pi/2) and RB < (3*pi/2) and abs(RB-HAD) < (pi/2):
            return "Safe"
        else:
            if HAD >= (7*pi/8) and HAD < (9*pi/8):
                return "Head-on"
            elif HAD >= (9*pi/8) and HAD < (13*pi/8):
                return "Starboard crossing"
            elif HAD >= (3*pi/8) and HAD < (7*pi/8):
                return "Port crossing"
            else:
                RB2 = (pi + RB - HAD) % (2*pi)
                if RB >= (5*pi/8) and RB < (11*pi/8):
                    return "Overtaking"
                elif RB2 >= (5*pi/8) and RB2 < (11*pi/8):
                    return "Overtaken"
                elif RB < pi:
                    return "Starboard crossing"
                else:
                    return "Port crossing"

    # def encounter_classification(self):
    #     RB = np.rad2deg(self.RB())
    #     RC = np.rad2deg(self.HAD())
    #     # R1
    #     if 0 <= RB <= 22.5:
    #         if 67.5 <= RC < 157.5:
    #             return "Safe"
    #         elif 157.5 <= RC <= 202.5:
    #             return "Head-on"
    #         elif 202.5 < RC <= 292.5:
    #             return "Starboard crossing"
    #         else:
    #             if self.Vo > self.Vt:
    #                 return "Overtaking"
    #             else:
    #                 return "Safe"

    #     elif 337.5 <= RB <= 360:
    #         if 67.5 <= RC < 157.5:
    #             return "Port crossing"
    #         elif 157.5 <= RC <= 202.5:
    #             return "Head-on"
    #         elif 202.5 < RC <= 292.5:
    #             return "Safe"
    #         else:
    #             if self.Vo > self.Vt:
    #                 return "Overtaking"
    #             else:
    #                 return "Safe"

    #     # R2
    #     elif 22.5 < RB <= 90:
    #         if 0 <= RC < 157.5:
    #             return "Safe"
    #         elif 157.5 <= RC <= 202.5:
    #             return "Head-on"
    #         else:
    #             return "Starboard crossing"

    #     # R3
    #     elif 90 < RB <= 112.5:
    #         if 270 <= RC < 360:
    #             if self.Vt > self.Vo:
    #                 return "Overtaken"
    #             else:
    #                 return "Safe"
    #         else:
    #             return "Safe"
        
    #     # R4
    #     elif 112.5 < RB < 247.5:
    #         if 67.5 < RC < 292.5:
    #             return "Safe"
    #         else:
    #             if self.Vo >= self.Vt:
    #                 return "Safe"
    #             else:
    #                 return "Overtaken"

    #     # R5
    #     elif 247.5 <= RB < 270:
    #         if 0 <= RC < 90:
    #             if self.Vt > self.Vo:
    #                 return "Overtaken"
    #             else:
    #                 return "Safe"
    #         else:
    #             return "Safe"
        
    #     # R6
    #     elif 247.5 <= RB < 337.5:
    #         if 0 <= RC < 157.5:
    #             return "Port crossing"
    #         elif 157.5 <= RC <= 202.5:
    #             return "Head-on"
    #         else:
    #             return "Safe"

    # def PARK(self):
    #     RB = np.rad2deg(self.RB())
    #     RC = np.rad2deg(self.HAD())
    #     Tp = rospy.get_param("Type_Factor/Container")
    #     Tf = rospy.get_param("Ton_Factor/500t")
    #     Lf1 = rospy.get_param("Length_Factor/70m")
    #     Wf = rospy.get_param("Width_Factor/10m")
    #     Caf = rospy.get_param("Career_Factor/1y")
    #     Lf2 = rospy.get_param("License_factor/2nd")
    #     Pf = rospy.get_param("Position_Factor/Captain")
    #     L = self.L
        
    #     if 22.5 <= RC <= 67.5:
    #         Crf = rospy.get_param("Crossing_Factor/CR135")
    #     elif 67.5 < RC <= 112.5:
    #         Crf = rospy.get_param("Crossing_Factor/CR90")
    #     elif 112.5 < RC <= 157.5:
    #         Crf = rospy.get_param("Crossing_Factor/CR45")
    #     else:
    #         Crf = rospy.get_param("Crossing_Factor/HO")

        
    #     if 10 <= RB < 180:
    #         Sf = rospy.get_param("Side_Factor/Stbd")
    #     else:
    #         Sf = rospy.get_param("Side_Factor/Port")
        
    #     H = rospy.get_param("Harbor_Factor/outer")
        
    #     if self.Vt > self.Vo:
    #         Sp = rospy.get_param("Speed_Factor/TS")
    #     elif self.Vt == self.Vo:
    #         Sp = rospy.get_param("Speed_Factor/eq")
    #     else:
    #         Sp = rospy.get_param("Speed_Factor/OS")
        
    #     Sd = abs(self.Vt - self.Vo) * 0.5144
    #     D = self.RD() / 1852

    #     result = 5.081905 + Tp + Tf + Lf1 + Wf + Caf + Lf2 + Pf - 0.002517 * L + Crf + Sf + H + Sp - 0.00493 * Sd - 0.43071 * D
    #     return result

    # def CoE(self):
    #     '''Coefficients of encounter situations'''
    #     if self.encounter_classification() == "Head-on":
    #         s = abs(2 - (self.Vo - self.Vt)/self.Vo)
    #         t = 0.2
    #     elif self.encounter_classification() == "Starboard crossing" or self.encounter_classification() == "Port crossing":
    #         s = 2 - self.HAD()/pi
    #         t = self.HAD()/pi
    #     elif self.encounter_classification() == "Overtaking":
    #         s = 1
    #         t = 0.2
    #     else:
    #         s = abs(1 + (self.Vo - self.Vt)/self.Vo)
    #         t = abs(0.5 + (self.Vo - self.Vt)/self.Vo)
    #     return s, t

    # def ship_domain(self):
    #     KAD = pow(10, (0.3591 * log10(self.Vo) + 0.0952))
    #     KDT = pow(10, (0.5411 * log10(self.Vo) - 0.0795))
    #     AD = self.L * KAD
    #     DT = self.L * KDT

    #     s, t = self.CoE()

    #     R_fore = self.L + (0.67 * (1 + s) * sqrt(pow(AD,2) + pow(DT/2,2)))
    #     R_aft = self.L + (0.67 * sqrt(pow(AD,2) + pow(DT/2,2)))
    #     R_stbd = self.B + DT * (1 + t)
    #     R_port = self.B + (0.75 * DT * (1 + t))

    #     return R_fore, R_aft, R_stbd, R_port

    # def Rf(self):
    #     R_fore, R_aft, R_stbd, R_port = self.ship_domain()
    #     return R_fore

    # def Ra(self):
    #     R_fore, R_aft, R_stbd, R_port = self.ship_domain()
    #     return R_aft

    # def Rs(self):
    #     R_fore, R_aft, R_stbd, R_port = self.ship_domain()
    #     return R_stbd

    # def Rp(self):
    #     R_fore, R_aft, R_stbd, R_port = self.ship_domain()
    #     return R_port