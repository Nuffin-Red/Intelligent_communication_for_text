import numpy as np
import time
import random
import math

#16根天线，7个信道，3个用户，16个反射面单元
IRS_Location=[75,100,20]
BS_Location=[0,0,40]
noise=math.pow(10,-169/ 10)

user_location=[[110.60324797205269, 61.720988988759, 0], [120.66997496909893, 58.844299918250286, 0], [120.0507004898175, 80.14087207918632, 0]]



class hd_channels(object):
    def __init__(self,fadding_factor_hd,M,N,K,C):
        self.BS_Location=[0,0,40]
        self.user=user_location     #3*3
        self.fadding_factor_hd=fadding_factor_hd          #直射链路的衰落因子
        self.M=M    #天线
        self.N=N    #反射面
        self.K=K    #用户
        self.C=C    #信道
    def Large_scale_fadding(self):
        fadding_hd=[]
        for i in range(self.K):
            B_u_distance=math.sqrt(math.pow(user_location[i][0],2)+math.pow(user_location[i][1],2)\
                         +math.pow(self.BS_Location[2],2))

            fadding_hd.append(math.sqrt(math.pow(10, 3)*math.pow(B_u_distance,-self.fadding_factor_hd)))
        h_d_rayleigh_component=np.sqrt(0.5)*(np.random.randn(self.C,self.M,self.K)
                                             +1j*np.random.randn(self.C,self.M,self.K))

        h_d=fadding_hd*h_d_rayleigh_component
        return h_d

class Reflection_path(object):
    def __init__(self,fadding_factor_BR,fadding_factor_Ru,M,N,K,C,K1,K2,N1,N2,M1,M2):
        self.fadding_factor_BR=fadding_factor_BR
        self.fadding_factor_Ru=fadding_factor_Ru
        self.BS_Location = [0, 0, 40]
        self.IRS_Location=[75,100,20]
        self.user=user_location
        self.K1=K1
        self.K2=K2
        self.M=M
        self.N=N
        self.K=K
        self.C=C
        self.N1=N1
        self.N2=N2
        self.M1=M1
        self.M2=M2
    def BS_IRS_fadding(self):
        d=0.5
        B_R_distance = math.sqrt(math.pow(self.IRS_Location[0], 2) + math.pow(self.IRS_Location[1], 2) + math.pow(self.IRS_Location[2]-self.BS_Location[2], 2))
        fadding_hBR=math.sqrt(math.pow(10, 3)* math.pow(B_R_distance, -self.fadding_factor_BR))
        hBR_tilde = np.sqrt(0.5) * (
                    np.random.randn(self.C, self.M, self.N) + 1j * np.random.randn(self.C, self.M, self.N))

        BS_AOD_x=math.atan(self.IRS_Location[0]/self.IRS_Location[1])
        BS_AOD_z=math.atan(-(0-math.sqrt(math.pow(self.IRS_Location[1],2)+math.pow(self.IRS_Location[0],2)))/(self.BS_Location[2]-self.IRS_Location[2]))
        IRS_AOA_x=math.atan(-self.IRS_Location[0]/self.IRS_Location[1])
        IRS_AOA_z=math.atan((0-math.pow(self.IRS_Location[1],2)+math.pow(self.IRS_Location[0],2))/(self.BS_Location[2]-self.IRS_Location[2]))
        #h_BR = np.zeros((self.C, self.M,self.N), dtype=complex)
        a1 = (1 / math.sqrt(self.M1)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.M1, 1).reshape(self.M1, 1))
                                               * d * math.cos(BS_AOD_z))
        a2 = (1 / math.sqrt(self.M2)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.M2, 1).reshape(self.M2, 1))
                                               * d * math.sin(BS_AOD_x)*math.sin(BS_AOD_z))
        a = np.kron(a1, a2)         #[0,1,2],[3,4,5]---[0,0,0,3,4,5,6,8,10]
        b1 = (1 / math.sqrt(self.N1)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.N1, 1).reshape(self.N1, 1))
                                               * d * math.cos(IRS_AOA_z))
        b2 = (1 / math.sqrt(self.N2)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.N2, 1).reshape(self.N2, 1))
                                               * d * math.sin(IRS_AOA_x) * math.sin(IRS_AOA_z))
        b = np.kron(b1, b2)
        hBR_hat=np.matmul(a,b.T)
        hBR_hat=np.expand_dims(hBR_hat,0)
        hBR_hat=np.repeat(hBR_hat,self.C,0)
        h_BR=math.sqrt(self.K1/(1+self.K1))*hBR_hat+math.sqrt(1/(self.K1+1))*hBR_tilde
        h_BR=fadding_hBR*h_BR
        return h_BR

    def IRS_user_fadding(self):
        d=0.5
        h_Ru=[]
        for k in range(self.K):
            R_u_distance = math.sqrt(math.pow(self.IRS_Location[0] - self.user[k][0], 2) + math.pow(self.IRS_Location[1]- self.user[k][1], 2) + math.pow(self.IRS_Location[2], 2))
            fadding_hRu = math.sqrt(math.pow(10, 3) * math.pow(R_u_distance, -self.fadding_factor_Ru))
            Ru_AOA_x=math.atan((self.IRS_Location[0]-self.user[k][0])/(self.IRS_Location[1]-self.user[k][1]))
            Ru_AOA_z=math.atan((-self.IRS_Location[2])/(math.sqrt(math.pow(self.IRS_Location[0]-self.user[k][0],2)+math.pow(self.IRS_Location[1]-self.user[k][1],2))))
            #单天线用户不考虑到达角
            a1 = (1 / math.sqrt(self.N1)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.N1, 1).reshape(self.N1, 1))
                                                   * d * math.cos(Ru_AOA_z))
            a2 = (1 / math.sqrt(self.N2)) * np.exp((-1j) * 2 * np.pi * (np.arange(0, self.N2, 1).reshape(self.N2, 1))
                                                   * d * math.sin(Ru_AOA_x) * math.sin(Ru_AOA_x))
            a=np.kron(a1,a2.T)
            h_R_uk_hat=np.reshape(a,(1,self.N))
            h_R_uk_hat=np.expand_dims(h_R_uk_hat, 0)
            h_R_uk_hat = np.repeat(h_R_uk_hat, self.C, 0)
            h_R_uk_hat=np.squeeze(h_R_uk_hat)

            h_R_uk_tilde = np.sqrt(0.5) * (
                    np.random.randn(self.C, self.N) + 1j * np.random.randn(self.C, self.N))
            h_R_uk=math.sqrt(self.K2/(1+self.K2))*h_R_uk_hat+math.sqrt(1/(self.K2+1))*h_R_uk_tilde

            h_R_uk=fadding_hRu*h_R_uk
            h_Ru.append(h_R_uk)
        h_Ru=np.reshape(h_Ru,(self.C, self.N, self.K))
        return h_Ru

#每一轮训练的初始状态
class reset_init_state(object):
    def __init__(self,C,K,M,N,PT,hd,hru,hbr,N1,N2,B,snr_min):
        self.C=C
        self.K=K
        self.M=M
        self.N=N
        self.B=B
        self.N1=N1
        self.N2=N2
        self.PT=PT
        self.hd=hd
        self.hru=hru
        self.hbr=hbr
        self.snr_min=snr_min
    #包括离散动作7+连续动作16+6*7+等效信道状态7*6*2+和速率1+用户传输速率3+是否满足最小速率要求3（0或1）,《《《实虚部分离》》》
    def get_init_state(self):
        h_=[]
        R_all=0
        R=[]

        snr=np.zeros((self.K), dtype=np.float64)
        d_action=np.zeros((self.C), dtype=np.int32)
        c_action=np.zeros((self.N+self.M*self.C), dtype=np.float64)
        hc=np.zeros((2*self.C*self.M), dtype=np.float64)
        r_require=[]
        for i in range(self.C):
            d_action[i]=random.randint(0,self.K-1)
        for j in range(self.N):
            c_action[j]=2*math.pi*random.random()
        for l in range(self.C):
            s=[random.random() for _ in range(self.M)]
            norm=math.pow(np.sum([math.pow(s[t],2) for t in range(self.M)],axis=0),0.5)
            for u in range(self.M):
                c_action[self.N+l*self.M+u]=(s[u]/norm)*math.pow(self.PT,0.5)
        for e in range(self.C):
            h_d=np.reshape(self.hd[e,:,d_action[e]],(self.M,1))            #第一个信道，使用者的信道状态，M*1
            h_br=np.reshape(self.hbr[e,:,:],(self.N,self.M))
            h_ru=np.reshape(self.hru[e,:,d_action[e]],(self.N,1))
            a = np.expand_dims(c_action[0:self.N], 1)
            a = np.repeat(a, self.N, 1)
            #h=h_d.T+h_ru.T*(np.exp(1j*np.diag(np.diag(a)))*np.eye(self.N))*h_br
            h=h_d.T+np.dot(h_ru.T,np.dot(np.exp(1j*np.diag(np.diag(a)))-1+np.eye(self.N),h_br))
            #h = h_d.T
            h_.append(h)
            for r in range(self.M):
                hc[e*self.M+r]=h[0][r].real
                hc[(self.C+e)*self.M+r]=h[0][r].imag
        for t in range(self.C):
            snr_c_1=h_[t] * c_action[self.N + t * self.M:self.N + (t + 1) * self.M]

            snr_c_2=math.sqrt(np.sum([math.pow(snr_c_1[0][t],2) for t in range(len(snr_c_1[0]))],axis=0))

            snr_c=(self.B/self.C)*math.log(1+(snr_c_2/noise),2)/1e10

            R_all+=snr_c
            for k in range(self.K):
                if d_action[t]==k:
                    snr[k]+=snr_c
        for y in range(self.K):
            if self.snr_min[y]<snr[y]:
                r_require.append(1)
            else:
                r_require.append(0)
        #SNR_min的判断，到底该设置在什么范围,该设置多大
        R.append(R_all)
        return  np.concatenate((d_action,c_action,hc,R,snr,r_require))

class calculation_SNR_and_get_next_state(object):
    def __init__(self,snr_min,hd,hru,hbr,B,M,K,N,C,PT):
        self.C = C
        self.K = K
        self.M = M
        self.N = N
        self.B=B
        self.snr_min=snr_min
        self.hd=hd
        self.hru=hru
        self.hbr=hbr
        self.PT=PT

    def SNR_and_get_next_state(self,daction,caction):
        r_require=[]
        R=[]
        r_per_channel=[]
        snr=np.zeros((self.K), dtype=np.float64)
        h_equivalent = np.zeros((2 * self.C * self.M), dtype=np.float64)
        SNR_all=0
        reward = 0
        u_num =[0,0,0]
        d_action=daction.copy()
        d_action=d_action[0]
        c_action=caction.copy()
        c_action=c_action[0]
        norm = math.pow(np.sum([math.pow(c_action[self.N+t], 2) for t in range(self.K*self.M)], axis=0), 0.5)
        for e in range(self.C):
            h_d=abs(np.reshape(self.hd[e,:,d_action[e]],(self.M,1))   )         #第一个信道，使用者的信道状态，M*1
            h_br=abs(np.reshape(self.hbr[e,:,:],(self.N,self.M)))
            h_ru=abs(np.reshape(self.hru[e,:,d_action[e]],(self.N,1)))
            a = np.expand_dims(c_action[0:self.N], 1)
            a = np.repeat(a, self.N, 1)
            #h=h_d.T+h_ru.T*np.exp(1j*math.pi*np.reshape(np.diag(np.diag(a))))*h_br
            h=h_d.T+np.dot(h_ru.T,np.dot(np.exp(1j*math.pi*np.diag(np.diag(a)))-1+np.eye(self.N),h_br))
            #h=h_d.T
            for r in range(self.M):
                h_equivalent[e*self.M+r]=h[0][r].real
                h_equivalent[(self.C+e)*self.M+r]=h[0][r].imag
            snr_c_1=(1/norm) * c_action[self.N + e * self.M:self.N + (e + 1) * self.M]*math.sqrt(self.PT)*h
            snr_c_2=math.sqrt(np.sum([math.pow(snr_c_1[0][t],2) for t in range(len(snr_c_1[0]))],axis=0))
            snr_c=(self.B/self.C)*math.log(1+(snr_c_2/noise),2)/1e10
            reward+=snr_c
            SNR_all+=snr_c
            for k in range(self.K):
                if d_action[e]==k:
                    snr[k]+=snr_c
                    u_num[k]+=1
        for i in range(self.K):
            if self.snr_min[i]>snr[i]:
                reward-=snr[i]
                r_require.append(0)
            else:
                r_require.append(1)
        R.append(SNR_all)
        return reward,h_equivalent,snr,SNR_all,np.concatenate((d_action,c_action,h_equivalent,R,snr,r_require))



















