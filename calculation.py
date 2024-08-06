from ATI import ATISensor
import time
import math

class cal_cantact_points():
    def __init__(self):
        self.a1 = 0
        self.b1 = 0
        self.e1 = math.sqrt(self.a1**2-self.b1**2)
        self.a2 = 0
        self.b2 = 0
        self.e2 = math.sqrt(self.a2**2-self.b2**2)

    def cal_cd(self,force):
        c1 = (self.a1 * force[1]) ** 2 + (self.b1 * force[0])
        c2 = (self.a2 * force[1]) ** 2 + (self.b2 * force[0])
        a12=self.a1**2
        a22=self.a2**2
        d1=math.sqrt((-2*a12*force[5]*force[1]-2*self.e1*(self.b1*force[0])**2)**2)


if __name__ == '__main__':
    sensor = ATISensor()
    t0 = time.time()
    for i in range(1000):

        ft = sensor.get_measurement()
        print(ft)
    print("The frequency of ATI is ", 1000/(time.time()-t0))