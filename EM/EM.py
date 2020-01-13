import numpy as np

def EM(data, para):
    # init parameters
    theta = para
    while(True):
        # expectation
        p = np.zeros(data.shape[0], dtype=float)
        for i in range(data.shape[0]):
            a = theta[1]**np.sum(data[i]==1)*(1.-theta[1])**np.sum(data[i]==0)
            b = theta[2]**np.sum(data[i]==1)*(1.-theta[2])**np.sum(data[i]==0)
            p[i] = a/(a+b)
        # maximization
        molecule_1, molecule_2 = 0., 0.
        denominator_1, denominator_2 = 0., 0.
        for i in range(data.shape[0]):
            molecule_1 += p[i]*np.sum(data[i]==1)
            denominator_1 += 10.*p[i]
            molecule_2 += (1.-p[i]) * np.sum(data[i] == 1)
            denominator_2 += 10. * (1.-p[i])
        tmp_1, tmp_2 = molecule_1/denominator_1, molecule_2/denominator_2
        # terminate condition
        if abs(tmp_1-theta[1])<=0.00001 and abs(tmp_2-theta[2])<=0.00001:
            break
        theta[1], theta[2] = tmp_1, tmp_2
    return theta

def GMM():
    return

if __name__ == '__main__':
    data = np.array([[1,0,0,0,1,1,0,1,0,1],
                     [1,1,1,1,0,1,1,1,1,1],
                     [1,0,1,1,1,1,1,0,1,1],
                     [1,0,1,0,0,0,1,1,0,0],
                     [0,1,1,1,0,1,1,1,0,1]], dtype=int)
    # 第一个参数固定，第二、三个参数"手动"初始化（初始化敏感）
    para = np.array([0.5,0.6,0.4], dtype=float)
    para = EM(data, para)
    print(para)