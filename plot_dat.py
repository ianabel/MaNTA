# %%
import numpy as np
import matplotlib.pyplot as plt
import parse
header_format = "# {0}\t{1} {2}\t{3} {4}\t{5} {6}"
time_format = "# t = {}"
data_format = "{}\t"

def solution_FISHER(x,t):
    c = 5.0/np.sqrt(6.0)
    z = x-c*t
    C = 1.0
    S = 1.0 + C * np.exp( z / np.sqrt( 6.0 ) )
    return 1.0 / ( S * S )

def solution_NonLinear(x,t):
    n = 2
    eta = x/np.sqrt(1+t)
    eta[eta>=1.0]=0.0
    return np.power(1-eta,1/n)

def main():
    with open("./Config/3VarCyl.dat",'rt') as data:
        count = 0
        time = 0
        index = 0
        nVars = 3
        headings = ""
        line_begin = False
        u = np.ndarray(shape=(301,nVars))
        U = np.ndarray(shape=(1,301,nVars))
        q =  np.ndarray(shape=(301,nVars))
        Q = np.ndarray(shape =(1,301,nVars))
        sigma = np.ndarray(shape=(301,nVars))
        Sigma = np.ndarray(shape=(1,301,nVars))
        x = np.ndarray(shape=(301))
        t =[]
        data_format_n=data_format*nVars*4+data_format.strip("\t")+"\n"
        for line in data:
            count += 1
            # headers = 
            # for i in range(0,nVars):

            if count == 3:
                h = line
                headings = parse.parse(header_format,line)

            elif count > 3:
                if(line.startswith("#")):
                    U = np.vstack((U,np.expand_dims(u,0)))
                    Q = np.vstack((Q,np.expand_dims(q,0)))
                    Sigma = np.vstack((Sigma,np.expand_dims(sigma,0)))
                    line_begin = True
                    index = 0
                    time = parse.parse(time_format,line)
                    t.append(float(time[0].strip("\n")))
                elif(line == "\n"):
                    line_begin = False
                   
                      
                elif(line_begin):
                    l = parse.parse(data_format_n,line)
                    x[index] = float(l[0])
                    for i in range(0,nVars):
                        u[index,i] = float(l[4*i+1])
                        q[index,i] = float(l[4*i+2])
                        sigma[index,i] = float(l[4*i+3].strip("\n"))
                    index += 1

        U = np.vstack((U,np.expand_dims(u,0)))
        Q = np.vstack((Q,np.expand_dims(q,0)))
        Sigma = np.vstack((Sigma,np.expand_dims(sigma,0)))
        ind = -1
        for i in range(0,nVars):
            plt.figure()
            ax = plt.axes()
            
            ax.plot(x,Sigma[2,:,i],label = "t=0")
            ax.plot(x,Sigma[3,:,i],label = "one time step")
            # ax.plot(x,Sigma[4,:,0],label = "two")
            # ax.plot(x,Sigma[5,:,0],label = "three")
            ax.plot(x,Sigma[-1,:,i],label = "end")
            ax.legend()
            plt.show()
        

            plt.figure()
            plt.plot(x,Q[2,:,i])
            plt.plot(x,Q[3,:,i])
            plt.show()
            plt.figure()
            plt.plot(x,U[2,:,i])
            plt.plot(x,U[3,:,i])
            plt.plot(x,U[-1,:,i])
            plt.show()
            plt.figure()

      
     
      #  [X,T] = np.meshgrid(x,t)
        # SOL = solution_NonLinear(X,T)

        # plt.figure()
        # for i in range(0,nVars):
        #     plt.plot(x,U[ind,:,i])
        #   #  plt.plot(x,SOL[ind,:])
        # plt.show()

        # plt.figure()
        # ax = plt.axes()
        # for i in range(0,nVars):
        #     ax.plot(x,Q[ind,:,i],label=str(i))
        # ax.legend()
        # plt.show()

        # plt.figure()

        # for i in range(0,nVars):
        #     plt.plot(x,Sigma[ind,:,i])
        # plt.show()

        data.close()
        

if __name__ == "__main__":
    main()




# %%
