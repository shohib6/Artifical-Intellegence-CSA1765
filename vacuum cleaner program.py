a=[[1,0,1,0],[1,1,1,1],[1,0,1,1],[1,0,1,1]]
print("Room With dust are represented as 1 and Room With NO Dust represented as 0\nRoom Structure with and without Dirt\n",a)
print("AGENT is Cleaning")
for i in range(4):
    for j in range(4):
        if(a[i][j]==1):
            print("Agent Cleaned Location",i,j)
            a[i][j]=0
    print("Agent Cleaned Room",i+1)
print("Room After Cleaning \n",a)
