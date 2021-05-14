with open('./data/inputData.txt') as f:
    lines = f.readlines()
ans = ""
#print(lines)
for i in range(0, len(lines)-1):
    if(len(lines[i]) >= 2 ):
        ans = ans + lines[i][0: len(lines[i])-2]
ans = ans + ". " + lines[len(lines)-1]
#print (ans)