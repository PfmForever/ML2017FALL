import sys
filename = sys.argv[1]
f = open(filename,'r',encoding = 'UTF-8')
fw = open('Q1.txt','w',encoding = 'UTF-8')
content = f.read()
contentSplit = content.split()
#print(contentSplit)
contentAppear = []
appear = 0
for idx in contentSplit:
    if(not(idx in contentAppear)):
        contentAppear.append(idx)
        fw.write(idx+' ')
        fw.write(str(appear)+' ')
        fw.write(str(contentSplit.count(idx))+'\n')
        #print(idx,appear,contentSplit.count(idx))
        appear = appear+1
fw.close()
f.close()
