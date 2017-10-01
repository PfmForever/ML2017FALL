import pandas
import sys
mydata = pandas.read_csv(sys.argv[1],encoding = 'big5')
print(mydata)