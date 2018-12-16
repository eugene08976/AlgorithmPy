def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
    
def fibDict(n, d):
    if n in d.keys():
        return d[n]
    elif n == 0:
        d[n] = 0
        print ("Entry key=", n)
        return d[n]
    elif n == 1:
        d[n] = 1
        print ("Entry key=", n)
        return d[n]
    else:
        d[n] = fibDict(n-2, d) + fibDict(n-1, d)
        print ("Entry key=", n)
        return d[n]

def fibArray(n):
    
    a = [0] * (n+1)
    a[0] = 0
    a[1] = 1
    for i in range(2, n+1):
        a[i] = a[i-2] + a[i-1]
    return a[-1]
    
    def lcs(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0
    if x[0] == y[0]:
        print (x[0])
        ans = 1 + lcs(x[1:], y[1:])
    else:
        ans = max(lcs(x, y[1:]), lcs(x[1:], y))
    return ans
    
    def lcsDict(x, y, d):
    key = str(len(x)) + "-" + str(len(y))
    if key in d:
        return d[key]
    if len(x) == 0 or len(y) == 0:
        d[key] = 0
        print ("Insert Key=", key, d[key])
        return d[key]
    if x[0] == y[0]:
        ans = 1 + lcsDict(x[1:], y[1:], d)
    else:
        ans = max(lcsDict(x, y[1:], d), lcsDict(x[1:], y, d))
    d[key] = ans
    print ("Insert Key=", key, d[key])
    return d[key]
    
    def lcsArray(x, y):
    arr = [[0 for j in range(len(y)+1)] for i in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            if x[len(x)-i] == y[len(y)-j]:
                arr[i][j] = 1 + arr[i-1][j-1]
            else:
                arr[i][j] = max(arr[i-1][j], arr[i][j-1])
    print (arr)
    return
    
    def lcsArrayPrefix(x, y):
    arr = [[0 for j in range(len(y)+1)] for i in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            if x[i-1] == y[j-1]:
                arr[i][j] = 1 + arr[i-1][j-1]
            else:
                arr[i][j] = max(arr[i-1][j], arr[i][j-1])
    print (arr)
    return
    
    def ed(x, y):
    inscost, delcost, replacecost = 2, 1, 4
    if len(x) == 0:
        cost = delcost * len(y)
    elif len(y) == 0:
        cost = inscost * len(x)
    else:
        if x[0] == y[0]:
            cost = 0 + ed(x[1:], y[1:])
        else:
            cost = min(inscost + ed(x[1:], y),
                      delcost + ed(x, y[1:]),
                      replacecost + ed(x[1:], y[1:]))
    return cost
    
    def edArray(x, y):
    inscost, delcost, replacecost = 2, 1, 4
    arr = [[0 for j in range(len(y)+1)] for i in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y) + 1):
            if x[i-1] == y[j-1]:
                arr[i][j] = arr[i-1][j-1]
            else:
                arr[i][j] = min(delcost + arr[i][j-1],
                               inscost + arr[i-1][j],
                               replacecost + arr[i-1][j-1])
    print (arr)
    return
    
    def mcm(chain):
    if len(chain) <= 1:
        return 0
    else:
        cost = min([mcm(chain[:k])+
                   mcm(chain[k:]) +
                   chain[0][0] * chain[k-1][1] * chain[-1][1]
                   for k in range(1, len(chain))])
        return cost
        
import numpy as np


def mcmLoop(chain):
    a = np.zeros((len(chain), len(chain)))
    
    
def tj(tl, lw):
    if len(tl) == 0:
        return 0
    elif len(tl) == 1:
        return (lw - len(tl[0])) ** 2
    else:
        usage = len(tl[0])
        waste = [(lw-usage)**2]
        maxwords = 1
        while usage + 1 + len(tl[maxwords]) <= lw:
            usage = usage + 1 + len(tl[maxwords])
            waste.append((lw-usage)**2)
            maxwords += 1
            if maxwords == len(tl):
                break
    return min([waste[k] + tj(tl[k+1:], lw) for k in range(maxwords)])
    
    
# Text Justification; Use a Dictionary to record solutions to subproblems

def tjDict(tl, lw, d, i, j):
    
    key = str(i)+"-"+str(j)
    print ("Call with Key=", key)
    if key in d:
        return d[key]
    elif i >= j:
        d[key] = 0
        print ("Insert a new entry to dictionary ", key, d[key])
        return d[key]
    else:
        usage = len(tl[i])
        waste = [(lw-usage)**2]
        k = 1
        while (i + k < j and
               usage + 1 + len(tl[i+k]) <= lw):
            usage = usage + 1 + len(tl[i+k])
            waste.append((lw - usage)**2)
            k += 1
            
        print ("Key and Waste List=", key, waste)
        d[key] = min([w + tjDict(tl, lw, d, i+r+1, j) 
                      for r, w in enumerate(waste)])
        print ("Insert a new entry to dictionary  ", key, d[key])
        return d[key]
    
 # Text Justification using an one dimensional array

def tjloop(textlist, lw):
    textlen = len(textlist)
    cost = [0] * (textlen + 1)
    wordsinline = cost[:]
    
    for i in range(textlen-1, -1, -1):
        usage = len(textlist[i])
        waste = [(lw - len(textlist[i])) ** 2]
        j = i + 1
        while (j <= textlen - 1 and
              usage + 1 + len(textlist[j]) <= lw):
            usage = usage + 1 + len(textlist[j])
            waste.append((lw - usage) ** 2)
            j += 1
            
        cost[i],wordsinline[i] = min([(w + cost[i+1+k], k+1)
                                     for k, w in 
                                      enumerate(waste)])
    print ("Cost", cost)
    print ("Wordsinline", wordsinline)
    
    jtext = ""
    i = 0
    while i <= textlen-1:
        line = str(textlist[i])
        if i < textlen-1:
            for j in range(i+1, i+wordsinline[i]):
                line += " " + str(textlist[j])
        line += "\n"
        jtext += line
        i += wordsinline[i]
        
    print (jtext)
        
        
