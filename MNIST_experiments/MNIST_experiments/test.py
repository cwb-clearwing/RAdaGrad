import sys

lines = []
while True:
    line = sys.stdin.readline().strip()
    if not line:
        break
    lines.append(line)

n = int(lines[0])
rpages=lines[1].split(" ")
for i in range(n):
    rpages[i] = int(rpages[i])
newr=[]
for r in rpages:
    if rpages.count(r) >= int(lines[2]):
        newr.append(r)
        while r in rpages:
            rpages.remove(r)
if len(newr) > 0:
    print(len(newr))
    for i in range(len(newr)):
        print(newr[i])
else:
    print(0)