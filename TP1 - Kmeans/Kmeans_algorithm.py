import random
import math

e1 = (2, 2)
e2 = (1.5, 1.5)
e3 = (7, 2)
e4 = (6.5, 1.5)
e5 = (3, 2.5)
e6 = (7.5, 2.5)

pointList = [e1, e2, e3, e4, e5, e6]
processedPointList = []
centerList = list([e1, e2]) #m1 & m2
pointsBelonging = []

#list of list of point by center
for center in centerList:
	pointsBelonging.append([])
print(pointsBelonging)
stable = False

while not stable:
    
	oldCenters = list(centerList)
    
	for i in range(len(pointList)):
		
        pointToProcess = random.choice(pointList)
        chosenCenter = centerList[0]
        centerIndex = 0
		
    	for i in range(len(centerList)):
        	
            distanceFromChosenCenter = math.sqrt((pointToProcess[0] - chosenCenter[0])**2 + (pointToProcess[1] - chosenCenter[1])**2)
        	distanceFromTestCenter = math.sqrt((pointToProcess[0] - centerList[i][0])**2 + (pointToProcess[1] - centerList[i][1])**2)
        	
            if(distanceFromTestCenter < distanceFromChosenCenter):
            	chosenCenter = centerList[i]
            	centerIndex = i
   	 
    	pointsBelonging[centerIndex].append(pointToProcess)
   	 
    	#recalculate center of gravity
    	newX = 0
    	newY = 0
    	for j in range(len(pointsBelonging[centerIndex])):
        	newX += pointsBelonging[centerIndex][j][0]
        	newY += pointsBelonging[centerIndex][j][1]
    	newX = newX/(len(pointsBelonging[centerIndex])+1)
    	newY = newY/(len(pointsBelonging[centerIndex])+1)
   	 
    	print("chosen center  for point ", pointToProcess, " is : ", chosenCenter)
    	chosenCenter = (newX, newY)
    	centerList[centerIndex] = chosenCenter
    	print("New center coordinates : ", centerList[centerIndex])

    	processedPointList.append(pointToProcess)
    	pointList.remove(pointToProcess)
    
	print("New centers : ", centerList)
	print("Old centers : ", oldCenters)
    
	#decide if stable or not, using ||Ci+1 - Ci|| / ||Ci|| < Epsilon, Epsilon may be 0.001 for example
	stable = True


