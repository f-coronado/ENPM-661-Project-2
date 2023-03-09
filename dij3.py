import time
from heapq import heappush, heapify, heappop, heappushpop, nlargest, nsmallest, heapreplace
import cv2
import numpy as np
from copy import deepcopy

startTime = time.time()

################################ Step One #########################################

# node = (C2C, node index, parent node index, (x, y)) .. type is tuple
def moveRight(node):
    x, y = node[3]
    newX = x + 1
    cost = 1
    newNode = (cost, node[1], node[2], (newX, y)) # only update location after performing move
    return newNode

def moveLeft(node):
    x, y = node[3]
    newX = x - 1
    cost = 1
    newNode = (cost, node[1], node[2], (newX, y))
    return newNode

def moveUp(node):
    x, y = node[3]
    newY = y + 1
    cost = 1
    newNode = (cost, node[1], node[2], (x, newY))
    return newNode

def moveDown(node):
    x, y = node[3]
    newY = y - 1
    cost = 1
    newNode = (cost, node[1], node[2], (x, newY))
    return newNode

def moveUpRight(node):
    x, y = node[3]
    newX, newY = x + 1, y + 1
    cost = 1.4
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveUpLeft(node):
    x, y = node[3]
    newX, newY = x - 1, y + 1
    cost = 1.4
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveDownRight(node):
    x, y = node[3]
    newX, newY = x + 1, y - 1
    cost = 1.4
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveDownLeft(node):
    x, y = node[3]
    newX, newY = x - 1, y - 1
    cost = 1.4
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

################################ Step One #########################################

################################ Step Two #########################################

canvas = np.zeros((250, 600, 3), dtype = np.uint8) # initializing 250x600 canvas with 3 channels, RGB, and a black background. zeros fills every element in array with a value of zero
                                                   # using uint8 type bc we're using RGB [0, 255]

red = (0, 0, 255) # used for obstacles
yellow = (0, 255, 255) # used for clearance

# creating wall clearance
cv2.rectangle(canvas, (0, 0), (600, 5), yellow, thickness = -1) # thickness fills in circle
cv2.rectangle(canvas, (0, 5), (5, 245), yellow, thickness = -1)
cv2.rectangle(canvas, (0, 250), (600, 245), yellow, thickness = -1)
cv2.rectangle(canvas, (595, 5), (600, 245), yellow, thickness = -1)

# creating yellow obstacle clearances then stacking the red obstacle on top
cv2.rectangle(canvas, (95, 0), (155, 105), yellow, thickness = -1)
cv2.rectangle(canvas, (100, 0), (150, 100), red, thickness = -1)
cv2.rectangle(canvas, (95, 145), (155, 250), yellow, thickness = -1)
cv2.rectangle(canvas, (100, 150), (150, 250), red, thickness = -1)

hexagonClearance = np.array([[300, 206], [370, 166], [370, 84], [300, 44], [230, 84], [230, 166]]) # rounded down bc fillPoly needs int values
cv2.fillPoly(canvas, [hexagonClearance], yellow)
hexagon = np.array([[300, 200], [364, 162], [364, 87], [300, 50], [235, 87], [235, 162]]) # rounded down bc fillPoly needs int values
cv2.fillPoly(canvas, [hexagon], red)

triangleClearance = np.array([[455, 20], [460, 20], [515, 125], [460, 230], [455, 230]])
cv2.fillPoly(canvas, [triangleClearance], yellow)
triangle = np.array([[460, 25], [460, 225], [510, 125]])
cv2.fillPoly(canvas, [triangle], red)

def checkObstacleSpace(node):

    x = node[3][0]
    y = node[3][1]

    # hexagon equations
    m1hex = ((166 - 206)/(370 - 300)) # slope of bottom right hexagon line
    b1hex = 166 - m1hex * 370
    h1hex = y - (m1hex * x + b1hex)
    m2hex = ((84 - 44)/(370 - 300)) # slope of top right hexagon line
    b2hex = 84 - m2hex * 370
    h2hex = y - (m2hex * x + b2hex)
    m3hex = m1hex # slope of top left hexagon line is the same as bottom right
    b3hex = 44 - m3hex * 300
    h3hex = y - (m3hex * x + b3hex)
    m4hex = m2hex # slope of bottom left is the same as top right
    b4hex = 166 - m4hex * 230
    h4hex = y - (m4hex * x + b4hex)
    # print("m1hex = ", m1hex, "b1hex = ", b1hex, "h1hex = ", h1hex)
    # print("m2hex = ", m2hex, "b2hex = ", b2hex, "h2hex = ", h2hex)
    # print("m3hex = ", m3hex, "b3hex = ", b3hex, "h3hex = ", h3hex)
    # print("m4hex = ", m4hex, "b4hex = ", b4hex, "h4hex = ", h4hex)

    # triangle equations
    m1tri = ((125 - 25)/(510 - 460))
    b1tri = 125 - m1tri * 510
    h1tri = y - (m1tri * x + b1tri)
    m2tri = ((125 - 225)/(510 - 460))
    b2tri = 225 - m2tri * 460
    h2tri = y - (m2tri * x + b2tri)
    # print("m1tri & b1 tri = ", m1tri, b1tri, "\nm2tri & b2tri = ", m2tri, b2tri)

    # checking if point is in obstacle space
    if x >= 95 and x <= 155 and y >= 145 and y <= 250 or \
        x >= 95 and x <= 155 and y >= 0 and y <= 105 or \
        x >= 230 and x <= 370 and y >= 44 and y <= 206 and h1hex <= 0 and h2hex >= 0 and h3hex >= 0 and h4hex <= 0 or \
        x >= 460 and h1tri >= 0 and h2tri <= 0 or \
        x <= 5 or x >= 595 or y <= 5 or y >= 245:
        return "In Obstacle Space"
    return "Not in obstacle space"

# cv2.imshow("Empty canvas", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################ Step Two #########################################

############################### Step Three ########################################

def findChildren(node):

    node1 = deepcopy(node)
    node2 = deepcopy(node)
    node3 = deepcopy(node)
    node4 = deepcopy(node)
    node5 = deepcopy(node)
    node6 = deepcopy(node)
    node7 = deepcopy(node)
    node8 = deepcopy(node)

    children = []

    newNodeUp = moveUp(node1)
    # heappush(children, newNodeUp)
    children.append(newNodeUp)

    newNodeUpRight = moveUpRight(node2)
    # heappush(children,newNodeUpRight)
    children.append(newNodeUpRight)

    newNodeRight = moveRight(node7)
    # heappush(children,newNodeRight)
    children.append(newNodeRight)

    newNodeDownRight = moveDownRight(node5)
    # heappush(children,newNodeDownRight)
    children.append(newNodeDownRight)

    newNodeDown = moveDown(node4)
    # heappush(children,newNodeDown)
    children.append(newNodeDown)

    newNodeDownLeft = moveDownLeft(node6)
    # heappush(children,newNodeDownLeft)
    children.append(newNodeDownLeft)

    newNodeLeft = moveLeft(node8)
    # heappush(children,newNodeLeft)
    children.append(newNodeLeft)

    newNodeUpLeft = moveUpLeft(node3)
    # heappush(children,newNodeUpLeft)
    children.append(newNodeUpLeft)


    # heapify(children)
    # print("children: ", children)

    return children

def goalNodeReached(node, goalNode):
    if node[3] == goalNode[3]:
        return True
    return False

def dijkstra(startNode, goalNode):

    openList = []
    openList.append(startNode)
    closedList = []
    locations = [startNode[3]]

    currentNode = openList[0]
    i = 0
    while openList and goalNodeReached(currentNode, goalNode) == False:
        i = i + 1
        # print("length before popping: ", len(openList))
        # nodeIndex = len(openList)
        currentNode = heappop(openList)
        # print("length after popping: ", len(openList))
        print("\nCurrent Node = ", currentNode) 
        closedList.append(currentNode)

        if goalNodeReached(currentNode, goalNode) == True:
                # insert backtrack function
                return "SUCCESS"
        else:
            index = 0
            for c in findChildren(currentNode):
                for cost in range(len(openList)):
                    if openList[cost][3] == c[3]:
                        print("duplicate location found!")
                        # c[1] = openList[cost][1] # 
                if c not in closedList and checkObstacleSpace(c) == "Not in obstacle space":
                    childC2C = currentNode[0] + c[0] # adding cost of action to current C2C
                    # print("cost: ", childC2C)
                    if c not in openList or childC2C < currentNode[0]:
                        if c not in openList:
                            nodeIndex = len(openList) + len(closedList)
                            # node = (C2C, node index, parent node index, (x, y)) .. type is tuple
                            childNode = (childC2C, nodeIndex, currentNode[1], c[3]) # if the child has not been checked OR we found a lower childC2C, assign it the currentNode as the parent
                            # print("childNode",index, ": ", childNode)
                            openList.append(childNode)
                        else:
                            index = openList.index(c)
                            print("index: ", index)
                            if childNode[0] < childC2C:
                                openList[index] = c 
                            else: 
                                openList[index]
                            print("updating: ", openList[index], "to c: ", c)
        # print("openList before heapify: ", openList)
                index = index + 1
                # print("childNode",index, ": ", childNode)
                print("c: ", c)
        # heapify(openList)
        # print("openList length: ", len(openList))
        # print("openList after heapify: ", openList, "\nclosedList: ", closedList)
        if i == 4:
            heapify(openList)
            print("closedList: ",closedList, "\nopenList length: ", len(openList))
            print("openList after heapify: ", openList, "\nclosedList: ", closedList)
            for node in openList:
                print(node[3])
            break
    return "FAILURE"

############################### Step Three ########################################

############################### Step Four #########################################
############################### Step Four #########################################

############################### Step Five #########################################
############################### Step Five #########################################

# startNode = (0, 0, None, (300, 125)) # initializing startNode so we enter the loop
# while checkObstacleSpace(startNode) == "In Obstacle Space":
#     xStart = int(input("enter the x coord of the start node: "))
#     yStart = int(input("enter the y coord of the start node: "))

#     # node = (C2C, node index, parent node index, (x, y)) .. type is tuple
#     startNode = (None, 0, None, (xStart, yStart))
#     if checkObstacleSpace(startNode) == "In Obstacle Space":
#         print("\nIn Obstacle space, please try again...")

# goalNode = (None, 0, None, (500, 125)) # initializing goalNode so we enter the loop
# while checkObstacleSpace(goalNode) == "In Obstacle Space":
#     xGoal = int(input("\nenter the x coord of the goal node: "))
#     yGoal = int(input("enter the y coord of the goal node: "))
#     goalNode = (None, None, None, (xGoal, yGoal))
#     if checkObstacleSpace(goalNode) == "In Obstacle Space":
#         print("\nIn Obstacle space, please try again...")

startNode = (0, 0, None, (7, 7))
print("Node format: (C2C, Node index, parent index, (x,y)) \nstartNode: ", startNode)
goalNode = (0, 0, None, (591, 245))
# findChildren(startNode)
dijkstra(startNode, goalNode)

# print("startNode: ", startNode, "\nnewNode after moving: ", newNode)

# print("\nlocation: ", startNode[3], checkObstacleSpace(startNode))

endTime = time.time()
print("\nrun time = ", endTime - startTime, "seconds")