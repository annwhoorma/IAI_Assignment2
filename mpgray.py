import cv2
import numpy as np 

import math

from random import randint, random

import time

RADIUS = 40
MAX_DIST_BTW_AGENTS = 40
PLG_MAX_SIZE = 50
PLG_MIN_SIZE = 30
INIT_AMOUNT_OF_AGENTS = 90
TIME_LIMIT_IN_SEC = 3600


def partition(arr, low, high): 
    i = (low - 1)
    pivot_score = arr[high].score
  
    for j in range(low, high):
        if arr[j].score >= pivot_score:
            i = i+1 
            arr[i], arr[j] = arr[j], arr[i] 
  
    arr[i+1], arr[high] = arr[high], arr[i+1] 
    return (i+1) 


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Circle:
    def __init__(self):
        pass

    def generate(self):
        self.radius = RADIUS
        self.center = Point(np.random.randint(self.radius, 512-self.radius), np.random.randint(self.radius, 512-self.radius))

    # method to create a frame around the circle and computes some parameters for technical purposes
    def form_rect(self):
        self.left_up = Point(self.center.x-self.radius, self.center.y+self.radius)
        self.bottom_right = Point(self.center.x+self.radius, self.center.y-self.radius)

        self.dy = self.left_up.y - self.bottom_right.y + 1
        self.dx = self.bottom_right.x - self.left_up.x + 1

    # method to check if a certain point is inside or not
    def is_inside(self, p):
        pp = Point(p.x + self.left_up.x, p.y + self.bottom_right.y)
        if math.sqrt(pow(self.center.x-pp.x, 2) + pow(self.center.y-pp.y, 2)) <= self.radius:
            return True
        return False

class Rectangle:
    def __init__(self):
        pass

    def generate(self):
        x0 = np.random.randint(0, 512)
        y0 = np.random.randint(0, 512)
        x1, y1 = -1, -1
        while x1 < 0 or x1 > 511:
            x1 = np.random.randint(x0, x0+PLG_MAX_SIZE+1)
        while y1 < 0 or y1 > 511:
            y1 = np.random.randint(y0 - PLG_MAX_SIZE, y0)
        self.p1, self.p2, self.center = Point(x0, y0), Point(x1, y1), Point((x0+x1)//2, (y0+y1)//2)

    # function that computes some parameters for technical purposes
    def form_rect(self):
        if self.p1.x <= self.p2.x and self.p1.y >= self.p2.y:
            self.left_up = Point(self.p1.x, self.p1.y)
            self.bottom_right = Point(self.p2.x, self.p2.y)
        else: 
            self.left_up = Point(self.p2.x, self.p2.y)
            self.bottom_right = Point(self.p1.x, self.p1.y)

        self.center = Point((self.left_up.x+self.bottom_right.x)//2, (self.left_up.y+self.bottom_right.y)//2)
        
        self.dy = self.left_up.y - self.bottom_right.y + 1
        self.dx = self.bottom_right.x - self.left_up.x + 1
    
    # method to check if a certain point is inside or not
    def is_inside(self, p):
        if p.x < self.dx and p.y < self.dy:
            return True
        return False

class MultiAgent:
    def __init__(self):
        self.agents = []

    # generates certain amount of agents
    def generate_agents(self, amount):
        for i in range (amount):
            agent = Agent()
            self.agents.append(agent.generate())
    
    # sorts agents in descending order
    def quicksort_agents(self, low, high): 
        if low < high:
            pi = partition(self.agents, low, high)
            self.quicksort_agents(low, pi-1)
            self.quicksort_agents(pi+1, high)
    
    # calulates fitness of the multiagent
    def fitness(self, src_img):
        for agent in self.agents:     
            agent.score = agent.calc_score(src_img)
        self.fit = 0
        self.quicksort_agents(0, len(self.agents)-1)
        for i in range(20):
            self.fit += self.agents[i].score


class Agent:
    def __init__(self):
        self.score = 0

    # generates a new agents with randomly placed polygon and random starting colors
    def generate(self):
        self.polygon = Circle()
        self.polygon.generate()
        self.polygon.form_rect()

        self.colors = generate_colors(self.polygon)
        return self

    # generates a pair of points inside the polygon
    def generate_pair_inside_polygon(self):
        while 1:
            p1 = Point(np.random.randint(self.polygon.dx), np.random.randint(self.polygon.dy))
            if (self.polygon.is_inside(p1)):
                p2 = Point(np.random.randint(self.polygon.dx), np.random.randint(self.polygon.dy))
                if (self.polygon.is_inside(p2)):
                    return p1, p2

    # swaps several pairs of pixels inside the polygon
    def swap_pixels(self):
        pairs = np.random.randint(0, 30)
        while pairs > 0:
            p1, p2 = self.generate_pair_inside_polygon()
            temp = self.colors[p1.x, p1.y]
            self.colors[p1.x, p1.y] = self.colors[p2.x, p2.y]
            self.colors[p2.x, p2.y] = temp
            pairs -= 1

    # changes a few randomly chosen pixels inside the polygon
    def change_pixel(self):
        pol = self.polygon
        for i in range(self.polygon.dx):
            for j in range(self.polygon.dy):
                if pol.is_inside(Point(i, j)):
                    flag = np.random.randint(0, 10)
                    if flag == 0:
                        # 10% to make it darker
                        self.colors[i][j] = randint(180, 255)
                    elif flag == 1:
                        # 10% to make it lighter
                        self.colors[i][j] = randint(0, 100)

    # changes position of agent's polygon with respect to the whole image
    def change_position(self):
        pol = self.polygon
        cx = np.random.randint(-8, 8)
        while(pol.center.x + cx < 0 or pol.center.x + cx > 511):
            cx = np.random.randint(-8, 8)
        pol.center.x += cx

        cy = np.random.randint(-8, 8)
        while(pol.center.y + cy < 0 or pol.center.y + cy > 511):
            cy = np.random.randint(-8, 8)
        pol.center.y += cy
        pol.form_rect()

    # copy colors from two parents to their kid - half from parent1, half from parent2
    def copy_colors(self, par1, par2):
        for i in range(self.polygon.dx):
            for j in range (self.polygon.dy):
                if self.polygon.is_inside(Point(i, j)):
                    if i < self.polygon.center.x:
                        self.colors[i][j] = par1.colors[i][j]
                    else:
                        self.colors[i][j] = par2.colors[i][j]
    
    # calculate score of the agent
    def calc_score(self, src_img):
        score = 0
        for i in range(self.polygon.dx):
            for j in range(self.polygon.dy):
                if self.polygon.is_inside(Point(i, j)):
                    real = src_img[i][j]
                    cur = self.colors[i][j]
                    if (cur >= real - 10 and cur <= real + 10):
                        score += 1
                    if (cur >= real - 3 and cur <= real + 3):
                        score += 3
                    if (cur == real):
                        score += 5
        self.score = score
        return score

    # create a copy of the agent
    def copy_from(self, agent):
        self.generate()
        self.polygon.center.x = agent.polygon.center.x
        self.polygon.center.y = agent.polygon.center.y
        self.polygon.form_rect()
        self.colors = np.copy(agent.colors)

# generate random colors for a given polygon 
def generate_colors(polygon):
    colors = np.zeros([polygon.dx, polygon.dy, 1])

    for i in range(polygon.dx):
            for j in range(polygon.dy):
                if polygon.is_inside(Point(i, j)):
                    colors[i][j] = np.random.randint(0, 256)

    return colors

# simulate evolution
def evolution(src_img):
    print("starting...")
    # create an initial multi-agent and generate initial amount of agents
    multiagent = MultiAgent()
    multiagent.generate_agents(INIT_AMOUNT_OF_AGENTS)
    # calculate fitness of initial multi-agent
    multiagent.fitness(src_img)

    save_image(multiagent, 'tmp1.jpg')
    time_init = time.time(); time_now = time_init

    # loop to keep the algorithm running
    while len(multiagent.agents) > 20 and time_now-time_init < TIME_LIMIT_IN_SEC:
        # form new generation
        new_multiagent = form_new_gen(multiagent)
        # calculate fitness of the newly formed generation
        new_multiagent.fitness(src_img)

        # newly formed generation will be next if and only if its fitness 
        # is higher that the previous generation's fitness
        if new_multiagent.fit > multiagent.fit:
            print("old fit: ", multiagent.fit, "; ", "found better: ", new_multiagent.fit)
            print("new generation has", len(new_multiagent.agents), "agents in multi-agent")
            multiagent = new_multiagent 
            save_image(multiagent, 'tmp.jpg')

        time_now = time.time()
    return multiagent

# forms new generation based on the old one
def form_new_gen(old_gen):
    old_best_agents = []
    # select best agents from the previous generation
    old_best_agents = old_gen.agents[:len(old_gen.agents)//4]
    # create new multi-agent
    new_gen = MultiAgent()
    # add a few of random agents to keep diversity and prevent "grouping" of polygons
    new_gen.generate_agents(55)

    crossed, mutated = [], []
    # control of amount of kids from crossover
    if len(old_best_agents) < 50:
        crossed = crossover(old_best_agents, 4)
    elif len(old_best_agents) < 100:
        crossed = crossover(old_best_agents, 2)
    else:
        crossed = crossover(old_best_agents, 1)

    # mutation probability is 66%
    mutated = mutation(old_best_agents)
    
    for each in crossed:
        new_gen.agents.append(each)
    for each in mutated:
        new_gen.agents.append(each)
    
    for each in new_gen.agents:
        each.score = 0

    return new_gen

# perform mutation
def mutation(agents):
    new_agents = []
    for agent in agents:
        flag = np.random.randint(0, 3)
        if flag == 0: 
            # probability of swapping colors inside the polygon - 33%
            new = Agent()
            new.copy_from(agent)
            new.swap_pixels()
            new_agents.append(new)
        elif flag == 1:
            # probability of small change of position (x+-7, y+-7) - 33%
            new = Agent()
            new.copy_from(agent)
            new.change_position()
            new_agents.append(new)
    return new_agents

# perform a crossover between two agents
def crossover(agents, max_kids):
    new_agents = []
    for i in range (1, len(agents)):
        if np.random.randint(0,2) == 0:
            # agents are sorted from best to not so best so let's cross best with best
            j = np.random.randint(0, 10)
            tmp_agents = cross_genes(agents[i], agents[j], max_kids)
            for k in range (len(tmp_agents)):
                new_agents.append(tmp_agents[k])
    return new_agents

# crossing genes of two agents with predefined max amount of kids
def cross_genes(agent1, agent2, max_kids):
    kids = []
    for i in range (max_kids):
        kid = Agent()
        kid.polygon = Circle()
        
        kid.polygon.center = Point()
        kid.polygon.center.x = agent1.polygon.center.x
        dx = np.random.randint(-2,2)
        while (kid.polygon.center.x + dx > 511 or kid.polygon.center.x + dx < 0):
            dx = np.random.randint(-2, 2)
        kid.polygon.center.x = agent1.polygon.center.x + dx

        kid.polygon.center.y = agent1.polygon.center.y 
        dy = np.random.randint(-2,2)
        while (kid.polygon.center.y + dy > 511 or kid.polygon.center.y + dy < 0):
            dy = np.random.randint(-2, 2)
        kid.polygon.center.y = agent1.polygon.center.y + dy

        kid.polygon.radius = agent1.polygon.radius

        kid.polygon.form_rect()
        kid.colors = generate_colors(kid.polygon)
        kid.copy_colors(agent1, agent2)
        
        kids.append(kid)
    return kids

# function to check if agents are close to each other or not. could've been used to cross only close agents
def agents_are_close(agent1, agent2):
    cx1 = agent1.polygon.center.x
    cy1 = agent1.polygon.center.y
    cx2 = agent2.polygon.center.x
    cy2 = agent2.polygon.center.y
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist = math.sqrt(pow(dx,2) + pow(dy,2))
    if dist > MAX_DIST_BTW_AGENTS:
        return False
    return True 

# add blur to the image
def blur(img):
    # sigma = 5.0 #coefficient
    # return gaussian_filter(img, sigma=3, multichannel=False)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img, -3, kernel)
    cv2.imwrite('perception.jpg', dst)
    return dst

# add black figures to the image
def blackout(img, times):   
    for i in range(0, times):
        dec = np.random.randint(0,2)
        if dec == 0: # rectange
            rect = Rectangle()
            rect.generate()
            rect.form_rect()
            img[rect.left_up.x:rect.bottom_right.x, rect.bottom_right.y:rect.left_up.y] = 0
        else: # circle
            circle = Circle()
            circle.generate()
            circle.form_rect()
            for i in range(circle.dx):
                for j in range(circle.dy):
                    if circle.is_inside(Point(i, j)):
                        img[i+circle.left_up.x, j+circle.bottom_right.y] = 0
    return img

def read_pic(path):
    original = cv2.imread(path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return blur(blackout(original, 25))

def save_image(ma, new_name):
    new_image = np.zeros([512,512,3])
    for agent in ma.agents:
        for i in range (agent.polygon.dx):
            for j in range (agent.polygon.dy):
                if agent.polygon.is_inside(Point(i, j)):
                    if i+agent.polygon.left_up.x > 0 and i+agent.polygon.left_up.x < 512 and j+agent.polygon.bottom_right.y > 0 and j+agent.polygon.bottom_right.y < 512:
                        new_image[i+agent.polygon.left_up.x, j+agent.polygon.bottom_right.y] = agent.colors[i, j]

    cv2.imwrite(new_name, new_image)

def main():
    print("Give me the path to the picture: ")
    path = input()
    src_img = read_pic(path)
    print("How to name the new file?")
    new_name = input()
    new_name = new_name + ".jpg"
    ma = evolution(src_img)
    save_image(ma, new_name)
main()




# bluring:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# point is in a triangle:
# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
# quicksort:
# https://www.geeksforgeeks.org/python-program-for-quicksort/