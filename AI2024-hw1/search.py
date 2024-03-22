"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
    
    # successor: 繼承者
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]



def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    initial_state = problem.getStartState()  # (34, 16)
    start_successors = problem.getSuccessors(problem.getStartState())  # [((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    """
    
    # 創建一個 Stack 作為 frontier
    frontier = util.Stack()
    
    # 第二項即為最後會回傳的 actions list (應該)
    # 第三項用不太到，就不 push 進 frontier
    frontier.push((problem.getStartState(), []))
   
    # 創建一個空集合 explored set 來儲存已訪問的節點
    explored_set = set()
   
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        
        if state not in explored_set:
            explored_set.add(state)
            
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in explored_set:
                    new_actions = actions + [action]
                    frontier.push((successor, new_actions))
    
    # 如果 frontier is empty，則無解
    return []
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
    
    
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    # 創建一個 Queue 作為 frontier
    frontier = util.Queue()
    
    # 第二項即為最後會回傳的 actions list (應該)
    # 第三項用不太到，就不 push 進 frontier
    frontier.push((problem.getStartState(), []))
    
    # 創建一個空集合 explored set 來儲存已訪問的節點
    explored_set = set()
    
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        
        if state not in explored_set:
            explored_set.add(state)
            
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in explored_set:
                    new_actions = actions + [action]
                    frontier.push((successor, new_actions))
   
    # 如果 frontier is empty，則無解
    return []
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
    
    
def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    # 創建一個 Priority Queue 作為 frontier
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), []), 0)
    
    # 創建一個空集合 explored set 來儲存已訪問的節點
    explored_set = set()
    
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        
        if state not in explored_set:
            explored_set.add(state)
            
            for successor, action, cost in problem.getSuccessors(state):
                if successor not in explored_set:
                    new_actions = actions + [action]
                    new_costs = problem.getCostOfActions(new_actions)
                    frontier.push((successor, new_actions), new_costs)
    
    # 如果 frontier is empty，則無解
    return []
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    
    
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    initial_state = problem.getStartState()
    # 創建一個 Priority Queue 作為 frontier
    frontier = util.PriorityQueue()
    frontier.push((initial_state, []), heuristic(initial_state, problem))
    
    # 創建一個空集合 explored set 來儲存已訪問的節點
    explored_set = set()
    # 記錄到達每個節點的最低成本
    costs = {initial_state: 0}
    
    while not frontier.isEmpty():
        state, actions = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        
        if state not in explored_set:
            explored_set.add(state)
            
            for successor, action, cost in problem.getSuccessors(state):
                new_actions = actions + [action]
                # 實際成本: 到達後繼節點的新成本
                new_cost = problem.getCostOfActions(new_actions)  
                if successor not in explored_set or new_cost < costs.get(successor, float('inf')):  
                    # 從 costs 字典中獲取 successor 的成本，若 successor 尚未有記錄，則返回 float('inf')
                    # 啟發式成本: 從後繼節點到目標的估計成本
                    h_cost = heuristic(successor, problem)
                    # 總成本
                    f_cost = new_cost + h_cost
                    frontier.push((successor, new_actions), f_cost)
                    # 更新到達後繼節點的成本
                    costs[successor] = new_cost
    
    # 如果 frontier is empty，則無解
    return []
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
