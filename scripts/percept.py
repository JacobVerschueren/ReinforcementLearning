class Percept:
    """
    This class is used to create percepts that are used to update the Learning Strategy
    """

    def __init__(self, old_state, action, new_state, reward, final_state):
        self.old_state = old_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.final_state = final_state

