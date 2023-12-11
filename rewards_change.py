import gym

class YPosBenefitWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(YPosBenefitWrapper, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        self.max_y = 0
        
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)

        ## reward Mario for increasing his Y POS
        self.max_y = max(info['y_pos'], self.max_y)
        reward += 0.2*info['y_pos'] / self.max_y
        
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True

        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

class ScoreBenefitWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(ScoreBenefitWrapper, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        self.max_delta_score = 0
        
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)

        ## reward Mario for increasing his delta score
        delta_score = info["score"]-self.current_score
        self.max_delta_score = max(self.max_delta_score, delta_score)
        reward += delta_score / self.max_delta_score
        
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True

        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info


class RewardClip(gym.Wrapper):
    def __init__(self, env=None, clip_value= 15.):
        super(RewardClip, self).__init__(env)
        self.clip_value = clip_value

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.clip_value < reward:
            return state, self.clip_value, done, info
        elif -self.clip_value > reward:
            return state, -self.clip_value, done, info
        else:
            return state, reward, done, info