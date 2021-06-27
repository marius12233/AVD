import numpy as np

class RoadHandler:

    def __init__(self):
        self._next_intersection = None
        self._nearest_intersection=None
        self._intersections_turn = None
        self._lanes = None #[[m1,b1],[m2,b2]]
        self._boundaries = [None, None]
    
    def set_next_intersection(self, next_intersection):
        self._next_intersection = next_intersection 

    def set_nearest_intersection(self, nearest_intersection):
        self._nearest_intersection = nearest_intersection
    
    def set_intersections_turn(self, intersections_turn):
        self._intersections_turn = intersections_turn

    def set_lanes(self, lanes):
        self._lanes = lanes

    def set_boundaries(self, boundaries):
        self._boundaries = boundaries    

    def is_turn(self):
        """
        Returns:
            True if at the nearest intersection we must afford a turn else False
        """
        is_turn = self._intersections_turn.get(str(self._nearest_intersection[:2]))
        if is_turn:
            return True
        return False
    
    def is_inrange_nearest_intersection(self, ego_state, ranging=15):
        return True if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=ranging else False
    
    def is_on_sidewalk(self, x, y):
        is_out_of_range = False
        for coefs in self._lanes:
            m,b = coefs
            if y > m*x + b:
                is_out_of_range = True
                break
        return is_out_of_range        
    

