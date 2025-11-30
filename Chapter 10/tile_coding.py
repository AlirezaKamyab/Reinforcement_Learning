import numpy as np


class TileCoder:
    def __init__(
            self, 
            low:np.ndarray, 
            high:np.ndarray, 
            num_tiles:int, 
            tile_per_dimension:np.ndarray,
            num_actions:int=None,
        ):

        if not isinstance(low, np.ndarray):
            low = np.array(low, dtype=np.float32)
            high = np.array(high, dtype=np.float32)

        if not isinstance(tile_per_dimension, np.ndarray):
            tile_per_dimension = np.array(tile_per_dimension)

        self.low = low
        self.high = high
        assert self.low.ndim == self.high.ndim
        self.dim = self.low.ndim
        self.num_tiles = num_tiles
        self.num_actions = num_actions

        if isinstance(tile_per_dimension, np.ndarray):
            self.tile_per_dimension = tile_per_dimension
        elif isinstance(tile_per_dimension, (list, tuple)):
            self.tile_per_dimension = np.array(tile_per_dimension, dtype=np.int32)
        elif isinstance(tile_per_dimension, int):
            self.tile_per_dimension = np.array([tile_per_dimension] * self.dim, dtype=np.int32)
        else:
            raise ValueError("Value for tile_per_dimension is invalid!")
        
        self.offset = (np.arange(num_tiles) / num_tiles)[:, None] * (1.0 / self.tile_per_dimension)

        if num_actions is not None:
            self.num_features = self.num_tiles * np.prod(self.tile_per_dimension) * num_actions
        else:
            self.num_features = self.num_tiles * np.prod(self.tile_per_dimension)

    def scale(self, state:np.ndarray):
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state = np.clip(state, min=self.low, max=self.high)
        state = (state - self.low) / (self.high - self.low)
        state = state * self.tile_per_dimension
        return state
    
    def get_indices(self, state:np.ndarray, action:int=None):
        if self.num_actions is not None and action is None:
            raise ValueError("Action cannot be None")
        
        state = self.scale(state)
        indices = np.empty(self.num_tiles, dtype=np.int32)

        for i in range(self.num_tiles):
            coordinate = np.floor(state + self.offset[i]).astype(np.int32)
            coordinate = np.minimum(coordinate, self.tile_per_dimension - 1)
            index = np.ravel_multi_index(tuple(coordinate), tuple(self.tile_per_dimension))
            indices[i] = i * np.prod(self.tile_per_dimension) + index
            if action is not None:
                indices[i] += action * np.prod(self.tile_per_dimension) * self.num_tiles
        return indices
    
    def get_vector(self, state:np.ndarray, action:int=None):
        x = self.get_indices(state=state, action=action)
        vec = np.zeros(self.num_features, dtype=np.float32)
        vec[x] = 1.0
        return vec
        

def main():
    coding = TileCoder(
        low=[-0.05, -0.07],
        high=[0.6, 0.07],
        num_tiles=8,
        tile_per_dimension=[8, 8],
        num_actions=3
    )

    state = [0.01, 0.03]
    action = 1
    print(coding.scale(state))
    print(coding.get_indices(state, action))
    print(coding.get_vector(state, action))


if __name__ == '__main__':
    main()
