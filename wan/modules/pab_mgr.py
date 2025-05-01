PAB_MANAGER = None


class PABConfig:
    def __init__(
        self,
        cross_broadcast: bool = False,
        cross_threshold: list = None,
        cross_range: int = None,
        spatial_broadcast: bool = False,
        spatial_threshold: list = None,
        spatial_range: int = None,
    ):
        self.steps = None

        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold
        self.cross_range = cross_range

        self.spatial_broadcast = spatial_broadcast
        self.spatial_threshold = spatial_threshold
        self.spatial_range = spatial_range


class PABManager:
    def __init__(self, config: PABConfig):
        self.config: PABConfig = config

        init_prompt = f"Init Pyramid Attention Broadcast."
        init_prompt += f" spatial broadcast: {config.spatial_broadcast}, spatial range: {config.spatial_range}, spatial threshold: {config.spatial_threshold}."        
        init_prompt += f" cross broadcast: {config.cross_broadcast}, cross range: {config.cross_range}, cross threshold: {config.cross_threshold}."
        print(init_prompt)

    def if_broadcast_cross(self, timestep: int, count: int):
        if (
            self.config.cross_broadcast
            and (timestep is not None)
            and (count % self.config.cross_range != 0)
            and (self.config.cross_threshold[0] < timestep < self.config.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

    def if_broadcast_spatial(self, timestep: int, count: int):
        if (
            self.config.spatial_broadcast
            and (timestep is not None)
            and (count % self.config.spatial_range != 0)
            and (self.config.spatial_threshold[0] < timestep < self.config.spatial_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

def set_pab_manager(config: PABConfig):
    global PAB_MANAGER
    PAB_MANAGER = PABManager(config)


def enable_pab():
    if PAB_MANAGER is None:
        return False
    return (
        PAB_MANAGER.config.cross_broadcast
        or PAB_MANAGER.config.spatial_broadcast
    )


def update_steps(steps: int):
    if PAB_MANAGER is not None:
        PAB_MANAGER.config.steps = steps


def if_broadcast_cross(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_cross(timestep, count)


def if_broadcast_spatial(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_spatial(timestep, count)

class WanV120PABConfig(PABConfig):
    def __init__(
        self,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [603, 976],
        spatial_range: int = 2,
        cross_broadcast: bool = True,
        cross_threshold: list = [603, 976],
        cross_range: int = 6,
    ):
        super().__init__(
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range,
        )