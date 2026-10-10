class _Config:
    def update(self, *args, **kwargs):
        return None


class _Run:
    def __init__(self):
        self.dir = None
        self.config = _Config()


config = _Config()
run = _Run()


def init(*args, **kwargs):
    return run


def watch(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None


def finish(*args, **kwargs):
    return None
