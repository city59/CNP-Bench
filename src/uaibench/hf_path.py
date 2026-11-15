

def add():
    import transformers.utils as tu
    import transformers.utils.import_utils as iu
    off = lambda: False
    for m in (tu, iu):
        m.is_vision_available = off
        m.is_torchvision_available = off
