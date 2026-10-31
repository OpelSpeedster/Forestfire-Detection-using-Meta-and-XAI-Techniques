def should_escalate(conf: float, threshold: float) -> bool:
    return conf < threshold

def route(local_label: str, local_conf: float, threshold: float, vlm_label: str = None) -> str:
    if should_escalate(local_conf, threshold) and vlm_label is not None:
        return vlm_label
    return local_label
