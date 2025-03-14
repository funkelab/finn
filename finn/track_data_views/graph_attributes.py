from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    POS: str = 'pos'
    TIME: str = 'time'
    SEG_ID: str = 'seg_id'
    SEG_HYPO: str = 'seg_hypo'
    TRACK_ID: str = 'track_id'
    AREA: str = 'area'


class EdgeAttr(Enum):
    """Edge attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    IOU: str = 'iou'
