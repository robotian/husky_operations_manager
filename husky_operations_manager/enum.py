from enum import Enum

class TaskEnum(Enum):
    """Tasks types in the system."""

    CHARGING_TASK = 0
    HARVESTING_TASK = 1
    UNLOADING_TASK = 2


class SubTaskEnum(Enum):
    """Sub-tasks types in the system."""

    MOVING = 1
    HARVESTING = 2
    DOCKING = 3
    CHARGING = 4
    LOADING = 5
    UNLOADING = 6
    UNDOCKING = 7


class OnlineFlagEnum(Enum):
    """The online status of a robot."""

    OFFLINE = 0
    ONLINE = 1
    EMERGENCY_STOP = 10
    ABNORMAL = 11


class NavigationStatus(Enum):
    """Navigation status enumeration."""
    IDLE = 0
    SENDING = 1
    ACCEPTED = 2
    REJECTED = 3
    ACTIVE = 4
    SUCCEEDED = 5
    ABORTED = 6
    CANCELED = 7
    FAILED = 8
    ERROR = 9


class ManipulatorStatus(Enum):
    IDLE             = 0
    PLANNING         = 1
    MOVING           = 2
    MOVING_COMPLETE  = 3
    FAILED           = 99


class RobotStatusEnum(Enum):
    """State of robot's operation."""

    IDLE = 0

    JOB_START = 1
    JOB_DONE = 2

    START_MOVING = 3
    MOVING = 4
    DESTINATION_REACHED = 5

    START_HARVESTING = 6
    HARVESTING = 7
    DONE_HARVESTING = 8

    START_DOCKING = 9
    DOCKING = 10
    DONE_DOCKING = 11

    START_LOADING = 12
    LOADING = 13
    DONE_LOADING = 14

    START_UNDOCKING = 15
    UNDOCKING = 16
    DONE_UNDOCKING = 17

    START_UNLOADING = 18
    UNLOADING = 19
    DONE_UNLOADING = 20

    START_CHARGING = 21
    CHARGING = 22
    DONE_CHARGING = 23

    ERROR = 94
    PAUSED = 95
    MAINTENANCE = 96
    OFFLINE = 97
    EMERGENCY_STOP = 98
    ABNORMAL = 99


class DriveStatus(Enum):
    """ Status enum for DriveClient. """
    IDLE      = 0
    FORWARD   = 1
    REVERSE   = 2
    DONE      = 3
    CANCELED  = 4
    ERROR     = 5
    
class ReverseDriveStatus(Enum):
    """ Status enum for ReverseDriveClient. """
    IDLE      = 0
    REVERSING = 1
    DONE      = 2
    CANCELED  = 3
    ERROR     = 4
 
 
class DockingParamFetcherStatus(Enum):
    """ Status enum for DockingParamFetcher. """
    IDLE      = 0
    LISTING   = 1
    FETCHING  = 2
    RESOLVING = 3
    COMPUTING = 4
    DONE      = 5
    ERROR     = 6