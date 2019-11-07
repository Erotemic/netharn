class NetharnSignal(Exception):
    """
    Abstract base class for exceptions that control program flow.
    These exceptions should always be caught and handled.
    They should not cause the system to crash.
    """


class NetharnError(Exception):
    """ Abstract base class for exceptions that indicate a crash """


class StopTraining(NetharnSignal):
    """ Thrown when training should terminate """


class CannotResume(NetharnSignal):
    """ Thrown when netharn cannot start from an existing snapshot """


class TrainingDiverged(NetharnError):
    """ Thrown when netharn detects divergence """
