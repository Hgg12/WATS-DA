import os
import enum
from datetime import datetime


class meta_singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "__unique_instance__"):
            cls.__unique_instance__ = super().__call__(*args, **kwargs)
        return cls.__unique_instance__


class log_level(enum.Enum):
    Record = enum.auto()
    Info = enum.auto()
    Warning = enum.auto()
    Error = enum.auto()


class _logger:
    def __init__(self) -> None:
        _log_path = "logs/{}.log".format(datetime.now()).replace(":", "-")
        if not os.path.exists(os.path.dirname(_log_path)):
            os.makedirs(os.path.dirname(_log_path))

        self._log_file = open(_log_path, mode="w+")
    
    def __del__(self):
        self._flush()
        if not self._log_file.closed:
            self._log_file.close()

    def _log(self, level: log_level = log_level.Info, *message: str, sep: str, end= "\n"):
        basic_info = "[%s][%s]" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level.name)
        log_msg = sep.join(message)
        
        self._log_file.write("%s %s%s" % (basic_info, log_msg, end))
        if level is log_level.Info:
            print("\033[32m%s %s\033[0m" % (basic_info, log_msg))
        elif level is log_level.Warning:
            print("\033[33m%s %s\033[0m" % (basic_info, log_msg))
        elif level is log_level.Error:
            print("\033[31m%s %s\033[0m" % (basic_info, log_msg))
    
    def _flush(self):
        self._log_file.flush()


class mylogger(metaclass = meta_singleton):
    __logger = _logger()

    @classmethod
    def record(cls, *message, sep = " "):
        cls.__logger._log(log_level.Record, *message, sep=sep)
    
    @classmethod
    def info(cls, *message, sep = " "):
        cls.__logger._log(log_level.Info, *message, sep=sep)

    @classmethod
    def warning(cls, *message, sep= " "):
        cls.__logger._log(log_level.Warning, *message, sep=sep)

    @classmethod
    def error(cls, *message, sep= " "):
        cls.__logger._log(log_level.Error, *message, sep=sep)

    @classmethod
    def record_dict(cls, *message, dict_msg: dict, sep = " "):
        msg = sep.join(message) + "\n"

        def _format_dict(_all_msg: dict, _prefix: str):
            _msg = ""
            for _key, _value in _all_msg.items():
                _msg += _prefix + _key + ": " 
                if isinstance(_value, dict):
                    _msg += "\n"
                    _msg += _format_dict(_value, _prefix + "  ")
                else:
                    _msg += str(_value) + "\n"
            return _msg
        
        msg += _format_dict(dict_msg, "  ")
        cls.__logger._log(log_level.Record, msg, sep="", end="")
    
    @classmethod
    def flush(cls):
        cls.__logger._flush()
    