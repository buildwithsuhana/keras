"""Logging utilities for PyTorch model parallelism in Keras.

This module provides structured logging for model parallelism operations,
making it easier to debug and monitor distributed training.
"""

import logging
import os
from typing import Optional


def _get_log_level() -> str:
    """Get the log level from environment variable or default to INFO.
    
    Returns:
        str: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    return os.environ.get('KERAS_TORCH_MODEL_PARALLEL_LOG_LEVEL', 'INFO')


def _setup_logger(name: str = 'keras.torch.model_parallel') -> logging.Logger:
    """Set up the model parallelism logger.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='[MODEL_PARALLEL:%(levelname)s:RANK:%(rank)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set log level
        log_level = _get_log_level()
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger


class DistributionLogger:
    """Logger for model parallelism operations.
    
    This class provides structured logging for model parallelism,
    including rank information, tensor shapes, and operation details.
    """
    
    def __init__(self, name: str = 'keras.torch.model_parallel'):
        """Initialize the distribution logger.
        
        Args:
            name: Logger name
        """
        self.logger = _setup_logger(name)
        
    def _get_rank(self) -> int:
        """Get current process rank.
        
        Returns:
            int: Current rank or 0 if not in distributed mode
        """
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except (ImportError, RuntimeError):
            pass
        return 0
    
    def debug(self, message: str, rank: Optional[int] = None):
        """Log debug message.
        
        Args:
            message: Log message
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        self.logger.debug(message, extra={'rank': rank})
    
    def info(self, message: str, rank: Optional[int] = None):
        """Log info message.
        
        Args:
            message: Log message
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        self.logger.info(message, extra={'rank': rank})
    
    def warning(self, message: str, rank: Optional[int] = None):
        """Log warning message.
        
        Args:
            message: Log message
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        self.logger.warning(message, extra={'rank': rank})
    
    def error(self, message: str, rank: Optional[int] = None):
        """Log error message.
        
        Args:
            message: Log message
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        self.logger.error(message, extra={'rank': rank})
    
    def log_operation(self, operation: str, details: dict, rank: Optional[int] = None):
        """Log a distributed operation with details.
        
        Args:
            operation: Name of the operation
            details: Dictionary with operation details
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        
        details_str = ', '.join(f'{k}={v}' for k, v in details.items())
        self.info(f'{operation} - {details_str}', rank)
    
    def log_tensor_distribution(self, tensor_name: str, original_shape: tuple, 
                               sharded_shape: tuple, axis: str, rank: Optional[int] = None):
        """Log tensor distribution operation.
        
        Args:
            tensor_name: Name of the tensor
            original_shape: Original tensor shape
            sharded_shape: Sharded tensor shape
            axis: Sharding axis
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        
        self.info(
            f'{tensor_name}: Distributing tensor - '
            f'original_shape={original_shape}, '
            f'sharded_shape={sharded_shape}, '
            f'axis={axis}',
            rank
        )
    
    def log_optimizer_step(self, optimizer_name: str, num_variables: int, 
                          learning_rate: float, rank: Optional[int] = None):
        """Log optimizer step operation.
        
        Args:
            optimizer_name: Name of the optimizer
            num_variables: Number of variables being updated
            learning_rate: Learning rate
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        
        self.info(
            f'{optimizer_name}: Optimizer step - '
            f'variables={num_variables}, '
            f'learning_rate={learning_rate}',
            rank
        )
    
    def log_collective_operation(self, operation: str, tensor_shape: tuple, 
                                axis_name: str, rank: Optional[int] = None):
        """Log collective operation (all_gather, all_reduce, etc.).
        
        Args:
            operation: Name of the collective operation
            tensor_shape: Shape of the tensor
            axis_name: Mesh axis name
            rank: Process rank (auto-detected if not provided)
        """
        if rank is None:
            rank = self._get_rank()
        
        self.info(
            f'{operation}: Collective operation - '
            f'tensor_shape={tensor_shape}, '
            f'axis={axis_name}',
            rank
        )


# Global logger instance
_dist_logger: Optional[DistributionLogger] = None


def get_logger() -> DistributionLogger:
    """Get the global distribution logger instance.
    
    Returns:
        DistributionLogger: Global logger instance
    """
    global _dist_logger
    if _dist_logger is None:
        _dist_logger = DistributionLogger()
    return _dist_logger


def log_distribution_setup(job_addresses: str, num_processes: int, 
                          process_id: int, backend: str):
    """Log distribution setup information.
    
    Args:
        job_addresses: Job addresses
        num_processes: Number of processes
        process_id: Current process ID
        backend: Communication backend
    """
    logger = get_logger()
    logger.log_operation(
        'distribution_setup',
        {
            'job_addresses': job_addresses,
            'num_processes': num_processes,
            'process_id': process_id,
            'backend': backend
        },
        process_id
    )


def log_tensor_distribution(tensor_name: str, original_shape: tuple, 
                           sharded_shape: tuple, axis: str, rank: int = 0):
    """Log tensor distribution operation.
    
    Args:
        tensor_name: Name of the tensor
        original_shape: Original tensor shape
        sharded_shape: Sharded tensor shape
        axis: Sharding axis
        rank: Process rank
    """
    logger = get_logger()
    logger.log_tensor_distribution(tensor_name, original_shape, sharded_shape, axis, rank)


def log_optimizer_step(optimizer_name: str, num_variables: int, 
                      learning_rate: float, rank: int = 0):
    """Log optimizer step operation.
    
    Args:
        optimizer_name: Name of the optimizer
        num_variables: Number of variables being updated
        learning_rate: Learning rate
        rank: Process rank
    """
    logger = get_logger()
    logger.log_optimizer_step(optimizer_name, num_variables, learning_rate, rank)


def log_collective_operation(operation: str, tensor_shape: tuple, 
                            axis_name: str, rank: int = 0):
    """Log collective operation.
    
    Args:
        operation: Name of the collective operation
        tensor_shape: Shape of the tensor
        axis_name: Mesh axis name
        rank: Process rank
    """
    logger = get_logger()
    logger.log_collective_operation(operation, tensor_shape, axis_name, rank)
