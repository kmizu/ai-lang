"""Color support for terminal output."""

import os
import sys


class Colors:
    """ANSI color codes for terminal output."""
    
    # Check if colors are supported
    _supports_color = (
        hasattr(sys.stdout, 'isatty') and 
        sys.stdout.isatty() and
        os.environ.get('TERM') != 'dumb' and
        os.environ.get('NO_COLOR') is None
    )
    
    # Color codes
    RESET = '\033[0m' if _supports_color else ''
    BOLD = '\033[1m' if _supports_color else ''
    DIM = '\033[2m' if _supports_color else ''
    
    # Foreground colors
    BLACK = '\033[30m' if _supports_color else ''
    RED = '\033[31m' if _supports_color else ''
    GREEN = '\033[32m' if _supports_color else ''
    YELLOW = '\033[33m' if _supports_color else ''
    BLUE = '\033[34m' if _supports_color else ''
    MAGENTA = '\033[35m' if _supports_color else ''
    CYAN = '\033[36m' if _supports_color else ''
    WHITE = '\033[37m' if _supports_color else ''
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m' if _supports_color else ''
    BRIGHT_RED = '\033[91m' if _supports_color else ''
    BRIGHT_GREEN = '\033[92m' if _supports_color else ''
    BRIGHT_YELLOW = '\033[93m' if _supports_color else ''
    BRIGHT_BLUE = '\033[94m' if _supports_color else ''
    BRIGHT_MAGENTA = '\033[95m' if _supports_color else ''
    BRIGHT_CYAN = '\033[96m' if _supports_color else ''
    BRIGHT_WHITE = '\033[97m' if _supports_color else ''
    
    @classmethod
    def success(cls, text: str) -> str:
        """Format success message."""
        return f"{cls.GREEN}✓{cls.RESET} {text}"
    
    @classmethod
    def error(cls, text: str) -> str:
        """Format error message."""
        return f"{cls.RED}✗{cls.RESET} {text}"
    
    @classmethod
    def warning(cls, text: str) -> str:
        """Format warning message."""
        return f"{cls.YELLOW}⚠{cls.RESET} {text}"
    
    @classmethod
    def info(cls, text: str) -> str:
        """Format info message."""
        return f"{cls.BLUE}ℹ{cls.RESET} {text}"
    
    @classmethod
    def bold(cls, text: str) -> str:
        """Format bold text."""
        return f"{cls.BOLD}{text}{cls.RESET}"
    
    @classmethod
    def dim(cls, text: str) -> str:
        """Format dimmed text."""
        return f"{cls.DIM}{text}{cls.RESET}"
    
    @classmethod
    def keyword(cls, text: str) -> str:
        """Format language keyword."""
        return f"{cls.MAGENTA}{text}{cls.RESET}"
    
    @classmethod
    def type_name(cls, text: str) -> str:
        """Format type name."""
        return f"{cls.CYAN}{text}{cls.RESET}"
    
    @classmethod
    def literal(cls, text: str) -> str:
        """Format literal value."""
        return f"{cls.GREEN}{text}{cls.RESET}"
    
    @classmethod
    def operator(cls, text: str) -> str:
        """Format operator."""
        return f"{cls.YELLOW}{text}{cls.RESET}"


def disable_colors():
    """Disable color output."""
    Colors._supports_color = False
    for attr in dir(Colors):
        if attr.isupper() and not attr.startswith('_'):
            setattr(Colors, attr, '')


def enable_colors():
    """Force enable color output."""
    Colors._supports_color = True
    # Re-initialize color codes
    Colors.RESET = '\033[0m'
    Colors.BOLD = '\033[1m'
    Colors.DIM = '\033[2m'
    Colors.RED = '\033[31m'
    Colors.GREEN = '\033[32m'
    Colors.YELLOW = '\033[33m'
    Colors.BLUE = '\033[34m'
    Colors.MAGENTA = '\033[35m'
    Colors.CYAN = '\033[36m'
    # ... etc