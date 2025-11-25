"""Configuration manager for smugVision."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from smugvision.config.defaults import (
    DEFAULT_CONFIG,
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """Manages configuration loading, validation, and user interaction.
    
    This class handles loading configuration from YAML files, validating
    required fields, prompting users for missing values, and providing
    access to configuration values with dot notation.
    
    Attributes:
        config: Dictionary containing all configuration values
        config_path: Path to the loaded configuration file
    
    Examples:
        >>> config = ConfigManager.load("config.yaml")
        >>> print(config.get("vision.model"))
        'llama3.2-vision'
        >>> print(config.get("smugmug.api_key"))
        'your_api_key'
    """
    
    def __init__(self, config: Dict[str, Any], config_path: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config: Configuration dictionary
            config_path: Path to the configuration file (optional)
        """
        self.config = config
        self.config_path = config_path
    
    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        interactive: bool = True,
        create_if_missing: bool = True
    ) -> "ConfigManager":
        """Load configuration from file or create new config.
        
        This method attempts to load configuration from the specified path,
        or searches for config.yaml in standard locations. If no config file
        is found and create_if_missing is True, it will prompt the user for
        required values and create a new configuration file.
        
        Args:
            config_path: Path to configuration file (optional)
            interactive: Whether to prompt user for missing values
            create_if_missing: Whether to create config file if not found
            
        Returns:
            ConfigManager instance with loaded configuration
            
        Raises:
            ConfigError: If configuration cannot be loaded or validated
        """
        # Determine config path
        if config_path:
            path = Path(config_path)
        else:
            # Search in standard locations
            path = cls._find_config_file()
        
        # Load existing config or create new one
        if path and path.exists():
            logger.info(f"Loading configuration from: {path}")
            config = cls._load_yaml(path)
            
            # Merge with defaults (in case new fields were added)
            config = cls._merge_with_defaults(config)
        else:
            if not create_if_missing:
                raise ConfigError(
                    f"Configuration file not found: {path or 'config.yaml'}\n"
                    "Run with create_if_missing=True to create a new config file."
                )
            
            logger.info("No configuration file found, creating new config")
            config = DEFAULT_CONFIG.copy()
            
            # Determine where to save new config
            if config_path:
                path = Path(config_path)
            else:
                # Default to user's .smugvision directory (consistent with other configs)
                path = Path.home() / ".smugvision" / "config.yaml"
        
        # Validate and prompt for missing required fields
        if interactive:
            config = cls._prompt_for_missing_fields(config)
        
        # Validate required fields
        cls._validate_required_fields(config)
        
        # Save config if it was created or modified
        if not path.exists() or interactive:
            cls._save_yaml(config, path)
            logger.info(f"Configuration saved to: {path}")
        
        return cls(config, path)
    
    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Search for config.yaml in standard locations.
        
        Search order:
        1. ~/.smugvision/config.yaml (user home directory - primary location)
        2. ./config.yaml (current directory - for development/testing)
        
        Returns:
            Path to config file if found, None otherwise
        """
        search_paths = [
            Path.home() / ".smugvision" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                logger.debug(f"Found config file: {path}")
                return path
        
        logger.debug("No config file found in standard locations")
        return None
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ConfigError(
                    f"Invalid configuration file: {path}\n"
                    "Configuration must be a YAML dictionary."
                )
            
            return config
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Failed to parse YAML configuration: {path}\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ConfigError(
                f"Failed to load configuration file: {path}\n"
                f"Error: {e}"
            ) from e
    
    @staticmethod
    def _save_yaml(config: Dict[str, Any], path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            path: Path to save YAML file
            
        Raises:
            ConfigError: If file cannot be saved
        """
        try:
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
        except Exception as e:
            raise ConfigError(
                f"Failed to save configuration to: {path}\n"
                f"Error: {e}"
            ) from e
    
    @staticmethod
    def _merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with defaults to add any new fields.
        
        User config values take precedence over defaults. This only adds
        missing keys from defaults, never overwrites user values.
        
        Args:
            config: Loaded configuration dictionary
            
        Returns:
            Merged configuration with defaults
        """
        def deep_merge(base: dict, updates: dict) -> dict:
            """Recursively merge two dictionaries, with updates taking precedence."""
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Both are dicts, merge recursively
                    result[key] = deep_merge(result[key], value)
                else:
                    # User value takes precedence (even if empty string)
                    result[key] = value
            return result
        
        # Merge with user config taking precedence
        return deep_merge(DEFAULT_CONFIG, config)
    
    @staticmethod
    def _prompt_for_missing_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        """Prompt user for missing required configuration fields.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with user-provided values
        """
        missing_fields = []
        
        # Check which required fields are missing
        for field_path in REQUIRED_FIELDS:
            value = ConfigManager._get_nested_value(config, field_path)
            if not value:
                missing_fields.append(field_path)
        
        if not missing_fields:
            return config
        
        # Prompt for missing fields
        print("\n" + "=" * 70)
        print("smugVision Configuration Setup")
        print("=" * 70)
        print("\nSome required configuration values are missing.")
        print("Please provide the following information:\n")
        
        for field_path in missing_fields:
            description = FIELD_DESCRIPTIONS.get(
                field_path,
                f"Value for {field_path}"
            )
            
            # Prompt with description
            while True:
                print(f"\n{description}")
                value = input(f"{field_path}: ").strip()
                
                if value:
                    ConfigManager._set_nested_value(config, field_path, value)
                    break
                else:
                    print("  Error: This field is required. Please provide a value.")
        
        print("\n" + "=" * 70)
        print("Configuration setup complete!")
        print("=" * 70 + "\n")
        
        return config
    
    @staticmethod
    def _validate_required_fields(config: Dict[str, Any]) -> None:
        """Validate that all required fields are present.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ConfigError: If required fields are missing
        """
        missing = []
        
        for field_path in REQUIRED_FIELDS:
            value = ConfigManager._get_nested_value(config, field_path)
            if not value:
                missing.append(field_path)
        
        if missing:
            raise ConfigError(
                f"Missing required configuration fields:\n"
                + "\n".join(f"  - {field}" for field in missing)
                + "\n\nPlease provide these values in your config.yaml file."
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "vision.model" or "smugmug.api_key")
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get("vision.model")
            'llama3.2-vision'
            >>> config.get("vision.temperature")
            0.7
            >>> config.get("nonexistent.key", "default_value")
            'default_value'
        """
        value = self._get_nested_value(self.config, key)
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "vision.model")
            value: Value to set
            
        Examples:
            >>> config.set("vision.temperature", 0.8)
            >>> config.set("processing.marker_tag", "processed")
        """
        self._set_nested_value(self.config, key, value)
    
    @staticmethod
    def _get_nested_value(config: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot-separated key path
            
        Returns:
            Value at key path, or None if not found
        """
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    @staticmethod
    def _set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split(".")
        current = config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration (uses loaded path if not specified)
            
        Raises:
            ConfigError: If path is not specified and no config was loaded
        """
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            raise ConfigError(
                "No configuration path specified. "
                "Provide a path or load config from a file first."
            )
        
        self._save_yaml(self.config, save_path)
        logger.info(f"Configuration saved to: {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __repr__(self) -> str:
        """Return string representation of configuration."""
        path_str = f" from {self.config_path}" if self.config_path else ""
        return f"<ConfigManager{path_str}>"

