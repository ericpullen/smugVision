"""Relationship configuration for generating contextual captions."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available, relationship context will be disabled")


class RelationshipManager:
    """Manages family/social relationships for contextual caption generation.
    
    This class loads relationship data from a YAML configuration file and
    provides methods to generate natural language descriptions of people
    in images based on their relationships.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize relationship manager.
        
        Args:
            config_path: Path to relationships YAML file.
                        Defaults to ~/.smugvision/relationships.yaml
        """
        self.config_path = config_path or str(Path.home() / ".smugvision" / "relationships.yaml")
        self.primary_person: Optional[str] = None
        self.people: Dict[str, Dict] = {}
        self.groups: Dict[str, Dict] = {}
        self.enabled = False
        
        if YAML_AVAILABLE:
            self.load_config()
    
    def load_config(self) -> bool:
        """Load relationship configuration from YAML file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.debug(f"Relationship config not found: {self.config_path}")
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.primary_person = config.get('primary_person')
            self.people = config.get('people', {})
            self.groups = config.get('groups', {})
            self.enabled = True
            
            logger.info(
                f"Loaded relationship config: {len(self.people)} people, "
                f"{len(self.groups)} groups"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load relationship config: {e}")
            return False
    
    def get_description(self, person_name: str) -> Optional[str]:
        """Get relationship description for a person.
        
        Args:
            person_name: Name of the person (e.g., "Kelly_Pullen")
            
        Returns:
            Relationship description (e.g., "your wife") or None
        """
        if not self.enabled or person_name not in self.people:
            return None
        
        return self.people[person_name].get('description')
    
    def get_group_description(self, person_names: List[str]) -> Optional[str]:
        """Get group description if all members are present.
        
        Args:
            person_names: List of person names in the image
            
        Returns:
            Group description if exact match found, None otherwise
        """
        if not self.enabled or not self.groups:
            return None
        
        person_set = set(person_names)
        
        # Check for exact group matches
        for group_name, group_data in self.groups.items():
            members = set(group_data.get('members', []))
            if members == person_set:
                return group_data.get('description')
        
        return None
    
    def generate_context(self, person_names: List[str]) -> Optional[str]:
        """Generate natural language context for people in an image.
        
        This method tries to find the most natural way to describe the people
        in the image, using group descriptions when appropriate or individual
        relationships when needed.
        
        Args:
            person_names: List of identified person names
            
        Returns:
            Natural language context string, or None if no context available
        """
        if not self.enabled or not person_names:
            return None
        
        # Try group description first (if all people form a known group)
        group_desc = self.get_group_description(person_names)
        if group_desc:
            return group_desc
        
        # Otherwise, build individual descriptions
        descriptions = []
        for name in person_names:
            desc = self.get_description(name)
            if desc:
                # Use the description (e.g., "your wife")
                descriptions.append(desc)
            else:
                # Fall back to just the name
                descriptions.append(name.replace('_', ' '))
        
        if not descriptions:
            return None
        
        # Format the list naturally
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"


# Global instance
_relationship_manager: Optional[RelationshipManager] = None


def get_relationship_manager(config_path: Optional[str] = None) -> RelationshipManager:
    """Get or create the global relationship manager instance.
    
    Args:
        config_path: Optional path to relationships config file
        
    Returns:
        RelationshipManager instance
    """
    global _relationship_manager
    
    if _relationship_manager is None:
        _relationship_manager = RelationshipManager(config_path)
    
    return _relationship_manager

