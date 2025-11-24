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
    in images based on their relationships to each other.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize relationship manager.
        
        Args:
            config_path: Path to relationships YAML file.
                        Defaults to ~/.smugvision/relationships.yaml
        """
        self.config_path = config_path or str(Path.home() / ".smugvision" / "relationships.yaml")
        self.relationships: List[tuple] = []  # List of (person1, person2, description) tuples
        self.groups: List[Dict] = []
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
            
            # Load relationships as list of tuples
            self.relationships = [
                (rel[0], rel[1], rel[2])
                for rel in config.get('relationships', [])
            ]
            
            self.groups = config.get('groups', [])
            self.enabled = True
            
            logger.info(
                f"Loaded relationship config: {len(self.relationships)} relationships, "
                f"{len(self.groups)} groups"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load relationship config: {e}")
            return False
    
    def get_relationships_for_people(self, person_names: List[str]) -> Dict[str, List[tuple]]:
        """Get all relationships for people in the image.
        
        Args:
            person_names: List of person names in the image
            
        Returns:
            Dictionary mapping relationship types to lists of (person1, person2) tuples
        """
        if not self.enabled or not person_names:
            return {}
        
        person_set = set(person_names)
        relationships_by_type = {}
        
        # Find all relationships between people in the image
        for person1, person2, rel_type in self.relationships:
            # Check if both people are in the image
            if person1 in person_set and person2 in person_set:
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append((person1, person2))
        
        return relationships_by_type
    
    def _format_names(self, names: List[str]) -> str:
        """Format a list of names nicely.
        
        Args:
            names: List of person names
            
        Returns:
            Formatted string like "Name1 and Name2" or "Name1, Name2, and Name3"
        """
        if len(names) == 1:
            return names[0].replace('_', ' ')
        elif len(names) == 2:
            return f"{names[0].replace('_', ' ')} and {names[1].replace('_', ' ')}"
        else:
            formatted = [n.replace('_', ' ') for n in names]
            return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
    
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
        
        # Check for exact group matches (try largest groups first)
        sorted_groups = sorted(self.groups, key=lambda g: len(g.get('members', [])), reverse=True)
        
        for group_data in sorted_groups:
            members = set(group_data.get('members', []))
            if members == person_set:
                return group_data.get('description')
        
        return None
    
    def generate_context(self, person_names: List[str]) -> Optional[str]:
        """Generate natural language context for people in an image.
        
        This method tries to find the most natural way to describe the people
        in the image, using group descriptions when appropriate or relationship
        descriptions when available.
        
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
        
        # Get relationship descriptions between people
        relationships_by_type = self.get_relationships_for_people(person_names)
        
        if not relationships_by_type:
            # No relationship info available
            return None
        
        # Build natural language description based on relationship types
        descriptions = []
        
        # Priority order for relationships (most specific first)
        if "spouse" in relationships_by_type and len(relationships_by_type["spouse"]) == 1:
            # Married couple
            pair = relationships_by_type["spouse"][0]
            descriptions.append(f"{pair[0].replace('_', ' ')} and {pair[1].replace('_', ' ')} (married couple)")
        
        if "parent" in relationships_by_type:
            # Parent-child relationships
            parents = set()
            children = set()
            for parent, child in relationships_by_type["parent"]:
                parents.add(parent)
                children.add(child)
            
            if len(parents) == 2 and len(children) == 1:
                descriptions.append(f"parents {self._format_names(list(parents))} with their child {children.pop().replace('_', ' ')}")
            elif len(parents) == 2 and len(children) > 1:
                descriptions.append(f"parents {self._format_names(list(parents))} with their children {self._format_names(list(children))}")
            elif len(parents) == 1 and len(children) == 1:
                descriptions.append(f"parent and child")
            elif len(parents) == 1 and len(children) > 1:
                descriptions.append(f"parent with children")
        
        if "grandparent" in relationships_by_type:
            grandparents = {gp for gp, _ in relationships_by_type["grandparent"]}
            grandchildren = {gc for _, gc in relationships_by_type["grandparent"]}
            if grandparents and grandchildren:
                descriptions.append(f"grandparent(s) and grandchild(ren)")
        
        if "sibling" in relationships_by_type and len(person_names) == 2:
            # Just siblings
            descriptions.append("siblings")
        
        if "cousin" in relationships_by_type:
            descriptions.append("cousins")
        
        if "partner" in relationships_by_type:
            descriptions.append("partners")
        
        if descriptions:
            return "; ".join(descriptions)
        
        # Fallback: just list the relationship types
        return f"related as: {', '.join(relationships_by_type.keys())}"


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

