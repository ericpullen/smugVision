"""User-defined pets: the animals face recognition can never learn.

Face recognition works from reference photos of human faces, so a dog is invisible to
it no matter how many pictures it appears in. This module holds the other half: a small
list of named animals and the sentence the model should be told about each one.

The file lives at ``~/.smugvision/pets.yaml`` and looks like this::

    pets:
      Biscuit: This is Biscuit, a Golden Retriever, and the family dog.
      Pepper: This is Pepper, the family's grey tabby cat.

The value is the whole sentence rather than a set of fields, because the sentence is
what reaches the model and the user is better placed than a template to decide how
their pet should be described.

Which pets are in which photo is NOT stored here - that is a per-photo assertion and
lives with the other assertions in ``hints.yaml`` (see :class:`HintManager`). This file
is the vocabulary; hints.yaml is the usage.
"""

import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without PyYAML
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_PETS_FILE = "~/.smugvision/pets.yaml"

#: A pet name has to survive being used as a keyword and as a YAML key, and is shown to
#: the model verbatim, so it is kept to plain readable characters.
VALID_NAME = re.compile(r"^[\w][\w .'-]{0,63}$", re.UNICODE)

_FILE_HEADER = """# smugVision pets
#
# Animals cannot be recognised from reference faces, so they are named here and
# ticked per photo (or per album) in the web UI.
#
# The value is the exact sentence given to the vision model as ground truth. Write it
# the way you would want it said:
#
#   pets:
#     Biscuit: This is Biscuit, a Golden Retriever, and the family dog.
#
# The key is also added as a keyword on any photo the pet is ticked for.

"""


class PetManager:
    """Reads and writes the user's pet vocabulary.

    Reloads when the file changes on disk, so hand edits and edits made by another
    smugVision process are both picked up without a restart.

    Attributes:
        pets_file: Path to the YAML file backing this manager
    """

    def __init__(self, pets_file: Optional[str] = None) -> None:
        """Initialize the manager.

        Args:
            pets_file: Path to the pets file. Defaults to ``~/.smugvision/pets.yaml``.
        """
        self.pets_file = Path(os.path.expanduser(pets_file or DEFAULT_PETS_FILE))
        self._pets: Dict[str, str] = {}
        self._mtime: Optional[float] = None
        self._loaded = False
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        """Load pets from disk, replacing anything held in memory.

        A missing file is normal - most users have no pets configured - and leaves an
        empty vocabulary rather than raising.
        """
        with self._lock:
            self._pets = {}
            self._loaded = True

            if not YAML_AVAILABLE:
                logger.debug("PyYAML unavailable; pets are disabled")
                self._mtime = None
                return

            if not self.pets_file.exists():
                self._mtime = None
                return

            try:
                with open(self.pets_file, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                self._mtime = self.pets_file.stat().st_mtime
            except Exception as e:
                logger.warning(f"Could not read {self.pets_file}: {e}")
                self._mtime = None
                return

            section = data.get("pets") if isinstance(data, dict) else None
            if section is None:
                return
            if not isinstance(section, dict):
                logger.warning(
                    f"Ignoring 'pets' in {self.pets_file}: expected a mapping of "
                    f"name to description, got {type(section).__name__}"
                )
                return

            for name, fact in section.items():
                clean_name = str(name).strip()
                clean_fact = str(fact).strip() if fact is not None else ""
                if not clean_name or not clean_fact:
                    logger.warning(
                        f"Skipping pet entry {name!r}: both a name and a description "
                        f"are required"
                    )
                    continue
                self._pets[clean_name] = clean_fact

            logger.info(f"Loaded {len(self._pets)} pet(s) from {self.pets_file}")

    def _ensure_fresh(self) -> None:
        """Reload if the file changed on disk since the last read."""
        try:
            exists = self.pets_file.exists()
            mtime = self.pets_file.stat().st_mtime if exists else None
        except OSError as e:  # pragma: no cover - unreadable parent directory
            logger.debug(f"Could not stat pets file {self.pets_file}: {e}")
            return

        if mtime != self._mtime or not self._loaded:
            if exists or self._mtime is not None:
                self.load()

    def all_pets(self) -> Dict[str, str]:
        """Return every configured pet.

        Returns:
            ``{name: description sentence}``, a copy safe to mutate
        """
        self._ensure_fresh()
        return dict(self._pets)

    @property
    def names(self) -> List[str]:
        """Configured pet names, sorted for a stable UI order."""
        self._ensure_fresh()
        return sorted(self._pets)

    def facts_for(self, names: Optional[List[str]]) -> List[str]:
        """Look up the sentences for the named pets.

        Unknown names are skipped rather than guessed at: a pet that was renamed or
        deleted should quietly stop contributing, not put its bare name in the prompt
        with no explanation of what it is.

        Args:
            names: Pet names, as stored against a photo

        Returns:
            Description sentences, in the order the names were given
        """
        if not names:
            return []
        self._ensure_fresh()

        facts: List[str] = []
        for name in names:
            fact = self._pets.get(str(name).strip())
            if fact:
                facts.append(fact)
            else:
                logger.debug(f"Ignoring unknown pet '{name}' (not in {self.pets_file})")
        return facts

    def known(self, names: Optional[List[str]]) -> List[str]:
        """Filter names down to the pets that actually exist.

        Args:
            names: Candidate pet names

        Returns:
            The subset that is configured, in the order given, without duplicates
        """
        if not names:
            return []
        self._ensure_fresh()

        kept: List[str] = []
        for name in names:
            clean = str(name).strip()
            if clean in self._pets and clean not in kept:
                kept.append(clean)
        return kept

    def set_pet(self, name: str, description: str) -> str:
        """Add or update one pet and persist immediately.

        Args:
            name: Pet name, also used as a keyword on tagged photos
            description: The sentence the model is told, e.g.
                "This is Biscuit, a Golden Retriever, and the family dog."

        Returns:
            The stored name, stripped

        Raises:
            ValueError: If the name or description is empty, or the name holds
                characters that would not survive being used as a keyword
            RuntimeError: If PyYAML is unavailable
            OSError: If the file cannot be written
        """
        clean_name = (name or "").strip()
        clean_description = (description or "").strip()

        if not clean_name:
            raise ValueError("A pet needs a name")
        if not VALID_NAME.match(clean_name):
            raise ValueError(
                "A pet name may only hold letters, numbers, spaces, apostrophes, "
                "dots and hyphens, and must be 64 characters or fewer"
            )
        if not clean_description:
            raise ValueError(
                "A pet needs a description - it is the sentence the model is told, "
                "so an empty one would say nothing"
            )

        self._ensure_fresh()
        with self._lock:
            self._pets[clean_name] = clean_description
            self._save()
        logger.info(f"Stored pet '{clean_name}'")
        return clean_name

    def remove_pet(self, name: str) -> bool:
        """Delete one pet and persist immediately.

        Photos that still name this pet keep the name in ``hints.yaml``; it simply
        stops contributing a sentence, and can be restored by adding the pet again.

        Args:
            name: Pet name to remove

        Returns:
            True if a pet was removed, False if the name was not configured

        Raises:
            RuntimeError: If PyYAML is unavailable
            OSError: If the file cannot be written
        """
        clean_name = (name or "").strip()
        self._ensure_fresh()
        with self._lock:
            if clean_name not in self._pets:
                return False
            del self._pets[clean_name]
            self._save()
        logger.info(f"Removed pet '{clean_name}'")
        return True

    def _save(self) -> None:
        """Write the pets file atomically. Must be called while holding the lock.

        Raises:
            RuntimeError: If PyYAML is unavailable
            OSError: If the file cannot be written
        """
        if not YAML_AVAILABLE:
            raise RuntimeError("Cannot save pets: PyYAML is not installed")

        payload = {"pets": dict(sorted(self._pets.items()))}
        self.pets_file.parent.mkdir(parents=True, exist_ok=True)

        existing_mode: Optional[int] = None
        if self.pets_file.exists():
            try:
                existing_mode = self.pets_file.stat().st_mode & 0o777
            except OSError:  # pragma: no cover - unreadable parent directory
                existing_mode = None

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.pets_file.parent),
            prefix=f".{self.pets_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_FILE_HEADER)
                yaml.dump(
                    payload,
                    handle,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            if existing_mode is not None:
                os.chmod(tmp_path, existing_mode)
            os.replace(tmp_path, self.pets_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - already gone
                pass
            raise

        try:
            self._mtime = self.pets_file.stat().st_mtime
        except OSError:  # pragma: no cover - vanished between write and stat
            self._mtime = None


_pet_manager: Optional[PetManager] = None


def get_pet_manager(pets_file: Optional[str] = None) -> PetManager:
    """Get or create the shared PetManager instance.

    Args:
        pets_file: Optional explicit path. Passing a different one rebuilds the
            manager, so a caller that configures a path is never served another's.

    Returns:
        The shared PetManager
    """
    global _pet_manager

    if _pet_manager is None:
        _pet_manager = PetManager(pets_file)
    elif pets_file is not None:
        wanted = Path(os.path.expanduser(pets_file))
        if wanted != _pet_manager.pets_file:
            _pet_manager = PetManager(pets_file)

    return _pet_manager
