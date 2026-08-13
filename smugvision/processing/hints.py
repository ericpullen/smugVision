"""User-supplied hints that ground caption generation in facts the model cannot see.

A vision model describes what a photo looks like, which is not always what it is.
gemma4 reliably called a ribbed white Nylabone dog chew "a long cracker": visually
defensible, factually wrong, and not fixable by prompt tuning because the model has no
way to know. Hints let the photo's owner assert the missing fact once and have every
subsequent caption of that photo (or album, or the whole library) respect it.

Three scopes, applied most-general-first so a narrower hint reads as a refinement of a
broader one rather than a contradiction:

* ``global`` - true of every photo ("Biscuit is a black Labrador.")
* ``album``  - true of one album, keyed by SmugMug album key
* ``image``  - true of one image, keyed by SmugMug image key

Storage is a hand-editable YAML file, ``~/.smugvision/hints.yaml`` by default::

    global: "Biscuit is a black Labrador. Ada and Sam are our children."
    albums:
      Ab3kZq: "This is Biscuit's 7th birthday party."
    images:
      Xy7NpQr: "The white ribbed object is a Nylabone dog chew, not food for people."

The file is optional at every level: a missing file, an empty file, a missing section
or a malformed entry degrades to "no hints" with a log line rather than failing the
image. Hints are advisory context, never a reason to lose a processing run.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available, hints will be disabled")

# Scope names accepted by set_hint/clear_hint, in the order resolve() applies them.
SCOPE_GLOBAL = "global"
SCOPE_ALBUM = "album"
SCOPE_IMAGE = "image"
VALID_SCOPES = (SCOPE_GLOBAL, SCOPE_ALBUM, SCOPE_IMAGE)

_FILE_HEADER = """# smugVision hints
#
# Facts about your photos that the vision model cannot see for itself. Hints are
# injected into the prompt as ground truth and outrank the model's own visual guess,
# so keep them factual - a wrong hint produces a confidently wrong caption.
#
# Scopes are combined most-general-first: global, then the album, then the image.
#
#   global: "Biscuit is a black Labrador."
#   albums:
#     Ab3kZq: "This is Biscuit's 7th birthday party."
#   images:
#     Xy7NpQr: "The white ribbed object is a Nylabone dog chew."
#
# A note reaches the prompt only. To correct a WRONG PLACE, use a location override
# instead: a note has to argue with the geocoded name that is in the prompt beside it,
# and it never reaches the keywords or the location shown in the UI. An override
# replaces the resolved place outright, so caption, keywords and UI all agree. The
# most specific scope wins - an image override beats an album one - and there is no
# global scope, because a location that applied to every photo would not be a location.
#
#   locations:
#     albums:
#       Kd2WmR: "Gorilla Enclosure, Louisville Zoo"
#     images:
#       Pv5LtBn: "Warthog Overlook, Louisville Zoo"
#
# WHO IS IN A PHOTO works the same way, for faces the recogniser missed - a mask, a hat,
# an odd angle, a child who looked different three years ago. Names are the
# reference_faces/ directory names, underscores intact, because relationships.yaml is
# keyed on that form. This REPLACES the recognised list, so name everyone in the photo
# including anyone it already got right, and unlike a note it reaches the keywords too.
#
#   people:
#     albums:
#       Qz8RcT: [Ada_Rivera, Nina_Rivera]
#     images:
#       Mw4HjXs: [Ada_Rivera]
#
# Edit by hand or through the web UI; either way the other side picks up the change.
"""


class HintManager:
    """Loads, resolves and persists user-supplied hints about photos.

    The manager owns one YAML file and keeps it in sync with in-memory state, so the
    web UI's edit-then-regenerate loop and a separate CLI process see the same hints.
    Reads re-stat the file and reload it when it changed underneath, which is what makes
    "edit the hint, re-run this one image" work without restarting anything.

    Examples:
        >>> manager = HintManager()
        >>> manager.set_hint("image", "That is a Nylabone dog chew.", key="Xy7NpQr")
        >>> manager.resolve(album_key="Ab3kZq", image_key="Xy7NpQr")
        'That is a Nylabone dog chew.'
    """

    def __init__(self, hints_file: Optional[str] = None) -> None:
        """Initialize the hint manager.

        Loading is attempted immediately but never raises: an absent or unreadable file
        simply yields no hints.

        Args:
            hints_file: Path to the hints YAML file. ``None`` (the default) uses
                ``~/.smugvision/hints.yaml``. ``~`` is expanded.
        """
        if hints_file:
            self.hints_file = Path(hints_file).expanduser()
        else:
            self.hints_file = Path.home() / ".smugvision" / "hints.yaml"

        self._global: str = ""
        self._albums: Dict[str, str] = {}
        self._images: Dict[str, str] = {}
        # Location overrides live in their own `locations:` section rather than turning
        # each note entry into a mapping, so every hints.yaml written before this
        # existed keeps loading unchanged and a bare string is never ambiguous.
        self._album_locations: Dict[str, str] = {}
        self._image_locations: Dict[str, str] = {}
        # People overrides: reference-face names, stored in the underscore form used by
        # reference_faces/ and matched by relationships.yaml.
        self._album_people: Dict[str, List[str]] = {}
        self._image_people: Dict[str, List[str]] = {}
        # Pets: names from pets.yaml. Kept beside people rather than inside them so a
        # dog is never counted as a detected face or announced as a person.
        self._album_pets: Dict[str, List[str]] = {}
        self._image_pets: Dict[str, List[str]] = {}
        self._mtime: Optional[float] = None
        self._loaded = False

        self.load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load hints from the YAML file, replacing any in-memory state.

        Never raises. A missing file, an empty file, a non-mapping document or an
        unparseable one all end with no hints loaded and a log line explaining why.

        Returns:
            Number of hints loaded, as counted by :attr:`hint_count`
        """
        self._global = ""
        self._albums = {}
        self._images = {}
        self._album_locations = {}
        self._image_locations = {}
        self._album_people = {}
        self._image_people = {}
        self._mtime = None
        self._loaded = True

        if not YAML_AVAILABLE:
            logger.debug("Hints disabled: PyYAML is not installed")
            return 0

        if not self.hints_file.exists():
            logger.debug(
                f"Hints file not found: {self.hints_file} "
                f"(create this file to give the model facts it cannot see)"
            )
            return 0

        try:
            self._mtime = self.hints_file.stat().st_mtime
            with open(self.hints_file, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except Exception as e:
            logger.warning(f"Failed to load hints from {self.hints_file}: {e}")
            return 0

        if data is None:
            logger.debug(f"Hints file is empty: {self.hints_file}")
            return 0

        if not isinstance(data, dict):
            logger.warning(
                f"Ignoring hints file {self.hints_file}: expected a mapping with "
                f"'global', 'albums' and/or 'images' keys, got {type(data).__name__}"
            )
            return 0

        self._global = self._coerce_text(data.get(SCOPE_GLOBAL), "global")
        self._albums = self._coerce_section(data.get("albums"), "albums")
        self._images = self._coerce_section(data.get("images"), "images")

        locations = data.get("locations")
        if locations is None:
            self._album_locations = {}
            self._image_locations = {}
        elif isinstance(locations, dict):
            self._album_locations = self._coerce_section(
                locations.get("albums"), "locations.albums"
            )
            self._image_locations = self._coerce_section(
                locations.get("images"), "locations.images"
            )
        else:
            logger.warning(
                f"Ignoring 'locations' in {self.hints_file}: expected a mapping with "
                f"'albums' and/or 'images' keys, got {type(locations).__name__}"
            )
            self._album_locations = {}
            self._image_locations = {}

        people = data.get("people")
        if people is None:
            self._album_people = {}
            self._image_people = {}
        elif isinstance(people, dict):
            self._album_people = self._coerce_people_section(people.get("albums"), "people.albums")
            self._image_people = self._coerce_people_section(people.get("images"), "people.images")
        else:
            logger.warning(
                f"Ignoring 'people' in {self.hints_file}: expected a mapping with "
                f"'albums' and/or 'images' keys, got {type(people).__name__}"
            )
            self._album_people = {}
            self._image_people = {}

        pets = data.get("pets")
        if pets is None:
            self._album_pets = {}
            self._image_pets = {}
        elif isinstance(pets, dict):
            self._album_pets = self._coerce_people_section(pets.get("albums"), "pets.albums")
            self._image_pets = self._coerce_people_section(pets.get("images"), "pets.images")
        else:
            logger.warning(
                f"Ignoring 'pets' in {self.hints_file}: expected a mapping with "
                f"'albums' and/or 'images' keys, got {type(pets).__name__}"
            )
            self._album_pets = {}
            self._image_pets = {}

        count = self.hint_count
        logger.debug(
            f"Loaded {count} hint(s) from {self.hints_file}: "
            f"global={'yes' if self._global else 'no'}, "
            f"albums={len(self._albums)}, images={len(self._images)}, "
            f"location overrides={len(self._album_locations) + len(self._image_locations)}"
        )
        return count

    def reload(self) -> int:
        """Force a re-read of the hints file.

        Returns:
            Number of hints loaded
        """
        return self.load()

    def _ensure_fresh(self) -> None:
        """Reload the hints file if it changed on disk since the last read.

        Keeps a long-lived manager (the web UI holds one for the life of the process)
        honest about hand edits and about writes made by another process. Costs one
        ``stat`` per call, which is nothing next to a vision inference.
        """
        try:
            exists = self.hints_file.exists()
            mtime = self.hints_file.stat().st_mtime if exists else None
        except OSError as e:
            logger.debug(f"Could not stat hints file {self.hints_file}: {e}")
            return

        if mtime != self._mtime or not self._loaded:
            if exists or self._mtime is not None:
                self.load()

    @staticmethod
    def _coerce_text(value: Any, where: str) -> str:
        """Coerce one YAML value into hint text.

        Accepts a string, a number, or a list of those (joined with spaces, so a user
        may write several short hints as a YAML list). Anything else is dropped with a
        warning rather than being stringified into the prompt.

        Args:
            value: Raw value from the YAML document
            where: Human-readable location, used only in log messages

        Returns:
            The hint text, stripped, or ``""`` if there is nothing usable
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            logger.warning(f"Ignoring hint at {where}: expected text, got a boolean")
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            parts: List[str] = []
            for item in value:
                text = HintManager._coerce_text(item, where)
                if text:
                    parts.append(text)
            return " ".join(parts)

        logger.warning(f"Ignoring hint at {where}: unsupported type {type(value).__name__}")
        return ""

    @staticmethod
    def _coerce_section(value: Any, section: str) -> Dict[str, str]:
        """Coerce one keyed YAML section (``albums`` or ``images``) into a mapping.

        Args:
            value: Raw value of the section from the YAML document
            section: Section name, used for log messages and key labelling

        Returns:
            Mapping of key to hint text, excluding blank entries
        """
        if value is None:
            return {}

        if not isinstance(value, dict):
            logger.warning(
                f"Ignoring hints section '{section}': expected a mapping of key to text, "
                f"got {type(value).__name__}"
            )
            return {}

        result: Dict[str, str] = {}
        for raw_key, raw_text in value.items():
            key = str(raw_key).strip()
            if not key:
                logger.warning(f"Ignoring hint in '{section}' with an empty key")
                continue
            text = HintManager._coerce_text(raw_text, f"{section}[{key}]")
            if text:
                result[key] = text
        return result

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_people_section(value: Any, section: str) -> Dict[str, List[str]]:
        """Coerce a ``people.albums`` / ``people.images`` YAML section into name lists.

        Accepts a YAML list of names, or a single string (split on commas, so
        hand-editing is forgiving). Names keep the underscore form used by
        ``reference_faces/`` directories, because that is what ``relationships.yaml``
        matches on and what the vision layer expects.

        Args:
            value: Raw value of the section from the YAML document
            section: Section name, used for log messages

        Returns:
            Mapping of key to list of names, excluding empty entries
        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            logger.warning(
                f"Ignoring '{section}': expected a mapping of key to a list of names, "
                f"got {type(value).__name__}"
            )
            return {}

        result: Dict[str, List[str]] = {}
        for raw_key, raw_names in value.items():
            key = str(raw_key).strip()
            if not key:
                logger.warning(f"Ignoring entry in '{section}' with an empty key")
                continue

            if isinstance(raw_names, str):
                candidates: List[Any] = list(raw_names.split(","))
            elif isinstance(raw_names, (list, tuple)):
                candidates = list(raw_names)
            else:
                logger.warning(
                    f"Ignoring '{section}[{key}]': expected a list of names, got "
                    f"{type(raw_names).__name__}"
                )
                continue

            names: List[str] = []
            for candidate in candidates:
                name = str(candidate).strip()
                if name and name not in names:
                    names.append(name)
            if names:
                result[key] = names
        return result

    def resolve_people(
        self, album_key: Optional[str], image_key: Optional[str]
    ) -> Optional[List[str]]:
        """Resolve who is in a photo, when the user has said so explicitly.

        Face recognition misses people - a mask, a hat, an odd angle, or a child who
        looked different three years ago. A free-text note can get a name into the
        caption but cannot get it into the keywords, because those come from the
        recognised-name list. An override replaces that list, so the caption, the
        keywords and the relationships lookup all see the same people.

        Replaces rather than adds, so there is one rule: list everyone in the photo,
        including anyone recognition already got right. The most specific scope wins
        outright - an image override beats an album one - and there is no global scope,
        since the same people are not in every photo.

        Args:
            album_key: SmugMug album key, or ``None`` to skip the album scope
            image_key: SmugMug image key, or ``None`` to skip the image scope

        Returns:
            Names in the underscore form (``["Ada_Rivera"]``), or ``None`` when no
            override applies. An empty list is never returned; clearing removes the entry.
        """
        self._ensure_fresh()

        if image_key:
            names = self._image_people.get(str(image_key).strip())
            if names:
                return list(names)
        if album_key:
            names = self._album_people.get(str(album_key).strip())
            if names:
                return list(names)
        return None

    def resolve_pets(
        self, album_key: Optional[str], image_key: Optional[str]
    ) -> Optional[List[str]]:
        """Resolve which pets are in a photo.

        Scoping matches :meth:`resolve_people` exactly - the most specific scope wins
        outright, an image list replaces an album one, and there is no global scope -
        so there is one rule to learn for "who is in this picture" whether the answer
        has two legs or four.

        Args:
            album_key: SmugMug album key, or ``None`` to skip the album scope
            image_key: SmugMug image key, or ``None`` to skip the image scope

        Returns:
            Pet names as stored, or ``None`` when no pets are named for this photo.
            Whether a name still exists in pets.yaml is not checked here.
        """
        self._ensure_fresh()

        if image_key:
            names = self._image_pets.get(str(image_key).strip())
            if names:
                return list(names)
        if album_key:
            names = self._album_pets.get(str(album_key).strip())
            if names:
                return list(names)
        return None

    def resolve_location(self, album_key: Optional[str], image_key: Optional[str]) -> Optional[str]:
        """Resolve the location override that applies to one image.

        A location is a single value, not something that concatenates, so the most
        specific scope wins outright: an image override beats an album override. There
        is deliberately no global scope - a location that applied to every photo you
        ever take would not be a location.

        Free-text notes cannot do this job. A note only reaches the prompt, where it
        has to argue with the geocoded place name the pipeline injects alongside it,
        and it never reaches the location tags, the appended caption suffix, or the
        location shown in the UI. An override replaces the resolved value at source so
        all four agree.

        Args:
            album_key: SmugMug album key, or ``None`` to skip the album scope
            image_key: SmugMug image key, or ``None`` to skip the image scope

        Returns:
            The overriding location string, or ``None`` when no override applies
        """
        self._ensure_fresh()

        if image_key:
            value = self._image_locations.get(str(image_key).strip())
            if value:
                return value
        if album_key:
            value = self._album_locations.get(str(album_key).strip())
            if value:
                return value
        return None

    def resolve(self, album_key: Optional[str], image_key: Optional[str]) -> str:
        """Resolve the hint text that applies to one image.

        Scopes are concatenated most-general-first - global, then album, then image -
        so a per-image hint reads as the last word on the subject. Blank scopes are
        skipped entirely, and unknown keys simply contribute nothing.

        Args:
            album_key: SmugMug album key, or ``None`` to skip the album scope. Prefer
                ``Album.album_key``; ``AlbumImage.album_key`` is empty for images
                fetched individually via ``SmugMugClient.get_image()``.
            image_key: SmugMug image key, or ``None`` to skip the image scope

        Returns:
            The combined hint text, or ``""`` when no hint applies
        """
        self._ensure_fresh()

        parts: List[str] = []
        if self._global:
            parts.append(self._global)
        if album_key:
            album_hint = self._albums.get(str(album_key).strip())
            if album_hint:
                parts.append(album_hint)
        if image_key:
            image_hint = self._images.get(str(image_key).strip())
            if image_hint:
                parts.append(image_hint)

        return " ".join(parts)

    def get_all(self) -> Dict[str, Any]:
        """Return every stored hint.

        Returns:
            ``{"global": str, "albums": {key: str}, "images": {key: str}}``. The
            mappings are copies, so mutating them does not change stored state.
        """
        self._ensure_fresh()
        return {
            SCOPE_GLOBAL: self._global,
            "albums": dict(self._albums),
            "images": dict(self._images),
            "locations": {
                "albums": dict(self._album_locations),
                "images": dict(self._image_locations),
            },
            "people": {
                "albums": {k: list(v) for k, v in self._album_people.items()},
                "images": {k: list(v) for k, v in self._image_people.items()},
            },
            "pets": {
                "albums": {k: list(v) for k, v in self._album_pets.items()},
                "images": {k: list(v) for k, v in self._image_pets.items()},
            },
        }

    def people_usage(self) -> Dict[str, int]:
        """Count how often each person has been named in a people override.

        This is the only record of who the user actually tags, which is a better
        ranking for a picker than how many reference photos a person happens to have:
        somebody can have the most reference photos and never once be picked.

        Returns:
            ``{person_name: times named}`` using raw underscore names, counting album
            and image scopes together. People never named are absent, not zero.
        """
        self._ensure_fresh()
        usage: Dict[str, int] = {}
        for names in list(self._album_people.values()) + list(self._image_people.values()):
            for name in names or []:
                usage[name] = usage.get(name, 0) + 1
        return usage

    @property
    def hint_count(self) -> int:
        """Number of stored hints, counting the global hint as one."""
        return (1 if self._global else 0) + len(self._albums) + len(self._images)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_hint(self, scope: str, text: str, key: Optional[str] = None) -> None:
        """Store a hint and persist it immediately.

        Blank text is treated as a removal, so the UI can clear a hint by submitting an
        empty textarea without needing a separate call.

        Args:
            scope: One of ``"global"``, ``"album"`` or ``"image"``
            text: The hint text. Blank (or whitespace-only) clears the hint instead.
            key: SmugMug album key or image key. Required for the ``"album"`` and
                ``"image"`` scopes, and rejected for ``"global"``.

        Raises:
            ValueError: If the scope is unknown, or ``key`` is missing for a keyed
                scope, or ``key`` is supplied for the global scope
            RuntimeError: If PyYAML is unavailable, so the hint cannot be persisted
        """
        scope, key = self._validate_scope(scope, key)

        clean = self._coerce_text(text, scope)
        if not clean:
            self.clear_hint(scope, key)
            return

        self._ensure_fresh()

        if scope == SCOPE_GLOBAL:
            self._global = clean
        elif key is not None:  # guaranteed by _validate_scope for the keyed scopes
            self._section(scope)[key] = clean

        self._save()
        logger.info(
            f"Stored {scope} hint{f' for {key}' if key else ''} "
            f"({len(clean)} chars) in {self.hints_file}"
        )

    def set_location(self, scope: str, location: str, key: Optional[str] = None) -> None:
        """Store a location override and persist it immediately.

        Blank text clears the override, so a UI can clear it with an empty input.

        Args:
            scope: ``"album"`` or ``"image"``. ``"global"`` is rejected - a location
                that applied to every photo would not be a location.
            location: The place name to use instead of the geocoded one. Blank clears.
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, or ``key`` is missing
            RuntimeError: If PyYAML is unavailable, so the value cannot be persisted
        """
        scope, key = self._validate_scope(scope, key)
        if scope == SCOPE_GLOBAL:
            raise ValueError(
                "A location override needs an album or image scope; 'global' is not "
                "meaningful for a location"
            )

        clean = self._coerce_text(location, f"locations.{scope}")
        self._ensure_fresh()
        section = self._location_section(scope)

        if not clean:
            removed = section.pop(key, None) is not None
            self._save()
            if removed:
                logger.info(f"Cleared {scope} location override for {key}")
            return

        section[key] = clean
        self._save()
        logger.info(f"Stored {scope} location override for {key}: {clean}")

    def clear_location(self, scope: str, key: Optional[str] = None) -> None:
        """Remove a location override and persist the removal immediately.

        Args:
            scope: ``"album"`` or ``"image"``
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, or ``key`` is missing
            RuntimeError: If PyYAML is unavailable
        """
        self.set_location(scope, "", key)

    def set_people(self, scope: str, names: Optional[List[str]], key: Optional[str] = None) -> None:
        """Store who is in a photo (or album) and persist it immediately.

        An empty list or ``None`` clears the override, so a UI can clear it by
        submitting nothing checked.

        Args:
            scope: ``"album"`` or ``"image"``. ``"global"`` is rejected - the same
                people are not in every photo.
            names: Reference-face names in the underscore form, e.g.
                ``["Ada_Rivera", "Nina_Rivera"]``. Order is preserved; duplicates and
                blanks are dropped. Names are NOT checked against ``reference_faces/``,
                so a name with no reference images still reaches the caption - it simply
                cannot be face-matched.
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, ``key`` is missing, or
                ``names`` is not a list of strings
            RuntimeError: If PyYAML is unavailable, so the value cannot be persisted
        """
        scope, key = self._validate_scope(scope, key)
        if scope == SCOPE_GLOBAL:
            raise ValueError(
                "A people override needs an album or image scope; 'global' is not "
                "meaningful for who is in a photo"
            )

        if names is None:
            names = []
        if isinstance(names, str) or not isinstance(names, (list, tuple)):
            raise ValueError("'names' must be a list of reference-face names")

        clean: List[str] = []
        for candidate in names:
            name = str(candidate).strip()
            if name and name not in clean:
                clean.append(name)

        self._ensure_fresh()
        section = self._people_section(scope)

        if not clean:
            removed = section.pop(key, None) is not None
            self._save()
            if removed:
                logger.info(f"Cleared {scope} people override for {key}")
            return

        section[key] = clean
        self._save()
        logger.info(f"Stored {scope} people override for {key}: {', '.join(clean)}")

    def set_pets(self, scope: str, names: Optional[List[str]], key: Optional[str] = None) -> None:
        """Store which pets are in a photo (or album) and persist it immediately.

        An empty list or ``None`` clears the entry, so a UI clears it by submitting
        nothing ticked.

        Args:
            scope: ``"album"`` or ``"image"``. ``"global"`` is rejected - the same pet
                is not in every photo you have ever taken.
            names: Pet names as defined in ``pets.yaml``. Order is preserved;
                duplicates and blanks are dropped. Names are NOT checked against
                pets.yaml here, so a pet can be ticked and defined in either order.
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, ``key`` is missing, or
                ``names`` is not a list of strings
            RuntimeError: If PyYAML is unavailable, so the value cannot be persisted
        """
        scope, key = self._validate_scope(scope, key)
        if scope == SCOPE_GLOBAL:
            raise ValueError(
                "A pet list needs an album or image scope; 'global' is not meaningful "
                "for which pets are in a photo"
            )

        if names is None:
            names = []
        if isinstance(names, str) or not isinstance(names, (list, tuple)):
            raise ValueError("'names' must be a list of pet names")

        clean: List[str] = []
        for candidate in names:
            name = str(candidate).strip()
            if name and name not in clean:
                clean.append(name)

        self._ensure_fresh()
        section = self._album_pets if scope == SCOPE_ALBUM else self._image_pets

        if not clean:
            removed = section.pop(key, None) is not None
            self._save()
            if removed:
                logger.info(f"Cleared {scope} pets for {key}")
            return

        section[key] = clean
        self._save()
        logger.info(f"Stored {scope} pets for {key}: {', '.join(clean)}")

    def clear_pets(self, scope: str, key: Optional[str] = None) -> None:
        """Remove a pet list and persist the removal immediately.

        Args:
            scope: ``"album"`` or ``"image"``
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, or ``key`` is missing
            RuntimeError: If PyYAML is unavailable
        """
        self.set_pets(scope, [], key)

    def clear_people(self, scope: str, key: Optional[str] = None) -> None:
        """Remove a people override and persist the removal immediately.

        Args:
            scope: ``"album"`` or ``"image"``
            key: SmugMug album key or image key. Required.

        Raises:
            ValueError: If the scope is unknown or ``"global"``, or ``key`` is missing
            RuntimeError: If PyYAML is unavailable
        """
        self.set_people(scope, [], key)

    def _people_section(self, scope: str) -> Dict[str, List[str]]:
        """Return the mutable people-override mapping for a keyed scope.

        Args:
            scope: ``"album"`` or ``"image"``, already validated

        Returns:
            The live dict for that scope
        """
        return self._album_people if scope == SCOPE_ALBUM else self._image_people

    def _location_section(self, scope: str) -> Dict[str, str]:
        """Return the mutable location-override mapping for a keyed scope.

        Args:
            scope: ``"album"`` or ``"image"``, already validated

        Returns:
            The live dict for that scope
        """
        return self._album_locations if scope == SCOPE_ALBUM else self._image_locations

    def clear_hint(self, scope: str, key: Optional[str] = None) -> None:
        """Remove a hint and persist the removal immediately.

        Clearing a hint that was never set is not an error; it just rewrites the file.

        Args:
            scope: One of ``"global"``, ``"album"`` or ``"image"``
            key: SmugMug album key or image key. Required for the ``"album"`` and
                ``"image"`` scopes, and rejected for ``"global"``.

        Raises:
            ValueError: If the scope is unknown, or ``key`` is missing for a keyed
                scope, or ``key`` is supplied for the global scope
            RuntimeError: If PyYAML is unavailable, so the removal cannot be persisted
        """
        scope, key = self._validate_scope(scope, key)

        self._ensure_fresh()

        if scope == SCOPE_GLOBAL:
            self._global = ""
        elif key is not None:  # guaranteed by _validate_scope for the keyed scopes
            self._section(scope).pop(key, None)

        self._save()
        logger.info(f"Cleared {scope} hint{f' for {key}' if key else ''} in {self.hints_file}")

    def _section(self, scope: str) -> Dict[str, str]:
        """Return the in-memory mapping backing one keyed scope.

        Args:
            scope: Either ``"album"`` or ``"image"``

        Returns:
            The live dict for that scope, so callers mutate stored state directly
        """
        return self._albums if scope == SCOPE_ALBUM else self._images

    @staticmethod
    def _validate_scope(scope: str, key: Optional[str]) -> Tuple[str, Optional[str]]:
        """Validate a scope/key pair.

        Args:
            scope: Scope name as supplied by the caller
            key: Key as supplied by the caller

        Returns:
            Tuple of (normalized scope, normalized key or None)

        Raises:
            ValueError: If the scope is unknown, or the scope/key combination is invalid
        """
        normalized = (scope or "").strip().lower()
        if normalized not in VALID_SCOPES:
            raise ValueError(
                f"Unknown hint scope {scope!r}: expected one of {', '.join(VALID_SCOPES)}"
            )

        clean_key = (key or "").strip()

        if normalized == SCOPE_GLOBAL:
            if clean_key:
                raise ValueError("The 'global' hint scope does not take a key")
            return normalized, None

        if not clean_key:
            raise ValueError(f"The {normalized!r} hint scope requires a key")

        return normalized, clean_key

    def _save(self) -> None:
        """Write current hints to the YAML file atomically.

        Writes a sibling temporary file in the same directory, flushes it to disk and
        then ``os.replace``s it over the target, so a crash mid-write leaves the
        previous file intact rather than a truncated one. The mode of an existing file
        is preserved; a new file is created private to the owner.

        Raises:
            RuntimeError: If PyYAML is unavailable
            OSError: If the file cannot be written
        """
        if not YAML_AVAILABLE:
            raise RuntimeError("Cannot save hints: PyYAML is not installed")

        payload: Dict[str, Any] = {
            SCOPE_GLOBAL: self._global,
            "albums": dict(sorted(self._albums.items())),
            "images": dict(sorted(self._images.items())),
            "locations": {
                "albums": dict(sorted(self._album_locations.items())),
                "images": dict(sorted(self._image_locations.items())),
            },
            "people": {
                "albums": {k: list(v) for k, v in sorted(self._album_people.items())},
                "images": {k: list(v) for k, v in sorted(self._image_people.items())},
            },
            "pets": {
                "albums": {k: list(v) for k, v in sorted(self._album_pets.items())},
                "images": {k: list(v) for k, v in sorted(self._image_pets.items())},
            },
        }

        self.hints_file.parent.mkdir(parents=True, exist_ok=True)

        existing_mode: Optional[int] = None
        if self.hints_file.exists():
            try:
                existing_mode = self.hints_file.stat().st_mode & 0o777
            except OSError:  # pragma: no cover - unreadable parent directory
                existing_mode = None

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.hints_file.parent),
            prefix=f".{self.hints_file.name}.",
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
            os.replace(tmp_path, self.hints_file)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - already gone
                pass
            raise

        try:
            self._mtime = self.hints_file.stat().st_mtime
        except OSError:  # pragma: no cover - vanished between write and stat
            self._mtime = None


# Module-level manager instance for convenience
_hint_manager: Optional[HintManager] = None


def get_hint_manager(hints_file: Optional[str] = None) -> HintManager:
    """Get or create the shared HintManager instance.

    The manager is cached for subsequent calls. Passing a different ``hints_file``
    rebuilds it, so a caller that configures an explicit path is never served a manager
    pointed at somewhere else.

    Args:
        hints_file: Optional path to the hints file. If provided and different from the
            cached manager's file, a new manager is created.

    Returns:
        HintManager instance
    """
    global _hint_manager

    if hints_file:
        path = Path(hints_file).expanduser()
        if _hint_manager is None or _hint_manager.hints_file != path:
            _hint_manager = HintManager(hints_file)
    elif _hint_manager is None:
        _hint_manager = HintManager()

    return _hint_manager
