"""SmugMug API client for gallery and image operations."""

import logging
import time
from typing import List, Optional, Dict, Any, Set, Tuple
from pathlib import Path

import requests
from requests_oauthlib import OAuth1

from smugvision.smugmug.models import (
    NODE_TYPE_FOLDER,
    Album,
    AlbumImage,
    AlbumSearchResult,
    BrowseNode,
    NodeListing,
    NodeRef,
)
from smugvision.smugmug.exceptions import (
    SmugMugAPIError,
    SmugMugAuthError,
    SmugMugNotFoundError,
    SmugMugRateLimitError,
)

logger = logging.getLogger(__name__)


class SmugMugClient:
    """Client for interacting with SmugMug API.
    
    This class handles authentication via OAuth 1.0a, API requests, and data
    retrieval from SmugMug galleries and images. It provides methods for
    listing albums, retrieving images, and updating image metadata.
    
    Attributes:
        api_key: SmugMug API key
        api_secret: SmugMug API secret
        access_token: OAuth access token
        access_token_secret: OAuth access token secret
        base_url: Base URL for SmugMug API v2
        auth: OAuth1 authentication object
    """
    
    API_VERSION = "v2"
    BASE_URL = f"https://api.smugmug.com/api/{API_VERSION}"

    #: Query parameters that nest the full Album resource inside each album node
    #: of a ``!children`` listing, so image counts and album keys arrive without
    #: an extra request per album. Verified essentially free in wall time; do NOT
    #: add ``HighlightImage`` here, it costs ~2.5x.
    _ALBUM_EXPAND_PARAMS = {"_expand": "Album", "_expandmethod": "inline"}

    #: Default bounds for search_albums(). These are a safety valve against a
    #: pathologically large account, NOT a latency knob: they are set high enough
    #: to cover a ~1200 album / ~115 folder account completely. Cost is roughly
    #: 0.2s per folder scanned (measured 25s for a complete 115-folder walk), so
    #: search_albums() must never run on a request thread. Use
    #: list_node_children() for anything interactive.
    DEFAULT_SEARCH_MAX_DEPTH = 6
    DEFAULT_SEARCH_MAX_FOLDERS = 300
    DEFAULT_SEARCH_MAX_RESULTS = 2000

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str,
        timeout: int = 30
    ) -> None:
        """Initialize SmugMug client with OAuth credentials.
        
        Args:
            api_key: SmugMug API key
            api_secret: SmugMug API secret
            access_token: OAuth access token
            access_token_secret: OAuth access token secret
            timeout: Request timeout in seconds
            
        Raises:
            SmugMugAuthError: If credentials are invalid
        """
        if not all([api_key, api_secret, access_token, access_token_secret]):
            raise SmugMugAuthError(
                "All OAuth credentials are required: "
                "api_key, api_secret, access_token, access_token_secret"
            )
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.timeout = timeout

        # Cached root node ID. It is immutable for a set of credentials but
        # costs an /api/v2!authuser round trip to look up, and the tree browser
        # asks for it on every listing.
        self._root_node_id: Optional[str] = None

        # Create OAuth1 authentication
        self.auth = OAuth1(
            api_key,
            api_secret,
            access_token,
            access_token_secret,
            signature_type='auth_header'
        )
        
        logger.info("SmugMug client initialized")
        
        # Verify authentication by getting user info
        try:
            self._verify_authentication()
        except Exception as e:
            raise SmugMugAuthError(
                f"Failed to authenticate with SmugMug: {e}"
            ) from e
    
    def _verify_authentication(self) -> Dict[str, Any]:
        """Verify authentication by retrieving authenticated user info.
        
        Returns:
            User data dictionary
            
        Raises:
            SmugMugAuthError: If authentication fails
        """
        try:
            response = self._request("GET", "/api/v2!authuser")
            user = response.get("Response", {}).get("User", {})
            nickname = user.get("NickName", "Unknown")
            logger.info(f"Successfully authenticated as: {nickname}")
            return user
        except SmugMugAPIError as e:
            raise SmugMugAuthError(
                f"Authentication failed: {e}"
            ) from e
    
    def get_user_root_node(self, use_cache: bool = True) -> str:
        """Get the authenticated user's root node ID.

        The value never changes for a set of credentials, so it is memoized on
        the client after the first lookup.

        Args:
            use_cache: If False, ignore the memoized value and re-query the API

        Returns:
            Root node ID

        Raises:
            SmugMugAPIError: If request fails
        """
        if use_cache and self._root_node_id:
            return self._root_node_id

        try:
            response = self._request("GET", "/api/v2!authuser")
            user = response.get("Response", {}).get("User", {})

            # Get node URI
            uris = user.get("Uris", {})
            node_uri = uris.get("Node", {})
            if node_uri:
                uri = node_uri.get("Uri", "")
                # Extract node ID from URI
                if "/node/" in uri:
                    node_id = uri.split("/node/")[-1]
                    logger.debug(f"User root node: {node_id}")
                    self._root_node_id = node_id
                    return node_id

            raise SmugMugAPIError("Could not find user root node")
        except Exception as e:
            raise SmugMugAPIError(f"Failed to get user root node: {e}") from e

    def find_node_by_path(self, path: str) -> Optional[str]:
        """Find a node ID by navigating a path from root.
        
        Args:
            path: Path like "Gallery/Year"
            
        Returns:
            Node ID if found, None otherwise
        """
        logger.info(f"Finding node by path: {path}")
        
        # Get root node
        try:
            current_node_id = self.get_user_root_node()
        except Exception as e:
            logger.error(f"Could not get root node: {e}")
            return None
        
        # Split path and navigate
        parts = [p for p in path.split('/') if p]
        
        for part in parts:
            logger.debug(f"Looking for '{part}' under node {current_node_id}")
            
            try:
                children = self.get_node_children(current_node_id)
            except Exception as e:
                logger.error(f"Could not get children of node {current_node_id}: {e}")
                return None
            
            # Find matching folder
            found = False
            for child in children:
                child_name = child.get("Name", "")
                child_url_name = child.get("UrlName", "")
                child_type = child.get("Type")
                
                if child_type == "Folder" and (child_name == part or child_url_name == part):
                    current_node_id = child.get("NodeID")
                    if current_node_id:
                        found = True
                        logger.debug(f"Found '{part}' -> node {current_node_id}")
                        break
            
            if not found:
                logger.warning(f"Could not find '{part}' under current node")
                return None
        
        logger.info(f"Resolved path '{path}' to node: {current_node_id}")
        return current_node_id
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to SmugMug API.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (with or without base URL)
            params: Query parameters
            json_data: JSON body data
            headers: Additional headers
            
        Returns:
            Response data dictionary
            
        Raises:
            SmugMugAPIError: If request fails
            SmugMugNotFoundError: If resource not found (404)
            SmugMugRateLimitError: If rate limit exceeded (429)
        """
        # Build full URL
        if endpoint.startswith("http"):
            url = endpoint
        elif endpoint.startswith("/api/v2"):
            url = f"https://api.smugmug.com{endpoint}"
        else:
            url = f"{self.BASE_URL}{endpoint}"
        
        # Default headers
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "smugVision/0.1.0"
        }
        if headers:
            request_headers.update(headers)
        
        # Log request
        logger.debug(f"SmugMug API {method} {url}")
        if params:
            logger.debug(f"  Params: {params}")
        
        try:
            # Disable automatic redirects to handle them manually
            # (redirects lose OAuth headers which causes 401 errors)
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                params=params,
                json=json_data,
                headers=request_headers,
                timeout=self.timeout,
                allow_redirects=False
            )
            
            # Log response status
            logger.debug(f"  Response: {response.status_code}")
            
            # Handle redirects manually to preserve auth
            redirect_count = 0
            max_redirects = 5
            while response.status_code in (301, 302, 303, 307, 308) and redirect_count < max_redirects:
                redirect_url = response.headers.get("Location")
                if not redirect_url:
                    break
                
                # Make absolute URL if relative
                if redirect_url.startswith("/"):
                    redirect_url = f"https://api.smugmug.com{redirect_url}"
                
                logger.debug(f"  Following redirect to: {redirect_url}")
                
                # For 303, always use GET; for others, preserve method
                redirect_method = "GET" if response.status_code == 303 else method
                
                response = requests.request(
                    method=redirect_method,
                    url=redirect_url,
                    auth=self.auth,
                    params=params if redirect_method == method else None,
                    json=json_data if redirect_method == method else None,
                    headers=request_headers,
                    timeout=self.timeout,
                    allow_redirects=False
                )
                
                logger.debug(f"  Redirect response: {response.status_code}")
                redirect_count += 1
            
            # Handle error status codes
            if response.status_code == 404:
                raise SmugMugNotFoundError(
                    f"Resource not found: {endpoint}",
                    status_code=404,
                    response=response.json() if response.content else None
                )
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = response.headers.get("Retry-After")
                retry_seconds = int(retry_after) if retry_after else None
                raise SmugMugRateLimitError(
                    "Rate limit exceeded",
                    retry_after=retry_seconds
                )
            elif response.status_code == 401:
                raise SmugMugAuthError(
                    "Authentication failed. Check your API credentials."
                )
            elif not response.ok:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "Message" in error_data:
                        error_msg = error_data["Message"]
                except:
                    pass
                raise SmugMugAPIError(
                    error_msg,
                    status_code=response.status_code,
                    response=response.json() if response.content else None
                )
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.Timeout as e:
            raise SmugMugAPIError(
                f"Request timeout after {self.timeout} seconds: {endpoint}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise SmugMugAPIError(
                f"Request failed: {e}"
            ) from e
    
    def get_node_children(
        self,
        node_id: str,
        start: int = 1,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Get children (albums and folders) of a node.
        
        This method handles pagination automatically to retrieve all children.
        
        Args:
            node_id: Node ID (e.g., from URL like n-ABC123)
            start: Starting position (1-indexed)
            count: Number of children per page (max 100)
            
        Returns:
            List of child nodes with their details
            
        Raises:
            SmugMugNotFoundError: If node not found
            SmugMugAPIError: If request fails
        """
        return self._fetch_node_children(node_id, start=start, count=count)

    def _fetch_node_children(
        self,
        node_id: str,
        start: int = 1,
        count: int = 100,
        extra_params: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO
    ) -> List[Dict[str, Any]]:
        """Fetch all children of a node, following pagination.

        Shared implementation behind :meth:`get_node_children` and the tree
        browsing methods. ``log_level`` exists so a recursive walk does not emit
        one INFO line per node while the single-level public method keeps its
        existing INFO logging.

        Args:
            node_id: Node ID
            start: Starting position (1-indexed)
            count: Number of children per page (max 100)
            extra_params: Additional SmugMug query parameters (e.g. ``_expand``)
            log_level: Logging level for the per-node "fetching" message

        Returns:
            List of raw child node dictionaries

        Raises:
            SmugMugNotFoundError: If node not found, or if node_id refers to an
                album (SmugMug has no ``!children`` endpoint for album nodes)
            SmugMugAPIError: If request fails
        """
        logger.log(log_level, f"Fetching children of node: {node_id}")

        all_children: List[Dict[str, Any]] = []
        current_start = start

        while True:
            endpoint = f"/node/{node_id}!children"
            params: Dict[str, Any] = {
                "start": current_start,
                "count": min(count, 100)  # Max 100 per page
            }
            if extra_params:
                params.update(extra_params)

            response = self._request("GET", endpoint, params=params)

            response_data = response.get("Response", {})
            children = response_data.get("Node", [])

            if not children:
                break

            all_children.extend(children)
            logger.debug(f"Retrieved {len(children)} children (start={current_start})")

            # Check if there are more pages
            pages = response_data.get("Pages", {})
            if not pages.get("NextPage"):
                break

            current_start += len(children)

        logger.debug(f"Found {len(all_children)} total children under node {node_id}")
        return all_children

    def get_node_breadcrumb(self, node_id: str) -> List[NodeRef]:
        """Get the ancestor chain of a node, ordered root-first.

        Uses SmugMug's ``!parents`` endpoint, which returns the whole chain in a
        single request (self first, root last) and works on album nodes as well
        as folders. The result is reversed so a UI can render it left to right,
        and the last entry is always the node itself.

        The root node's SmugMug ``Name`` is the empty string; ``NodeRef``
        substitutes ``ROOT_NODE_LABEL`` for it.

        Args:
            node_id: Node ID to build a breadcrumb for

        Returns:
            List of NodeRef from root to the node itself. Empty if the chain
            could not be retrieved.

        Raises:
            SmugMugNotFoundError: If the node does not exist
            SmugMugAPIError: If the request fails
        """
        response = self._request(
            "GET",
            f"/node/{node_id}!parents",
            params={"_shorturis": 1}
        )
        nodes = response.get("Response", {}).get("Node", [])
        if not nodes:
            logger.warning(f"No parent chain returned for node {node_id}")
            return []

        # !parents is ordered self-first / root-last; a breadcrumb reads the
        # other way round.
        crumbs = [NodeRef.from_api_response(n) for n in reversed(nodes)]
        logger.debug(f"Breadcrumb for {node_id}: {' / '.join(c.name for c in crumbs)}")
        return crumbs

    def list_node_children(
        self,
        node_id: Optional[str] = None,
        include_breadcrumb: bool = True,
        include_album_details: bool = True,
        count: int = 100
    ) -> NodeListing:
        """List one level of the node tree: child folders and albums.

        This is the lazy, per-level primitive a gallery picker should use. It
        costs two GETs (~0.3-0.7s measured) regardless of how large the account
        is, because it never descends. A full recursive walk of a large account
        takes tens of seconds and must not sit behind a page load - see
        :meth:`search_albums` for a bounded alternative.

        ``System Album`` nodes (SmugMug's own Profile Images / Cover Images) are
        excluded even though they expose real album keys, and a child that cannot
        be interpreted is logged and skipped rather than failing the listing.

        Args:
            node_id: Node to list. Defaults to the authenticated user's root node.
            include_breadcrumb: If True, also fetch the ancestor chain (one extra
                GET) and use it to verify the node is listable. When False the
                returned breadcrumb holds only the node itself.
            include_album_details: If True, inline-expand each album so
                ``image_count`` and album metadata are populated without an extra
                request per album. Set False for a slightly smaller payload;
                ``album_key`` is still resolved either way.
            count: Children per page (max 100); pagination is automatic.

        Returns:
            NodeListing with the node, its breadcrumb, and its child folders and
            albums in separate lists. ``partial`` is True when part of the
            listing could not be retrieved.

        Raises:
            SmugMugNotFoundError: If the node does not exist
            SmugMugAPIError: If node_id refers to an album (albums have no
                children) or the request fails
        """
        if not node_id:
            node_id = self.get_user_root_node()

        node_ref: Optional[NodeRef] = None
        breadcrumb: List[NodeRef] = []
        partial = False

        if include_breadcrumb:
            try:
                breadcrumb = self.get_node_breadcrumb(node_id)
            except SmugMugNotFoundError:
                # A genuinely missing node should surface as 404, not as a
                # confusing empty listing.
                raise
            except Exception as e:
                logger.warning(f"Could not build breadcrumb for node {node_id}: {e}")
                partial = True

            if breadcrumb:
                node_ref = breadcrumb[-1]
                if not node_ref.is_folder:
                    # !children 404s on album nodes, which is indistinguishable
                    # from a bad node ID. Fail with something actionable instead.
                    raise SmugMugAPIError(
                        f"Node {node_id} is a {node_ref.node_type!r} "
                        f"({node_ref.name!r}), not a folder, so it has no children. "
                        "Albums are selected, not browsed into."
                    )

        extra_params = dict(self._ALBUM_EXPAND_PARAMS) if include_album_details else None
        children = self._fetch_node_children(
            node_id,
            count=count,
            extra_params=extra_params,
            log_level=logging.DEBUG
        )

        folders: List[BrowseNode] = []
        albums: List[BrowseNode] = []
        for child in children:
            try:
                entry = BrowseNode.from_api_response(child)
            except Exception as e:
                # One malformed row must not lose the whole level.
                logger.warning(f"Skipping unreadable child of node {node_id}: {e}")
                partial = True
                continue
            if entry is None:
                continue
            if entry.is_folder:
                folders.append(entry)
            else:
                albums.append(entry)

        if node_ref is None:
            # Reached only when the breadcrumb is unavailable (not requested, or
            # the !parents lookup failed), so there is no real name to use.
            node_ref = NodeRef(
                node_id=node_id,
                name=node_id,
                node_type=NODE_TYPE_FOLDER,
                is_root=node_id == self._root_node_id
            )
            breadcrumb = [node_ref]

        logger.info(
            f"Listed node {node_id} ({node_ref.name!r}): "
            f"{len(folders)} folder(s), {len(albums)} album(s)"
        )
        return NodeListing(
            node=node_ref,
            breadcrumb=breadcrumb,
            folders=folders,
            albums=albums,
            partial=partial
        )

    def search_albums(
        self,
        query: Optional[str] = None,
        node_id: Optional[str] = None,
        exact_match: bool = False,
        max_depth: int = DEFAULT_SEARCH_MAX_DEPTH,
        max_folders: int = DEFAULT_SEARCH_MAX_FOLDERS,
        max_results: int = DEFAULT_SEARCH_MAX_RESULTS
    ) -> AlbumSearchResult:
        """Walk the node tree breadth-first, with hard bounds, collecting albums.

        A full walk of a ~1200 album account measured ~25s over ~116 requests
        (median 0.19s per request), so this method is bounded on THREE axes and
        reports when a bound stopped it. Callers must check
        ``AlbumSearchResult.truncated`` before telling a user "no matches" - a
        truncated walk returns a partial tree, not a complete answer.

        Nodes that cannot be listed are logged as warnings, recorded in
        ``AlbumSearchResult.errors`` and skipped; the walk still returns whatever
        it found.

        Args:
            query: Case-insensitive substring to match against album name and
                URL name. None or empty returns every album found within bounds.
            node_id: Node to search under. Defaults to the user's root node.
            exact_match: If True, require the whole name (or URL name) to equal
                ``query`` case-insensitively instead of containing it.
            max_depth: Maximum levels below the starting node to descend. 0
                lists only the starting node's own children.
            max_folders: Maximum number of folders to list. This is the real cost
                bound: roughly 0.2-0.5s per folder.
            max_results: Maximum albums to return.

        Returns:
            AlbumSearchResult with the albums found and truncation information.

        Raises:
            SmugMugAPIError: If the starting node cannot be resolved
        """
        if not node_id:
            node_id = self.get_user_root_node()

        search_term = (query or "").strip().lower()
        result = AlbumSearchResult()
        # Breadth-first, so shallow (usually more relevant) albums come first and
        # a truncated walk returns a sensible prefix rather than one deep branch.
        frontier: List[Tuple[str, int]] = [(node_id, 0)]
        seen: Set[str] = {node_id}
        reasons: List[str] = []
        depth_pruned = 0
        stop = False

        logger.info(
            f"Searching albums under node {node_id} "
            f"(query={query!r}, max_depth={max_depth}, max_folders={max_folders}, "
            f"max_results={max_results})"
        )

        while frontier and not stop:
            current_node_id, depth = frontier.pop(0)

            if result.folders_scanned >= max_folders:
                reasons.append(
                    f"folder limit reached (scanned {max_folders}); "
                    f"{len(frontier) + 1} folder(s) not searched"
                )
                stop = True
                break

            try:
                children = self._fetch_node_children(
                    current_node_id,
                    extra_params=dict(self._ALBUM_EXPAND_PARAMS),
                    log_level=logging.DEBUG
                )
            except Exception as e:
                logger.warning(f"Could not list node {current_node_id}, skipping: {e}")
                result.errors.append(f"{current_node_id}: {e}")
                continue

            result.folders_scanned += 1
            result.max_depth_reached = max(result.max_depth_reached, depth)

            for child in children:
                try:
                    entry = BrowseNode.from_api_response(child)
                except Exception as e:
                    logger.warning(f"Skipping unreadable child of {current_node_id}: {e}")
                    result.errors.append(f"child of {current_node_id}: {e}")
                    continue
                if entry is None:
                    continue

                if entry.is_folder:
                    if entry.node_id in seen:
                        continue
                    if depth >= max_depth:
                        # Pruned by max_depth. This is truncation: albums below
                        # here exist and were never looked at.
                        depth_pruned += 1
                        continue
                    seen.add(entry.node_id)
                    frontier.append((entry.node_id, depth + 1))
                    continue

                if search_term and not self._album_matches(entry, search_term, exact_match):
                    continue

                if len(result.albums) >= max_results:
                    reasons.append(f"result limit reached ({max_results} albums)")
                    stop = True
                    break
                result.albums.append(entry)

        if depth_pruned:
            reasons.append(
                f"depth limit reached (max_depth={max_depth}); "
                f"{depth_pruned} subfolder(s) not searched"
            )
        if reasons:
            result.truncated = True
            result.truncation_reason = "; ".join(reasons)

        if result.truncated:
            logger.warning(
                f"Album search under node {node_id} was TRUNCATED and is INCOMPLETE: "
                f"{result.truncation_reason}. Found {len(result.albums)} album(s) in "
                f"{result.folders_scanned} folder(s)."
            )
        else:
            logger.info(
                f"Album search complete: {len(result.albums)} album(s) in "
                f"{result.folders_scanned} folder(s), depth {result.max_depth_reached}"
            )
        if result.errors:
            logger.warning(f"{len(result.errors)} node(s) could not be listed during search")

        return result

    @staticmethod
    def _album_matches(album: BrowseNode, search_term: str, exact_match: bool) -> bool:
        """Test an album against a lowercased search term.

        Args:
            album: Album entry to test
            search_term: Already-lowercased search term
            exact_match: If True, require equality rather than containment

        Returns:
            True if the album's name or URL name matches
        """
        candidates = [album.name.lower(), (album.url_name or "").lower()]
        if exact_match:
            return any(c == search_term for c in candidates)
        return any(search_term in c for c in candidates)

    def find_albums_by_name(
        self,
        node_id: str,
        album_name: str,
        exact_match: bool = False,
        recursive: bool = True,
        max_depth: int = 3
    ) -> List[Album]:
        """Find albums by name under a specific node.
        
        Args:
            node_id: Node ID to search under
            album_name: Album name to search for
            exact_match: If True, require exact name match (case-insensitive)
            recursive: If True, search in subfolders too
            max_depth: Maximum depth to recurse (default 3)
            
        Returns:
            List of matching Album objects
            
        Raises:
            SmugMugAPIError: If request fails
        """
        logger.info(f"Searching for album '{album_name}' under node {node_id} (recursive={recursive})")
        
        def search_node(current_node_id: str, depth: int = 0) -> List[Album]:
            """Recursively search for albums."""
            if depth > max_depth:
                return []
            
            try:
                children = self.get_node_children(current_node_id)
            except Exception as e:
                logger.warning(f"Could not access node {current_node_id}: {e}")
                return []
            
            matching_albums = []
            search_lower = album_name.lower()
            folders_to_search = []
            
            for child in children:
                child_type = child.get("Type")
                
                if child_type == "Album":
                    child_name = child.get("Name", "")
                    url_name = child.get("UrlName", "")
                    
                    # Check if it matches
                    matches = False
                    if exact_match:
                        matches = (child_name.lower() == search_lower or 
                                  url_name.lower() == search_lower)
                    else:
                        matches = (search_lower in child_name.lower() or 
                                  search_lower in url_name.lower())
                    
                    if matches:
                        # Extract album key from URI
                        uris = child.get("Uris", {})
                        album_uri = uris.get("Album", {})
                        if album_uri:
                            uri = album_uri.get("Uri", "")
                            if "/album/" in uri:
                                album_key = uri.split("/album/")[-1]
                                try:
                                    album = self.get_album(album_key)
                                    matching_albums.append(album)
                                    logger.debug(f"Found matching album: {album.name} ({album_key})")
                                except Exception as e:
                                    logger.warning(f"Could not fetch album {album_key}: {e}")
                
                elif child_type == "Folder" and recursive:
                    # Add folder to search list
                    child_node_id = child.get("NodeID")
                    if child_node_id:
                        folders_to_search.append(child_node_id)
            
            # Search subfolders
            if recursive and folders_to_search:
                logger.debug(f"Searching {len(folders_to_search)} subfolder(s) at depth {depth + 1}")
                for folder_node_id in folders_to_search:
                    matching_albums.extend(search_node(folder_node_id, depth + 1))
            
            return matching_albums
        
        matching_albums = search_node(node_id)
        logger.info(f"Found {len(matching_albums)} matching album(s)")
        return matching_albums
    
    def resolve_album_key(
        self,
        identifier: str,
        node_id: Optional[str] = None
    ) -> str:
        """Resolve an album identifier to an album key.
        
        This method tries to determine if the identifier is:
        1. Already an album key (try to fetch it)
        2. An album name (search for it under node_id)
        
        Args:
            identifier: Album key or album name
            node_id: Node ID to search under (required if identifier is a name)
            
        Returns:
            Album key
            
        Raises:
            SmugMugNotFoundError: If album not found
            SmugMugAPIError: If resolution fails
        """
        # First, try as album key
        try:
            album = self.get_album(identifier)
            logger.debug(f"'{identifier}' is a valid album key")
            return identifier
        except SmugMugNotFoundError:
            # Not a valid album key, try as name
            if not node_id:
                raise SmugMugAPIError(
                    f"'{identifier}' is not a valid album key. "
                    "Provide node_id to search by name."
                )
            
            # Search for album by name
            logger.debug(f"'{identifier}' not found as key, searching as name...")
            albums = self.find_albums_by_name(node_id, identifier)
            
            if not albums:
                raise SmugMugNotFoundError(
                    f"No albums found matching '{identifier}' under node {node_id}",
                    status_code=404
                )
            
            if len(albums) > 1:
                names = [a.name for a in albums]
                raise SmugMugAPIError(
                    f"Multiple albums found matching '{identifier}': {names}\n"
                    f"Please be more specific or use the album key directly."
                )
            
            logger.info(f"Resolved '{identifier}' to album: {albums[0].name}")
            return albums[0].album_key
    
    def get_album(self, album_key: str) -> Album:
        """Get album details by album key.
        
        Args:
            album_key: Album key (unique identifier)
            
        Returns:
            Album object
            
        Raises:
            SmugMugNotFoundError: If album not found
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching album: {album_key}")
        
        endpoint = f"/album/{album_key}"
        response = self._request("GET", endpoint)
        
        album_data = response.get("Response", {}).get("Album", {})
        album = Album.from_api_response(album_data)
        
        logger.debug(f"Retrieved album: {album.name} ({album.image_count} images)")
        return album
    
    def get_album_images(
        self,
        album_key: str,
        start: int = 1,
        count: int = 100
    ) -> List[AlbumImage]:
        """Get images from an album.
        
        This method handles pagination automatically to retrieve all images.
        SmugMug returns images in pages (default 100 per page).
        The _expandmethod parameter requests full image size URIs.
        
        Args:
            album_key: Album key (unique identifier)
            start: Starting position (1-indexed)
            count: Number of images per page (max 100)
            
        Returns:
            List of AlbumImage objects
            
        Raises:
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching images from album: {album_key}")
        
        all_images = []
        current_start = start
        
        while True:
            endpoint = f"/album/{album_key}!images"
            params = {
                "start": current_start,
                "count": min(count, 100),  # Max 100 per page
                "_expandmethod": "inline",  # Request full URIs
                "_expand": "ImageDownload,ImageSizes"  # Include download URLs and sizes
            }
            
            response = self._request("GET", endpoint, params=params)
            
            response_data = response.get("Response", {})
            images_data = response_data.get("AlbumImage", [])
            
            if not images_data:
                break
            
            # Convert to AlbumImage objects
            for image_data in images_data:
                image = AlbumImage.from_api_response(image_data, album_key)
                all_images.append(image)
            
            logger.debug(f"Retrieved {len(images_data)} images (start={current_start})")
            
            # Check if there are more pages
            pages = response_data.get("Pages", {})
            if not pages.get("NextPage"):
                break
            
            current_start += len(images_data)
        
        logger.info(f"Retrieved {len(all_images)} total images from album {album_key}")
        return all_images
    
    def get_image(self, image_key: str, expand_sizes: bool = False) -> AlbumImage:
        """Get image details by image key.
        
        Args:
            image_key: Image key (unique identifier)
            expand_sizes: If True, include expanded image size URLs
            
        Returns:
            AlbumImage object
            
        Raises:
            SmugMugNotFoundError: If image not found
            SmugMugAPIError: If request fails
        """
        logger.info(f"Fetching image: {image_key}")
        
        endpoint = f"/image/{image_key}"
        params = {}
        if expand_sizes:
            params["_expandmethod"] = "inline"
            params["_expand"] = "ImageSizes"
        
        response = self._request("GET", endpoint, params=params if params else None)
        
        image_data = response.get("Response", {}).get("Image", {})
        image = AlbumImage.from_api_response(image_data)
        
        logger.debug(f"Retrieved image: {image.file_name}")
        return image
    
    def update_image_metadata(
        self,
        image_key: str,
        caption: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> AlbumImage:
        """Update image metadata (caption, keywords, title).
        
        Args:
            image_key: Image key (unique identifier)
            caption: New caption (optional)
            keywords: New keywords list (optional)
            title: New title (optional)
            
        Returns:
            Updated AlbumImage object
            
        Raises:
            SmugMugAPIError: If update fails
        """
        logger.info(f"Updating metadata for image: {image_key}")
        
        # Build update data
        update_data = {}
        if caption is not None:
            update_data["Caption"] = caption
        if keywords is not None:
            # Convert list to comma-separated string
            update_data["Keywords"] = ", ".join(keywords) if keywords else ""
        if title is not None:
            update_data["Title"] = title
        
        if not update_data:
            logger.warning("No metadata to update")
            return self.get_image(image_key)
        
        logger.debug(f"Update data: {update_data}")
        
        endpoint = f"/image/{image_key}"
        response = self._request("PATCH", endpoint, json_data=update_data)
        
        image_data = response.get("Response", {}).get("Image", {})
        image = AlbumImage.from_api_response(image_data)
        
        logger.info(f"Successfully updated metadata for: {image.file_name}")
        return image
    
    def download_image(
        self,
        image: AlbumImage,
        destination: str,
        size: str = "Medium",
        skip_if_exists: bool = True
    ) -> Optional[Path]:
        """Download image or video to local file.
        
        Args:
            image: AlbumImage object (can also be a video)
            destination: Destination directory path
            size: Image size (Medium, Large, XLarge, X2Large, X3Large, Original, etc.).
                  Matched case-insensitively, so "medium" and "Medium" both work.
                  Note: For videos, only Original is supported
            skip_if_exists: If True, skip download if file already exists
            
        Returns:
            Path to downloaded file, or None if skipped
            
        Raises:
            SmugMugAPIError: If download fails
        """
        dest_path = Path(destination) / image.file_name
        
        # Check if already exists
        if skip_if_exists and dest_path.exists():
            logger.debug(f"Skipping {image.file_name} (already cached)")
            return None
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get download URL
        download_url = None
        
        # For videos, use the LargestVideo endpoint to get the actual video file
        # The ArchivedUri and ImageSizes endpoints only provide thumbnail/poster images for videos
        if image.is_video:
            if image.uris and "LargestVideo" in image.uris:
                largest_video_data = image.uris["LargestVideo"]
                if isinstance(largest_video_data, dict):
                    video_uri = largest_video_data.get("Uri")
                    if video_uri:
                        try:
                            # Convert relative URI to full URL
                            if video_uri.startswith("/"):
                                video_uri = f"https://api.smugmug.com{video_uri}"
                            
                            # Fetch the video details
                            video_response = self._request("GET", video_uri)
                            response_data = video_response.get("Response", video_response)
                            video_data = response_data.get("LargestVideo", {})
                            
                            # Get the actual video URL
                            download_url = video_data.get("Url")
                            if download_url:
                                video_size_mb = video_data.get("Size", 0) / (1024 * 1024)
                                logger.info(
                                    f"Downloading video {image.file_name} "
                                    f"({video_size_mb:.1f} MB) to {dest_path}"
                                )
                        except Exception as e:
                            logger.warning(f"Could not fetch video details: {e}")
            
            if not download_url:
                raise SmugMugAPIError(
                    f"No video download URL available for {image.file_name}. "
                    f"Video may still be processing or download may not be enabled."
                )
        else:
            # For images, use the ImageSizes endpoint
            if image.uris and "ImageSizes" in image.uris:
                image_sizes_data = image.uris["ImageSizes"]
                if isinstance(image_sizes_data, dict):
                    sizes_uri = image_sizes_data.get("Uri")
                    if sizes_uri:
                        # Fetch the available sizes for this image
                        try:
                            # Convert relative URI to full URL
                            if sizes_uri.startswith("/"):
                                sizes_uri = f"https://api.smugmug.com{sizes_uri}"
                            
                            sizes_response = self._request("GET", sizes_uri)
                            # Extract ImageSizes from Response wrapper
                            response_data = sizes_response.get("Response", sizes_response)
                            sizes_data = response_data.get("ImageSizes", {})
                            
                            # Try to find the requested size. SmugMug's keys are
                            # capitalized ("MediumImageUrl") while the configured size
                            # is commonly lowercase ("medium"), so match case-insensitively.
                            size_key = f"{size}ImageUrl".lower()
                            matched_key = next(
                                (k for k in sizes_data if k.lower() == size_key), None
                            )
                            if matched_key:
                                download_url = sizes_data[matched_key]
                            # Fall back to LargestImageUrl if requested size not available
                            elif "LargestImageUrl" in sizes_data:
                                logger.warning(f"Size '{size}' not available for {image.file_name}, using Largest")
                                download_url = sizes_data["LargestImageUrl"]
                        except Exception as e:
                            logger.warning(f"Could not fetch image sizes: {e}")
            
            # For Original size, try ArchivedUri as fallback
            if not download_url and size.lower() == "original" and image.archived_uri:
                download_url = image.archived_uri
            
            # Last resort: try the largest available size from image metadata
            if not download_url:
                # Check if we have direct size URLs in the image data
                for size_attr in ["original_url", "largest_url", "large_url", "medium_url"]:
                    url = getattr(image, size_attr, None)
                    if url:
                        download_url = url
                        break
            
            if download_url:
                logger.info(f"Downloading {image.file_name} ({size}) to {dest_path}")
        
        if not download_url:
            media_type = "video" if image.is_video else "image"
            raise SmugMugAPIError(
                f"No download URL available for {media_type} {image.file_name}. "
                f"Try a different size or check permissions."
            )
        
        try:
            # Download media - use authenticated request since we have OAuth
            response = requests.get(
                download_url,
                auth=self.auth,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Verify we got media data, not HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                raise SmugMugAPIError(
                    f"Received HTML instead of media for {image.file_name}. "
                    f"URL: {download_url}"
                )
            
            # Write to file
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = dest_path.stat().st_size
            logger.debug(f"Downloaded {dest_path} ({file_size} bytes)")
            return dest_path
            
        except Exception as e:
            # Clean up partial file
            if dest_path.exists():
                dest_path.unlink()
            raise SmugMugAPIError(
                f"Failed to download image {image.file_name}: {e}"
            ) from e
    
    def download_album_images(
        self,
        album_key: str,
        destination: str,
        size: str = "Medium",
        skip_if_exists: bool = True,
        skip_videos: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """Download all images from an album.
        
        Args:
            album_key: Album key
            destination: Destination directory path
            size: Image size (Medium, Large, XLarge, X2Large, Original, etc.)
            skip_if_exists: If True, skip download if file already exists
            skip_videos: If True, skip video files (default: True)
            progress_callback: Optional callback function(current, total, image)
            
        Returns:
            List of downloaded file paths (excludes skipped files)
            
        Raises:
            SmugMugAPIError: If download fails
        """
        logger.info(f"Downloading images from album {album_key}")
        
        # Get album and images
        album = self.get_album(album_key)
        all_items = self.get_album_images(album_key)
        
        if not all_items:
            logger.warning(f"No images found in album {album_key}")
            return []
        
        # Filter out videos if requested
        images = all_items
        if skip_videos:
            images = [img for img in all_items if not img.is_video]
            videos_skipped = len(all_items) - len(images)
            if videos_skipped > 0:
                logger.info(f"Skipping {videos_skipped} video file(s)")
        
        if not images:
            logger.warning(f"No images to download from album {album_key} after filtering")
            return []
        
        logger.info(f"Downloading {len(images)} images from '{album.name}'")
        
        downloaded_paths = []
        skipped_count = 0
        error_count = 0
        
        for i, image in enumerate(images, 1):
            try:
                if progress_callback:
                    progress_callback(i, len(images), image)
                
                path = self.download_image(
                    image,
                    destination,
                    size,
                    skip_if_exists
                )
                
                if path:
                    downloaded_paths.append(path)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to download {image.file_name}: {e}")
                error_count += 1
                # Continue with next image
        
        logger.info(
            f"Download complete: {len(downloaded_paths)} downloaded, "
            f"{skipped_count} skipped, {error_count} errors"
        )
        
        return downloaded_paths

