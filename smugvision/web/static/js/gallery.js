/**
 * smugVision - gallery browser for the landing page.
 *
 * Walks the SmugMug node tree one level at a time via GET /api/galleries,
 * lets the user select an album, and starts a preview (dry-run) job.
 *
 * Nothing here can write to SmugMug: the only write endpoint is
 * POST /api/commit, which lives on the proof-sheet page behind a
 * confirmation latch.
 */
(function () {
    'use strict';

    const S = window.smugvision;
    const el = S.el;

    /* Seconds per frame for a dry run, used only for the "about N minutes" estimate.
       Measured at ~13s with gemma4:26b on an M-series Mac (a 43-frame album took just
       over nine minutes). It scales with model size and hardware, so treat it as an
       order of magnitude: the previous 2.5 was off by 5x and made a nine-minute run
       look like a two-minute one. */
    const SECONDS_PER_IMAGE = 13;
    const BIG_ALBUM = 200;

    const dom = {};
    let selected = null;      // BrowseNode of the chosen album
    let currentNode = null;   // node_id being listed (null = root)
    let running = false;

    /* Proof state for the level on screen: {album_key: {state, tagged, total}}.
       Filled in after the list is drawn, because the scan costs a request per 100
       images per album. */
    let proofStates = {};

    /* Bumped on every loadLevel. A proof-state scan of a big folder takes seconds, so
       by the time it answers the user may have navigated on; the token lets a late
       reply recognise that its level is gone and drop itself instead of painting
       badges onto somebody else's albums. */
    let levelToken = 0;

    document.addEventListener('DOMContentLoaded', function () {
        dom.crumbs = document.getElementById('crumbs');
        dom.catalog = document.getElementById('catalog');
        dom.catalogStatus = document.getElementById('catalog-status');
        dom.catalogError = document.getElementById('catalog-error');
        dom.refreshBtn = document.getElementById('refresh-btn');
        dom.cacheNote = document.getElementById('cache-note');
        dom.returnNotice = document.getElementById('return-notice');

        dom.selection = document.getElementById('selection');
        dom.selectionName = document.getElementById('selection-name');
        dom.selectionMeta = document.getElementById('selection-meta');
        dom.selectionNotice = document.getElementById('selection-notice');
        dom.runBtn = document.getElementById('run-btn');
        dom.forceReprocess = document.getElementById('force-reprocess');
        dom.replaceExisting = document.getElementById('replace-existing');
        dom.generateTitles = document.getElementById('generate-titles');
        dom.clearSelectionBtn = document.getElementById('clear-selection-btn');

        dom.progress = document.getElementById('run-progress');
        dom.progressBar = document.getElementById('progress-bar');
        dom.progressTitle = document.getElementById('progress-title');
        dom.progressLine = document.getElementById('progress-line');

        dom.urlForm = document.getElementById('url-form');
        dom.urlInput = document.getElementById('album-url');
        dom.urlBtn = document.getElementById('url-btn');
        dom.urlError = document.getElementById('url-error');

        dom.refreshBtn.addEventListener('click', function () {
            loadLevel(currentNode, {refresh: true});
        });
        dom.clearSelectionBtn.addEventListener('click', clearSelection);
        // Ticking re-proof changes how many frames a run covers, so the estimate and
        // the notice have to follow it.
        dom.forceReprocess.addEventListener('change', refreshSelectionMeta);
        dom.runBtn.addEventListener('click', function () {
            if (!selected) return;
            startRun({album_key: selected.album_key}, selected.name);
        });
        dom.urlForm.addEventListener('submit', function (event) {
            event.preventDefault();
            const url = dom.urlInput.value.trim();
            setNotice(dom.urlError, '');
            if (!url) {
                setNotice(dom.urlError, 'Paste a SmugMug album URL first.', 'error');
                dom.urlInput.focus();
                return;
            }
            startRun({url: url}, 'the pasted album');
        });

        const opening = new URLSearchParams(window.location.search);
        showReturnNotice(opening);
        loadLevel(opening.get('node') || null, {});
        loadServiceStatus();
    });

    /**
     * Report a write that happened on the proof sheet we just came back from.
     *
     * The proof sheet navigates here after a clean write, so the confirmation has to
     * survive the trip. It is carried in the query string and then stripped from the
     * address bar, so reloading this page does not re-announce a write that is over.
     */
    function showReturnNotice(params) {
        const wrote = params.get('wrote');
        if (wrote === null) return;

        const count = parseInt(wrote, 10);
        const album = params.get('album') || 'that album';
        setNotice(
            dom.returnNotice,
            (Number.isFinite(count) ? S.plural(count, 'image') : 'Changes') +
            ' written to ' + album + '. Its badge below is up to date.',
            'ok'
        );

        params.delete('wrote');
        params.delete('album');
        const rest = params.toString();
        window.history.replaceState({}, '', rest ? '/?' + rest : '/');
    }

    /* -------------------------------------------------------------- *
     * Listing one level
     * -------------------------------------------------------------- */

    async function loadLevel(nodeId, options) {
        const refresh = options && options.refresh;
        currentNode = nodeId || null;
        const token = ++levelToken;
        proofStates = {};

        showSkeleton();
        setNotice(dom.catalogError, '');
        dom.refreshBtn.disabled = true;
        S.announce(dom.catalogStatus, 'Loading galleries from SmugMug…');

        let query = '/api/galleries';
        const params = [];
        if (nodeId) params.push('node=' + encodeURIComponent(nodeId));
        if (refresh) params.push('refresh=1');
        if (params.length) query += '?' + params.join('&');

        try {
            const data = await S.apiGet(query);
            renderCrumbs(data);
            renderCatalog(data);
            loadProofState(data.albums, refresh, token);

            const counts = [];
            if (data.folders.length) {
                counts.push(S.plural(data.folders.length, 'folder'));
            }
            if (data.albums.length) {
                counts.push(S.plural(data.albums.length, 'album'));
            }
            S.announce(
                dom.catalogStatus,
                data.node.name + ': ' +
                    (counts.length ? counts.join(', ') : 'nothing here')
            );

            dom.cacheNote.textContent = data.cached
                ? 'from cache (up to ' + data.cache_ttl_seconds + 's old)'
                : '';

            if (data.partial) {
                showCatalogError(
                    'Some entries in this folder could not be read and are not ' +
                    'listed. Try Refresh, or check the server log.',
                    'caution'
                );
            }
        } catch (error) {
            S.setChildren(dom.catalog, []);
            S.announce(dom.catalogStatus, 'Could not load galleries.');
            showCatalogError(error.message, 'error');
        } finally {
            dom.refreshBtn.disabled = false;
        }
    }

    function showSkeleton() {
        const rows = [1, 2, 3, 4].map(function () {
            return el('div', {className: 'skeleton-row'});
        });
        S.setChildren(dom.catalog, [
            el('li', {className: 'catalog-loading'}, rows)
        ]);
    }

    /**
     * Show or hide a .notice element. An empty message hides it, so the same
     * call clears a stale error.
     */
    function setNotice(node, message, kind) {
        if (!node) return;
        node.className = 'notice' + (kind ? ' ' + kind : '');
        node.textContent = message || '';
        node.classList.toggle('hidden', !message);
    }

    function showCatalogError(message, kind) {
        setNotice(dom.catalogError, message, kind === 'caution' ? 'caution' : 'error');
    }

    function renderCrumbs(data) {
        const crumbs = data.breadcrumb || [];
        const items = crumbs.map(function (crumb, index) {
            const isLast = index === crumbs.length - 1;
            if (isLast) {
                return el('li', {}, [
                    el('span', {'aria-current': 'page', text: crumb.name})
                ]);
            }
            const button = el('button', {
                type: 'button',
                text: crumb.name,
                title: 'Go to ' + crumb.name
            });
            button.addEventListener('click', function () {
                loadLevel(crumb.is_root ? null : crumb.node_id, {});
            });
            return el('li', {}, [button]);
        });
        S.setChildren(dom.crumbs, items);
    }

    function renderCatalog(data) {
        const rows = [];

        /* The folder/album badge only earns its ink when a level actually
           holds both kinds. On a level of 47 albums it is 47 repetitions of
           the same word, so it stays in the DOM for screen readers and is
           hidden visually. */
        const mixed = data.folders.length > 0 && data.albums.length > 0;

        data.folders.forEach(function (folder) {
            rows.push(el('li', {}, [folderRow(folder, mixed)]));
        });
        data.albums.forEach(function (album) {
            rows.push(el('li', {}, [albumRow(album, mixed)]));
        });

        if (!rows.length) {
            rows.push(el('li', {className: 'catalog-loading'}, [
                el('p', {
                    className: 'muted',
                    text: 'This folder is empty.'
                })
            ]));
        }

        S.setChildren(dom.catalog, rows);
    }

    /**
     * The kind badge, shown only on a level holding both kinds. It is always
     * aria-hidden: each row carries an explicit aria-label instead, because
     * the adjacent spans have no whitespace between them and a screen reader
     * would otherwise announce "folderFamily Photos".
     */
    function kindBadge(word, mixed) {
        if (!mixed) return null;
        return el('span', {className: 'kind', 'aria-hidden': 'true', text: word});
    }

    function folderRow(folder, mixed) {
        const button = el('button', {
            type: 'button',
            className: 'catalog-row is-folder',
            'aria-label': 'Folder: ' + folder.name
        }, [
            kindBadge('folder', mixed),
            el('span', {className: 'row-name', text: folder.name}),
            el('span', {className: 'row-go', 'aria-hidden': 'true', text: '›'})
        ]);
        button.addEventListener('click', function () {
            loadLevel(folder.node_id, {});
        });
        return button;
    }

    function albumRow(album, mixed) {
        const count = album.image_count;
        const known = typeof count === 'number';
        const classes = ['catalog-row', 'is-album'];
        if (known && count === 0) classes.push('is-empty');
        if (known && count >= BIG_ALBUM) classes.push('is-big');

        const label = 'Album: ' + album.name +
            (known ? ', ' + S.plural(count, 'image') : ', image count unknown');

        const children = [
            kindBadge('album', mixed),
            el('span', {className: 'row-name', text: album.name}),
            el('span', {
                className: 'row-meta',
                text: known ? S.plural(count, 'image') : 'image count unknown'
            })
        ];

        /* An empty album can never have been proofed, so it gets no badge and is left
           out of the scan entirely. */
        if (!known || count > 0) {
            children.push(el('span', {
                className: 'row-proof is-checking',
                text: 'checking…'
            }));
        }

        const button = el('button', {
            type: 'button',
            className: classes.join(' '),
            'aria-pressed': 'false',
            'aria-label': label,
            dataset: {albumKey: album.album_key, baseLabel: label}
        }, children);

        button.addEventListener('click', function () {
            selectAlbum(album, button);
        });
        return button;
    }

    /* -------------------------------------------------------------- *
     * "Have I already proofed this one?" badges
     * -------------------------------------------------------------- */

    /**
     * Fill in the proof badges for a level that has already been drawn.
     *
     * Kept out of the listing request on purpose: reading keywords costs one request
     * per 100 images per album, so a folder of 19 albums takes seconds. The list stays
     * usable throughout, and a failure here degrades to "unknown" rather than
     * breaking album selection.
     */
    async function loadProofState(albums, refresh, token) {
        const scannable = (albums || []).filter(function (album) {
            return album.album_key && album.image_count !== 0;
        });

        if (!scannable.length) return;

        let query = '/api/albums/proof-state?keys=' +
            encodeURIComponent(scannable.map(function (a) { return a.album_key; }).join(','));
        if (refresh) query += '&refresh=1';

        let data;
        try {
            data = await S.apiGet(query);
        } catch (error) {
            if (token !== levelToken) return;
            markProofUnknown('Could not read: ' + error.message);
            return;
        }

        if (token !== levelToken) return;   // user has navigated on; this reply is stale

        proofStates = data.states || {};
        Object.keys(proofStates).forEach(function (key) {
            paintProofBadge(key, proofStates[key], null);
        });
        Object.keys(data.errors || {}).forEach(function (key) {
            paintProofBadge(key, null, data.errors[key]);
        });
        (data.unscanned || []).forEach(function (key) {
            paintProofBadge(key, null, 'Not scanned: too many albums on this level.');
        });

        if (selected) refreshSelectionMeta();
    }

    function albumRowFor(albumKey) {
        return dom.catalog.querySelector(
            '.catalog-row.is-album[data-album-key="' + albumKey + '"]'
        );
    }

    /** Describe a proof state as {text, className, spoken}. */
    function proofLabel(state) {
        if (!state) return null;
        /* Albums listing zero items are never scanned, so "empty" coming back from the
           scan means the opposite: the album holds things, but none of them are photos
           (a folder of .MOV files does this). Saying so beats leaving a row that reads
           "10 images" with no badge and no explanation. */
        if (state.state === 'empty') {
            return {
                text: 'no photos',
                className: 'is-unknown',
                spoken: 'no photos to proof',
                title: 'Nothing here to proof - videos are not processed.'
            };
        }
        if (state.state === 'all') {
            return {text: '✓ proofed', className: 'is-proofed', spoken: 'already proofed'};
        }
        if (state.state === 'partial') {
            return {
                text: state.tagged + ' of ' + state.total + ' proofed',
                className: 'is-partial',
                spoken: state.tagged + ' of ' + state.total + ' images already proofed'
            };
        }
        return {text: 'not proofed', className: 'is-unproofed', spoken: 'not proofed yet'};
    }

    function paintProofBadge(albumKey, state, errorMessage) {
        const row = albumRowFor(albumKey);
        if (!row) return;

        const badge = row.querySelector('.row-proof');
        if (!badge) return;

        if (errorMessage) {
            badge.className = 'row-proof is-unknown';
            badge.textContent = 'unknown';
            badge.title = errorMessage;
            return;
        }

        const label = proofLabel(state);
        if (!label) {
            badge.remove();
            return;
        }

        badge.className = 'row-proof ' + label.className;
        badge.textContent = label.text;
        if (label.title) badge.title = label.title;
        else badge.removeAttribute('title');
        row.setAttribute('aria-label', (row.dataset.baseLabel || '') + ', ' + label.spoken);
    }

    function markProofUnknown(message) {
        dom.catalog.querySelectorAll('.row-proof').forEach(function (badge) {
            badge.className = 'row-proof is-unknown';
            badge.textContent = 'unknown';
            badge.title = message;
        });
    }

    /* -------------------------------------------------------------- *
     * Selecting an album
     * -------------------------------------------------------------- */

    function selectAlbum(album, button) {
        selected = album;

        dom.catalog.querySelectorAll('.catalog-row.is-album').forEach(function (row) {
            row.setAttribute('aria-pressed', String(row === button));
        });

        dom.selectionName.textContent = album.name;
        refreshSelectionMeta();

        dom.selection.classList.remove('hidden');
        dom.runBtn.focus();
    }

    /**
     * Rewrite the selection panel's meta line and notice.
     *
     * Re-run whenever anything it depends on moves: a new selection, the proof-state
     * scan landing, or the re-proof checkbox being toggled. It reads the album's proof
     * state so the estimate counts the frames a run would actually process, not the
     * whole album - "about 2 minutes" is a lie when 45 of the 47 frames are already
     * done and will be left out.
     */
    function refreshSelectionMeta() {
        const album = selected;
        if (!album) return;

        const state = proofStates[album.album_key];
        const force = dom.forceReprocess.checked;
        const knownCount = typeof album.image_count === 'number';

        // How many frames a run would actually work on.
        let toProof = knownCount ? album.image_count : null;
        if (state && !force) toProof = state.untagged;
        else if (state) toProof = state.total;

        const bits = ['album ' + album.album_key];
        if (knownCount) bits.push(S.plural(album.image_count, 'image'));
        if (state && state.tagged) {
            bits.push(state.tagged + ' already proofed');
        }
        if (typeof toProof === 'number' && toProof > 0) {
            bits.push('about ' + estimate(toProof) + ' to proof ' +
                S.plural(toProof, 'frame'));
        }
        if (album.privacy) bits.push(String(album.privacy).toLowerCase());
        dom.selectionMeta.textContent = bits.join('  ·  ');

        const empty = album.image_count === 0;
        const nothingLeft = Boolean(state && !force && state.untagged === 0 &&
            state.total > 0);
        const big = typeof toProof === 'number' && toProof >= BIG_ALBUM;

        // Clear the text as well as hiding it: a stale sentence left in a hidden node
        // reappears the moment anything unhides it.
        dom.selectionNotice.className = 'notice hidden';
        dom.selectionNotice.textContent = '';
        if (empty) {
            setSelectionNotice(
                '', 'This album has no images, so there is nothing to proof.'
            );
        } else if (nothingLeft) {
            setSelectionNotice(
                'caution',
                'Every image here is already proofed. A run would find nothing to do ' +
                'unless you tick "Re-proof images that smugVision already tagged".'
            );
        } else if (state && state.tagged && !force) {
            setSelectionNotice(
                '',
                state.tagged + ' of ' + state.total + ' images are already proofed and ' +
                'will be left out. This run covers the remaining ' +
                S.plural(state.untagged, 'image') + '.'
            );
        } else if (big) {
            setSelectionNotice(
                'caution',
                'Large run: ' + S.plural(toProof, 'image') + ' will take roughly ' +
                estimate(toProof) +
                '. The run streams progress and you can leave this tab open.'
            );
        }

        dom.runBtn.disabled = Boolean(empty);
    }

    function setSelectionNotice(kind, message) {
        dom.selectionNotice.className = 'notice' + (kind ? ' ' + kind : '');
        dom.selectionNotice.textContent = message;
        dom.selectionNotice.classList.remove('hidden');
    }

    function clearSelection() {
        selected = null;
        dom.selection.classList.add('hidden');
        dom.catalog.querySelectorAll('.catalog-row.is-album').forEach(function (row) {
            row.setAttribute('aria-pressed', 'false');
        });
    }

    /** Human estimate of dry-run duration for a frame count. */
    function estimate(imageCount) {
        const seconds = Math.round(imageCount * SECONDS_PER_IMAGE);
        if (seconds < 90) return seconds + ' seconds';
        const minutes = Math.round(seconds / 60);
        if (minutes < 60) return minutes + ' minutes';
        const hours = Math.floor(minutes / 60);
        const rest = minutes % 60;
        return hours + 'h ' + rest + 'm';
    }

    /* -------------------------------------------------------------- *
     * Starting a proof run  (always dry-run; never writes)
     * -------------------------------------------------------------- */

    async function startRun(body, label) {
        if (running) return;
        running = true;

        const trigger = body.url ? dom.urlBtn : dom.runBtn;
        const restore = S.busy(trigger, 'Starting');
        dom.runBtn.disabled = true;
        dom.urlBtn.disabled = true;
        setNotice(dom.urlError, '');
        setNotice(dom.catalogError, '');

        dom.progress.classList.remove('hidden');
        dom.progressBar.value = 0;
        dom.progressBar.removeAttribute('value');
        dom.progressTitle.textContent = 'Starting proof run…';
        S.announce(dom.progressLine, 'Asking SmugMug for ' + label + '.');

        const force = dom.forceReprocess.checked;
        const payload = Object.assign({
            force_reprocess: force,
            origin_node: currentNode || '',
            replace_existing: dom.replaceExisting.checked,
            generate_titles: dom.generateTitles.checked
        }, body);

        let job;
        try {
            job = await S.apiPost('/api/preview', payload);
        } catch (error) {
            failRun(error.message, body.url ? dom.urlError : null);
            restore();
            running = false;
            dom.urlBtn.disabled = false;
            dom.runBtn.disabled = !selected;
            return;
        }

        // Already-proofed frames are left out of the run unless "re-proof" is ticked,
        // so an album that is fully done has nothing to stream. Say so plainly rather
        // than opening an empty proof sheet.
        if (!job.total_images) {
            dom.progressBar.value = 100;
            dom.progressTitle.textContent = 'Nothing left to proof';
            S.announce(
                dom.progressLine,
                job.excluded_count
                    ? S.plural(job.excluded_count, 'image') + ' in ' + job.album_name +
                      ' already carry the smugVision tag. Tick "Re-proof images that ' +
                      'smugVision already tagged" to go through them again.'
                    : 'This album has no images to proof.'
            );
            restore();
            running = false;
            dom.urlBtn.disabled = false;
            dom.runBtn.disabled = !selected;
            return;
        }

        dom.progressTitle.textContent = 'Proofing ' + job.album_name;
        dom.progressBar.value = 0;
        dom.progressBar.max = 100;
        S.announce(
            dom.progressLine,
            '0 of ' + job.total_images + ' images' +
            (job.excluded_count
                ? ' (' + job.excluded_count + ' already tagged, left out)'
                : '') +
            '. Nothing is being written.'
        );

        streamProgress(job, restore);
    }

    /**
     * Consume the SSE progress stream, then hand off to the proof sheet.
     * The write mode chosen here is passed on in the URL: it only decides
     * whether the proof sheet opens its write panel, never whether a write
     * happens - that still needs the latch and the confirm dialog.
     */
    function streamProgress(job, restore) {
        // No force_reprocess here on purpose: which images are in the run was settled
        // when the job was created, and the server reads it back off the job.
        const source = new EventSource(
            '/api/preview/status?job_id=' + encodeURIComponent(job.job_id)
        );
        let finished = false;

        source.addEventListener('progress', function (event) {
            const data = JSON.parse(event.data);
            dom.progressBar.value = data.percent;
            S.announce(
                dom.progressLine,
                data.current + ' of ' + data.total + ': ' + data.filename
            );
        });

        source.addEventListener('complete', function (event) {
            const stats = JSON.parse(event.data);
            finished = true;
            source.close();
            dom.progressBar.value = 100;
            dom.progressTitle.textContent = 'Proof run complete';
            S.announce(
                dom.progressLine,
                stats.processed + ' proposed, ' + stats.skipped + ' skipped, ' +
                stats.errors + ' failed' +
                (stats.excluded ? ', ' + stats.excluded + ' already tagged' : '') +
                '. Opening the proof sheet…'
            );

            const target = '/preview/' + encodeURIComponent(job.job_id);
            window.setTimeout(function () {
                window.location.href = target;
            }, 250);
        });

        source.addEventListener('error', function (event) {
            if (event.data) {
                let message = 'Processing failed.';
                try {
                    message = JSON.parse(event.data).message || message;
                } catch (parseError) {
                    /* keep the default */
                }
                finished = true;
                source.close();
                failRun(message, null);
                restore();
                running = false;
                dom.urlBtn.disabled = false;
                dom.runBtn.disabled = !selected;
            }
        });

        source.onerror = function () {
            if (finished || source.readyState === EventSource.CLOSED) return;
            source.close();
            failRun(
                'Lost the connection to the server while processing. The job may ' +
                'still be running - reload and check.',
                null
            );
            restore();
            running = false;
            dom.urlBtn.disabled = false;
            dom.runBtn.disabled = !selected;
        };
    }

    function failRun(message, extraTarget) {
        dom.progressTitle.textContent = 'Proof run failed';
        S.announce(dom.progressLine, message);
        showCatalogError(message, 'error');
        if (extraTarget) setNotice(extraTarget, message, 'error');
    }

    /* -------------------------------------------------------------- *
     * Service status
     * -------------------------------------------------------------- */

    async function loadServiceStatus() {
        const target = document.getElementById('service-status');

        let status;
        try {
            status = await S.apiGet('/api/status');
        } catch (error) {
            S.setChildren(target, [
                el('p', {
                    className: 'status-error',
                    text: 'Could not load service status: ' + error.message
                })
            ]);
            return;
        }

        // Reflect the configured defaults, so an untouched checkbox agrees with the
        // config instead of silently overriding it on the next run.
        if (dom.generateTitles && typeof status.generate_titles_default === 'boolean') {
            dom.generateTitles.checked = status.generate_titles_default;
        }
        if (dom.replaceExisting && status.preserve_existing_default === false) {
            dom.replaceExisting.checked = true;
        }

        const rows = [
            row('SmugMug', status.smugmug, status.smugmug === 'connected'),
            row(
                'Vision model',
                status.vision_model,
                String(status.vision_model || '').indexOf('connected') !== -1
            ),
            row(
                'Face recognition',
                status.face_recognition,
                String(status.face_recognition || '').indexOf('enabled') !== -1
            )
        ];

        const list = el('dl', {className: 'status-list'});
        rows.forEach(function (pair) {
            list.appendChild(pair[0]);
            list.appendChild(pair[1]);
        });
        S.setChildren(target, [list]);
    }

    function row(label, value, ok) {
        return [
            el('dt', {text: label}),
            el('dd', {
                className: ok ? 'status-ok' : 'status-warn',
                text: (ok ? '✓ ' : '⚠ ') + (value || 'unknown')
            })
        ];
    }
})();
