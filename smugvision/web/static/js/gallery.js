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

    /* Measured on this account: roughly 2-3s per image for a dry run. */
    const SECONDS_PER_IMAGE = 2.5;
    const BIG_ALBUM = 200;

    const dom = {};
    let selected = null;      // BrowseNode of the chosen album
    let currentNode = null;   // node_id being listed (null = root)
    let running = false;

    document.addEventListener('DOMContentLoaded', function () {
        dom.crumbs = document.getElementById('crumbs');
        dom.catalog = document.getElementById('catalog');
        dom.catalogStatus = document.getElementById('catalog-status');
        dom.catalogError = document.getElementById('catalog-error');
        dom.refreshBtn = document.getElementById('refresh-btn');
        dom.cacheNote = document.getElementById('cache-note');

        dom.selection = document.getElementById('selection');
        dom.selectionName = document.getElementById('selection-name');
        dom.selectionMeta = document.getElementById('selection-meta');
        dom.selectionNotice = document.getElementById('selection-notice');
        dom.runBtn = document.getElementById('run-btn');
        dom.forceReprocess = document.getElementById('force-reprocess');
        dom.replaceExisting = document.getElementById('replace-existing');
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

        loadLevel(null, {});
        loadServiceStatus();
    });

    /* -------------------------------------------------------------- *
     * Listing one level
     * -------------------------------------------------------------- */

    async function loadLevel(nodeId, options) {
        const refresh = options && options.refresh;
        currentNode = nodeId || null;

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

        const button = el('button', {
            type: 'button',
            className: classes.join(' '),
            'aria-pressed': 'false',
            'aria-label': label,
            dataset: {albumKey: album.album_key}
        }, [
            kindBadge('album', mixed),
            el('span', {className: 'row-name', text: album.name}),
            el('span', {
                className: 'row-meta',
                text: known ? S.plural(count, 'image') : 'image count unknown'
            })
        ]);

        button.addEventListener('click', function () {
            selectAlbum(album, button);
        });
        return button;
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

        const bits = ['album ' + album.album_key];
        if (typeof album.image_count === 'number') {
            bits.push(S.plural(album.image_count, 'image'));
            bits.push('about ' + estimate(album.image_count) + ' to proof');
        }
        if (album.privacy) bits.push(String(album.privacy).toLowerCase());
        dom.selectionMeta.textContent = bits.join('  ·  ');

        const empty = album.image_count === 0;
        const big = typeof album.image_count === 'number' &&
            album.image_count >= BIG_ALBUM;

        dom.selectionNotice.classList.add('hidden');
        dom.selectionNotice.className = 'notice hidden';
        if (empty) {
            dom.selectionNotice.className = 'notice';
            dom.selectionNotice.textContent =
                'This album has no images, so there is nothing to proof.';
            dom.selectionNotice.classList.remove('hidden');
        } else if (big) {
            dom.selectionNotice.className = 'notice caution';
            dom.selectionNotice.textContent =
                'Large album: ' + S.plural(album.image_count, 'image') +
                ' will take roughly ' + estimate(album.image_count) +
                '. The run streams progress and you can leave this tab open.';
            dom.selectionNotice.classList.remove('hidden');
        }

        dom.runBtn.disabled = Boolean(empty);
        dom.selection.classList.remove('hidden');
        dom.runBtn.focus();
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

    function selectedWriteMode() {
        const checked = document.querySelector('input[name="run-mode"]:checked');
        return checked ? checked.value : 'dry';
    }

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
            replace_existing: dom.replaceExisting.checked
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

        dom.progressTitle.textContent = 'Proofing ' + job.album_name;
        dom.progressBar.value = 0;
        dom.progressBar.max = 100;
        S.announce(
            dom.progressLine,
            '0 of ' + job.total_images + ' images. Nothing is being written.'
        );

        streamProgress(job, force, restore);
    }

    /**
     * Consume the SSE progress stream, then hand off to the proof sheet.
     * The write mode chosen here is passed on in the URL: it only decides
     * whether the proof sheet opens its write panel, never whether a write
     * happens - that still needs the latch and the confirm dialog.
     */
    function streamProgress(job, force, restore) {
        const source = new EventSource(
            '/api/preview/status?job_id=' + encodeURIComponent(job.job_id) +
            '&force_reprocess=' + (force ? 'true' : 'false')
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
                stats.errors + ' failed. Opening the proof sheet…'
            );

            let target = '/preview/' + encodeURIComponent(job.job_id);
            if (selectedWriteMode() === 'write') target += '?write=1';
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
