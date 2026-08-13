/**
 * smugVision - proof sheet for one preview job.
 *
 * Renders a card per image (frame, proposed caption, tags, what the pipeline
 * saw), lets the user write a margin note (a hint) and re-read that single
 * frame, and gates the one action that writes to SmugMug behind a latch and
 * a confirmation dialog.
 *
 * Write safety, in this file:
 *   - the only call to /api/commit is in doCommit()
 *   - it always sends confirm: true AND an explicit image_keys list
 *   - it is reachable only from the dialog's confirm button, which is only
 *     reachable once the latch checkbox is ticked
 *   - the image_keys sent are the ones the dialog NAMED, frozen when the
 *     dialog opened, so a re-read that lands while it is open cannot add an
 *     image to the write set the user answered for
 *   - the write button stays disabled while any frame is being re-read, so
 *     the write path and the re-read path never overlap
 */
(function () {
    'use strict';

    const S = window.smugvision;
    const el = S.el;

    const dom = {};
    let jobId = null;
    let albumKey = null;
    let albumName = '';
    let images = [];          // latest image result objects, in order
    let hints = null;         // {enabled, global, album, images:{key:text}}
    let committed = false;
    let knownFaces = [];      // [{name, display_name, reference_count}] for the picker
    let albumPicker = null;   // album-scope people picker, built after results load
    let albumPetPicker = null;    // album-scope pet picker
    let knownPets = [];           // [{name, description}] from pets.yaml
    let originNode = '';          // folder the picker was on when this run started
    let pendingWriteKeys = null;  // image keys the open confirm dialog named
    let regenerating = 0;         // in-flight single-frame re-reads
    let sweeping = false;         // an album-wide re-read is running
    let stopSweep = false;        // asked to stop after the frame in flight

    /* Most frames are the same few people, so the picker pins those and files the rest
       behind a drawer. The pinned set is the user's, kept in localStorage: it is a
       preference about this browser's UI, not a fact about the photos, so it does not
       belong in ~/.smugvision alongside hints and relationships. */
    const FAVOURITES_KEY = 'smugvision.favouritePeople';
    const FAVOURITES_SEED = 4;
    let favourites = null;        // array of names; lazily loaded, then authoritative
    const livePickers = [];       // built pickers, re-rendered when the pinned set moves

    document.addEventListener('DOMContentLoaded', function () {
        const root = document.getElementById('proof-root');
        jobId = root.dataset.jobId;

        dom.backLink = document.getElementById('back-to-albums');
        dom.albumTitle = document.getElementById('album-title');
        dom.albumKeyEl = document.getElementById('album-key');
        dom.tally = document.getElementById('tally');
        dom.grid = document.getElementById('image-grid');
        dom.gridStatus = document.getElementById('grid-status');
        dom.loadError = document.getElementById('load-error');

        dom.albumHintBox = document.getElementById('album-hint-box');
        dom.albumHint = document.getElementById('album-hint');
        dom.albumHintBtn = document.getElementById('album-hint-btn');
        dom.albumRereadBtn = document.getElementById('album-reread-btn');
        dom.albumRereadStop = document.getElementById('album-reread-stop');
        dom.albumHintStatus = document.getElementById('album-hint-status');
        dom.albumLocation = document.getElementById('album-location');
        dom.albumPeople = document.getElementById('album-people');
        dom.globalHintSummary = document.getElementById('global-hint-summary');

        dom.writePanel = document.getElementById('write-panel');
        dom.writeLatch = document.getElementById('write-latch');
        dom.writeBtn = document.getElementById('write-btn');
        dom.writeSummary = document.getElementById('write-summary');
        dom.writeList = document.getElementById('write-list');
        dom.writeResult = document.getElementById('write-result');
        dom.writeBusyNote = document.getElementById('write-busy-note');

        dom.confirmModal = document.getElementById('confirm-write-modal');
        dom.confirmText = document.getElementById('confirm-text');
        dom.confirmList = document.getElementById('confirm-list');
        dom.confirmBtn = document.getElementById('confirm-write-btn');
        dom.cancelBtn = document.getElementById('cancel-write-btn');

        dom.writeLatch.addEventListener('change', syncWriteButton);
        dom.writeBtn.addEventListener('click', openConfirm);
        dom.cancelBtn.addEventListener('click', function () {
            dom.confirmModal.close();
        });
        /* Covers Cancel, Escape and the close after a write: the frozen write
           set never outlives the dialog that named it. */
        dom.confirmModal.addEventListener('close', function () {
            pendingWriteKeys = null;
        });
        dom.confirmBtn.addEventListener('click', doCommit);
        dom.albumHintBtn.addEventListener('click', saveAlbumHint);
        dom.albumRereadBtn.addEventListener('click', rereadAlbum);
        dom.albumRereadStop.addEventListener('click', function () {
            stopSweep = true;
            dom.albumRereadStop.disabled = true;
            S.announce(dom.albumHintStatus, 'Stopping after this frame…');
        });

        loadResults();
    });

    /* -------------------------------------------------------------- *
     * Load
     * -------------------------------------------------------------- */

    async function loadResults() {
        let data;
        try {
            data = await S.apiGet(
                '/api/preview/results?job_id=' + encodeURIComponent(jobId)
            );
        } catch (error) {
            S.setChildren(dom.grid, []);
            S.announce(dom.gridStatus, 'Could not load this proof run.');
            dom.loadError.textContent = error.message;
            dom.loadError.classList.remove('hidden');
            return;
        }

        // The people picker offers exactly the reference faces that exist, so a name
        // typed by hand cannot drift from a reference_faces/ directory. A failure here
        // only costs the picker, never the proof sheet.
        try {
            const faceData = await S.apiGet('/api/faces');
            knownFaces = (faceData && faceData.faces) || [];
        } catch (error) {
            knownFaces = [];
        }

        // Pets are optional and independent: failing to read them must not cost the
        // people picker or the proof sheet.
        try {
            const petData = await S.apiGet('/api/pets');
            knownPets = (petData && petData.pets) || [];
        } catch (error) {
            knownPets = [];
        }

        albumKey = data.album_key;
        albumName = data.album_name;
        originNode = data.origin_node || '';
        if (dom.backLink) dom.backLink.href = albumsUrl();
        images = data.images || [];
        hints = data.hints || {enabled: false, global: '', album: '', images: {}};

        document.title = 'smugVision - ' + albumName;
        dom.albumTitle.textContent = albumName;
        dom.albumKeyEl.textContent = 'album ' + albumKey + '  ·  job ' + jobId
            + (data.replace_existing ? '  ·  replacing existing metadata' : '');

        renderTally(data.stats);
        renderRereadButton();
        renderHintEditors();
        renderGrid();
        renderWritePanel();

        S.announce(
            dom.gridStatus,
            S.plural(images.length, 'frame') + ' in this proof run.'
        );
    }

    /* Say how much work the button is about to do: "every frame" reads very
       differently on a 5-frame run and on a 330-frame one. */
    function renderRereadButton() {
        if (!dom.albumRereadBtn) return;
        dom.albumRereadBtn.textContent = images.length
            ? 'Save & re-read all ' + S.plural(images.length, 'frame')
            : 'Save & re-read every frame';
        dom.albumRereadBtn.disabled = !images.length;
    }

    function renderTally(stats) {
        const entries = [
            ['processed', stats.processed, 'proposed'],
            ['skipped', stats.skipped, 'skipped'],
            ['errors', stats.errors, 'failed'],
            ['total', stats.total, 'total']
        ];
        // Only worth a slot when some frames were held back; a zero here would read as
        // "nothing was already tagged", which is noise on a first pass.
        if (stats.excluded) {
            entries.push(['excluded', stats.excluded, 'already tagged']);
        }
        S.setChildren(dom.tally, entries.map(function (entry) {
            const classes = [entry[0]];
            if (!entry[1]) classes.push('is-zero');
            return el('li', {className: classes.join(' ')}, [
                el('span', {className: 'tally-n', text: String(entry[1])}),
                el('span', {text: entry[2]})
            ]);
        }));
    }

    /* -------------------------------------------------------------- *
     * Hint editors: album scope here, global scope on /hints
     * -------------------------------------------------------------- */

    function renderHintEditors() {
        if (!hints.enabled) {
            dom.albumHintBox.classList.add('hidden');
            return;
        }

        dom.albumHint.value = hints.album || '';
        dom.albumLocation.value = hints.album_location || '';

        albumPicker = buildPeoplePicker(
            'album-who',
            hints.album_people || [],
            'Who is in this album?',
            'album-people-hint'
        );
        albumPetPicker = buildPetPicker(
            'album-who',
            hints.album_pets || [],
            'Any pets in this album?'
        );
        S.setChildren(dom.albumPeople, [albumPicker.node, albumPetPicker.node]);
        syncAlbumPeopleDrawer();

        if (hints.global) {
            S.setChildren(dom.globalHintSummary, [
                el('span', {className: 'scope-badge global', text: 'global'}),
                document.createTextNode(' '),
                el('span', {text: hints.global})
            ]);
        } else {
            S.setChildren(dom.globalHintSummary, [
                el('span', {
                    className: 'muted',
                    text: 'No global hint set. A global hint applies to every album.'
                })
            ]);
        }
    }

    /**
     * Keep the album-scope drawer honest about itself.
     *
     * Naming people for a whole album is the rare case, so the drawer starts closed -
     * but an override that IS set has to be visible, or the user cannot see why every
     * frame is claiming the same person. When something is set the drawer says so and
     * opens itself; when nothing is, it stays a single quiet line.
     */
    function syncAlbumPeopleDrawer() {
        const drawer = document.getElementById('album-people-drawer');
        const summary = document.getElementById('album-people-summary');
        if (!drawer || !summary) return;

        const set = (hints.album_people || []).length;
        summary.textContent = set
            ? 'Who is in this whole album? (' + set + ' set, applies to every frame)'
            : 'Who is in this whole album?';
        drawer.classList.toggle('is-set', Boolean(set));
        if (set) drawer.open = true;
    }

    /**
     * Write the album-scope note, location and people, and fold the stored result back
     * into the page. Shared by "Save album note" and the re-read-everything sweep, so
     * both persist exactly the same thing.
     *
     * @returns {Promise<Object>} the stored hint as the server echoed it
     */
    async function persistAlbumHint() {
        const result = await S.apiPut('/api/hints', {
            scope: 'album',
            key: albumKey,
            text: dom.albumHint.value,
            location: dom.albumLocation.value,
            people: albumPicker ? albumPicker.selected() : [],
            pets: albumPetPicker ? albumPetPicker.selected() : []
        });
        hints.album = result.text;
        dom.albumHint.value = result.text;
        hints.album_location = result.location || '';
        dom.albumLocation.value = hints.album_location;
        hints.album_people = result.people || [];
        hints.album_pets = result.pets || [];
        syncAlbumPeopleDrawer();
        return result;
    }

    async function saveAlbumHint() {
        const restore = S.busy(dom.albumHintBtn, 'Saving album note');
        S.announce(dom.albumHintStatus, '');
        try {
            const result = await persistAlbumHint();
            S.announce(
                dom.albumHintStatus,
                result.cleared
                    ? 'Album note cleared. Re-read a frame to see the effect.'
                    : 'Album note saved. Nothing on screen has changed yet - a note only ' +
                      'reaches the model on the next read.',
                'ok'
            );
        } catch (error) {
            S.announce(dom.albumHintStatus, error.message, 'error');
        } finally {
            restore();
        }
    }

    /**
     * Save the album note and then re-read every frame in this run with it applied.
     *
     * Sequential on purpose: the model is a single local Ollama process, so firing
     * frames at it in parallel would not finish sooner and would make the progress
     * line meaningless. The whole sweep counts as one in-flight re-read, which holds
     * the write path shut from the first frame to the last.
     */
    async function rereadAlbum() {
        if (sweeping) return;
        if (!images.length) {
            S.announce(dom.albumHintStatus, 'This run has no frames to re-read.');
            return;
        }

        sweeping = true;
        stopSweep = false;
        regenerating += 1;
        syncWriteButton();
        setFrameNotesDisabled(true);
        dom.albumRereadStop.disabled = false;
        dom.albumRereadStop.classList.remove('hidden');
        const restore = S.busy(dom.albumRereadBtn, 'Re-reading every frame');

        // Snapshot the keys: each frame's card is rebuilt as it lands, and a sweep of a
        // long album should cover the run it started on.
        const keys = images.map(function (image) { return image.image_key; });
        let done = 0;
        let failed = 0;
        let stopped = false;

        try {
            S.announce(dom.albumHintStatus, 'Saving album note…');
            try {
                await persistAlbumHint();
            } catch (error) {
                S.announce(
                    dom.albumHintStatus,
                    'Could not save the note, so nothing was re-read: ' + error.message,
                    'error'
                );
                return;
            }

            for (let i = 0; i < keys.length; i++) {
                if (stopSweep) {
                    stopped = true;
                    break;
                }
                S.announce(
                    dom.albumHintStatus,
                    'Re-reading frame ' + (i + 1) + ' of ' + keys.length + '…'
                );
                if (await rereadFrame(keys[i], null)) done += 1;
                else failed += 1;

                // Each landed frame rebuilds its card, which brings its own buttons back.
                if (sweeping) setFrameNotesDisabled(true);
            }

            const parts = [S.plural(done, 'frame') + ' re-read with the album note'];
            if (failed) parts.push(S.plural(failed, 'frame') + ' failed');
            if (stopped) {
                parts.push('stopped early, ' +
                    (keys.length - done - failed) + ' left untouched');
            }
            S.announce(
                dom.albumHintStatus,
                parts.join('  ·  '),
                failed ? 'error' : 'ok'
            );
        } finally {
            restore();
            setFrameNotesDisabled(false);
            dom.albumRereadStop.classList.add('hidden');
            regenerating -= 1;
            syncWriteButton();
            sweeping = false;
        }
    }

    /**
     * Disable the per-frame save buttons for the duration of a sweep.
     *
     * Without this, a frame could be re-read by the sweep and by its own button at the
     * same time, and the two replies would race to replace the same card.
     */
    function setFrameNotesDisabled(disabled) {
        // Only the save buttons: the picker's star buttons are inside .margin-note too,
        // and pinning somebody mid-sweep is harmless.
        dom.grid.querySelectorAll('.margin-note button.note-save').forEach(function (button) {
            button.disabled = disabled;
        });
    }

    /* -------------------------------------------------------------- *
     * The grid of frames
     * -------------------------------------------------------------- */

    function renderGrid() {
        if (!images.length) {
            S.setChildren(dom.grid, [
                el('li', {}, [
                    el('div', {
                        className: 'info-box',
                        text: 'This run produced no frames.'
                    })
                ])
            ]);
            return;
        }
        S.setChildren(dom.grid, images.map(buildCard));
    }

    function buildCard(image) {
        const card = el('li', {
            className: 'image-card ' + image.status,
            dataset: {imageKey: image.image_key}
        }, [
            buildFrame(image),
            buildBody(image)
        ]);
        return card;
    }

    function buildFrame(image) {
        const img = el('img', {
            src: image.thumbnail_url,
            alt: 'Photograph ' + image.filename,
            loading: 'lazy',
            decoding: 'async'
        });

        const frame = el('div', {className: 'frame'}, [
            img,
            el('div', {className: 'frame-label'}, [
                el('span', {className: 'frame-name', text: image.filename}),
                el('span', {
                    className: 'status-badge ' + image.status,
                    text: statusWord(image.status)
                })
            ])
        ]);

        /* Videos and anything not downloaded have no cached file to serve. */
        img.addEventListener('error', function () {
            img.remove();
            frame.insertBefore(
                el('p', {
                    className: 'frame-missing',
                    text: 'No cached image for this file'
                }),
                frame.firstChild
            );
        });

        return frame;
    }

    function statusWord(status) {
        if (status === 'processed') return 'proposed';
        if (status === 'skipped') return 'skipped';
        if (status === 'error') return 'failed';
        return status;
    }

    function buildBody(image) {
        const parts = [];

        if (image.status === 'skipped') {
            parts.push(el('p', {
                className: 'no-change',
                text: image.skip_reason || 'Already processed.'
            }));
            if (image.current.caption) {
                parts.push(el('div', {className: 'slip'}, [
                    el('p', {className: 'slip-label', text: 'Caption on SmugMug'}),
                    el('p', {className: 'caption-text', text: image.current.caption})
                ]));
            }
        } else if (image.status === 'error') {
            parts.push(el('p', {
                className: 'notice error',
                text: image.error || 'Processing failed for this frame.'
            }));
        } else {
            parts.push(buildCaptionSlip(image));
            parts.push(buildTags(image));
        }

        parts.push(buildEvidence(image));

        if (hints && hints.enabled) {
            parts.push(buildMarginNote(image));
        }

        return el('div', {className: 'card-body'}, parts);
    }

    /**
     * The caption slip.
     *
     * preserve_existing makes the proposed caption literally
     * "<existing> | <new>", so when the proposal starts with what is already
     * on SmugMug only the appended tail is actually new. Showing that split
     * is far more readable than marking the whole paragraph as changed.
     */
    function buildCaptionSlip(image) {
        const current = (image.current.caption || '').trim();
        const proposed = (image.proposed.caption || '').trim();
        const rows = [];

        if (!proposed) {
            rows.push(el('p', {className: 'no-change', text: 'No caption proposed.'}));
        } else if (!current) {
            rows.push(el('p', {className: 'diff-line added', text: proposed}));
        } else if (current === proposed) {
            rows.push(el('p', {className: 'diff-line unchanged', text: proposed}));
        } else if (proposed.indexOf(current) === 0) {
            const tail = proposed.slice(current.length).replace(/^\s*\|\s*/, '');
            rows.push(el('p', {className: 'diff-line unchanged', text: current}));
            rows.push(el('p', {className: 'diff-line added', text: tail}));
        } else {
            rows.push(el('p', {className: 'diff-line removed', text: current}));
            rows.push(el('p', {className: 'diff-line added', text: proposed}));
        }

        const blocks = [el('p', {className: 'slip-label', text: 'Proposed caption'})];
        // Titles are opt-in (processing.generate_titles), so the row only appears when
        // one was actually produced - an absent title means Title is left untouched.
        const title = (image.proposed.title || '').trim();
        if (title) {
            blocks.push(el('p', {className: 'slip-label', text: 'Proposed title'}));
            blocks.push(el('p', {className: 'diff-line added title-line', text: title}));
            blocks.push(el('p', {className: 'slip-label', text: 'Caption'}));
        }
        return el('div', {className: 'slip'}, blocks.concat(rows));
    }

    function buildTags(image) {
        const current = image.current.keywords || [];
        const proposed = image.proposed.keywords || [];
        const currentLower = current.map(lower);
        const proposedLower = proposed.map(lower);

        const tags = [];
        proposed.forEach(function (keyword) {
            const isNew = currentLower.indexOf(lower(keyword)) === -1;
            tags.push(el('span', {
                className: 'tag' + (isNew ? ' added' : ''),
                text: keyword
            }));
        });
        current.forEach(function (keyword) {
            if (proposedLower.indexOf(lower(keyword)) === -1) {
                tags.push(el('span', {className: 'tag removed', text: keyword}));
            }
        });

        return el('div', {className: 'slip'}, [
            el('p', {className: 'slip-label', text: 'Keywords'}),
            tags.length
                ? el('div', {className: 'tag-row'}, tags)
                : el('p', {className: 'no-change', text: 'No keywords proposed.'})
        ]);
    }

    function lower(value) {
        return String(value).toLowerCase();
    }

    function buildEvidence(image) {
        const faces = image.details.faces_detected || [];
        const location = image.details.location;

        return el('div', {className: 'evidence'}, [
            el('div', {}, [
                el('span', {className: 'evidence-label', text: 'People recognised'}),
                el('span', {
                    className: 'evidence-value' + (faces.length ? '' : ' none'),
                    text: faces.length ? faces.join(', ') : 'none'
                })
            ]),
            el('div', {}, [
                el('span', {className: 'evidence-label', text: 'Location'}),
                el('span', {
                    className: 'evidence-value' + (location ? '' : ' none'),
                    text: location || 'none'
                })
            ])
        ]);
    }

    /* -------------------------------------------------------------- *
     * Margin note (per-image hint) + re-read one frame
     * -------------------------------------------------------------- */

    /**
     * A checkbox per known reference face. Checkboxes rather than a custom widget so it
     * is keyboard-navigable and screen-reader-sane for free, and the face sample gives
     * the visual recognition a name list does not.
     */
    /* -------------------------------------------------------------- *
     * Who is in this picture: pinned people, everyone else in a drawer
     * -------------------------------------------------------------- */

    /**
     * The pinned set, read once from localStorage and then held in memory.
     *
     * With nothing stored yet it is seeded from how often each person has actually been
     * picked before (picker_count), falling back to reference-photo count for people
     * with no history. Neither is truth - somebody you photograph constantly may never
     * have needed a manual override - so the seed only has to beat an empty row until
     * the first star is clicked.
     */
    function loadFavourites() {
        if (favourites) return favourites;

        const known = {};
        knownFaces.forEach(function (face) { known[face.name] = true; });

        let stored = null;
        try {
            stored = JSON.parse(window.localStorage.getItem(FAVOURITES_KEY));
        } catch (error) {
            stored = null;   // unreadable or disabled storage is not worth a failure
        }

        if (Array.isArray(stored)) {
            // Drop anyone whose reference folder has since gone, so a renamed person
            // cannot leave a permanently dead tile pinned.
            favourites = stored.filter(function (name) { return known[name]; });
        } else {
            favourites = knownFaces.slice()
                .sort(function (a, b) {
                    return (b.picker_count || 0) - (a.picker_count || 0) ||
                        b.reference_count - a.reference_count;
                })
                .slice(0, FAVOURITES_SEED)
                .map(function (face) { return face.name; });
        }
        return favourites;
    }

    function saveFavourites() {
        try {
            window.localStorage.setItem(FAVOURITES_KEY, JSON.stringify(favourites));
        } catch (error) {
            /* Private mode or a full quota: the pinned set still works for this page. */
        }
    }

    function toggleFavourite(name) {
        const list = loadFavourites();
        const at = list.indexOf(name);
        if (at === -1) list.push(name);
        else list.splice(at, 1);
        saveFavourites();

        // Every picker on the page shows the same pinned set, so they all move together.
        for (let i = livePickers.length - 1; i >= 0; i--) {
            if (!livePickers[i].node.isConnected) livePickers.splice(i, 1);
            else livePickers[i].rebuild();
        }
    }

    function faceThumb(name) {
        const img = el('img', {
            className: 'picker-face',
            src: '/api/face-sample/' + encodeURIComponent(name),
            alt: ''
        });
        img.addEventListener('error', function () { img.remove(); });
        return img;
    }

    /**
     * Build a people picker whose pinned people are always on screen as large tiles
     * and whose remaining people sit in a collapsed drawer.
     *
     * Selection state lives in a closure rather than in the DOM, so re-rendering after
     * a star click cannot lose a tick the user has already made.
     */
    function buildPeoplePicker(idPrefix, selected, legendText, describedBy) {
        const chosen = {};
        (selected || []).forEach(function (name) { chosen[name] = true; });

        if (!knownFaces.length) {
            return {
                node: el('p', {
                    className: 'field-hint',
                    text: 'No reference faces are set up yet, so there is nobody to ' +
                          'pick from. Add folders under reference_faces/ first.'
                }),
                selected: function () { return []; }
            };
        }

        const pinnedRow = el('div', {className: 'picker-pinned'});
        const drawerList = el('div', {className: 'picker-list'});
        const drawerSummary = el('summary', {});
        const drawer = el('details', {className: 'picker-drawer'}, [
            drawerSummary, drawerList
        ]);

        const fieldset = el('fieldset', {className: 'people-picker'}, [
            el('legend', {text: legendText}),
            pinnedRow,
            drawer
        ]);
        if (describedBy) fieldset.setAttribute('aria-describedby', describedBy);

        /* `tile` is layout (big, in the top row) and `pinned` is state (the star).
           They usually agree, but a ticked-but-unpinned person is hoisted into the top
           row and must look like its neighbours while still offering an empty star. */
        function personItem(face, tile, pinned) {
            const id = idPrefix + '-' + face.name;
            const box = el('input', {
                type: 'checkbox',
                id: id,
                value: face.name,
                /* Hidden only on a tile, where the whole tile is the control and the
                   checked state is drawn on the frame. A drawer row is a plain list, so
                   it keeps its real checkbox: hiding that left a row that ticked
                   silently and looked identical either way. */
                className: tile ? 'visually-hidden' : ''
            });
            box.checked = !!chosen[face.name];
            box.addEventListener('change', function () {
                chosen[face.name] = box.checked;
            });

            const star = el('button', {
                type: 'button',
                className: 'picker-star' + (pinned ? ' is-on' : ''),
                'aria-pressed': pinned ? 'true' : 'false',
                'aria-label': (pinned ? 'Unpin ' : 'Pin ') + face.display_name,
                title: pinned
                    ? 'Unpin ' + face.display_name
                    : 'Pin ' + face.display_name + ' so they always show here',
                text: pinned ? '★' : '☆'
            });
            star.addEventListener('click', function () {
                toggleFavourite(face.name);
            });

            return el('div', {
                className: 'picker-item' + (tile ? ' is-tile' : '')
            }, [
                box,
                el('label', {for: id}, [
                    faceThumb(face.name),
                    el('span', {className: 'picker-name', text: face.display_name})
                ]),
                star
            ]);
        }

        function rebuild() {
            const pinned = loadFavourites();
            const isPinned = {};
            pinned.forEach(function (name) { isPinned[name] = true; });

            const byName = {};
            knownFaces.forEach(function (face) { byName[face.name] = face; });

            // Anyone already ticked is hoisted out of the drawer when the picker is
            // built, so a saved selection is visible on load instead of hiding inside a
            // collapsed drawer. Ticking someone mid-session does not move them: pulling
            // a control out from under the cursor is worse than leaving it where it is.
            const upFront = pinned.slice();
            Object.keys(chosen).forEach(function (name) {
                if (chosen[name] && byName[name] && !isPinned[name]) upFront.push(name);
            });

            const upFrontSet = {};
            upFront.forEach(function (name) { upFrontSet[name] = true; });

            S.setChildren(pinnedRow, upFront
                .filter(function (name) { return byName[name]; })
                .map(function (name) {
                    return personItem(byName[name], true, !!isPinned[name]);
                }));

            const rest = knownFaces.filter(function (face) {
                return !upFrontSet[face.name];
            });
            S.setChildren(drawerList, rest.map(function (face) {
                return personItem(face, false, false);
            }));

            drawerSummary.textContent = rest.length
                ? 'Everyone else (' + rest.length + ')'
                : 'Everyone is shown above';
            drawer.classList.toggle('hidden', !rest.length);

            if (!upFront.length) {
                S.setChildren(pinnedRow, [
                    el('p', {
                        className: 'field-hint',
                        text: 'Nobody pinned. Open the drawer and use ☆ to pin the ' +
                              'people you tag most.'
                    })
                ]);
            }
        }

        rebuild();

        const picker = {
            node: fieldset,
            rebuild: rebuild,
            selected: function () {
                return Object.keys(chosen).filter(function (name) {
                    return chosen[name];
                });
            }
        };
        livePickers.push(picker);
        return picker;
    }

    /* -------------------------------------------------------------- *
     * Pets: the subjects recognition can never learn
     * -------------------------------------------------------------- */

    /**
     * Build a pet picker: one chip per configured pet.
     *
     * Separate from the people picker on purpose. A pet has no reference face and must
     * never be counted as one, so ticking a pet adds its description to the prompt as
     * ground truth and his name to the keywords - it does not claim a face was found.
     *
     * @param {string} idPrefix unique per picker instance
     * @param {Array<string>} selected pet names already stored for this scope
     * @param {string} legendText fieldset legend
     * @returns {{node: Element, selected: function(): Array<string>}}
     */
    function buildPetPicker(idPrefix, selected, legendText) {
        const chosen = {};
        (selected || []).forEach(function (name) { chosen[name] = true; });

        if (!knownPets.length) {
            /* Still say the feature exists: a pet nobody has defined is invisible
               otherwise, and the place to define one is not on this page. */
            return {
                node: el('p', {className: 'field-hint'}, [
                    document.createTextNode('No pets set up yet. '),
                    el('a', {href: '/hints', text: 'Add a pet'}),
                    document.createTextNode(
                        ' to tick it here - useful for animals, which face recognition ' +
                        'can never learn.'
                    )
                ]),
                selected: function () { return []; }
            };
        }

        const chips = knownPets.map(function (pet) {
            const id = idPrefix + '-pet-' + pet.name;
            const box = el('input', {
                type: 'checkbox',
                id: id,
                value: pet.name,
                className: 'visually-hidden'
            });
            box.checked = !!chosen[pet.name];
            box.addEventListener('change', function () {
                chosen[pet.name] = box.checked;
            });
            return el('div', {className: 'pet-item'}, [
                box,
                el('label', {for: id, title: pet.description, text: pet.name})
            ]);
        });

        const fieldset = el('fieldset', {className: 'people-picker pet-picker'}, [
            el('legend', {text: legendText})
        ].concat([el('div', {className: 'pet-chips'}, chips)]));

        return {
            node: fieldset,
            selected: function () {
                return knownPets
                    .map(function (pet) { return pet.name; })
                    .filter(function (name) { return chosen[name]; });
            }
        };
    }

    function buildMarginNote(image) {
        const textareaId = 'hint-' + image.image_key;
        const locationId = 'hint-loc-' + image.image_key;
        const statusId = 'hint-status-' + image.image_key;
        const stored = (hints.images && hints.images[image.image_key]) || '';
        const storedLocation =
            (hints.image_locations && hints.image_locations[image.image_key]) || '';

        const textarea = el('textarea', {
            id: textareaId,
            rows: '2',
            placeholder: 'e.g. The white ribbed object is a Nylabone dog chew, not food.'
        });
        textarea.value = stored;

        // A location is a value to replace, not a fact to argue with. A note only
        // reaches the prompt; this replaces the geocoded place outright so the caption,
        // the keywords and the LOCATION field above all agree.
        const locationInput = el('input', {
            id: locationId,
            type: 'text',
            placeholder: 'e.g. Gorilla Enclosure, Louisville Zoo'
        });
        locationInput.value = storedLocation;

        const hintId = 'people-hint-' + image.image_key;
        const picker = buildPeoplePicker(
            'who-' + image.image_key,
            (hints.image_people && hints.image_people[image.image_key]) || [],
            'Who is in this frame?',
            hintId
        );

        const petPicker = buildPetPicker(
            'who-' + image.image_key,
            (hints.image_pets && hints.image_pets[image.image_key]) || [],
            'Any pets in this frame?'
        );

        const status = el('p', {
            className: 'note-status',
            id: statusId,
            role: 'status',
            'aria-live': 'polite'
        });

        const button = el('button', {
            type: 'button',
            className: 'secondary note-save',
            text: 'Save note & re-read'
        });
        button.addEventListener('click', function () {
            regenerate(
                image.image_key, textarea, locationInput, picker, petPicker, status, button
            );
        });

        return el('div', {className: 'margin-note'}, [
            el('label', {for: textareaId}, [
                document.createTextNode('Note for this frame'),
                el('span', {className: 'scope-tag', text: 'image'})
            ]),
            textarea,
            el('p', {
                className: 'field-hint',
                text: 'A fact the model must accept over its own guess. Saving ' +
                      're-reads only this frame.'
            }),
            el('label', {for: locationId}, [
                document.createTextNode('Correct location for this frame'),
                el('span', {className: 'scope-tag', text: 'overrides GPS'})
            ]),
            locationInput,
            el('p', {
                className: 'field-hint',
                text: 'Replaces the place name resolved from GPS, for the caption and ' +
                      'the keywords both. Leave blank to keep the resolved one.'
            }),
            picker.node,
            el('p', {
                className: 'field-hint',
                id: hintId,
                text: 'For faces the recogniser missed. Tick everyone in the frame, ' +
                      'including anyone it already got right - this replaces its list, ' +
                      'and feeds the caption and the keywords. Leave all unticked to ' +
                      'trust the recogniser.'
            }),
            petPicker.node,
            el('div', {className: 'note-actions'}, [button, status])
        ]);
    }

    /**
     * Re-read one frame while holding the write path shut.
     *
     * A re-read changes what would be written, so it must never be in flight
     * at the same time as a write. Counting them and re-syncing the write
     * button keeps the two mutually exclusive: the button cannot be pressed
     * (so the dialog cannot open) until every re-read has landed, and the
     * dialog makes the rest of the page inert, so no re-read can start while
     * it is open.
     */
    async function regenerate(
        imageKey, textarea, locationInput, picker, petPicker, status, button
    ) {
        regenerating += 1;
        syncWriteButton();
        try {
            await regenerateFrame(
                imageKey, textarea, locationInput, picker, petPicker, status, button
            );
        } finally {
            regenerating -= 1;
            syncWriteButton();
        }
    }

    /**
     * Save the image-scope hint, then re-run that one image.
     * Only this card shows a spinner and only this card is replaced.
     */
    async function regenerateFrame(
        imageKey, textarea, locationInput, picker, petPicker, status, button
    ) {
        const card = dom.grid.querySelector(
            '.image-card[data-image-key="' + cssEscape(imageKey) + '"]'
        );
        const restore = S.busy(button, 'Saving note and re-reading frame');
        S.announce(status, 'Saving note…');
        setCardBusy(card, true, 'Re-reading…');

        try {
            await S.apiPut('/api/hints', {
                scope: 'image',
                key: imageKey,
                text: textarea.value,
                location: locationInput.value,
                people: picker.selected(),
                pets: petPicker.selected()
            });
            hints.images[imageKey] = textarea.value.trim();
            if (!hints.image_locations) {
                hints.image_locations = {};
            }
            hints.image_locations[imageKey] = locationInput.value.trim();
            if (!hints.image_people) {
                hints.image_people = {};
            }
            hints.image_people[imageKey] = picker.selected();
            if (!hints.image_pets) {
                hints.image_pets = {};
            }
            hints.image_pets[imageKey] = petPicker.selected();
        } catch (error) {
            S.announce(status, 'Could not save the note: ' + error.message, 'error');
            setCardBusy(card, false);
            restore();
            return;
        }

        S.announce(status, 'Re-reading this frame with the model…');
        const ok = await rereadFrame(imageKey, status);
        restore();
        return ok;
    }

    /**
     * Re-read one frame and swap its card in place.
     *
     * The hint save is deliberately NOT part of this: a single frame saves its own
     * note first, and an album-wide sweep saves the album note once and then reuses
     * this for every frame. One re-read path, two callers.
     *
     * @param {string} imageKey frame to re-read
     * @param {Element} status where to report a failure that leaves no card to write on
     * @returns {Promise<boolean>} true when the frame came back cleanly
     */
    async function rereadFrame(imageKey, status) {
        const card = dom.grid.querySelector(
            '.image-card[data-image-key="' + cssEscape(imageKey) + '"]'
        );
        setCardBusy(card, true, 'Re-reading…');

        let payload;
        try {
            payload = await S.apiPost(
                '/api/preview/' + encodeURIComponent(jobId) + '/regenerate',
                {image_key: imageKey}
            );
        } catch (error) {
            /* 502 still carries the image object so the card can show it. */
            if (error.payload && error.payload.image) {
                applyRegenerated(error.payload, imageKey);
                announceOnCard(imageKey, error.message, 'error');
            } else {
                setCardBusy(card, false);
                if (status) S.announce(status, error.message, 'error');
            }
            return false;
        }

        applyRegenerated(payload, imageKey);
        announceOnCard(
            imageKey,
            payload.hint_applied
                ? 'Re-read using: ' + payload.hint_applied
                : 'Re-read with no note applied.',
            'ok'
        );
        return true;
    }

    /** Swap one card's data in place, leaving every other card untouched. */
    function applyRegenerated(payload, imageKey) {
        const updated = payload.image;
        const index = images.findIndex(function (item) {
            return item.image_key === imageKey;
        });
        if (index !== -1) images[index] = updated;

        const oldCard = dom.grid.querySelector(
            '.image-card[data-image-key="' + cssEscape(imageKey) + '"]'
        );
        const newCard = buildCard(updated);
        if (oldCard) {
            oldCard.replaceWith(newCard);
        } else {
            dom.grid.appendChild(newCard);
        }

        if (payload.stats) renderTally(payload.stats);
        renderWritePanel();
    }

    /** Re-announce on the rebuilt card (the old status node is gone). */
    function announceOnCard(imageKey, message, kind) {
        const node = document.getElementById('hint-status-' + imageKey);
        if (node) S.announce(node, message, kind);
    }

    function setCardBusy(card, isBusy, label) {
        if (!card) return;
        card.classList.toggle('is-busy', isBusy);
        const existing = card.querySelector('.card-veil');
        if (existing) existing.remove();
        if (isBusy) {
            card.appendChild(el('div', {className: 'card-veil', 'aria-hidden': 'true'}, [
                el('span', {className: 'veil-text', text: label || 'Working…'})
            ]));
        }
    }

    /** CSS.escape with a conservative fallback for attribute selectors. */
    function cssEscape(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/["\\]/g, '\\$&');
    }

    /* -------------------------------------------------------------- *
     * The safelight: writing to SmugMug
     * -------------------------------------------------------------- */

    /** Image keys that have something worth writing. */
    function writableImages() {
        return images.filter(function (image) {
            if (image.status !== 'processed') return false;
            const caption = (image.proposed.caption || '').trim();
            const keywords = image.proposed.keywords || [];
            return Boolean(caption) || keywords.length > 0;
        });
    }

    function renderWritePanel() {
        const targets = writableImages();

        if (committed) return;

        if (!targets.length) {
            dom.writePanel.classList.add('hidden');
            return;
        }
        dom.writePanel.classList.remove('hidden');

        dom.writeSummary.textContent =
            S.plural(targets.length, 'frame') + ' would be updated on SmugMug';

        S.setChildren(dom.writeList, targets.map(function (image) {
            return el('li', {text: image.filename + '  (' + image.image_key + ')'});
        }));

        syncWriteButton();
    }

    function syncWriteButton() {
        const busy = regenerating > 0 && !committed;
        dom.writeBtn.disabled = !dom.writeLatch.checked || committed || busy;
        if (dom.writeBusyNote) {
            dom.writeBusyNote.classList.toggle('hidden', !busy);
        }
    }

    function openConfirm() {
        const targets = writableImages();
        if (!targets.length || !dom.writeLatch.checked || regenerating > 0) return;

        /* Freeze the write set the moment the dialog names it. doCommit sends
           these keys and nothing else, so a re-read that lands while the
           dialog is open cannot smuggle an unconfirmed image into the write. */
        pendingWriteKeys = targets.map(function (image) {
            return image.image_key;
        });

        dom.confirmText.textContent =
            'This writes captions and keywords for ' +
            S.plural(targets.length, 'image') + ' to "' + albumName +
            '" on SmugMug. This cannot be undone from smugVision.';

        S.setChildren(dom.confirmList, targets.map(function (image) {
            return el('li', {text: image.filename});
        }));

        dom.confirmModal.showModal();
        dom.cancelBtn.focus();
    }

    /**
     * The one and only write. Sends confirm: true plus the explicit list of
     * image keys captured when the dialog opened, so the server writes exactly
     * what the dialog listed - never a set recomputed after the user read it.
     */
    async function doCommit() {
        const targetKeys = pendingWriteKeys;
        if (!targetKeys || !targetKeys.length) return;

        const restore = S.busy(dom.confirmBtn, 'Writing to SmugMug');
        dom.cancelBtn.disabled = true;

        let result;
        try {
            result = await S.apiPost('/api/commit', {
                job_id: jobId,
                confirm: true,
                image_keys: targetKeys.slice()
            });
        } catch (error) {
            restore();
            dom.cancelBtn.disabled = false;
            dom.confirmModal.close();
            showWriteResult(
                'Nothing was written: ' + error.message,
                'error'
            );
            return;
        }

        committed = true;
        restore();
        dom.cancelBtn.disabled = false;
        dom.confirmModal.close();

        const bits = [S.plural(result.committed, 'image') + ' written to SmugMug'];
        if (result.errors) bits.push(result.errors + ' failed');
        if (result.skipped_keys && result.skipped_keys.length) {
            bits.push(result.skipped_keys.length + ' skipped (nothing to write)');
        }
        showWriteResult(bits.join('. ') + '.', result.errors ? 'caution' : 'ok');

        dom.writeLatch.checked = false;
        dom.writeLatch.disabled = true;
        dom.writeBtn.disabled = true;
        dom.writeBtn.textContent = 'Written';
        S.setChildren(dom.writeList, []);
        dom.writeSummary.textContent =
            'Written. Re-run the album to propose further changes.';

        returnToAlbums(result);
    }

    /**
     * URL of the picker, restored to the folder this run was started from.
     *
     * A run started from a pasted URL has no browsing position, so it falls back to
     * the root rather than inventing one.
     *
     * @param {Object} [extra] additional query parameters
     * @returns {string}
     */
    function albumsUrl(extra) {
        const params = [];
        if (originNode) params.push('node=' + encodeURIComponent(originNode));
        Object.keys(extra || {}).forEach(function (key) {
            params.push(key + '=' + encodeURIComponent(extra[key]));
        });
        return '/' + (params.length ? '?' + params.join('&') : '');
    }

    /**
     * Send the user back to the album list after a clean write.
     *
     * Only when nothing failed: a partial write is exactly when the user needs to stay
     * and read which frames did not make it. The pause is long enough to see the
     * result on this page, and the summary is repeated on the picker so leaving does
     * not lose it.
     */
    function returnToAlbums(result) {
        if (result.errors) return;

        const target = albumsUrl({
            wrote: result.committed,
            album: albumName
        });
        window.setTimeout(function () {
            window.location.href = target;
        }, 1800);
    }

    function showWriteResult(message, kind) {
        dom.writeResult.className = 'notice ' +
            (kind === 'error' ? 'error' : (kind === 'caution' ? 'caution' : ''));
        dom.writeResult.textContent = message;
        dom.writeResult.classList.remove('hidden');
    }
})();
