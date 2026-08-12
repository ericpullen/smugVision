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
    let pendingWriteKeys = null;  // image keys the open confirm dialog named
    let regenerating = 0;         // in-flight single-frame re-reads

    document.addEventListener('DOMContentLoaded', function () {
        const root = document.getElementById('proof-root');
        jobId = root.dataset.jobId;

        dom.albumTitle = document.getElementById('album-title');
        dom.albumKeyEl = document.getElementById('album-key');
        dom.tally = document.getElementById('tally');
        dom.grid = document.getElementById('image-grid');
        dom.gridStatus = document.getElementById('grid-status');
        dom.loadError = document.getElementById('load-error');

        dom.albumHintBox = document.getElementById('album-hint-box');
        dom.albumHint = document.getElementById('album-hint');
        dom.albumHintBtn = document.getElementById('album-hint-btn');
        dom.albumHintStatus = document.getElementById('album-hint-status');
        dom.albumLocation = document.getElementById('album-location');
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

        albumKey = data.album_key;
        albumName = data.album_name;
        images = data.images || [];
        hints = data.hints || {enabled: false, global: '', album: '', images: {}};

        document.title = 'smugVision - ' + albumName;
        dom.albumTitle.textContent = albumName;
        dom.albumKeyEl.textContent = 'album ' + albumKey + '  ·  job ' + jobId;

        renderTally(data.stats);
        renderHintEditors();
        renderGrid();
        renderWritePanel();

        S.announce(
            dom.gridStatus,
            S.plural(images.length, 'frame') + ' in this proof run.'
        );
    }

    function renderTally(stats) {
        const entries = [
            ['processed', stats.processed, 'proposed'],
            ['skipped', stats.skipped, 'skipped'],
            ['errors', stats.errors, 'failed'],
            ['total', stats.total, 'total']
        ];
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

    async function saveAlbumHint() {
        const restore = S.busy(dom.albumHintBtn, 'Saving album note');
        S.announce(dom.albumHintStatus, '');
        try {
            const result = await S.apiPut('/api/hints', {
                scope: 'album',
                key: albumKey,
                text: dom.albumHint.value,
                location: dom.albumLocation.value
            });
            hints.album = result.text;
            dom.albumHint.value = result.text;
            hints.album_location = result.location || '';
            dom.albumLocation.value = hints.album_location;
            S.announce(
                dom.albumHintStatus,
                result.cleared
                    ? 'Album note cleared. Re-read a frame to see the effect.'
                    : 'Album note saved. Re-read a frame to apply it.',
                'ok'
            );
        } catch (error) {
            S.announce(dom.albumHintStatus, error.message, 'error');
        } finally {
            restore();
        }
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

        return el('div', {className: 'slip'}, [
            el('p', {className: 'slip-label', text: 'Proposed caption'})
        ].concat(rows));
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

        const status = el('p', {
            className: 'note-status',
            id: statusId,
            role: 'status',
            'aria-live': 'polite'
        });

        const button = el('button', {
            type: 'button',
            className: 'secondary',
            text: 'Save note & re-read'
        });
        button.addEventListener('click', function () {
            regenerate(image.image_key, textarea, locationInput, status, button);
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
    async function regenerate(imageKey, textarea, locationInput, status, button) {
        regenerating += 1;
        syncWriteButton();
        try {
            await regenerateFrame(imageKey, textarea, locationInput, status, button);
        } finally {
            regenerating -= 1;
            syncWriteButton();
        }
    }

    /**
     * Save the image-scope hint, then re-run that one image.
     * Only this card shows a spinner and only this card is replaced.
     */
    async function regenerateFrame(imageKey, textarea, locationInput, status, button) {
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
                location: locationInput.value
            });
            hints.images[imageKey] = textarea.value.trim();
            if (!hints.image_locations) {
                hints.image_locations = {};
            }
            hints.image_locations[imageKey] = locationInput.value.trim();
        } catch (error) {
            S.announce(status, 'Could not save the note: ' + error.message, 'error');
            setCardBusy(card, false);
            restore();
            return;
        }

        S.announce(status, 'Re-reading this frame with the model…');

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
                S.announce(status, error.message, 'error');
            }
            restore();
            return;
        }

        applyRegenerated(payload, imageKey);
        announceOnCard(
            imageKey,
            payload.hint_applied
                ? 'Re-read using: ' + payload.hint_applied
                : 'Re-read with no note applied.',
            'ok'
        );
        restore();
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
    }

    function showWriteResult(message, kind) {
        dom.writeResult.className = 'notice ' +
            (kind === 'error' ? 'error' : (kind === 'caution' ? 'caution' : ''));
        dom.writeResult.textContent = message;
        dom.writeResult.classList.remove('hidden');
    }
})();
