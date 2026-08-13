/**
 * smugVision - hint editor.
 *
 * Hints are facts the person who took the photo asserts, and they outrank
 * the vision model's own guess. Three scopes, most general first:
 *
 *   global  - every album, every image, forever
 *   album   - every image in one album
 *   image   - one frame
 *
 * This page edits all three. It never triggers processing and never writes
 * to SmugMug; hints live in ~/.smugvision/hints.yaml.
 */
(function () {
    'use strict';

    const S = window.smugvision;
    const el = S.el;

    const dom = {};
    let store = null;

    document.addEventListener('DOMContentLoaded', function () {
        dom.globalText = document.getElementById('global-hint');
        dom.globalBtn = document.getElementById('global-hint-btn');
        dom.globalStatus = document.getElementById('global-hint-status');

        dom.savedList = document.getElementById('saved-list');
        dom.savedStatus = document.getElementById('saved-status');
        dom.disabledNotice = document.getElementById('hints-disabled');
        dom.fileNote = document.getElementById('hints-file');
        dom.loadError = document.getElementById('load-error');

        dom.addForm = document.getElementById('add-form');
        dom.addKey = document.getElementById('add-key');
        dom.addText = document.getElementById('add-text');
        dom.addStatus = document.getElementById('add-status');
        dom.addBtn = document.getElementById('add-btn');

        dom.petsList = document.getElementById('pets-list');
        dom.petsStatus = document.getElementById('pets-status');
        dom.petForm = document.getElementById('pet-form');
        dom.petName = document.getElementById('pet-name');
        dom.petDescription = document.getElementById('pet-description');
        dom.petStatus = document.getElementById('pet-status');
        dom.petSaveBtn = document.getElementById('pet-save-btn');

        dom.petForm.addEventListener('submit', function (event) {
            event.preventDefault();
            savePet();
        });

        dom.globalBtn.addEventListener('click', saveGlobal);
        dom.addForm.addEventListener('submit', function (event) {
            event.preventDefault();
            addHint();
        });

        load();
        loadPets();
    });

    /* -------------------------------------------------------------- *
     * Pets: the vocabulary the proof sheet ticks against
     * -------------------------------------------------------------- */

    async function loadPets() {
        let data;
        try {
            data = await S.apiGet('/api/pets');
        } catch (error) {
            S.setChildren(dom.petsList, [
                el('p', {className: 'notice error', text: error.message})
            ]);
            return;
        }
        renderPets(data.pets || []);
    }

    function renderPets(pets) {
        if (!pets.length) {
            S.setChildren(dom.petsList, [
                el('p', {
                    className: 'muted',
                    text: 'No pets yet. Add one below and it appears on every proof ' +
                          'sheet, ready to tick.'
                })
            ]);
            S.announce(dom.petsStatus, 'No pets configured.');
            return;
        }

        S.setChildren(dom.petsList, [
            el('ul', {className: 'pet-list'}, pets.map(function (pet) {
                const remove = el('button', {
                    type: 'button',
                    className: 'secondary',
                    text: 'Remove',
                    'aria-label': 'Remove ' + pet.name
                });
                remove.addEventListener('click', function () {
                    removePet(pet.name, remove);
                });
                /* Clicking the name loads it into the form, so editing a description is
                   "save over it" rather than "delete and retype". */
                const edit = el('button', {
                    type: 'button',
                    className: 'link-button pet-name',
                    text: pet.name,
                    title: 'Edit ' + pet.name
                });
                edit.addEventListener('click', function () {
                    dom.petName.value = pet.name;
                    dom.petDescription.value = pet.description;
                    dom.petDescription.focus();
                });
                return el('li', {}, [
                    edit,
                    el('span', {className: 'pet-fact', text: pet.description}),
                    remove
                ]);
            }))
        ]);
        S.announce(dom.petsStatus, S.plural(pets.length, 'pet') + ' configured.');
    }

    async function savePet() {
        const name = dom.petName.value.trim();
        const description = dom.petDescription.value.trim();

        if (!name || !description) {
            S.announce(
                dom.petStatus,
                'A pet needs both a name and a description.',
                'error'
            );
            return;
        }

        const restore = S.busy(dom.petSaveBtn, 'Saving pet');
        try {
            await S.apiPut('/api/pets', {name: name, description: description});
            dom.petName.value = '';
            dom.petDescription.value = '';
            S.announce(dom.petStatus, 'Saved ' + name + '.', 'ok');
            await loadPets();
        } catch (error) {
            S.announce(dom.petStatus, error.message, 'error');
        } finally {
            restore();
        }
    }

    async function removePet(name, button) {
        const restore = S.busy(button, 'Removing pet');
        try {
            await S.apiDelete('/api/pets/' + encodeURIComponent(name));
            S.announce(dom.petStatus, 'Removed ' + name + '.', 'ok');
            await loadPets();
        } catch (error) {
            S.announce(dom.petStatus, error.message, 'error');
        } finally {
            restore();
        }
    }

    async function load() {
        let data;
        try {
            data = await S.apiGet('/api/hints');
        } catch (error) {
            dom.loadError.textContent = error.message;
            dom.loadError.classList.remove('hidden');
            return;
        }

        store = data;

        if (!data.enabled) {
            dom.disabledNotice.textContent = data.message ||
                'Hints are disabled in your config (hints.enabled=false), so ' +
                'nothing can be saved.';
            dom.disabledNotice.classList.remove('hidden');
            disableEverything();
            return;
        }

        dom.fileNote.textContent = data.file || '';
        dom.globalText.value = data.global || '';
        renderSaved();
    }

    function disableEverything() {
        [dom.globalText, dom.globalBtn, dom.addKey, dom.addText, dom.addBtn]
            .forEach(function (node) {
                if (node) node.disabled = true;
            });
    }

    /* -------------------------------------------------------------- *
     * Global scope
     * -------------------------------------------------------------- */

    async function saveGlobal() {
        const restore = S.busy(dom.globalBtn, 'Saving global note');
        S.announce(dom.globalStatus, '');
        try {
            const result = await S.apiPut('/api/hints', {
                scope: 'global',
                text: dom.globalText.value
            });
            store.global = result.text;
            dom.globalText.value = result.text;
            S.announce(
                dom.globalStatus,
                result.cleared
                    ? 'Global note cleared.'
                    : 'Global note saved. It now applies to every album.',
                'ok'
            );
        } catch (error) {
            S.announce(dom.globalStatus, error.message, 'error');
        } finally {
            restore();
        }
    }

    /* -------------------------------------------------------------- *
     * Saved album/image notes
     * -------------------------------------------------------------- */

    function renderSaved() {
        const rows = [];

        Object.keys(store.albums || {}).sort().forEach(function (key) {
            rows.push(hintRow('album', key, store.albums[key]));
        });
        Object.keys(store.images || {}).sort().forEach(function (key) {
            rows.push(hintRow('image', key, store.images[key]));
        });

        if (!rows.length) {
            S.setChildren(dom.savedList, [
                el('div', {
                    className: 'info-box',
                    text: 'No album or image notes saved yet. Add one below, or ' +
                          'write a note straight onto a frame from a proof sheet.'
                })
            ]);
            S.announce(dom.savedStatus, 'No album or image notes saved.');
            return;
        }

        S.setChildren(dom.savedList, rows);
        S.announce(dom.savedStatus, S.plural(rows.length, 'saved note') + '.');
    }

    function hintRow(scope, key, text) {
        const textareaId = 'saved-' + scope + '-' + key;
        const textarea = el('textarea', {id: textareaId, rows: '2'});
        textarea.value = text || '';

        const status = el('span', {
            className: 'note-status',
            role: 'status',
            'aria-live': 'polite'
        });

        const saveBtn = el('button', {
            type: 'button',
            className: 'secondary',
            text: 'Save'
        });
        const clearBtn = el('button', {
            type: 'button',
            className: 'quiet',
            text: 'Clear'
        });

        saveBtn.addEventListener('click', function () {
            writeHint(scope, key, textarea.value, saveBtn, status);
        });
        clearBtn.addEventListener('click', function () {
            textarea.value = '';
            writeHint(scope, key, '', clearBtn, status);
        });

        return el('div', {className: 'scope-card ' + scope}, [
            el('div', {className: 'scope-head'}, [
                el('span', {className: 'scope-badge ' + scope, text: scope + ' scope'}),
                el('span', {className: 'mono', text: key})
            ]),
            el('p', {
                className: 'scope-reach',
                text: scope === 'album'
                    ? 'Applies to every image in album ' + key + '.'
                    : 'Applies to image ' + key + ' only.'
            }),
            el('div', {className: 'field'}, [
                el('label', {for: textareaId, className: 'visually-hidden',
                    text: scope + ' note for ' + key}),
                textarea
            ]),
            el('div', {className: 'note-actions'}, [saveBtn, clearBtn, status])
        ]);
    }

    async function writeHint(scope, key, text, button, status) {
        const restore = S.busy(button, 'Saving');
        S.announce(status, '');
        try {
            const result = await S.apiPut('/api/hints', {
                scope: scope,
                key: key,
                text: text
            });
            const section = scope === 'album' ? 'albums' : 'images';
            if (result.cleared) {
                delete store[section][key];
                restore();
                renderSaved();
                return;
            }
            store[section][key] = result.text;
            S.announce(status, 'Saved.', 'ok');
        } catch (error) {
            S.announce(status, error.message, 'error');
        } finally {
            restore();
        }
    }

    /* -------------------------------------------------------------- *
     * Adding a note by key
     * -------------------------------------------------------------- */

    async function addHint() {
        const scopeInput = document.querySelector('input[name="add-scope"]:checked');
        const scope = scopeInput ? scopeInput.value : 'album';
        const key = dom.addKey.value.trim();
        const text = dom.addText.value;

        S.announce(dom.addStatus, '');

        if (!key) {
            S.announce(dom.addStatus, 'Enter an album or image key.', 'error');
            dom.addKey.focus();
            return;
        }
        if (!text.trim()) {
            S.announce(dom.addStatus, 'Enter the note text.', 'error');
            dom.addText.focus();
            return;
        }

        const restore = S.busy(dom.addBtn, 'Saving note');
        try {
            const result = await S.apiPut('/api/hints', {
                scope: scope,
                key: key,
                text: text
            });
            const section = scope === 'album' ? 'albums' : 'images';
            store[section][key] = result.text;
            dom.addKey.value = '';
            dom.addText.value = '';
            S.announce(dom.addStatus, 'Saved a ' + scope + ' note for ' + key + '.', 'ok');
            renderSaved();
        } catch (error) {
            S.announce(dom.addStatus, error.message, 'error');
        } finally {
            restore();
        }
    }
})();
