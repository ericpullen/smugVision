/**
 * smugVision - known reference faces.
 *
 * Reads GET /api/faces and renders one portrait per person. When a sample
 * image cannot be served, the portrait falls back to the person's initials
 * drawn in CSS rather than an inline SVG data URI, so this app references no
 * URLs of any kind and works with no network.
 */
(function () {
    'use strict';

    const S = window.smugvision;
    const el = S.el;

    document.addEventListener('DOMContentLoaded', load);

    async function load() {
        const grid = document.getElementById('faces-grid');
        const status = document.getElementById('faces-status');
        const errorBox = document.getElementById('faces-error');
        const intro = document.getElementById('faces-intro');

        let data;
        try {
            data = await S.apiGet('/api/faces');
        } catch (error) {
            S.setChildren(grid, []);
            errorBox.textContent = error.message;
            errorBox.classList.remove('hidden');
            S.announce(status, 'Could not load reference faces.');
            return;
        }

        if (!data.enabled) {
            S.setChildren(grid, [
                el('li', {}, [
                    el('div', {className: 'info-box'}, [
                        el('p', {
                            text: data.message ||
                                'Face recognition is not enabled or configured.'
                        }),
                        el('p', {className: 'muted'}, [
                            document.createTextNode('Add reference images under '),
                            el('code', {text: '~/.smugvision/reference_faces/PersonName/'}),
                            document.createTextNode(' to switch it on.')
                        ])
                    ])
                ])
            ]);
            S.announce(status, 'Face recognition is not enabled.');
            return;
        }

        if (!data.faces.length) {
            S.setChildren(grid, [
                el('li', {}, [
                    el('div', {className: 'info-box'}, [
                        el('p', {text: 'No reference faces found.'}),
                        el('p', {className: 'muted'}, [
                            document.createTextNode('Add reference images under '),
                            el('code', {text: '~/.smugvision/reference_faces/PersonName/'})
                        ])
                    ])
                ])
            ]);
            S.announce(status, 'No reference faces found.');
            return;
        }

        intro.textContent =
            'smugVision can recognise ' + S.plural(data.total, 'person', 'people') +
            '. Their names are passed to the vision model as context, which is why ' +
            'captions can say who is in the picture.';

        S.setChildren(grid, data.faces.map(faceCard));
        S.announce(status, S.plural(data.total, 'known face') + ' loaded.');
    }

    function faceCard(face) {
        const portrait = el('div', {className: 'face-portrait'});

        const img = el('img', {
            src: '/api/face-sample/' + encodeURIComponent(face.name),
            alt: 'Reference photograph of ' + face.display_name,
            loading: 'lazy',
            decoding: 'async'
        });
        img.addEventListener('error', function () {
            img.remove();
            portrait.appendChild(el('span', {
                className: 'initials',
                'aria-hidden': 'true',
                text: initials(face.display_name)
            }));
        });
        portrait.appendChild(img);

        return el('li', {className: 'face-card'}, [
            portrait,
            el('h4', {text: face.display_name}),
            el('p', {
                className: 'reference-count',
                text: S.plural(face.reference_count, 'reference')
            })
        ]);
    }

    /** "Ada Rivera" -> "AR". Falls back to the first character. */
    function initials(name) {
        const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return '?';
        if (parts.length === 1) return parts[0].charAt(0).toUpperCase();
        return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
    }
})();
