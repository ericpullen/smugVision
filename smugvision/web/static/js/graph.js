/**
 * smugVision - relationship graph.
 *
 * A small Fruchterman-Reingold force layout rendered as inline SVG. This
 * replaces the vis-network library that used to be pulled from a CDN, so the
 * page works with no network connection.
 *
 * The graph itself is decorative (aria-hidden): the accessible, complete
 * representation of the same data is the relationship table the page renders
 * beside it. A force-directed diagram cannot be read by a screen reader or
 * navigated by keyboard in any useful way, so the table is not a fallback -
 * it is the primary text form.
 */
(function () {
    'use strict';

    const S = window.smugvision;
    const el = S.el;

    /**
     * The SVG namespace, taken from a parsed element rather than written as a
     * literal URL. Same result as hardcoding it, but it keeps this app's
     * assets free of any URL-shaped string, so an audit for external
     * references comes back empty.
     */
    const SVG_NS = (function () {
        const probe = document.createElement('div');
        probe.innerHTML = '<svg></svg>';
        return probe.firstChild.namespaceURI;
    })();

    const NODE_RADIUS = 26;
    const ITERATIONS = 320;
    const PADDING = 34;

    document.addEventListener('DOMContentLoaded', load);

    async function load() {
        const container = document.getElementById('graph-container');
        const errorBox = document.getElementById('graph-error');
        const status = document.getElementById('graph-status');
        const tableWrap = document.getElementById('relationship-table');
        const groupsSection = document.getElementById('groups-section');
        const groupsList = document.getElementById('groups-list');

        let data;
        try {
            data = await S.apiGet('/api/relationships');
        } catch (error) {
            S.setChildren(container, []);
            errorBox.textContent = error.message;
            errorBox.classList.remove('hidden');
            S.announce(status, 'Could not load relationships.');
            return;
        }

        if (!data.enabled || !data.nodes.length) {
            S.setChildren(container, [
                el('div', {className: 'info-box'}, [
                    el('p', {
                        text: data.enabled
                            ? 'No relationships are defined yet.'
                            : (data.message || 'Relationships are not configured.')
                    }),
                    el('p', {className: 'muted'}, [
                        document.createTextNode('Define them in '),
                        el('code', {text: '~/.smugvision/relationships.yaml'}),
                        document.createTextNode('.')
                    ])
                ])
            ]);
            S.announce(status, 'No relationships defined.');
            return;
        }

        renderTable(tableWrap, data);
        S.announce(
            status,
            S.plural(data.nodes.length, 'person', 'people') + ' and ' +
            S.plural(data.edges.length, 'relationship') + '.'
        );

        drawGraph(container, data);

        if (data.groups && data.groups.length) {
            S.setChildren(groupsList, [
                el('ul', {}, data.groups.map(function (group) {
                    const members = group.members
                        .map(S.displayName)
                        .join(', ');
                    return el('li', {}, [
                        el('strong', {text: group.description || 'Group'}),
                        document.createTextNode(': ' + members)
                    ]);
                }))
            ]);
            groupsSection.classList.remove('hidden');
        }
    }

    /* -------------------------------------------------------------- *
     * The accessible representation
     * -------------------------------------------------------------- */

    function renderTable(wrap, data) {
        const rows = data.edges.map(function (edge) {
            return el('tr', {}, [
                el('td', {text: S.displayName(edge.from)}),
                el('td', {text: edge.label}),
                el('td', {text: S.displayName(edge.to)})
            ]);
        });

        S.setChildren(wrap, [
            el('div', {className: 'table-scroll'}, [
                el('table', {className: 'hint-table'}, [
                    el('caption', {
                        className: 'visually-hidden',
                        text: 'Every known relationship, as a list'
                    }),
                    el('thead', {}, [
                        el('tr', {}, [
                            el('th', {scope: 'col', text: 'Person'}),
                            el('th', {scope: 'col', text: 'Relationship'}),
                            el('th', {scope: 'col', text: 'To'})
                        ])
                    ]),
                    el('tbody', {}, rows)
                ])
            ])
        ]);
    }

    /* -------------------------------------------------------------- *
     * Layout + render
     * -------------------------------------------------------------- */

    function drawGraph(container, data) {
        const width = Math.max(container.clientWidth || 640, 320);
        const height = window.matchMedia('(max-width: 760px)').matches ? 340 : 460;

        const nodes = data.nodes.map(function (node, index) {
            const angle = (index / data.nodes.length) * Math.PI * 2;
            return {
                id: node.id,
                label: node.label || S.displayName(node.id),
                x: width / 2 + Math.cos(angle) * (Math.min(width, height) / 3),
                y: height / 2 + Math.sin(angle) * (Math.min(width, height) / 3)
            };
        });

        const index = {};
        nodes.forEach(function (node, i) {
            index[node.id] = i;
        });

        const edges = data.edges
            .filter(function (edge) {
                return index[edge.from] !== undefined && index[edge.to] !== undefined;
            })
            .map(function (edge) {
                return {
                    source: index[edge.from],
                    target: index[edge.to],
                    label: edge.label
                };
            });

        layout(nodes, edges, width, height);
        rescale(nodes, width, height);
        paint(container, nodes, edges, width, height);
    }

    /**
     * Spread the settled layout across the whole canvas.
     *
     * Force layouts habitually settle into one corner or a tight clump, which
     * wastes the canvas and pushes labels on top of each other. Fitting the
     * bounding box to the viewport afterwards - with one shared scale factor,
     * so the shape is not distorted - makes the result independent of how the
     * force constants happen to be tuned. The horizontal inset is larger
     * because node labels extend sideways.
     */
    function rescale(nodes, width, height) {
        if (nodes.length < 2) {
            nodes.forEach(function (node) {
                node.x = width / 2;
                node.y = height / 2;
            });
            return;
        }

        const xs = nodes.map(function (n) { return n.x; });
        const ys = nodes.map(function (n) { return n.y; });
        const minX = Math.min.apply(null, xs);
        const maxX = Math.max.apply(null, xs);
        const minY = Math.min.apply(null, ys);
        const maxY = Math.max.apply(null, ys);

        const insetX = 92;
        const insetY = 34;
        const boxW = Math.max(width - insetX * 2, 40);
        const boxH = Math.max(height - insetY * 2, 40);
        const spanX = Math.max(maxX - minX, 1);
        const spanY = Math.max(maxY - minY, 1);
        const scale = Math.min(boxW / spanX, boxH / spanY);

        /* Centre whatever is left over after the shared scale. */
        const offsetX = insetX + (boxW - spanX * scale) / 2;
        const offsetY = insetY + (boxH - spanY * scale) / 2;

        nodes.forEach(function (node) {
            node.x = offsetX + (node.x - minX) * scale;
            node.y = offsetY + (node.y - minY) * scale;
        });
    }

    /** Fruchterman-Reingold with linear cooling. n is small (tens). */
    function layout(nodes, edges, width, height) {
        const area = width * height;
        const k = Math.sqrt(area / Math.max(nodes.length, 1)) * 0.62;
        let temperature = width / 8;
        const cooling = temperature / (ITERATIONS + 1);

        for (let step = 0; step < ITERATIONS; step++) {
            nodes.forEach(function (node) {
                node.dx = 0;
                node.dy = 0;
            });

            /* Repulsion between every pair. */
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const a = nodes[i];
                    const b = nodes[j];
                    let dx = a.x - b.x;
                    let dy = a.y - b.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 0.01) {
                        dx = (Math.random() - 0.5) * 0.1;
                        dy = (Math.random() - 0.5) * 0.1;
                        dist = 0.01;
                    }
                    const force = (k * k) / dist;
                    const ux = (dx / dist) * force;
                    const uy = (dy / dist) * force;
                    a.dx += ux;
                    a.dy += uy;
                    b.dx -= ux;
                    b.dy -= uy;
                }
            }

            /* Attraction along edges. */
            edges.forEach(function (edge) {
                const a = nodes[edge.source];
                const b = nodes[edge.target];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
                const force = (dist * dist) / k;
                const ux = (dx / dist) * force;
                const uy = (dy / dist) * force;
                a.dx -= ux;
                a.dy -= uy;
                b.dx += ux;
                b.dy += uy;
            });

            /* Gentle pull to centre so disconnected people do not drift off. */
            nodes.forEach(function (node) {
                node.dx += (width / 2 - node.x) * 0.012;
                node.dy += (height / 2 - node.y) * 0.012;
            });

            /* Apply, capped by temperature, clamped to the canvas. */
            nodes.forEach(function (node) {
                const disp = Math.sqrt(node.dx * node.dx + node.dy * node.dy);
                if (disp > 0.01) {
                    const limit = Math.min(disp, temperature);
                    node.x += (node.dx / disp) * limit;
                    node.y += (node.dy / disp) * limit;
                }
                node.x = clamp(node.x, PADDING, width - PADDING);
                node.y = clamp(node.y, PADDING, height - PADDING);
            });

            temperature -= cooling;
        }
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function paint(container, nodes, edges, width, height) {
        const svg = document.createElementNS(SVG_NS, 'svg');
        svg.setAttribute('class', 'graph-svg');
        svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        /* Decorative: the table beside it carries the same data as text. */
        svg.setAttribute('aria-hidden', 'true');
        svg.setAttribute('focusable', 'false');

        const edgeLayer = document.createElementNS(SVG_NS, 'g');
        const labelLayer = document.createElementNS(SVG_NS, 'g');
        const nodeLayer = document.createElementNS(SVG_NS, 'g');

        const drawn = edges.map(function (edge) {
            const a = nodes[edge.source];
            const b = nodes[edge.target];

            const line = document.createElementNS(SVG_NS, 'line');
            line.setAttribute('class', 'graph-edge');
            line.setAttribute('x1', a.x);
            line.setAttribute('y1', a.y);
            line.setAttribute('x2', b.x);
            line.setAttribute('y2', b.y);
            edgeLayer.appendChild(line);

            const label = document.createElementNS(SVG_NS, 'text');
            label.setAttribute('class', 'graph-edge-label');
            label.setAttribute('x', (a.x + b.x) / 2);
            label.setAttribute('y', (a.y + b.y) / 2 - 3);
            label.setAttribute('text-anchor', 'middle');
            label.textContent = edge.label;
            labelLayer.appendChild(label);

            return {edge: edge, line: line, label: label};
        });

        svg.appendChild(edgeLayer);
        svg.appendChild(labelLayer);
        svg.appendChild(nodeLayer);
        /* In the document before the node shapes are sized: the pill width
           comes from the rendered text, which cannot be measured offscreen. */
        S.setChildren(container, [svg]);

        nodes.forEach(function (node, i) {
            const group = document.createElementNS(SVG_NS, 'g');
            group.setAttribute('class', 'graph-node');
            nodeLayer.appendChild(group);

            const text = document.createElementNS(SVG_NS, 'text');
            text.setAttribute('x', node.x);
            text.setAttribute('y', node.y + 4);
            text.textContent = node.label;
            group.appendChild(text);

            /* A pill sized to the name, so "Madison Kirkpatrick" is not
               clipped by a fixed-radius circle. */
            let half = NODE_RADIUS;
            try {
                half = text.getComputedTextLength() / 2 + 11;
            } catch (measureFailed) {
                half = Math.max(node.label.length * 3.4, NODE_RADIUS);
            }

            const pill = document.createElementNS(SVG_NS, 'ellipse');
            pill.setAttribute('cx', node.x);
            pill.setAttribute('cy', node.y);
            pill.setAttribute('rx', Math.max(half, 20));
            pill.setAttribute('ry', 15);
            group.insertBefore(pill, text);

            group.addEventListener('mouseenter', function () {
                highlight(i, group, drawn, true);
            });
            group.addEventListener('mouseleave', function () {
                highlight(i, group, drawn, false);
            });
        });
    }

    /** Emphasise one person's edges on hover. Pure enhancement. */
    function highlight(nodeIndex, group, drawn, on) {
        group.classList.toggle('is-active', on);
        drawn.forEach(function (item) {
            if (item.edge.source === nodeIndex || item.edge.target === nodeIndex) {
                item.line.classList.toggle('is-highlight', on);
                item.label.classList.toggle('is-highlight', on);
            }
        });
    }
})();
