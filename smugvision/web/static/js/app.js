/**
 * smugVision Web UI - shared helpers.
 *
 * Vanilla JS, no framework, no build step. Loaded on every page by
 * base.html; page-specific logic lives in its own file and reads these
 * off window.smugvision.
 */

/* ------------------------------------------------------------------ *
 * Text helpers
 * ------------------------------------------------------------------ */

/**
 * Escape text for safe interpolation into HTML.
 * Prefer setting .textContent; this is for the few template literals left.
 */
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/** Format an ISO date as a short local date, or '' when absent/unparseable. */
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return '';
    return date.toLocaleDateString();
}

/** Truncate to maxLength characters, adding an ellipsis when cut. */
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) return text || '';
    return text.substring(0, maxLength) + '…';
}

/** "1 image" / "2 images" - pluralize a count with its noun. */
function plural(count, singular, pluralForm) {
    const word = count === 1 ? singular : (pluralForm || singular + 's');
    return count + ' ' + word;
}

/** Turn a reference-face folder name into a display name (John_Doe -> John Doe). */
function displayName(name) {
    return String(name || '').replace(/_/g, ' ');
}

/* ------------------------------------------------------------------ *
 * Fetch helpers
 *
 * Every endpoint in this app reports failure as {"error": "..."} with a
 * real HTTP status, so one error path covers all of them.
 * ------------------------------------------------------------------ */

/**
 * Call a JSON endpoint. Resolves with the parsed body; rejects with an
 * Error whose .status is the HTTP status and .payload the parsed body
 * (regenerate returns the failed image alongside its error, and callers
 * need it to render the card).
 */
async function apiCall(url, options = {}) {
    const opts = Object.assign({}, options);
    opts.headers = Object.assign(
        {'Content-Type': 'application/json'},
        options.headers || {}
    );

    let response;
    try {
        response = await fetch(url, opts);
    } catch (networkError) {
        const err = new Error(
            'Could not reach the smugVision server. Is it still running?'
        );
        err.status = 0;
        throw err;
    }

    let data = null;
    try {
        data = await response.json();
    } catch (parseError) {
        data = null;
    }

    if (!response.ok) {
        const message = (data && data.error)
            ? data.error
            : 'HTTP ' + response.status + ' ' + response.statusText;
        const err = new Error(message);
        err.status = response.status;
        err.payload = data;
        throw err;
    }

    return data;
}

/** GET a JSON endpoint. */
async function apiGet(url) {
    return apiCall(url);
}

/** POST a JSON body. */
async function apiPost(url, body) {
    return apiCall(url, {method: 'POST', body: JSON.stringify(body || {})});
}

/** PUT a JSON body. */
async function apiPut(url, body) {
    return apiCall(url, {method: 'PUT', body: JSON.stringify(body || {})});
}

/* ------------------------------------------------------------------ *
 * DOM helpers
 * ------------------------------------------------------------------ */

/**
 * Create an element.
 * @param {string} tag
 * @param {Object} [attrs] - className, textContent, dataset, aria-*, etc.
 * @param {Array} [children]
 */
function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);

    Object.keys(attrs).forEach(function (key) {
        const value = attrs[key];
        if (value === null || value === undefined || value === false) return;

        if (key === 'className') {
            node.className = value;
        } else if (key === 'text') {
            node.textContent = value;
        } else if (key === 'dataset') {
            Object.keys(value).forEach(function (dataKey) {
                node.dataset[dataKey] = value[dataKey];
            });
        } else if (key === 'html') {
            node.innerHTML = value;
        } else {
            node.setAttribute(key, value === true ? '' : value);
        }
    });

    (Array.isArray(children) ? children : [children]).forEach(function (child) {
        if (child === null || child === undefined || child === false) return;
        node.appendChild(
            typeof child === 'string' ? document.createTextNode(child) : child
        );
    });

    return node;
}

/** Replace all children of a node. */
function setChildren(node, children) {
    node.replaceChildren.apply(
        node,
        (Array.isArray(children) ? children : [children]).filter(Boolean)
    );
}

/**
 * Put a button into its loading state and return a function that restores it.
 * Keeps the accessible label while the visual spinner runs.
 */
function busy(button, busyLabel) {
    const originalLabel = button.textContent;
    const originalDisabled = button.disabled;
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    if (busyLabel) button.setAttribute('aria-label', busyLabel);

    return function restore() {
        button.disabled = originalDisabled;
        button.removeAttribute('aria-busy');
        button.removeAttribute('aria-label');
        button.textContent = originalLabel;
    };
}

/** Announce a message in a live region, styled by kind. */
function announce(node, message, kind) {
    if (!node) return;
    node.textContent = message || '';
    node.classList.remove('error', 'ok');
    if (kind) node.classList.add(kind);
}

/** Debounce a function by `wait` ms. */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(function () {
            func.apply(null, args);
        }, wait);
    };
}

window.smugvision = {
    escapeHtml: escapeHtml,
    formatDate: formatDate,
    truncateText: truncateText,
    plural: plural,
    displayName: displayName,
    apiCall: apiCall,
    apiGet: apiGet,
    apiPost: apiPost,
    apiPut: apiPut,
    el: el,
    setChildren: setChildren,
    busy: busy,
    announce: announce,
    debounce: debounce
};
