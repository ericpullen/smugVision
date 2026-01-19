/**
 * smugVision Web UI - Common JavaScript utilities
 */

// Utility function to escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Utility function to format dates
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Utility function to truncate text
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// API helper with error handling
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error(`API call failed: ${url}`, error);
        throw error;
    }
}

// POST helper
async function apiPost(url, body) {
    return apiCall(url, {
        method: 'POST',
        body: JSON.stringify(body)
    });
}

// GET helper
async function apiGet(url) {
    return apiCall(url);
}

// Show notification (simple alert for now)
function showNotification(message, type = 'info') {
    // Could be enhanced with a toast library later
    if (type === 'error') {
        console.error(message);
    } else {
        console.log(message);
    }
}

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for use in templates (globals)
window.smugvision = {
    escapeHtml,
    formatDate,
    truncateText,
    apiCall,
    apiPost,
    apiGet,
    showNotification,
    debounce
};
