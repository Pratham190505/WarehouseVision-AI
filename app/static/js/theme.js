/**
 * WarehouseVision AI - Theme Manager
 *
 * Manages Light/Dark theme switching with:
 * - localStorage persistence
 * - System preference detection
 * - Smooth transitions
 * - Chart theme updates
 */

const ThemeManager = (() => {
    const STORAGE_KEY = 'warehousevision-theme';

    function init() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            setTheme(saved, false);
        } else {
            // Respect system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            setTheme(prefersDark ? 'dark' : 'light', false);
        }

        // Listen for system preference changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(STORAGE_KEY)) {
                setTheme(e.matches ? 'dark' : 'light', true);
            }
        });
    }

    function setTheme(theme, updateCharts = true) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(STORAGE_KEY, theme);
        updateToggleIcons(theme);
        if (updateCharts && typeof ChartsManager !== 'undefined') {
            ChartsManager.updateTheme();
        }
    }

    function toggle() {
        const current = getTheme();
        const next = current === 'dark' ? 'light' : 'dark';
        setTheme(next, true);
    }

    function getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    }

    function updateToggleIcons(theme) {
        document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
            btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
            btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        });
    }

    function isDark() {
        return getTheme() === 'dark';
    }

    return { init, toggle, getTheme, isDark, setTheme };
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
});
