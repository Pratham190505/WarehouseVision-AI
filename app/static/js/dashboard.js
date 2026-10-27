/**
 * WarehouseVision AI - Dashboard Controller
 *
 * Handles:
 * - File upload with drag & drop
 * - Analysis processing and status
 * - Dashboard data loading from backend
 * - History table population
 * - Insights rendering
 * - URL-backed top navigation
 * - Modal/detail views
 */

const Dashboard = (() => {
    // State
    let selectedFile = null;
    let currentAnalysis = null;
    let isProcessing = false;

    // DOM Elements
    const els = {};

    function init() {
        cacheDom();
        bindEvents();
        checkCurrentUser();
        ChartsManager.initCharts();
        loadDashboardData();
    }

    function cacheDom() {
        els.dropZone = document.getElementById('dropZone');
        els.fileInput = document.getElementById('fileInput');
        els.fileInfo = document.getElementById('fileInfo');
        els.fileName = document.getElementById('fileName');
        els.fileMeta = document.getElementById('fileMeta');
        els.fileRemove = document.getElementById('fileRemove');
        els.analyzeBtn = document.getElementById('analyzeBtn');
        els.chooseBtn = document.getElementById('chooseBtn');
        els.boxVideoInput = document.getElementById('boxVideoInput');
        els.boxRunBtn = document.getElementById('boxRunBtn');
        els.boxTrackingStatus = document.getElementById('boxTrackingStatus');
        els.boxTrackingResults = document.getElementById('boxTrackingResults');

        els.processingStatus = document.getElementById('processingStatus');
        els.statusIcon = document.getElementById('statusIcon');
        els.statusText = document.getElementById('statusText');
        els.statusDetail = document.getElementById('statusDetail');
        els.progressBar = document.getElementById('progressBar');

        els.resultSection = document.getElementById('resultSection');
        els.resultTitle = document.getElementById('resultTitle');
        els.resultMedia = document.getElementById('resultMedia');
        els.resultCounts = document.getElementById('resultCounts');

        els.insightsList = document.getElementById('insightsList');

        els.kpiWorkers = document.getElementById('kpiWorkers');
        els.kpiForklifts = document.getElementById('kpiForklifts');
        els.kpiBoxes = document.getElementById('kpiBoxes');
        els.kpiPallets = document.getElementById('kpiPallets');
        els.kpiUtilization = document.getElementById('kpiUtilization');
        els.kpiWorkersMeta = document.getElementById('kpiWorkersMeta');
        els.kpiForkMeta = document.getElementById('kpiForkMeta');
        els.kpiBoxesMeta = document.getElementById('kpiBoxesMeta');
        els.kpiPalletsMeta = document.getElementById('kpiPalletsMeta');
        els.kpiUtilMeta = document.getElementById('kpiUtilMeta');

        els.analyticsOccupancy = document.getElementById('analyticsOccupancy');
        els.analyticsUtilization = document.getElementById('analyticsUtilization');
        els.analyticsViolations = document.getElementById('analyticsViolations');
        els.analyticsObjects = document.getElementById('analyticsObjects');
        els.barOccupancy = document.getElementById('barOccupancy');
        els.barUtilization = document.getElementById('barUtilization');
        els.barViolations = document.getElementById('barViolations');
        els.barObjects = document.getElementById('barObjects');

        els.historyBody = document.getElementById('historyBody');

        els.mobileNavToggle = document.getElementById('mobileNavToggle');
        els.primaryNavigation = document.getElementById('primaryNavigation');

        els.modalOverlay = document.getElementById('modalOverlay');
        els.modalTitle = document.getElementById('modalTitle');
        els.modalBody = document.getElementById('modalBody');

        els.toastContainer = document.getElementById('toastContainer');

        els.logoutBtn = document.getElementById('logoutBtn');
        els.userNameDisplay = document.getElementById('userNameDisplay');
        els.userEmailDisplay = document.getElementById('userEmailDisplay');
        els.userAvatar = document.getElementById('userAvatar');

        // Settings Elements
        els.headerUserProfile = document.getElementById('headerUserProfile');
        els.settingsCancelBtn = document.getElementById('settingsCancelBtn');
        els.settingsForm = document.getElementById('settingsForm');
        els.settingsNameInput = document.getElementById('settingsNameInput');
        els.settingsDeptInput = document.getElementById('settingsDeptInput');
        els.settingsOrgInput = document.getElementById('settingsOrgInput');
        els.settingsEmailInput = document.getElementById('settingsEmailInput');
        els.settingsRoleBadge = document.getElementById('settingsRoleBadge');
        els.settingsAvatarPreview = document.getElementById('settingsAvatarPreview');
        els.settingsNamePreview = document.getElementById('settingsNamePreview');
        els.settingsEmailPreview = document.getElementById('settingsEmailPreview');
        els.settingsSaveBtn = document.getElementById('settingsSaveBtn');
        els.settingsSaveText = document.getElementById('settingsSaveText');
        els.settingsFormMessage = document.getElementById('settingsFormMessage');
        els.settingsThemeValue = document.getElementById('settingsThemeValue');
        els.settingsSecurityEmail = document.getElementById('settingsSecurityEmail');

        // Safety Elements
        els.activeSafetyAlertsBanner = document.getElementById('activeSafetyAlertsBanner');
        els.navAlertBadge = document.getElementById('navAlertBadge');
        els.kpiSafetyActive = document.getElementById('kpiSafetyActive');
        els.kpiSafetyActiveMeta = document.getElementById('kpiSafetyActiveMeta');
        els.kpiSafetyTotal = document.getElementById('kpiSafetyTotal');
        els.kpiSafetyResolved = document.getElementById('kpiSafetyResolved');
        els.kpiSafetyMode = document.getElementById('kpiSafetyMode');
        els.kpiSafetyModeMeta = document.getElementById('kpiSafetyModeMeta');
        els.safetyEventsBody = document.getElementById('safetyEventsBody');
        els.refreshSafetyBtn = document.getElementById('refreshSafetyBtn');
    }

    function bindEvents() {
        // Upload
        if (els.dropZone) {
            els.dropZone.addEventListener('click', () => els.fileInput?.click());
            els.dropZone.addEventListener('dragover', handleDragOver);
            els.dropZone.addEventListener('dragleave', handleDragLeave);
            els.dropZone.addEventListener('drop', handleDrop);
        }
        if (els.fileInput) {
            els.fileInput.addEventListener('change', handleFileSelect);
        }
        if (els.fileRemove) {
            els.fileRemove.addEventListener('click', removeFile);
        }
        if (els.analyzeBtn) {
            els.analyzeBtn.addEventListener('click', startAnalysis);
        }
        els.boxVideoInput?.addEventListener('change', () => {
            els.boxRunBtn.disabled = !els.boxVideoInput.files?.length || isProcessing;
        });
        els.boxRunBtn?.addEventListener('click', runBoxTracking);
        if (els.chooseBtn) {
            els.chooseBtn.addEventListener('click', () => els.fileInput?.click());
        }
        if (els.refreshSafetyBtn) {
            els.refreshSafetyBtn.addEventListener('click', () => {
                loadSafetyData();
                showToast('Safety alerts refreshed.', 'info');
            });
        }

        // Shared top navigation and URL-backed views
        els.mobileNavToggle?.addEventListener('click', () => {
            const isOpen = els.primaryNavigation?.classList.toggle('open') || false;
            els.mobileNavToggle.setAttribute('aria-expanded', String(isOpen));
            els.mobileNavToggle.setAttribute('aria-label', isOpen ? 'Close navigation menu' : 'Open navigation menu');
        });
        document.querySelectorAll('.nav-link[data-route], .brand[data-route]').forEach(link => {
            link.addEventListener('click', event => {
                if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                event.preventDefault();
                navigateToRoute(link.dataset.route);
            });
        });
        window.addEventListener('popstate', () => renderRoute(window.location.pathname));
        renderRoute(window.location.pathname);

        // Settings page and profile navigation
        if (els.headerUserProfile) {
            els.headerUserProfile.addEventListener('click', () => navigateToRoute('/settings'));
            els.headerUserProfile.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    navigateToRoute('/settings');
                }
            });
        }
        els.settingsCancelBtn?.addEventListener('click', () => {
            if (currentProfileUser) populateSettingsFields(currentProfileUser);
            clearSettingsMessage();
        });
        els.settingsForm?.addEventListener('submit', handleSaveSettings);
        els.settingsForm?.addEventListener('input', clearSettingsMessage);
        window.addEventListener('warehousevision-theme-change', renderSettingsTheme);
        renderSettingsTheme();

        // Theme toggles
        document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => ThemeManager.toggle());
        });

        // Modal close
        if (els.modalOverlay) {
            els.modalOverlay.addEventListener('click', (e) => {
                if (e.target === els.modalOverlay) closeModal();
            });
        }
        const modalClose = document.getElementById('modalClose');
        if (modalClose) {
            modalClose.addEventListener('click', closeModal);
        }

        // Logout
        if (els.logoutBtn) {
            els.logoutBtn.addEventListener('click', handleLogout);
        }

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
                els.primaryNavigation?.classList.remove('open');
                els.mobileNavToggle?.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // ===== Settings & Profile Logic =====
    let currentProfileUser = null;

    function renderSettingsTheme() {
        if (!els.settingsThemeValue || typeof ThemeManager === 'undefined') return;
        const theme = ThemeManager.getTheme();
        els.settingsThemeValue.textContent = theme === 'dark' ? 'Dark' : 'Light';
    }

    function showSettingsMessage(message, type = 'success') {
        if (!els.settingsFormMessage) return;
        els.settingsFormMessage.textContent = message;
        els.settingsFormMessage.className = `settings-message settings-message-${type}`;
        els.settingsFormMessage.hidden = false;
        els.settingsFormMessage.setAttribute('role', type === 'error' ? 'alert' : 'status');
    }

    function clearSettingsMessage() {
        if (!els.settingsFormMessage) return;
        els.settingsFormMessage.textContent = '';
        els.settingsFormMessage.hidden = true;
        els.settingsFormMessage.className = 'settings-message';
        els.settingsFormMessage.setAttribute('role', 'status');
    }

    function populateSettingsFields(user) {
        if (!user) return;
        currentProfileUser = user;
        if (els.settingsNameInput) els.settingsNameInput.value = user.name || '';
        if (els.settingsDeptInput) els.settingsDeptInput.value = user.department || '';
        if (els.settingsOrgInput) els.settingsOrgInput.value = user.organization || '';
        if (els.settingsEmailInput) els.settingsEmailInput.value = user.email || '';
        if (els.settingsRoleBadge) els.settingsRoleBadge.textContent = user.role || 'user';
        if (els.settingsSecurityEmail) els.settingsSecurityEmail.textContent = user.email || '';
        if (els.settingsNamePreview) els.settingsNamePreview.textContent = user.name || 'Warehouse User';
        if (els.settingsEmailPreview) els.settingsEmailPreview.textContent = user.email || '';
        if (els.settingsAvatarPreview) els.settingsAvatarPreview.textContent = (user.name || 'VO').slice(0, 2).toUpperCase();
    }

    async function handleSaveSettings(e) {
        e.preventDefault();
        const name = (els.settingsNameInput?.value || '').trim();
        const department = (els.settingsDeptInput?.value || '').trim();
        const organization = (els.settingsOrgInput?.value || '').trim();

        clearSettingsMessage();
        if (!name || name.length < 2 || name.length > 100) {
            const message = 'Display name must contain between 2 and 100 characters.';
            showSettingsMessage(message, 'error');
            showToast(message, 'error');
            els.settingsNameInput?.focus();
            return;
        }

        const originalBtnText = els.settingsSaveText ? els.settingsSaveText.textContent : 'Save Changes';
        if (els.settingsSaveBtn) els.settingsSaveBtn.disabled = true;
        if (els.settingsSaveText) els.settingsSaveText.textContent = 'Saving...';

        try {
            const resp = await fetch('/api/auth/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, department, organization }),
            });

            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                throw new Error(data.detail || 'Failed to update profile.');
            }

            const updated = data.user || { name, department, organization };
            currentProfileUser = updated;

            // Immediately update the Dashboard UI
            if (els.userNameDisplay) els.userNameDisplay.textContent = updated.name;
            if (els.userAvatar) els.userAvatar.textContent = (updated.name || 'VO').slice(0, 2).toUpperCase();
            if (els.userEmailDisplay) {
                els.userEmailDisplay.textContent = updated.organization 
                    ? `${updated.organization} · ${updated.email || ''}`
                    : (updated.email || '');
            }

            populateSettingsFields(updated);
            showSettingsMessage('Profile changes saved successfully.', 'success');
            showToast('Profile updated successfully!', 'success');
        } catch (err) {
            console.error('Settings save error:', err);
            const message = err.message || 'Could not update profile.';
            showSettingsMessage(message, 'error');
            showToast(message, 'error');
        } finally {
            if (els.settingsSaveBtn) els.settingsSaveBtn.disabled = false;
            if (els.settingsSaveText) els.settingsSaveText.textContent = originalBtnText;
        }
    }

    // ===== Authentication & User State =====
    async function checkCurrentUser() {
        try {
            const res = await fetch('/api/auth/me');
            if (res.status === 401) {
                window.location.href = '/login';
                return;
            }
            if (res.ok) {
                const user = await res.json();
                currentProfileUser = user;
                if (els.userNameDisplay && user.name) els.userNameDisplay.textContent = user.name;
                if (els.userEmailDisplay && user.email) {
                    els.userEmailDisplay.textContent = user.organization 
                        ? `${user.organization} · ${user.email}`
                        : user.email;
                }
                if (els.userAvatar && user.name) {
                    els.userAvatar.textContent = user.name.slice(0, 2).toUpperCase();
                }
                populateSettingsFields(user);
            }
        } catch (e) {
            console.warn('Current user check failed:', e);
        }
    }

    async function handleLogout() {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
        } finally {
            window.location.href = '/login';
        }
    }

    // ===== File Upload =====
    function handleDragOver(e) {
        e.preventDefault();
        els.dropZone.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        els.dropZone.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        els.dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            setFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            setFile(files[0]);
        }
    }

    function setFile(file) {
        const validExts = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.wmv'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!validExts.includes(ext)) {
            showToast('Unsupported file type. Use JPG, PNG, MP4, AVI, MOV, or MKV.', 'error');
            return;
        }

        selectedFile = file;
        const isVideo = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'].includes(ext);
        const icon = isVideo ? '🎬' : '🖼️';
        const type = isVideo ? 'Video' : 'Image';

        if (els.fileName) els.fileName.textContent = file.name;
        if (els.fileMeta) els.fileMeta.textContent = `${type} · ${formatFileSize(file.size)}`;
        if (els.fileInfo) els.fileInfo.classList.add('visible');

        const fileIcon = els.fileInfo?.querySelector('.file-icon');
        if (fileIcon) fileIcon.textContent = icon;

        if (els.analyzeBtn) els.analyzeBtn.disabled = false;
    }

    function removeFile() {
        selectedFile = null;
        if (els.fileInfo) els.fileInfo.classList.remove('visible');
        if (els.fileInput) els.fileInput.value = '';
        if (els.analyzeBtn) els.analyzeBtn.disabled = true;
    }

    let boxTrackingAnalysisId = null;

    async function runBoxTracking() {
        const file = els.boxVideoInput?.files?.[0];
        if (!file || isProcessing) return;
        isProcessing = true;
        els.boxRunBtn.disabled = true;
        els.boxTrackingStatus.textContent = 'Uploading recorded video…';
        els.boxTrackingResults.innerHTML = '';
        try {
            const form = new FormData();
            form.append('file', file);
            const uploadResponse = await fetch('/api/upload', { method: 'POST', body: form });
            const uploaded = await uploadResponse.json();
            if (!uploadResponse.ok) throw new Error(uploaded.detail || 'Video upload failed.');
            if (uploaded.media_type !== 'video') throw new Error('Choose a recorded video file.');

            els.boxTrackingStatus.textContent = 'Video uploaded. Detecting and tracking boxes… This can take a while for long videos.';
            const response = await fetch(`/api/box-tracking/${encodeURIComponent(uploaded.analysis_id)}/analyze`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Box tracking failed.');
            boxTrackingAnalysisId = data.analysis_id;
            renderBoxTrackingResults(data);
            els.boxTrackingStatus.textContent = 'Processing complete.';
            loadDashboardData();
        } catch (error) {
            els.boxTrackingStatus.textContent = error.message;
            showToast(error.message, 'error');
        } finally {
            isProcessing = false;
            els.boxRunBtn.disabled = !els.boxVideoInput?.files?.length;
        }
    }

    function renderBoxTrackingResults(data) {
        const result = data?.box_tracking;
        if (!result || !els.boxTrackingResults) return;
        boxTrackingAnalysisId = data.analysis_id || boxTrackingAnalysisId;
        const tracks = result.confirmed_tracks || [];
        const ids = tracks.slice(0, 30).map(track =>
            `<li><strong>${escapeHtml(track.box_id)}</strong> <span>${Number(track.average_confidence || 0).toFixed(2)} confidence</span></li>`
        ).join('');
        const sourceWarning = result.box_detection_source === 'COCO backpack/suitcase proxies'
            ? '<p class="box-tracking-note box-model-warning">This model detects backpack/suitcase proxies, not cardboard boxes. Counts are approximate.</p>'
            : result.box_detection_source === 'no dedicated box class configured'
                ? '<p class="box-tracking-note box-model-warning">The configured model has no dedicated box class. Configure BOX_CLASS_IDS or a model labeled for boxes.</p>'
                : '';
        els.boxTrackingResults.innerHTML = `
            <div class="box-count-summary">
                <div class="box-count-primary"><strong>${Number(result.total_unique_boxes || 0)}</strong><span>Total unique boxes tracked</span></div>
                <div><strong>${Number(result.currently_visible_boxes || 0)}</strong><span>Currently visible boxes</span></div>
            </div>
            <div class="box-output-video"><video controls playsinline preload="metadata" src="${escapeHtml(data.result_path || '')}" poster="${escapeHtml(data.thumbnail_path || '')}"></video></div>
            <h3 class="box-tracking-list-title">Confirmed box IDs${tracks.length > 30 ? ` (first 30 of ${tracks.length})` : ''}</h3>
            <ul class="box-confirmed-list">${ids || '<li>No tracks met the confirmation threshold.</li>'}</ul>
            ${sourceWarning}
            ${result.untracked_box_detections ? `<p class="box-tracking-note">${Number(result.untracked_box_detections)} box detections had no tracker ID and were not counted.</p>` : ''}`;
    }
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // ===== Analysis =====
    async function startAnalysis() {
        if (!selectedFile || isProcessing) return;

        isProcessing = true;
        els.analyzeBtn.disabled = true;

        // Show processing status
        updateStatus('uploading', '⬆️', 'Uploading...', `Uploading ${selectedFile.name}`);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Upload + analyze in one step
            updateStatus('processing', '⚙️', 'Processing...', 'Analyzing warehouse media with AI...');

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const result = await response.json();

            if (result.status === 'failed') {
                updateStatus('failed', '❌', 'Analysis Failed', result.error_message || 'Unknown error');
                showToast('Analysis failed. Please try again.', 'error');
            } else {
                updateStatus('completed', '✅', 'Analysis Complete',
                    `Processed in ${result.processing_time || 0}s`);
                showToast('Analysis completed successfully!', 'success');

                currentAnalysis = result;
                if (result.analysis_id) {
                    localStorage.setItem('wv_active_analysis_id', result.analysis_id);
                }
                renderAnalysisResult(result);
            }
        } catch (err) {
            console.error('Analysis error:', err);
            updateStatus('failed', '❌', 'Error', err.message || 'An error occurred');
            showToast(err.message || 'Analysis failed', 'error');
        } finally {
            isProcessing = false;
            els.analyzeBtn.disabled = false;
            removeFile();
            loadDashboardData(); // Refresh history and stats
        }
    }

    function updateStatus(state, icon, text, detail) {
        if (els.processingStatus) els.processingStatus.classList.add('visible');
        if (els.statusIcon) els.statusIcon.textContent = icon;
        if (els.statusText) els.statusText.textContent = text;
        if (els.statusDetail) els.statusDetail.textContent = detail;

        if (els.progressBar) {
            els.progressBar.classList.remove('indeterminate');
            switch (state) {
                case 'uploading':
                    els.progressBar.style.width = '20%';
                    break;
                case 'processing':
                    els.progressBar.classList.add('indeterminate');
                    els.progressBar.style.width = '30%';
                    break;
                case 'completed':
                    els.progressBar.classList.remove('indeterminate');
                    els.progressBar.style.width = '100%';
                    break;
                case 'failed':
                    els.progressBar.classList.remove('indeterminate');
                    els.progressBar.style.width = '100%';
                    els.progressBar.style.background = 'var(--danger)';
                    setTimeout(() => {
                        els.progressBar.style.background = '';
                    }, 3000);
                    break;
            }
        }
    }

    // ===== Render Results =====
    function renderAnalysisResult(data) {
        if (data.box_tracking && data.analysis_id) {
            boxTrackingAnalysisId = data.analysis_id;
            renderBoxTrackingResults(data);
        }
        const counts = data.object_counts || {};
        const analytics = data.analytics || {};

        // KPI Cards
        updateKPI('kpiWorkers', counts.workers, 'kpiWorkersMeta');
        updateKPI('kpiForklifts', counts.forklifts, 'kpiForkMeta');
        updateKPI('kpiBoxes', counts.boxes, 'kpiBoxesMeta');
        updateKPI('kpiPallets', counts.pallets, 'kpiPalletsMeta');

        const util = analytics.space_utilization || 0;
        if (els.kpiUtilization) els.kpiUtilization.textContent = util > 0 ? `${util}%` : '--';
        if (els.kpiUtilMeta) els.kpiUtilMeta.textContent = util > 0 ? 'Latest analysis' : 'No data yet';

        // Analytics cards
        const occ = analytics.occupancy || 0;
        const violations = analytics.restricted_zone_violations || 0;
        const totalObj = (counts.workers || 0) + (counts.forklifts || 0) + (counts.boxes || 0) + (counts.pallets || 0);

        if (els.analyticsOccupancy) els.analyticsOccupancy.textContent = occ > 0 ? `${occ}%` : '--';
        if (els.analyticsUtilization) els.analyticsUtilization.textContent = util > 0 ? `${util}%` : '--';
        if (els.analyticsViolations) els.analyticsViolations.textContent = violations.toString();
        if (els.analyticsObjects) els.analyticsObjects.textContent = totalObj.toString();

        if (els.barOccupancy) els.barOccupancy.style.width = `${Math.min(occ, 100)}%`;
        if (els.barUtilization) els.barUtilization.style.width = `${Math.min(util, 100)}%`;
        if (els.barViolations) els.barViolations.style.width = violations > 0 ? '100%' : '0%';
        if (els.barObjects) els.barObjects.style.width = totalObj > 0 ? `${Math.min((totalObj / 100) * 100, 100)}%` : '0%';

        // Charts
        ChartsManager.updateAll(data);

        // Detection result
        if (data.result_path) {
            showResultMedia(data);
        }

        // Insights
        renderInsights(analytics.insights || []);

        // Machine Proximity Safety Alerts
        if (data.safety_alerts && data.safety_alerts.length > 0) {
            renderActiveSafetyBanner(data.safety_alerts);
            showToast(`🚨 Machine Proximity Alert: ${data.safety_alerts.length} danger breach detected!`, 'error');
        }
        loadSafetyData();
    }

    function updateKPI(elId, value, metaId) {
        const el = els[elId];
        const meta = els[metaId];
        if (el) el.textContent = (value !== undefined && value !== null) ? value : '--';
        if (meta) meta.textContent = (value !== undefined && value !== null && value > 0) ? 'Latest analysis' : 'No data yet';
    }

    function showResultMedia(data) {
        if (!els.resultSection) return;
        els.resultSection.classList.add('visible');

        const isVideo = (data.media_type === 'video') || (data.result_path && data.result_path.includes('/videos/'));
        const fileName = data.original_filename || data.filename || '';

        if (els.resultTitle) {
            els.resultTitle.textContent = isVideo 
                ? `Video Analysis & Tracking ${fileName ? '— ' + fileName : ''}`
                : `Detection Result ${fileName ? '— ' + fileName : ''}`;
        }

        if (els.resultMedia) {
            if (isVideo) {
                const videoSrc = data.result_path || '';
                const posterSrc = data.thumbnail_path || '';
                els.resultMedia.innerHTML = `
                    <div style="position:relative;width:100%;background:#090D16;border-radius:var(--radius-md);overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.25);">
                        <video id="mainResultVideo" controls playsinline preload="auto" ${posterSrc ? `poster="${posterSrc}"` : ''} style="width:100%;max-height:600px;display:block;">
                            <source src="${videoSrc}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div style="padding:10px 14px;background:var(--bg-input);border-top:1px solid var(--border-color);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                            <span style="font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;">
                                <span>📹</span> Processed Video with Bounding Boxes & Object Tracking
                            </span>
                            <a href="${videoSrc}" download class="view-btn" style="text-decoration:none;padding:5px 12px;font-size:12px;display:inline-flex;align-items:center;gap:4px;">
                                <span>⬇️</span> Download Video
                            </a>
                        </div>
                    </div>
                `;
            } else {
                els.resultMedia.innerHTML = `
                    <div style="position:relative;width:100%;border-radius:var(--radius-md);overflow:hidden;">
                        <img src="${data.result_path}" alt="Detection result" style="width:100%;border-radius:var(--radius-md);display:block;">
                        <div style="padding:10px 14px;background:var(--bg-input);border:1px solid var(--border-color);border-top:none;border-radius:0 0 var(--radius-md) var(--radius-md);display:flex;align-items:center;justify-content:space-between;">
                            <span style="font-size:12px;color:var(--text-secondary);">Annotated Image with AI Detections</span>
                            <a href="${data.result_path}" download class="view-btn" style="text-decoration:none;padding:5px 12px;font-size:12px;display:inline-flex;align-items:center;gap:4px;">
                                <span>⬇️</span> Download Image
                            </a>
                        </div>
                    </div>
                `;
            }
        }

        if (els.resultCounts) {
            const counts = data.object_counts || {};
            let badgesHtml = Object.entries(counts)
                .map(([key, val]) => `
                    <div class="result-count-badge">
                        <span class="count-label">${capitalize(key)}</span>
                        <span class="count-value">${val}</span>
                    </div>
                `).join('');

            // If video tracking data exists, add tracking badges
            if (data.tracking) {
                const tr = data.tracking;
                badgesHtml += `
                    <div class="result-count-badge" style="border-left:3px solid var(--info);">
                        <span class="count-label">Unique Workers</span>
                        <span class="count-value">${tr.unique_workers ?? 0}</span>
                    </div>
                    <div class="result-count-badge" style="border-left:3px solid var(--warning);">
                        <span class="count-label">Unique Forklifts</span>
                        <span class="count-value">${tr.unique_forklifts ?? 0}</span>
                    </div>
                    <div class="result-count-badge" style="border-left:3px solid var(--success);">
                        <span class="count-label">Frames</span>
                        <span class="count-value">${tr.processed_frames ?? 0} / ${tr.total_frames ?? 0}</span>
                    </div>
                `;
            }

            // If safety breaches exist, add safety badge
            if ((data.safety_incidents && data.safety_incidents.length > 0) || (data.safety_alerts && data.safety_alerts.length > 0)) {
                const sCount = (data.safety_incidents ? data.safety_incidents.length : 0) || (data.safety_alerts ? data.safety_alerts.length : 0);
                badgesHtml += `
                    <div class="result-count-badge" style="border-left:3px solid var(--danger);background:var(--danger-alert-badge);">
                        <span class="count-label" style="color:var(--danger-alert-badge-text);font-weight:700;">⚠️ Safety Proximity Alert</span>
                        <span class="count-value" style="color:var(--danger);font-weight:800;">${sCount} Hazard${sCount > 1 ? 's' : ''}</span>
                    </div>
                `;
            }

            els.resultCounts.innerHTML = badgesHtml;
        }
    }

    function renderInsights(insights) {
        if (!els.insightsList) return;
        if (!insights || insights.length === 0) {
            els.insightsList.innerHTML = '<div class="insight-empty">Upload and analyze media to see insights.</div>';
            return;
        }
        els.insightsList.innerHTML = insights.map(text => `
            <li class="insight-item">
                <span class="insight-icon">${text.includes('⚠️') ? '⚠️' : '💡'}</span>
                <span>${escapeHtml(text.replace('⚠️ ', ''))}</span>
            </li>
        `).join('');
    }

    // ===== Dashboard Data =====
    async function loadDashboardData() {
        let stats = {};
        let history = [];

        try {
            const response = await fetch('/api/analytics');
            if (response.status === 401) {
                window.location.href = '/login';
                return;
            }
            if (response.ok) {
                const data = await response.json();
                stats = data.stats || {};
                history = data.history || [];
            } else {
                console.error('Analytics fetch returned non-200:', response.status);
            }
        } catch (err) {
            console.error('Could not fetch analytics:', err);
        }

        // Update KPIs from latest analysis
        const latest = stats.latest_analysis;
        if (latest && latest.status === 'completed') {
            const counts = latest.object_counts || {};
            const analytics = latest.analytics || {};

            updateKPI('kpiWorkers', counts.workers, 'kpiWorkersMeta');
            updateKPI('kpiForklifts', counts.forklifts, 'kpiForkMeta');
            updateKPI('kpiBoxes', counts.boxes, 'kpiBoxesMeta');
            updateKPI('kpiPallets', counts.pallets, 'kpiPalletsMeta');

            const util = analytics.space_utilization || 0;
            if (els.kpiUtilization) els.kpiUtilization.textContent = util > 0 ? `${util}%` : '--';
            if (els.kpiUtilMeta) els.kpiUtilMeta.textContent = util > 0 ? 'Latest analysis' : 'No data yet';

            // Update analytics cards
            const occ = analytics.occupancy || 0;
            const violations = analytics.restricted_zone_violations || 0;
            const totalObj = (counts.workers || 0) + (counts.forklifts || 0) + (counts.boxes || 0) + (counts.pallets || 0);

            if (els.analyticsOccupancy) els.analyticsOccupancy.textContent = occ > 0 ? `${occ}%` : '--';
            if (els.analyticsUtilization) els.analyticsUtilization.textContent = util > 0 ? `${util}%` : '--';
            if (els.analyticsViolations) els.analyticsViolations.textContent = violations.toString();
            if (els.analyticsObjects) els.analyticsObjects.textContent = totalObj.toString();

            if (els.barOccupancy) els.barOccupancy.style.width = `${Math.min(occ, 100)}%`;
            if (els.barUtilization) els.barUtilization.style.width = `${Math.min(util, 100)}%`;
            if (els.barViolations) els.barViolations.style.width = violations > 0 ? '100%' : '0%';
            if (els.barObjects) els.barObjects.style.width = totalObj > 0 ? `${Math.min((totalObj / 100) * 100, 100)}%` : '0%';

            // Update charts
            ChartsManager.updateAll({ object_counts: counts, analytics: analytics });

            // Update insights
            renderInsights(analytics.insights || []);
        }

        // Restore active analysis on page refresh
        let analysisToDisplay = null;
        const activeId = localStorage.getItem('wv_active_analysis_id');

        if (activeId) {
            try {
                const activeResp = await fetch(`/api/analysis/${activeId}`);
                if (activeResp.ok) {
                    const fetched = await activeResp.json();
                    if (fetched && fetched.status === 'completed') {
                        analysisToDisplay = fetched;
                    }
                }
            } catch (e) {
                console.warn('Could not restore saved analysis:', e);
            }
        }

        // Fallback to latest analysis if none restored from localStorage
        if (!analysisToDisplay && stats.latest_analysis && stats.latest_analysis.status === 'completed') {
            analysisToDisplay = stats.latest_analysis;
        }

        // Display restored analysis in the results view
        if (analysisToDisplay) {
            currentAnalysis = analysisToDisplay;
            renderAnalysisResult(analysisToDisplay);
        }

        // Render history
        renderHistory(history);

        // Load machine safety events & alerts
        loadSafetyData();
    }

    function renderHistory(history) {
        if (!els.historyBody) return;

        if (!history || history.length === 0) {
            els.historyBody.innerHTML = `
                <tr>
                    <td colspan="9">
                        <div class="history-empty">
                            <div class="empty-icon">📋</div>
                            <div>No analysis history yet. Upload and analyze media to get started.</div>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        els.historyBody.innerHTML = history.map(record => {
            const counts = record.object_counts || {};
            const status = record.status || 'pending';
            const date = record.timestamp ? formatDate(record.timestamp) : '--';
            const filename = record.original_filename || record.filename || '--';
            const type = (record.media_type || 'unknown').charAt(0).toUpperCase() + (record.media_type || 'unknown').slice(1);

            return `
                <tr>
                    <td title="${escapeHtml(filename)}">${truncate(filename, 25)}</td>
                    <td>${type}</td>
                    <td>${date}</td>
                    <td><span class="status-badge ${status}">${capitalize(status)}</span></td>
                    <td>${counts.workers ?? '--'}</td>
                    <td>${counts.forklifts ?? '--'}</td>
                    <td>${counts.boxes ?? '--'}</td>
                    <td>${counts.pallets ?? '--'}</td>
                    <td>
                        ${status === 'completed'
                            ? `<button class="view-btn" onclick="Dashboard.viewAnalysis('${record.analysis_id}')">View</button>`
                            : '--'
                        }
                    </td>
                </tr>
            `;
        }).join('');
    }

    async function viewAnalysis(analysisId) {
        try {
            showToast('Loading analysis...', 'info');
            const resp = await fetch(`/api/analysis/${analysisId}`);
            if (!resp.ok) {
                showToast('Could not load analysis details.', 'error');
                return;
            }
            const data = await resp.json();

            // Store as active analysis so refresh preserves this selection
            currentAnalysis = data;
            localStorage.setItem('wv_active_analysis_id', analysisId);

            // Update main dashboard view (KPIs, charts, and Detection Results section)
            renderAnalysisResult(data);

            // Show in modal
            const fileName = data.original_filename || data.filename || analysisId;
            if (els.modalTitle) els.modalTitle.textContent = `Analysis: ${fileName}`;

            let html = '<div style="display:flex;flex-direction:column;gap:var(--space-md);">';

            const isVideo = (data.media_type === 'video') || (data.result_path && data.result_path.includes('/videos/'));

            // Result image/video
            if (data.result_path) {
                if (isVideo) {
                    const posterSrc = data.thumbnail_path || '';
                    html += `
                        <div style="background:#090D16;border-radius:var(--radius-md);overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.3);">
                            <video id="modalVideoPlayer" controls playsinline preload="auto" ${posterSrc ? `poster="${posterSrc}"` : ''} style="width:100%;max-height:500px;display:block;">
                                <source src="${data.result_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    `;
                } else {
                    html += `<img src="${data.result_path}" alt="Result" style="width:100%;border-radius:var(--radius-md);display:block;box-shadow:0 4px 14px rgba(0,0,0,0.1);">`;
                }
            }

            // Counts
            const counts = data.object_counts || {};
            html += `<div style="display:flex;gap:var(--space-sm);flex-wrap:wrap;">`;
            for (const [key, val] of Object.entries(counts)) {
                html += `<div class="result-count-badge"><span class="count-label">${capitalize(key)}</span><span class="count-value">${val}</span></div>`;
            }
            if (data.tracking) {
                const tr = data.tracking;
                html += `
                    <div class="result-count-badge" style="border-left:3px solid var(--info);"><span class="count-label">Unique Workers</span><span class="count-value">${tr.unique_workers ?? 0}</span></div>
                    <div class="result-count-badge" style="border-left:3px solid var(--warning);"><span class="count-label">Unique Forklifts</span><span class="count-value">${tr.unique_forklifts ?? 0}</span></div>
                    <div class="result-count-badge" style="border-left:3px solid var(--success);"><span class="count-label">Frames</span><span class="count-value">${tr.processed_frames ?? 0} / ${tr.total_frames ?? 0}</span></div>
                `;
            }
            html += `</div>`;

            // Analytics
            const analytics = data.analytics || {};
            html += `
                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:var(--space-sm);margin-top:var(--space-xs);">
                    <div class="kpi-card" style="padding:10px 14px;">
                        <div class="kpi-label">Occupancy</div>
                        <div class="kpi-value" style="font-size:18px;">${analytics.occupancy ?? 0}%</div>
                    </div>
                    <div class="kpi-card" style="padding:10px 14px;">
                        <div class="kpi-label">Space Utilization</div>
                        <div class="kpi-value" style="font-size:18px;">${analytics.space_utilization ?? 0}%</div>
                    </div>
                    <div class="kpi-card" style="padding:10px 14px;">
                        <div class="kpi-label">Zone Violations</div>
                        <div class="kpi-value" style="font-size:18px;color:${(analytics.restricted_zone_violations || 0) > 0 ? 'var(--danger)' : 'var(--success)'};">${analytics.restricted_zone_violations ?? 0}</div>
                    </div>
                </div>
            `;

            if (analytics.insights && analytics.insights.length > 0) {
                html += `<div><strong style="color:var(--text-primary);font-size:13px;">Key Insights:</strong><ul class="insight-list" style="margin-top:var(--space-xs);">`;
                analytics.insights.forEach(i => {
                    html += `<li class="insight-item"><span class="insight-icon">${i.includes('⚠️') ? '⚠️' : '💡'}</span><span>${escapeHtml(i.replace('⚠️ ', ''))}</span></li>`;
                });
                html += `</ul></div>`;
            }

            // Meta & Actions
            html += `
                <div style="display:flex;align-items:center;justify-content:space-between;padding-top:var(--space-sm);border-top:1px solid var(--border-color);flex-wrap:wrap;gap:var(--space-sm);">
                    <div style="font-size:var(--font-size-xs);color:var(--text-tertiary);">
                        ⏱️ ${data.processing_time || 0}s ${data.timestamp ? '· 📅 ' + formatDate(data.timestamp) : ''}
                    </div>
                    <div style="display:flex;gap:var(--space-xs);">
                        ${data.result_path ? `<a href="${data.result_path}" download class="view-btn" style="text-decoration:none;">⬇️ Download Media</a>` : ''}
                        <button class="view-btn" onclick="Dashboard.closeModal(); Dashboard.scrollToSection('resultSection');" style="background:var(--primary);color:#fff;border-color:var(--primary);">🎯 View on Dashboard</button>
                    </div>
                </div>
            `;

            html += '</div>';

            if (els.modalBody) els.modalBody.innerHTML = html;
            openModal();

            // Auto-load modal video if present
            setTimeout(() => {
                const vid = document.getElementById('modalVideoPlayer');
                if (vid) vid.load();
            }, 60);

            showToast(`Loaded analysis: ${fileName}`, 'success');
        } catch (err) {
            console.error('viewAnalysis error:', err);
            showToast('Failed to load analysis.', 'error');
        }
    }

    // ===== URL-backed application views =====
    const routeViews = {
        '/dashboard': 'dashboard',
        '/analysis': 'analysis',
        '/safety': 'safety',
        '/box-counting': 'box-counting',
        '/settings': 'settings',
    };
    const viewRoutes = Object.fromEntries(Object.entries(routeViews).map(([route, view]) => [view, route]));

    function renderRoute(pathname) {
        const path = routeViews[pathname] ? pathname : '/dashboard';
        const activeView = routeViews[path];
        document.querySelectorAll('[data-view]').forEach(element => {
            element.hidden = element.dataset.view !== activeView;
        });
        document.querySelectorAll('.nav-link[data-route]').forEach(link => {
            const active = link.dataset.route === path;
            link.classList.toggle('active', active);
            if (active) link.setAttribute('aria-current', 'page');
            else link.removeAttribute('aria-current');
        });
        els.primaryNavigation?.classList.remove('open');
        if (els.mobileNavToggle) {
            els.mobileNavToggle.setAttribute('aria-expanded', 'false');
            els.mobileNavToggle.setAttribute('aria-label', 'Open navigation menu');
        }
        if (window.location.pathname !== path) window.history.replaceState({}, '', path);
        if (activeView === 'dashboard' && typeof Chart !== 'undefined') {
            requestAnimationFrame(() => {
                document.querySelectorAll('canvas').forEach(canvas => Chart.getChart(canvas)?.resize());
            });
        }
    }

    function navigateToRoute(path) {
        if (!routeViews[path]) return;
        if (window.location.pathname !== path) window.history.pushState({}, '', path);
        renderRoute(path);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function scrollToSection(id) {
        const el = document.getElementById(id);
        if (!el) return;
        const view = el.dataset.view;
        const route = viewRoutes[view];
        if (route && route !== window.location.pathname) navigateToRoute(route);
        requestAnimationFrame(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    }

    // ===== Modal =====
    function openModal() {
        els.modalOverlay?.classList.add('visible');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        els.modalOverlay?.classList.remove('visible');
        document.body.style.overflow = '';
    }

    // ===== Toast =====
    function showToast(message, type = 'info') {
        if (!els.toastContainer) return;

        const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
            <span class="toast-message">${escapeHtml(message)}</span>
        `;

        els.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ===== Utilities =====
    function capitalize(s) {
        return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
    }

    function truncate(s, max) {
        return s && s.length > max ? s.substring(0, max) + '...' : s;
    }

    function escapeHtml(s) {
        const div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
    }

    function formatDate(isoStr) {
        try {
            const d = new Date(isoStr);
            return d.toLocaleDateString('en-US', { day: '2-digit', month: 'short', year: 'numeric' }) +
                ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        } catch {
            return isoStr;
        }
    }

    // ===== Machine Safety Monitoring =====
    async function loadSafetyData() {
        try {
            const [alertsRes, eventsRes, configRes, summaryRes] = await Promise.allSettled([
                fetch('/api/safety/alerts'),
                fetch('/api/safety/events?limit=50'),
                fetch('/api/safety/config'),
                fetch('/api/safety/summary'),
            ]);

            let activeAlerts = [];
            let events = [];
            let config = null;
            let summary = null;

            if (alertsRes.status === 'fulfilled' && alertsRes.value.ok) {
                const data = await alertsRes.value.json();
                activeAlerts = data.alerts || [];
            }

            if (eventsRes.status === 'fulfilled' && eventsRes.value.ok) {
                const data = await eventsRes.value.json();
                events = data.events || [];
            }

            if (configRes.status === 'fulfilled' && configRes.value.ok) {
                config = await configRes.value.json();
            }

            if (summaryRes.status === 'fulfilled' && summaryRes.value.ok) {
                summary = await summaryRes.json();
            }

            // Render prominent red alert banner
            renderActiveSafetyBanner(activeAlerts);

            // Render safety table
            renderSafetyEventsTable(events);

            // Update safety KPIs derived directly from persisted data
            const activeCount = summary ? summary.active : activeAlerts.length;
            const totalCount = summary ? summary.total : events.length;
            const resolvedCount = summary ? summary.resolved : events.filter(e => e.status === 'RESOLVED').length;

            if (els.kpiSafetyActive) {
                els.kpiSafetyActive.textContent = activeCount.toString();
            }
            if (els.kpiSafetyActiveMeta) {
                els.kpiSafetyActiveMeta.textContent = activeCount > 0 
                    ? `${activeCount} hazard${activeCount > 1 ? 's' : ''} currently active!` 
                    : 'No active breaches';
            }
            if (els.kpiSafetyTotal) {
                els.kpiSafetyTotal.textContent = totalCount.toString();
            }
            if (els.kpiSafetyResolved) {
                els.kpiSafetyResolved.textContent = resolvedCount.toString();
            }
            if (els.kpiSafetyMode) {
                const isCalibrated = config && config.camera_calibration && config.camera_calibration.enabled;
                els.kpiSafetyMode.textContent = isCalibrated ? 'Metric (m)' : 'Image-Space';
            }
            if (els.kpiSafetyModeMeta) {
                const isCalibrated = config && config.camera_calibration && config.camera_calibration.enabled;
                els.kpiSafetyModeMeta.textContent = isCalibrated ? 'Calibrated Ground-Plane' : 'Perimeter danger zone';
            }

        } catch (err) {
            console.warn('Error loading safety data:', err);
        }
    }

    function renderActiveSafetyBanner(alerts) {
        if (!els.activeSafetyAlertsBanner) return;
        if (!alerts || alerts.length === 0) {
            els.activeSafetyAlertsBanner.innerHTML = '';
            if (els.navAlertBadge) els.navAlertBadge.classList.remove('active');
            return;
        }

        if (els.navAlertBadge) els.navAlertBadge.classList.add('active');

        els.activeSafetyAlertsBanner.innerHTML = alerts.map(alert => {
            const isZoneViolation = alert.event_type === 'restricted_zone_violation';
            const zoneName = alert.zone_name;
            const mId = alert.machine_id ? alert.machine_id : (alert.machine_type ? capitalize(alert.machine_type) + (alert.machine_track_id ? ' #' + alert.machine_track_id : '') : null);
            const wId = alert.worker_id ? alert.worker_id : ('Worker #' + (alert.worker_track_id || ''));
            const isAck = alert.is_acknowledged;
            const statusLabel = isAck ? 'ACTIVE (ACKNOWLEDGED)' : 'ACTIVE';
            const statusBadgeClass = isAck ? 'status-badge acknowledged' : 'status-badge danger';

            const title = isZoneViolation ? 'RESTRICTED-ZONE VIOLATION' : 'MACHINE PROXIMITY ALERT';
            const subtitle = isZoneViolation 
                ? `Worker detected inside the ${escapeHtml(zoneName || 'Loading Bay')} restricted area.`
                : `Worker detected inside a hazardous machine's safety zone.`;

            // Distance: Show ONLY if reliably estimated in metres
            let distanceDisplay = '';
            if (alert.estimated_distance_m !== null && alert.estimated_distance_m !== undefined) {
                distanceDisplay = `${alert.estimated_distance_m} m ${alert.configured_threshold_m ? `(Threshold: ${alert.configured_threshold_m}m)` : ''}`;
            } else if (isZoneViolation) {
                distanceDisplay = `Restricted area breach: ${escapeHtml(zoneName || 'Loading Bay')}`;
            } else {
                distanceDisplay = alert.proximity_label || 'Zone breach (Image-space proximity)';
            }

            const timeStr = alert.last_detected_at ? formatDate(alert.last_detected_at) : (alert.timestamp ? formatDate(alert.timestamp) : 'Active now');
            const eventId = alert.event_id || alert.incident_id || '';

            return `
                <div class="safety-alert-banner" data-event-id="${escapeHtml(eventId)}">
                    <div class="safety-alert-header">
                        <div class="safety-alert-title-wrap">
                            <span class="safety-alert-icon">⚠️</span>
                            <div>
                                <div class="safety-alert-title">
                                    <span>${escapeHtml(title)}</span>
                                    <span class="${statusBadgeClass}">${statusLabel}</span>
                                    <span class="status-badge danger" style="font-size:11px;">${escapeHtml(alert.severity || 'HIGH')}</span>
                                </div>
                                <div class="safety-alert-subtitle">${subtitle}</div>
                            </div>
                        </div>
                    </div>

                    <div class="safety-alert-grid">
                        ${zoneName ? `
                            <div class="safety-meta-item">
                                <span class="safety-meta-label">Zone</span>
                                <span class="safety-meta-val">📍 ${escapeHtml(zoneName)}</span>
                            </div>
                        ` : ''}
                        ${mId ? `
                            <div class="safety-meta-item">
                                <span class="safety-meta-label">Machine</span>
                                <span class="safety-meta-val">🏗️ ${escapeHtml(mId)}</span>
                            </div>
                        ` : ''}
                        <div class="safety-meta-item">
                            <span class="safety-meta-label">Worker</span>
                            <span class="safety-meta-val">👷 ${escapeHtml(wId)}</span>
                        </div>
                        <div class="safety-meta-item">
                            <span class="safety-meta-label">Proximity / Distance</span>
                            <span class="safety-meta-val">📏 ${escapeHtml(distanceDisplay)}</span>
                        </div>
                        <div class="safety-meta-item">
                            <span class="safety-meta-label">Status</span>
                            <span class="safety-meta-val" style="color:var(--danger);">${statusLabel}</span>
                        </div>
                    </div>

                    <div class="safety-alert-footer">
                        <span class="safety-alert-time">
                            <span>🕒</span> Detected: ${timeStr}
                        </span>
                        <div>
                            ${isAck 
                                ? `<span style="font-size:12px;color:var(--text-secondary);display:inline-flex;align-items:center;gap:4px;">
                                    <span>🛡️</span> Acknowledged — Physical Hazard Remains Active
                                   </span>`
                                : `<button class="btn-ack-alert" onclick="Dashboard.handleAcknowledgeAlert('${escapeHtml(eventId)}', this)">
                                    <span>✋</span> Acknowledge Alert
                                   </button>`
                            }
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    function renderSafetyEventsTable(events) {
        if (!els.safetyEventsBody) return;
        if (!events || events.length === 0) {
            els.safetyEventsBody.innerHTML = `
                <tr>
                    <td colspan="8">
                        <div class="history-empty">
                            <div class="empty-icon">🛡️</div>
                            <div>No safety incidents recorded yet. Analyze media to monitor machine proximity.</div>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        els.safetyEventsBody.innerHTML = events.map(ev => {
            const isZone = ev.event_type === 'restricted_zone_violation';
            let targetDisplay = '';
            if (isZone && ev.zone_name) {
                targetDisplay = `📍 ${escapeHtml(ev.zone_name)}`;
                if (ev.machine_type || ev.machine_track_id) {
                    targetDisplay += `<br><small style="color:var(--text-secondary);">🏗️ ${capitalize(ev.machine_type || 'Machine')} #${ev.machine_track_id || ''}</small>`;
                }
            } else {
                const mName = ev.machine_type ? capitalize(ev.machine_type) : 'Machine';
                const mTrack = ev.machine_track_id ? ` #${ev.machine_track_id}` : '';
                targetDisplay = `🏗️ ${escapeHtml(mName + mTrack)}`;
            }

            const wId = ev.worker_track_id ? `Worker #${ev.worker_track_id}` : 'Worker';
            let dist = 'Perimeter Breach (Image-space)';
            if (ev.estimated_distance_m !== null && ev.estimated_distance_m !== undefined) {
                dist = `${ev.estimated_distance_m} m`;
            } else if (isZone && ev.zone_name) {
                dist = `Restricted Zone (${escapeHtml(ev.zone_name)})`;
            }

            const status = ev.status || 'ACTIVE';
            const isAck = ev.is_acknowledged;
            const timeStr = ev.last_detected_at ? formatDate(ev.last_detected_at) : (ev.first_detected_at ? formatDate(ev.first_detected_at) : '--');
            const eventId = ev.event_id || '';

            return `
                <tr>
                    <td><strong>${targetDisplay}</strong></td>
                    <td>👷 ${escapeHtml(wId)}</td>
                    <td><span title="${escapeHtml(ev.proximity_method || '')}">📏 ${escapeHtml(dist)}</span></td>
                    <td><span class="status-badge ${ev.severity === 'CRITICAL' ? 'danger' : 'warning'}">${escapeHtml(ev.severity || 'HIGH')}</span></td>
                    <td><span class="status-badge ${status === 'ACTIVE' ? 'danger' : 'completed'}">${status}</span></td>
                    <td><span class="status-badge ${isAck ? 'acknowledged' : 'pending'}">${isAck ? 'Acknowledged' : 'Unacknowledged'}</span></td>
                    <td>${timeStr}</td>
                    <td>
                        ${isAck 
                            ? `<span style="font-size:12px;color:var(--text-tertiary);">Ack'd ✓</span>`
                            : `<button class="view-btn" style="padding:4px 10px;font-size:12px;" onclick="Dashboard.handleAcknowledgeAlert('${escapeHtml(eventId)}', this)">
                                <span>✋</span> Ack
                               </button>`
                        }
                    </td>
                </tr>
            `;
        }).join('');
    }

    async function handleAcknowledgeAlert(eventId, buttonEl) {
        if (!eventId) return;
        if (buttonEl) {
            buttonEl.disabled = true;
            buttonEl.textContent = 'Acknowledging...';
        }

        try {
            const res = await fetch(`/api/safety/events/${encodeURIComponent(eventId)}/acknowledge`, {
                method: 'POST',
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'Could not acknowledge alert.');
            }
            showToast('Safety alert acknowledged. Note: Physical hazard remains active until cleared.', 'info');
            await loadSafetyData();
        } catch (e) {
            console.error('Failed to acknowledge alert:', e);
            showToast(e.message || 'Error acknowledging alert.', 'error');
            if (buttonEl) {
                buttonEl.disabled = false;
                buttonEl.textContent = 'Acknowledge Alert';
            }
        }
    }

    // Public API
    return {
        init,
        viewAnalysis,
        showToast,
        loadDashboardData,
        scrollToSection,
        closeModal,
        loadSafetyData,
        handleAcknowledgeAlert,
    };
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    Dashboard.init();
});
