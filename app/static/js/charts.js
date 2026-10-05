/**
 * WarehouseVision AI - Charts Manager
 *
 * Manages Chart.js instances that display real backend data.
 * All charts automatically adapt to theme changes.
 */

const ChartsManager = (() => {
    let objectDistChart = null;
    let warehouseAnalyticsChart = null;
    let spaceUtilChart = null;

    function getThemeColors() {
        const style = getComputedStyle(document.documentElement);
        return {
            chart1: style.getPropertyValue('--chart-1').trim() || '#4F6AFF',
            chart2: style.getPropertyValue('--chart-2').trim() || '#10B981',
            chart3: style.getPropertyValue('--chart-3').trim() || '#F59E0B',
            chart4: style.getPropertyValue('--chart-4').trim() || '#8B5CF6',
            chart5: style.getPropertyValue('--chart-5').trim() || '#EC4899',
            chart6: style.getPropertyValue('--chart-6').trim() || '#06B6D4',
            grid: style.getPropertyValue('--chart-grid').trim() || 'rgba(0,0,0,0.06)',
            text: style.getPropertyValue('--chart-text').trim() || '#6B7280',
            bg: style.getPropertyValue('--bg-card').trim() || '#FFFFFF',
        };
    }

    function initCharts() {
        initObjectDistribution();
        initWarehouseAnalytics();
        initSpaceUtilization();
    }

    function initObjectDistribution() {
        const ctx = document.getElementById('objectDistChart');
        if (!ctx) return;

        const colors = getThemeColors();

        objectDistChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Workers', 'Forklifts', 'Boxes', 'Pallets'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [colors.chart1, colors.chart2, colors.chart3, colors.chart4],
                    borderWidth: 0,
                    hoverOffset: 8,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: colors.text,
                            padding: 16,
                            usePointStyle: true,
                            pointStyleWidth: 8,
                            font: { family: "'Inter', sans-serif", size: 12, weight: '500' },
                        },
                    },
                    tooltip: {
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.grid,
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        titleFont: { family: "'Inter', sans-serif", weight: '600' },
                        bodyFont: { family: "'Inter', sans-serif" },
                    },
                },
            },
        });
    }

    function initWarehouseAnalytics() {
        const ctx = document.getElementById('warehouseAnalyticsChart');
        if (!ctx) return;

        const colors = getThemeColors();

        warehouseAnalyticsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Workers', 'Forklifts', 'Boxes', 'Pallets'],
                datasets: [{
                    label: 'Detected Objects',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        colors.chart1 + '90',
                        colors.chart2 + '90',
                        colors.chart3 + '90',
                        colors.chart4 + '90',
                    ],
                    borderColor: [colors.chart1, colors.chart2, colors.chart3, colors.chart4],
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.grid,
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        titleFont: { family: "'Inter', sans-serif", weight: '600' },
                        bodyFont: { family: "'Inter', sans-serif" },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: colors.grid, drawBorder: false },
                        ticks: {
                            color: colors.text,
                            font: { family: "'Inter', sans-serif", size: 11 },
                            stepSize: 1,
                        },
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: colors.text,
                            font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
                        },
                    },
                },
            },
        });
    }

    function initSpaceUtilization() {
        const ctx = document.getElementById('spaceUtilChart');
        if (!ctx) return;

        const colors = getThemeColors();

        spaceUtilChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Utilized', 'Available'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [colors.chart2, colors.grid],
                    borderWidth: 0,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',
                rotation: -90,
                circumference: 180,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: colors.bg,
                        titleColor: colors.text,
                        bodyColor: colors.text,
                        borderColor: colors.grid,
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                    },
                },
            },
        });
    }

    function updateObjectDistribution(counts) {
        if (!objectDistChart) return;
        const data = [
            counts.workers || 0,
            counts.forklifts || 0,
            counts.boxes || 0,
            counts.pallets || 0,
        ];
        objectDistChart.data.datasets[0].data = data;
        objectDistChart.update('none');
    }

    function updateWarehouseAnalytics(counts) {
        if (!warehouseAnalyticsChart) return;
        warehouseAnalyticsChart.data.datasets[0].data = [
            counts.workers || 0,
            counts.forklifts || 0,
            counts.boxes || 0,
            counts.pallets || 0,
        ];
        warehouseAnalyticsChart.update('none');
    }

    function updateSpaceUtilization(utilPercent) {
        if (!spaceUtilChart) return;
        const util = Math.min(Math.max(utilPercent || 0, 0), 100);
        spaceUtilChart.data.datasets[0].data = [util, 100 - util];
        spaceUtilChart.update('none');
    }

    function updateAll(data) {
        const counts = data.object_counts || data.counts || {};
        updateObjectDistribution(counts);
        updateWarehouseAnalytics(counts);

        const analytics = data.analytics || {};
        const util = analytics.space_utilization || 0;
        updateSpaceUtilization(util);
    }

    function updateTheme() {
        const colors = getThemeColors();

        // Recreate charts with new theme colors
        if (objectDistChart) {
            objectDistChart.data.datasets[0].backgroundColor = [
                colors.chart1, colors.chart2, colors.chart3, colors.chart4,
            ];
            objectDistChart.options.plugins.legend.labels.color = colors.text;
            objectDistChart.options.plugins.tooltip.backgroundColor = colors.bg;
            objectDistChart.options.plugins.tooltip.titleColor = colors.text;
            objectDistChart.options.plugins.tooltip.bodyColor = colors.text;
            objectDistChart.update('none');
        }

        if (warehouseAnalyticsChart) {
            warehouseAnalyticsChart.data.datasets[0].backgroundColor = [
                colors.chart1 + '90', colors.chart2 + '90', colors.chart3 + '90', colors.chart4 + '90',
            ];
            warehouseAnalyticsChart.data.datasets[0].borderColor = [
                colors.chart1, colors.chart2, colors.chart3, colors.chart4,
            ];
            warehouseAnalyticsChart.options.scales.y.grid.color = colors.grid;
            warehouseAnalyticsChart.options.scales.y.ticks.color = colors.text;
            warehouseAnalyticsChart.options.scales.x.ticks.color = colors.text;
            warehouseAnalyticsChart.options.plugins.tooltip.backgroundColor = colors.bg;
            warehouseAnalyticsChart.options.plugins.tooltip.titleColor = colors.text;
            warehouseAnalyticsChart.options.plugins.tooltip.bodyColor = colors.text;
            warehouseAnalyticsChart.update('none');
        }

        if (spaceUtilChart) {
            spaceUtilChart.data.datasets[0].backgroundColor = [colors.chart2, colors.grid];
            spaceUtilChart.options.plugins.tooltip.backgroundColor = colors.bg;
            spaceUtilChart.update('none');
        }
    }

    return { initCharts, updateAll, updateTheme, updateObjectDistribution, updateWarehouseAnalytics, updateSpaceUtilization };
})();
