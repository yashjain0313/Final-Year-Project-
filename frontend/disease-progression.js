// Disease Progression Detection - Frontend JavaScript

// State management
const state = {
    userId: localStorage.getItem('agrosmart_user_id') || 'anonymous',
    uploadedDays: new Set(),
    uploadedImages: {},
    analysisResults: null,
    chartInstance: null
};

// Trigger file upload
function triggerUpload(day) {
    document.getElementById(`file-${day}`).click();
}

// Handle file selection
async function handleFileSelect(day) {
    const fileInput = document.getElementById(`file-${day}`);
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    // Show preview immediately
    const reader = new FileReader();
    reader.onload = async function(e) {
        const preview = document.getElementById(`preview-${day}`);
        preview.src = e.target.result;
        
        // Update UI state
        document.getElementById(`card-${day}`).classList.add('uploaded');
        document.getElementById(`step-${day}`).classList.add('completed');
        
        // Update timeline progress
        updateTimelineProgress();
        
        // Upload to backend
        await uploadDayImage(day, e.target.result);
    };
    reader.readAsDataURL(file);
}

// Update timeline progress bar
function updateTimelineProgress() {
    const maxDay = Math.max(...Array.from(state.uploadedDays), 0);
    const progress = (maxDay / 5) * 100;
    document.getElementById('timeline-progress').style.width = `${progress}%`;
    
    // Activate next step
    if (maxDay < 5) {
        document.getElementById(`step-${maxDay + 1}`).classList.add('active');
    }
}

// Upload image to backend
async function uploadDayImage(day, imageData) {
    try {
        const response = await fetch('/api/disease-progression/upload-day', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.userId,
                day: day,
                image: imageData
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            state.uploadedDays.add(day);
            state.uploadedImages[day] = imageData;
            
            // Enable analyze button if 3+ images
            if (state.uploadedDays.size >= 3) {
                const btn = document.getElementById('analyze-btn');
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-microscope"></i> Analyze Progression';
            }
            
            updateTimelineProgress();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Failed to upload Day ${day} image: ${error.message}`);
        // Revert UI changes on error
        document.getElementById(`card-${day}`).classList.remove('uploaded');
        document.getElementById(`preview-${day}`).src = '';
    }
}

// Analyze progression
async function analyzeProgression() {
    if (state.uploadedDays.size < 3) {
        alert('Please upload at least 3 images');
        return;
    }
    
    document.getElementById('loading-overlay').classList.add('active');
    
    try {
        const response = await fetch('/api/disease-progression/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: state.userId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result.analysis, result.recommendation);
            // Refresh history after a successful analysis
            loadProgressionHistory();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        document.getElementById('loading-overlay').classList.remove('active');
    }
}

// Display results
function displayResults(analysis, recommendation) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.add('show');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Update text metrics
    document.getElementById('disease-name').textContent = analysis.disease.replace(/_/g, ' ');
    document.getElementById('confidence-badge').innerHTML = `<i class="fas fa-check-circle"></i> Confidence: ${(analysis.confidence * 100).toFixed(1)}%`;
    document.getElementById('severity-value').textContent = `${(analysis.severity_score * 100).toFixed(0)}%`;
    document.getElementById('progression-value').textContent = `${analysis.progression_rate.toFixed(3)}/day`;
    
    // Update Chart
    updateChart(analysis.severity_timeline, analysis.days_analyzed);
    
    // Update Recommendations
    updateRecommendations(recommendation);
}

// Update Chart.js chart
function updateChart(timeline, days) {
    const ctx = document.getElementById('progressionChart').getContext('2d');
    
    if (state.chartInstance) {
        state.chartInstance.destroy();
    }
    
    state.chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days.map(d => `Day ${d}`),
            datasets: [{
                label: 'Disease Severity',
                data: timeline.map(v => v * 100),
                borderColor: '#2e7d32', // Primary Green
                backgroundColor: 'rgba(46, 125, 50, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#2e7d32',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Severity: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Severity (%)' },
                    grid: { color: '#f0f0f0' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

// Update recommendations list
function updateRecommendations(rec) {
    const list = document.getElementById('rec-list');
    const badge = document.getElementById('urgency-badge');
    
    list.innerHTML = '';
    
    // Update badge
    badge.textContent = `${rec.urgency.toUpperCase()} PRIORITY`;
    badge.style.backgroundColor = getUrgencyColor(rec.urgency, true);
    badge.style.color = getUrgencyColor(rec.urgency, false);
    
    // Helper to add items
    const addItem = (icon, title, text) => {
        const div = document.createElement('div');
        div.className = 'rec-item';
        div.innerHTML = `
            <div class="rec-icon"><i class="${icon}"></i></div>
            <div class="rec-content">
                <h4 style="margin-bottom: 5px; color: var(--text-color);">${title}</h4>
                <p style="color: var(--text-secondary); font-size: 0.95rem;">${text}</p>
            </div>
        `;
        list.appendChild(div);
    };
    
    // Add Actions
    rec.actions.forEach(action => addItem('fas fa-exclamation-circle', 'Action Required', action));
    
    // Add Treatments
    rec.treatments.forEach(treatment => addItem('fas fa-spray-can', 'Treatment', treatment));
    
    // Add Monitoring
    if (rec.monitoring) {
        rec.monitoring.forEach(m => addItem('fas fa-eye', 'Monitoring', m));
    }
}

function getUrgencyColor(urgency, isBg) {
    switch(urgency.toLowerCase()) {
        case 'high': return isBg ? '#ffebee' : '#c62828'; // Red
        case 'medium': return isBg ? '#fff3e0' : '#ef6c00'; // Orange
        case 'low': return isBg ? '#e8f5e9' : '#2e7d32'; // Green
        default: return isBg ? '#f5f5f5' : '#757575';
    }
}

// Progression History Loader
async function loadProgressionHistory() {
    const listEl = document.getElementById('progressionHistoryList');
    const emptyMsg = document.getElementById('progHistoryEmptyMsg');
    if (!state.userId || !listEl) return;

    try {
        const resp = await fetch(`/api/disease-history?user_id=${state.userId}`);
        const data = await resp.json();
        
        // Filter specifically for progression scans (instant scans stay on the other page)
        const progHistory = (data.history || []).filter(item => item.scan_type === 'progression');
        
        if (progHistory.length === 0) {
            if (emptyMsg) emptyMsg.style.display = 'block';
            return;
        }
        if (emptyMsg) emptyMsg.style.display = 'none';

        // Clear old cards (keep emptyMsg)
        listEl.querySelectorAll('.history-card').forEach(c => c.remove());

        progHistory.forEach(item => {
            const date = item.created_at ? new Date(item.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
            const conf = item.confidence ? Math.round(item.confidence * 100) : '--';
            const card = document.createElement('div');
            card.className = 'history-card card';
            card.style.cssText = 'padding: 20px; animation: fadeIn 0.3s;';
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background: rgba(46,125,50,0.1); color: var(--primary-color); padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: 600;">
                        <i class="fas fa-chart-line" style="margin-right: 4px;"></i>Progression
                    </span>
                    <span style="font-size: 0.8rem; color: var(--text-secondary);">${date}</span>
                </div>
                <h4 style="color: var(--text-color); margin-bottom: 6px;">${item.disease_label || 'Unknown Disease'}</h4>
                <span style="background: #e8f5e9; color: var(--primary-color); padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: 600;">Confidence: ${conf}%</span>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 8px; line-height: 1.4;">${(item.description || '').substring(0, 120)}${item.description && item.description.length > 120 ? '...' : ''}</p>
            `;
            listEl.appendChild(card);
        });
    } catch (err) {
        console.error('Failed to load progression history:', err);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Disease Progression Tracker Ready');
    loadProgressionHistory();
});
