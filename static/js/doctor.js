// Placeholder for future interactivity (e.g., filtering, sorting patients)
console.log('Doctor Dashboard loaded.');

// Initialize Bootstrap modal
const historyModal = new bootstrap.Modal(document.getElementById('historyModal'));

// Search and filter functionality
document.getElementById('searchInput').addEventListener('input', filterPatients);
document.getElementById('filterStatus').addEventListener('change', filterPatients);
document.getElementById('sortBy').addEventListener('change', sortPatients);

function filterPatients() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const statusFilter = document.getElementById('filterStatus').value;
    const patientCards = document.querySelectorAll('.patient-card');

    patientCards.forEach(card => {
        const patientName = card.querySelector('h3').textContent.toLowerCase();
        const patientStatus = card.dataset.status;
        const matchesSearch = patientName.includes(searchTerm);
        const matchesStatus = statusFilter === 'all' || patientStatus === statusFilter;

        card.closest('.col-md-6').style.display = matchesSearch && matchesStatus ? 'block' : 'none';
    });
}

function sortPatients() {
    const sortBy = document.getElementById('sortBy').value;
    const patientList = document.getElementById('patient-list');
    const patientCards = Array.from(patientList.getElementsByClassName('col-md-6'));

    patientCards.sort((a, b) => {
        const cardA = a.querySelector('.patient-card');
        const cardB = b.querySelector('.patient-card');

        switch (sortBy) {
            case 'name':
                return cardA.querySelector('h3').textContent.localeCompare(cardB.querySelector('h3').textContent);
            case 'critical':
                return cardB.dataset.status.localeCompare(cardA.dataset.status);
            case 'recent':
            default:
                const dateA = new Date(cardA.querySelector('.vital-signs p:last-child').textContent.split(': ')[1]);
                const dateB = new Date(cardB.querySelector('.vital-signs p:last-child').textContent.split(': ')[1]);
                return dateB - dateA;
        }
    });

    patientCards.forEach(card => patientList.appendChild(card));
}

// Function to view patient history
function viewPatientHistory(registrationId) {
    // Show loading state
    const modal = new bootstrap.Modal(document.getElementById('historyModal'));
    modal.show();
    
    // Fetch patient history
    fetch(`/patient_history/${registrationId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error loading patient history: ' + data.error);
                return;
            }
            
            // Populate patient information
            const patientInfo = document.getElementById('patientInfo');
            patientInfo.innerHTML = `
                <div class="mb-3">
                    <h5>${data.patient_info.basic_info.name}</h5>
                    <p class="mb-1"><strong>Age:</strong> ${data.patient_info.basic_info.age}</p>
                    <p class="mb-1"><strong>Gender:</strong> ${data.patient_info.basic_info.gender}</p>
                    <p class="mb-1"><strong>ID:</strong> ${data.patient_info.basic_info.registration_id}</p>
                    <p class="mb-1"><strong>Total Visits:</strong> ${data.patient_info.basic_info.visit_count}</p>
                    <p class="mb-1"><strong>Last Visit:</strong> ${data.patient_info.basic_info.last_visit}</p>
                </div>
                <div class="mb-3">
                    <h6>Medical Information</h6>
                    <p class="mb-1"><strong>Comorbidities:</strong></p>
                    <ul class="list-unstyled ms-3">
                        ${data.patient_info.medical_info.comorbidities.map(c => `<li>• ${c}</li>`).join('') || '<li>None recorded</li>'}
                    </ul>
                    <p class="mb-1"><strong>Medications:</strong></p>
                    <ul class="list-unstyled ms-3">
                        ${data.patient_info.medical_info.medications.map(m => `<li>• ${m}</li>`).join('') || '<li>None recorded</li>'}
                    </ul>
                </div>
            `;
            
            // Populate current status
            const currentStatus = document.getElementById('currentStatus');
            const latestVitals = data.current_status.latest_vitals;
            currentStatus.innerHTML = `
                <div class="mb-3">
                    <h6>Risk Assessment</h6>
                    <p class="mb-1"><strong>Level:</strong> <span class="badge ${getRiskBadgeClass(data.current_status.risk_level)}">${data.current_status.risk_level}</span></p>
                    <p class="mb-1"><strong>Score:</strong> ${data.current_status.risk_score.toFixed(2)}</p>
                </div>
                ${latestVitals ? `
                <div class="mb-3">
                    <h6>Latest Vitals</h6>
                    <p class="mb-1"><strong>BP:</strong> ${latestVitals.systolic_bp}/${latestVitals.diastolic_bp}</p>
                    <p class="mb-1"><strong>Temperature:</strong> ${latestVitals.temp}°C</p>
                    <p class="mb-1"><strong>Pulse:</strong> ${latestVitals.pulse} bpm</p>
                    <p class="mb-1"><strong>BMI:</strong> ${latestVitals.bmi.toFixed(1)}</p>
                </div>
                ` : ''}
            `;
            
            // Populate comprehensive summary
            const comprehensiveSummary = document.getElementById('comprehensiveSummary');
            comprehensiveSummary.innerHTML = `
                <div class="mb-3">
                    <p>${data.comprehensive_summary.patient_overview}</p>
                    <p>${data.comprehensive_summary.risk_assessment}</p>
                    <p>${data.comprehensive_summary.vital_trends}</p>
                    <p>${data.comprehensive_summary.alert_summary}</p>
                    <p>${data.comprehensive_summary.recommendation_summary}</p>
                </div>
            `;
            
            // Populate vital signs history table
            const vitalSignsTable = document.getElementById('vitalSignsTable').getElementsByTagName('tbody')[0];
            vitalSignsTable.innerHTML = data.history.map(record => `
                <tr>
                    <td>${new Date(record.created_at).toLocaleString()}</td>
                    <td>${record.systolic_bp}/${record.diastolic_bp}</td>
                    <td>${record.temp}°C</td>
                    <td>${record.pulse}</td>
                    <td>${record.bmi.toFixed(1)}</td>
                </tr>
            `).join('');
            
            // Populate trend analysis
            const trendAnalysis = document.getElementById('trendAnalysis');
            trendAnalysis.innerHTML = `
                <div class="mb-3">
                    <h6>Vital Signs Trends</h6>
                    <p>${data.trend_analysis.vital_trends || 'No significant trends detected.'}</p>
                </div>
                <div class="mb-3">
                    <h6>Alert History</h6>
                    <p>${data.trend_analysis.alert_summary || 'No significant alerts in history.'}</p>
                </div>
                <div class="mb-3">
                    <h6>Recommendation History</h6>
                    <p>${data.trend_analysis.recommendation_summary || 'No active recommendations.'}</p>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error loading patient history. Please try again.');
        });
}

// Helper function to get appropriate badge class for risk level
function getRiskBadgeClass(riskLevel) {
    switch(riskLevel) {
        case 'CRITICAL':
            return 'bg-danger';
        case 'HIGH':
            return 'bg-warning';
        case 'MODERATE':
            return 'bg-info';
        case 'LOW':
            return 'bg-success';
        default:
            return 'bg-secondary';
    }
}

// Auto-refresh dashboard every 5 minutes
setInterval(() => {
    window.location.reload();
}, 300000);

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});