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

// View patient history
function viewPatientHistory(registrationId) {
    fetch(`/patient_history/${registrationId}`)
        .then(response => response.json())
        .then(data => {
            const historyContent = document.getElementById('historyContent');
            historyContent.innerHTML = '';

            if (data.error) {
                historyContent.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                return;
            }

            // Create patient info section
            const patientInfo = document.createElement('div');
            patientInfo.className = 'card mb-4';
            patientInfo.innerHTML = `
                <div class="card-header">
                    <h5 class="mb-0">Patient Information</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3">
                            <p><strong>Name:</strong> ${data.patient_info.name}</p>
                        </div>
                        <div class="col-md-3">
                            <p><strong>Age:</strong> ${data.patient_info.age}</p>
                        </div>
                        <div class="col-md-3">
                            <p><strong>Gender:</strong> ${data.patient_info.gender}</p>
                        </div>
                        <div class="col-md-3">
                            <p><strong>Last Risk Level:</strong> 
                                <span class="badge bg-${getRiskLevelColor(data.patient_info.last_risk_level)}">
                                    ${data.patient_info.last_risk_level}
                                </span>
                            </p>
                        </div>
                    </div>
                </div>
            `;
            historyContent.appendChild(patientInfo);

            // Create vital signs history table
            const table = document.createElement('table');
            table.className = 'table table-striped table-hover';
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>BMI</th>
                        <th>BP</th>
                        <th>Temp</th>
                        <th>Pulse</th>
                        <th>Risk Level</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.history.map(record => `
                        <tr>
                            <td>${record.date}</td>
                            <td>${record.time}</td>
                            <td>${record.bmi.toFixed(1)}</td>
                            <td>${record.systolic_bp}/${record.diastolic_bp}</td>
                            <td>${record.temp}°F</td>
                            <td>${record.pulse}</td>
                            <td>
                                <span class="badge bg-${getRiskLevelColor(record.risk_level)}">
                                    ${record.risk_level}
                                </span>
                            </td>
                            <td>
                                <button class="btn btn-sm btn-info" onclick="showVitalDetails(${JSON.stringify(record).replace(/"/g, '&quot;')})">
                                    View Details
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
            historyContent.appendChild(table);

            // Create trend analysis section
            if (data.trend_analysis) {
                const trendSection = document.createElement('div');
                trendSection.className = 'card mt-4';
                trendSection.innerHTML = `
                    <div class="card-header">
                        <h5 class="mb-0">Trend Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Vital Signs Trends</h6>
                                <ul class="list-group">
                                    ${Object.entries(data.trend_analysis).map(([key, value]) => `
                                        <li class="list-group-item d-flex justify-content-between align-items-center">
                                            ${key}
                                            <span class="badge bg-${getTrendColor(value)}">${value}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                `;
                historyContent.appendChild(trendSection);
            }

            historyModal.show();
        })
        .catch(error => {
            console.error('Error fetching patient history:', error);
            document.getElementById('historyContent').innerHTML = 
                '<div class="alert alert-danger">Error loading patient history</div>';
            historyModal.show();
        });
}

function showVitalDetails(record) {
    const modal = new bootstrap.Modal(document.getElementById('vitalDetailsModal'));
    const modalBody = document.getElementById('vitalDetailsBody');
    
    modalBody.innerHTML = `
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="mb-0">Summary</h6>
            </div>
            <div class="card-body">
                <pre style="white-space: pre-wrap; font-family: inherit; background: none; border: none; padding: 0; margin: 0;">${record.summary || 'No summary available.'}</pre>
            </div>
        </div>
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="mb-0">Alerts</h6>
            </div>
            <div class="card-body">
                ${record.alerts && record.alerts.length > 0 ? `<ul class='list-unstyled'>${record.alerts.map(alert => `<li class='alert ${alert.includes('Critical') ? 'alert-danger' : 'alert-warning'} mb-2'>${alert}</li>`).join('')}</ul>` : '<div class="alert alert-success">No alerts</div>'}
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Recommendations</h6>
            </div>
            <div class="card-body">
                ${record.recommendations && record.recommendations.length > 0 ? `<ul class='list-unstyled'>${record.recommendations.map(rec => `<li class='recommendation-item'>${rec}</li>`).join('')}</ul>` : '<div class="alert alert-success">No recommendations</div>'}
            </div>
        </div>
    `;
    
    modal.show();
}

function getRiskLevelColor(level) {
    switch(level) {
        case 'CRITICAL': return 'danger';
        case 'HIGH': return 'warning';
        case 'MODERATE': return 'info';
        case 'LOW': return 'success';
        default: return 'secondary';
    }
}

function getTrendColor(trend) {
    if (trend.includes('increasing')) return 'danger';
    if (trend.includes('decreasing')) return 'success';
    return 'info';
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