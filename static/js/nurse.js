document.getElementById('vitals-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    const submitButton = e.target.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';

    try {
        const formData = {
            registration_id: document.getElementById('registration_id').value,
            name: document.getElementById('name').value,
            gender: document.getElementById('gender').value,
            age: parseInt(document.getElementById('age').value),
            date: document.getElementById('date').value,
            time: document.getElementById('time').value,
            height: parseFloat(document.getElementById('height').value),
            weight: parseFloat(document.getElementById('weight').value),
            temp: parseFloat(document.getElementById('temp').value),
            systolic_bp: parseInt(document.getElementById('systolic_bp').value),
            diastolic_bp: parseInt(document.getElementById('diastolic_bp').value),
            pulse: parseInt(document.getElementById('pulse').value),
            pain_scale: parseInt(document.getElementById('pain_scale').value)
        };

        const response = await fetch('/submit_vitals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }

        // Update BMI display
        document.getElementById('bmi').textContent = result.vitals.bmi;
        
        // Show success message
        showNotification('Vitals submitted successfully!', 'success');

        // Reset form
        e.target.reset();

    } catch (error) {
        console.error('Error:', error);
        showNotification(error.message || 'Error submitting vitals. Please try again.', 'error');
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalButtonText;
    }
});

// Function to show notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    notification.style.zIndex = '1050';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.body.appendChild(notification);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Add input validation
const numericInputs = document.querySelectorAll('input[type="number"]');
numericInputs.forEach(input => {
    input.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        const min = parseFloat(e.target.min);
        const max = parseFloat(e.target.max);

        if (value < min) {
            e.target.value = min;
        } else if (value > max) {
            e.target.value = max;
        }
    });
});

// Add date validation
document.getElementById('date').addEventListener('input', (e) => {
    const selectedDate = new Date(e.target.value);
    const today = new Date();
    
    if (selectedDate > today) {
        e.target.value = today.toISOString().split('T')[0];
        showNotification('Date cannot be in the future', 'warning');
    }
});