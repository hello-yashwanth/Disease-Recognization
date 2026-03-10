async function postPredict(payload) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data.error || 'Prediction failed';
    throw new Error(errorMsg);
  }
  return data;
}

// Add smooth animations
function animateElement(element, animationClass) {
  element.classList.add(animationClass);
  setTimeout(() => element.classList.remove(animationClass), 1000);
}

document.addEventListener('click', (e) => {
  if (e.target && e.target.id === 'add-symptom') {
    addRow();
  }
  if (e.target && e.target.classList.contains('remove')) {
    const row = e.target.closest('.symptom-row');
    if (row) {
      row.style.transition = 'all 0.3s ease';
      row.style.opacity = '0';
      row.style.transform = 'translateX(-20px)';
      setTimeout(() => row.remove(), 300);
    }
  }
});

function addRow() {
  const div = document.createElement('div');
  div.className = 'symptom-row d-flex gap-2 mb-3 align-items-center';
  div.style.opacity = '0';
  div.style.transform = 'translateY(-10px)';
  div.innerHTML = `
    <div class="flex-grow-1">
      <label class="form-label small text-muted mb-1">
        <i class="fas fa-search me-1"></i>Symptom Name
      </label>
      <input name="symptom" list="symptoms-list" class="form-control" placeholder="e.g., fever, headache, cough" />
    </div>
    <div style="width: 120px;">
      <label class="form-label small text-muted mb-1">
        <i class="fas fa-exclamation-triangle me-1"></i>Severity (1-5)
      </label>
      <input name="severity" class="form-control" type="number" min="1" max="5" placeholder="1-5" />
    </div>
    <div style="width: 140px;">
      <label class="form-label small text-muted mb-1">
        <i class="fas fa-calendar-alt me-1"></i>Duration (days)
      </label>
      <input name="duration" class="form-control" type="number" min="1" placeholder="Days" />
    </div>
    <div class="d-flex align-items-end" style="height: 58px;">
      <button class="btn btn-outline-danger remove" type="button" title="Remove symptom">
        <i class="fas fa-times"></i>
      </button>
    </div>
  `;
  document.getElementById('symptom-list').appendChild(div);
  
  // Animate in
  setTimeout(() => {
    div.style.transition = 'all 0.3s ease';
    div.style.opacity = '1';
    div.style.transform = 'translateY(0)';
  }, 10);
  
  // Focus on the new symptom input
  setTimeout(() => {
    div.querySelector('input[name="symptom"]').focus();
  }, 350);
}

function gatherInput() {
  const rows = Array.from(document.querySelectorAll('.symptom-row'));
  const out = {};
  rows.forEach(r => {
    const s = r.querySelector('input[name=symptom]').value.trim().toLowerCase();
    if (!s) return;
    const sev = r.querySelector('input[name=severity]').value;
    const dur = r.querySelector('input[name=duration]').value;
    if (sev || dur) {
      out[s] = { presence: 1 };
      if (sev) out[s].severity = Number(sev);
      if (dur) out[s].duration = Number(dur);
    } else {
      out[s] = 1;
    }
  });
  return out;
}

function showLoading() {
  const submitBtn = document.getElementById('submit');
  const form = document.getElementById('symptom-form');
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
  form.classList.add('loading');
}

function hideLoading() {
  const submitBtn = document.getElementById('submit');
  const form = document.getElementById('symptom-form');
  submitBtn.disabled = false;
  submitBtn.innerHTML = '<i class="fas fa-brain me-2"></i>Analyze & Predict';
  form.classList.remove('loading');
}

function displayResults(res) {
  const resultEl = document.getElementById('result');
  resultEl.classList.remove('d-none');
  resultEl.style.opacity = '0';
  resultEl.style.transform = 'translateY(20px)';
  
  // Animate result section in
  setTimeout(() => {
    resultEl.style.transition = 'all 0.5s ease';
    resultEl.style.opacity = '1';
    resultEl.style.transform = 'translateY(0)';
  }, 10);
  
  // Update prediction
  const predictionEl = document.getElementById('prediction');
  const confidence = Math.round((res.confidence || 0) * 100);
  const confidenceColor = confidence >= 70 ? 'var(--accent)' : confidence >= 50 ? 'var(--primary)' : 'var(--muted)';
  predictionEl.innerHTML = `
    <div>
      <strong>${res.prediction}</strong>
      <span style="color: ${confidenceColor}; font-size: 1rem; margin-left: 1rem;">
        ${confidence}% confidence
      </span>
    </div>
  `;
  
  // Update top features
  const tf = document.getElementById('top-features');
  tf.innerHTML = '';
  const features = res.top_features_by_shap || res.top_features_by_model || [];
  if (features.length === 0) {
    tf.innerHTML = '<li style="color: var(--muted);">No feature data available</li>';
  } else {
    features.forEach((s, index) => {
      const li = document.createElement('li');
      li.textContent = s;
      li.style.opacity = '0';
      li.style.transform = 'translateX(-10px)';
      tf.appendChild(li);
      setTimeout(() => {
        li.style.transition = 'all 0.3s ease';
        li.style.opacity = '1';
        li.style.transform = 'translateX(0)';
      }, index * 100);
    });
  }
  
  // Update tests
  const tt = document.getElementById('tests');
  tt.innerHTML = '';
  const tests = res.recommended_tests || [];
  if (tests.length === 0) {
    tt.innerHTML = '<li style="color: var(--muted);">No specific tests recommended at this time</li>';
  } else {
    tests.forEach((t, index) => {
      const li = document.createElement('li');
      li.textContent = t;
      li.style.opacity = '0';
      li.style.transform = 'translateX(-10px)';
      tt.appendChild(li);
      setTimeout(() => {
        li.style.transition = 'all 0.3s ease';
        li.style.opacity = '1';
        li.style.transform = 'translateX(0)';
      }, index * 100);
    });
  }
  
  // Scroll to results
  setTimeout(() => {
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

document.addEventListener('submit', async (ev) => {
  if (ev.target && ev.target.id === 'symptom-form') {
    ev.preventDefault();
    const symptoms = gatherInput();
    
    if (Object.keys(symptoms).length === 0) {
      alert('Please enter at least one symptom');
      return;
    }
    
    try {
      showLoading();
      const res = await postPredict({ symptoms });
      hideLoading();
      displayResults(res);
    } catch (error) {
      hideLoading();
      const errorMessage = error.message || 'An error occurred while predicting. Please try again.';
      alert(errorMessage);
      console.error('Prediction error:', error);
    }
  }
});

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});
