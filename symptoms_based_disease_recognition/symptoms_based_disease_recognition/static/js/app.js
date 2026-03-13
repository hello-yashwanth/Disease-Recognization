/* =========================================
   API CALL
========================================= */

async function postPredict(payload) {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const errorMsg = data.error || `HTTP ${res.status}: Prediction failed`;
      throw new Error(errorMsg);
    }

    return await res.json();
  } catch (error) {
    console.error("Prediction request failed:", error);
    throw error;
  }
}

/* =========================================
   DOM READY
========================================= */

document.addEventListener("DOMContentLoaded", () => {

  /* ==============================
     PREDICTOR PAGE
  ============================== */

  const form = document.getElementById("symptom-form");

  if (form) {

    const addBtn = document.getElementById("add-symptom");

    if (addBtn) {
      addBtn.addEventListener("click", () => addRow());
    }

    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("remove") || e.target.closest(".remove")) {

        const row = e.target.closest(".symptom-row");
        if (!row) return;

        row.style.transition = "all 0.3s ease";
        row.style.opacity = "0";
        row.style.transform = "translateX(-20px)";

        setTimeout(() => row.remove(), 300);
      }
    });

    form.addEventListener("submit", async (e) => {

      e.preventDefault();

      const symptoms = gatherInput();

      if (Object.keys(symptoms).length === 0) {
        alert("Please enter at least one symptom");
        return;
      }

      const patientName =
        document.getElementById("patient-name")?.value || "";

      try {

        showLoading();

        const res = await postPredict({
          symptoms: symptoms,
          patient_name: patientName
        });

        hideLoading();

        displayResults(res);

      } catch (err) {

        hideLoading();

        console.error(err);

        const msg = String(err?.message || "").toLowerCase();

        if (
          msg.includes("login") ||
          msg.includes("unauthorized") ||
          msg.includes("401")
        ) {
          alert("Please login first.");
          window.location.href = "/auth";
        } else {
          alert("Prediction failed");
        }

      }

    });

  }

  /* ==============================
     CONTACT PAGE
  ============================== */

  const contactForm = document.getElementById("contactForm");

  if (contactForm) {

    contactForm.addEventListener("submit", async function (e) {

      e.preventDefault();

      const name = document.getElementById("contact-name")?.value || "";
      const email = document.getElementById("contact-email")?.value || "";
      const message = document.getElementById("contact-message")?.value || "";

      try {

        const res = await fetch("/api/contact", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            name,
            email,
            message
          })
        });

        const data = await res.json();

        if (data.success) {

          alert("Message Submitted Successfully!");

          contactForm.reset();

        } else {

          alert(data.message || "Submission Failed");

        }

      } catch (err) {

        console.error(err);

        alert("Server error while sending message");

      }

    });

  }

});


/* =========================================
   ADD ROW
========================================= */

function addRow() {

  const container = document.getElementById("symptom-list");

  if (!container) return;

  const div = document.createElement("div");

  div.className = "symptom-row d-flex gap-2 mb-3 align-items-center";

  div.innerHTML = `
  <div class="flex-grow-1">
    <input name="symptom"
      list="symptoms-list"
      class="form-control"
      placeholder="fever, cough, headache">
  </div>

  <div style="width:120px">
    <input name="severity"
      class="form-control"
      type="number"
      min="1"
      max="5"
      placeholder="1-5">
  </div>

  <div style="width:140px">
    <input name="duration"
      class="form-control"
      type="number"
      min="1"
      placeholder="Days">
  </div>

  <div>
    <button class="btn btn-outline-danger remove">
      <i class="fas fa-times"></i>
    </button>
  </div>
  `;

  container.appendChild(div);

}


/* =========================================
   COLLECT INPUT
========================================= */

function gatherInput() {

  const rows = document.querySelectorAll(".symptom-row");

  const symptoms = {};

  const used = new Set();

  rows.forEach(row => {

    const s = row.querySelector("input[name=symptom]")
      ?.value.trim().toLowerCase();

    if (!s || s.length < 2) return;

    if (used.has(s)) return;

    used.add(s);

    const sev = row.querySelector("input[name=severity]")?.value;

    const dur = row.querySelector("input[name=duration]")?.value;

    if (sev || dur) {

      symptoms[s] = { presence: 1 };

      if (sev) symptoms[s].severity = Number(sev);

      if (dur) symptoms[s].duration = Number(dur);

    } else {

      symptoms[s] = 1;

    }

  });

  return symptoms;

}


/* =========================================
   LOADING STATE
========================================= */

function showLoading() {

  const btn = document.getElementById("submit");

  if (!btn) return;

  btn.disabled = true;

  btn.innerHTML =
    '<i class="fas fa-spinner fa-spin"></i> Predicting...';

}


function hideLoading() {

  const btn = document.getElementById("submit");

  if (!btn) return;

  btn.disabled = false;

  btn.innerHTML =
    '<i class="fas fa-brain"></i> Analyze & Predict';

}


/* =========================================
   DISPLAY RESULTS
========================================= */

function displayResults(res) {

  const result = document.getElementById("result");

  if (!result) return;

  result.classList.remove("d-none");

  const prediction = document.getElementById("prediction");

  const confidence = Math.round((res.confidence || 0) * 100);

  if (prediction) {
    prediction.innerHTML = `
      <strong>${res.prediction}</strong>
      <span style="margin-left:10px">(${confidence}%)</span>
    `;
  }

  const dl = document.getElementById("download-report");

  if (dl && res.report_id) {

    dl.href = `/api/report/${res.report_id}/download`;

    dl.classList.remove("d-none");

  }

  const tf = document.getElementById("top-features");

  if (tf) {

    tf.innerHTML = "";

    const features = res.top_features_by_shap || [];

    features.forEach(f => {

      const li = document.createElement("li");

      li.textContent = f;

      tf.appendChild(li);

    });

  }

  const tt = document.getElementById("tests");

  if (tt) {

    tt.innerHTML = "";

    const tests = res.recommended_tests || [];

    tests.forEach(t => {

      const li = document.createElement("li");

      li.textContent = t;

      tt.appendChild(li);

    });

  }

  result.scrollIntoView({
    behavior: "smooth"
  });

}