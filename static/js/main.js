/**
 * Breast Cancer Detection System - Client Scripts
 * Handles sample preloading, input form validation, and interactive dynamic charts.
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Example Record Loader
  const loadBenignBtn = document.getElementById("btn-load-benign");
  const loadMalignantBtn = document.getElementById("btn-load-malignant");
  const resetFormBtn = document.getElementById("btn-reset-form");

  if (loadBenignBtn) {
    loadBenignBtn.addEventListener("click", function () {
      loadSampleRecord("benign");
    });
  }

  if (loadMalignantBtn) {
    loadMalignantBtn.addEventListener("click", function () {
      loadSampleRecord("malignant");
    });
  }

  if (resetFormBtn) {
    resetFormBtn.addEventListener("click", function () {
      const inputs = document.querySelectorAll("#prediction-form input[type='number']");
      inputs.forEach((input) => (input.value = ""));
      const alertArea = document.getElementById("sample-alert");
      if (alertArea) alertArea.innerHTML = "";
    });
  }

  function loadSampleRecord(type) {
    const alertArea = document.getElementById("sample-alert");
    if (alertArea) {
      alertArea.innerHTML = `
        <div class="alert alert-info py-2 px-3 small d-flex align-items-center">
          <span class="spinner-border spinner-border-sm me-2" role="status"></span>
          Loading ${type} benchmark sample from Wisconsin dataset...
        </div>`;
    }

    fetch(`/api/sample/${type}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load sample");
        return res.json();
      })
      .then((data) => {
        const dict = data.features_dict;
        for (const [key, value] of Object.entries(dict)) {
          const input = document.getElementById(`input-${key.replace(/\s+/g, "_")}`);
          if (input) {
            input.value = value;
          }
        }
        if (alertArea) {
          alertArea.innerHTML = `
            <div class="alert alert-${type === 'benign' ? 'success' : 'danger'} py-2 px-3 small d-flex align-items-center justify-content-between">
              <span><strong>Sample Loaded:</strong> Successfully populated inputs with a validated <em>${type.toUpperCase()}</em> case from the WDBC dataset.</span>
              <span class="badge bg-secondary">Demo Record</span>
            </div>`;
        }
      })
      .catch((err) => {
        if (alertArea) {
          alertArea.innerHTML = `<div class="alert alert-warning py-2 px-3 small">Error loading sample: ${err.message}</div>`;
        }
      });
  }

  // 2. Client-side Form Validation
  const form = document.getElementById("prediction-form");
  if (form) {
    form.addEventListener("submit", function (e) {
      const inputs = form.querySelectorAll("input[type='number']");
      let hasEmpty = false;
      inputs.forEach((input) => {
        if (!input.value || isNaN(parseFloat(input.value))) {
          input.classList.add("is-invalid");
          hasEmpty = true;
        } else {
          input.classList.remove("is-invalid");
        }
      });

      if (hasEmpty) {
        e.preventDefault();
        const alertArea = document.getElementById("sample-alert");
        if (alertArea) {
          alertArea.innerHTML = `
            <div class="alert alert-danger py-2 px-3 small">
              Please enter valid numerical values for all 30 cytological features before running prediction.
            </div>`;
        }
      }
    });
  }
});
