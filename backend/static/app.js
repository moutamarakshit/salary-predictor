const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");

const resultEmpty = document.getElementById("result-empty");
const resultFilled = document.getElementById("result-filled");
const resultLoading = document.getElementById("result-loading");

const resultNumber = document.getElementById("result-number");
const resultRange = document.getElementById("result-range");
const factR2 = document.getElementById("fact-r2");
const factMae = document.getElementById("fact-mae");

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function showPanel(panel) {
  [resultEmpty, resultFilled, resultLoading].forEach((el) => {
    el.hidden = el !== panel;
  });
}

function populateSelect(selectEl, options, { includeBlank = false } = {}) {
  selectEl.innerHTML = "";
  if (includeBlank) {
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Select…";
    selectEl.appendChild(blank);
  }
  options.forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    selectEl.appendChild(opt);
  });
}

function populateDatalist(datalistEl, options) {
  datalistEl.innerHTML = "";
  options.forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    datalistEl.appendChild(opt);
  });
}

async function loadMetadata() {
  try {
    const res = await fetch("/api/metadata");
    if (!res.ok) throw new Error("metadata unavailable");
    const meta = await res.json();

    populateDatalist(document.getElementById("title-options"), meta.dropdowns.title || []);
    populateDatalist(document.getElementById("company-options"), meta.dropdowns.company || []);
    populateSelect(document.getElementById("nation"), meta.dropdowns.nation || []);
    populateSelect(document.getElementById("education"), meta.dropdowns.education || []);
    populateSelect(document.getElementById("gender"), meta.dropdowns.gender || []);
    populateSelect(document.getElementById("race"), meta.dropdowns.race || []);

    if (meta.metrics) {
      factR2.textContent = meta.metrics.r2 != null ? meta.metrics.r2.toFixed(2) : "—";
      factMae.textContent =
        meta.metrics.mae != null ? currencyFormatter.format(meta.metrics.mae) : "—";
    }
  } catch (err) {
    formError.textContent =
      "Couldn't load model metadata. Have you run `python src/train.py` yet? See the README.";
    formError.hidden = false;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  submitBtn.disabled = true;
  showPanel(resultLoading);

  const payload = {
    title: document.getElementById("title").value.trim(),
    company: document.getElementById("company").value.trim(),
    nation: document.getElementById("nation").value,
    education: document.getElementById("education").value,
    gender: document.getElementById("gender").value,
    race: document.getElementById("race").value,
    yearsofexperience: document.getElementById("yearsofexperience").value,
    yearsatcompany: document.getElementById("yearsatcompany").value,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong");
    }

    resultNumber.textContent = currencyFormatter.format(data.predicted_salary);
    resultRange.textContent = `Likely range: ${currencyFormatter.format(
      data.range_low
    )} – ${currencyFormatter.format(data.range_high)}`;
    showPanel(resultFilled);
  } catch (err) {
    formError.textContent = err.message;
    formError.hidden = false;
    showPanel(resultEmpty);
  } finally {
    submitBtn.disabled = false;
  }
});

loadMetadata();
