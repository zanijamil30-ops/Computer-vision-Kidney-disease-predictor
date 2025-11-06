const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');
const previewImg = document.getElementById('preview-img');
const noImageText = document.getElementById('no-image-text');
const resultDiv = document.getElementById('result');
const topPredP = document.getElementById('top-pred');
const probsList = document.getElementById('probs-list');
const fileHint = document.getElementById('file-hint');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];

  if (!file) {
    // If user cleared selection, revert UI
    selectedFile = null;
    previewImg.style.display = "none";
    noImageText.style.display = "block";
    resultDiv.style.display = "none";
    return;
  }

  // Save selected file but DO NOT display its name anywhere
  selectedFile = file;

  // show preview image and hide "No image selected"
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    previewImg.style.display = "block";
    noImageText.style.display = "none";
    resultDiv.style.display = "none"; // hide previous results
  };
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert("Please choose an image first.");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      body: form
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      alert("Error: " + (err.error || resp.statusText));
      console.error(err);
      return;
    }
    const data = await resp.json();
    // Show results
    topPredP.textContent = `${data.prediction} (probability: ${data.probability})`;
    probsList.innerHTML = "";
    for (const [label, prob] of Object.entries(data.all_probabilities)) {
      const li = document.createElement('li');
      li.textContent = `${label}: ${prob}`;
      probsList.appendChild(li);
    }
    resultDiv.style.display = "block";
  } catch (err) {
    alert("Network or server error. See console for details.");
    console.error(err);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});
