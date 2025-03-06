document.getElementById("uploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    
    const input = document.getElementById("imageInput");
    if (input.files.length === 0) {
      alert("Please select an image.");
      return;
    }
    
    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    // Show a loading message in the result div.
    const resultDiv = document.getElementById("result");
    resultDiv.style.display = 'block';
    resultDiv.textContent = "Uploading and processing image...";
    
    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) {
        resultDiv.textContent = "Error: " + response.statusText;
        return;
      }
      
      const data = await response.json();
      let outputHTML = `<h3>Prediction Result:</h3>`;
      outputHTML += `<p><strong>Predicted Class:</strong> ${data.predicted_class}</p>`;
      outputHTML += `<p><strong>Confidence:</strong> ${data.confidence}</p>`;
      
      if (data.predictions) {
        outputHTML += "<p><strong>Raw Predictions:</strong></p><ul>";
        data.predictions.forEach((pred, index) => {
          const predNumber = Number(pred);
          if (!isNaN(predNumber)) {
            outputHTML += `<li>Class ${index}: ${predNumber.toFixed(4)}</li>`;
          } else {
            outputHTML += `<li>Class ${index}: ${pred}</li>`;
          }
        });
        outputHTML += "</ul>";
      }
      resultDiv.innerHTML = outputHTML;
    } catch (error) {
      resultDiv.textContent = "Error: " + error;
    }
  });
  