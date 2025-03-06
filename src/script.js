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
    
    // Show a loading message.
    const resultDiv = document.getElementById("result");
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
      resultDiv.innerHTML = `<strong>Prediction:</strong> ${data.predicted_class} <br> <strong>Confidence:</strong> ${data.confidence}`;
    } catch (error) {
      resultDiv.textContent = "Error: " + error;
    }
  });
  