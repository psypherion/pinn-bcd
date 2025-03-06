// Handle image upload and prediction
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
    
    // Show the result container and set a loading message.
    const resultDiv = document.getElementById("result");
    const predictionContent = resultDiv.querySelector(".prediction-content");
    resultDiv.style.display = 'flex';
    predictionContent.textContent = "Uploading and processing image...";
    
    // Use FileReader to preview the image in the designated preview container.
    const reader = new FileReader();
    reader.onload = function(e) {
      const previewImg = document.getElementById("preview-image");
      previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) {
        predictionContent.textContent = "Error: " + response.statusText;
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
      
      // Update prediction content with the results.
      predictionContent.innerHTML = outputHTML;
      
      // Dynamically create and append the NGOs Near Me button.
      const ngosButton = document.createElement("button");
      ngosButton.textContent = "NGOs Near Me";
      ngosButton.classList.add("ngos-button");
      ngosButton.addEventListener("click", handleNgosClick);
      predictionContent.appendChild(ngosButton);
      
    } catch (error) {
      predictionContent.textContent = "Error: " + error;
    }
  });
  
  // Function to handle the NGOs Near Me button click.
  function handleNgosClick() {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(function(position) {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        // Use OSM Nominatim for reverse geocoding.
        fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`)
          .then(response => response.json())
          .then(data => {
            // Extract city, town, or village as available.
            const address = data.address;
            let city = address.city || address.town || address.village || address.county;
            if (city) {
              if (confirm(`We detected your location as ${city}. Do you want to find NGOs near you?`)) {
                window.location.href = `https://www.justdial.com/${encodeURIComponent(city)}/NGOS-For-Cancer-Patient`;
              }
            } else {
              alert("Could not determine your nearest city.");
            }
          })
          .catch(error => {
            console.error("Reverse geocoding error:", error);
            alert("Error retrieving location information.");
          });
      }, function(error) {
        console.error("Geolocation error:", error);
        alert("Error retrieving your location. Please ensure location services are enabled.");
      });
    } else {
      alert("Geolocation is not supported by your browser.");
    }
  }
  