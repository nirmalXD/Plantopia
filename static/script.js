function uploadImage() {
    let input = document.getElementById("imageInput");
    if (input.files.length === 0) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultElement = document.getElementById("result");
        if (data.error) {
            resultElement.innerHTML = `<p style="color:red;">${data.error}</p>`;
        } else {
            // Display prediction, confidence, description, and possible steps
            let resultHtml = `<p style="font-size: 1.5em; font-weight: bold;">Prediction: ${data.prediction}</p>`;
            if (data.message) {
                resultHtml += `<p style="color:orange; font-size: 1.5em;">${data.message}</p>`;
            }
            resultHtml += `<p style="font-size: 1.5em; font-weight: bold;">Confidence: ${data.confidence}</p>`;
            resultHtml += `<p style="text-align: justify; font-size: 18px;"><strong>Description:</strong> ${data.description}</p>`;
            resultHtml += `<p style="text-align: justify; font-size: 18px;"><strong>Possible Steps:</strong> ${data.possible_steps}</p>`;            
            if (data.image_url) {
                resultHtml += `<img src="${data.image_url}" alt="Disease Image" style="max-width: 100%; height: auto; margin-top: 10px;">`;
            }
            resultElement.innerHTML = resultHtml;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `<p style="color:red;">Error occurred during image prediction. Please try again later.</p>`;
    });
}
