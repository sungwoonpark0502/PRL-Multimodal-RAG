document.addEventListener("DOMContentLoaded", () => {
    console.log("Dark Mode Active");

    // Handle query input
    document.getElementById("query").addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            askQuestion();
        }
    });
});

function askQuestion() {
    let queryText = document.getElementById("query").value;
    let responseElement = document.getElementById("response");

    if (!queryText.trim()) {
        responseElement.innerText = "Please enter a question.";
        return;
    }

    // Display loading effect
    responseElement.innerHTML = "Thinking<span class='dots'></span>";

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            responseElement.innerText = data.response;
        } else {
            responseElement.innerText = "Error fetching response.";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        responseElement.innerText = "Failed to fetch response.";
    });
}