document.addEventListener("DOMContentLoaded", () => {
    console.log("Multimodal RAG loaded!");

    // Enable 'Enter' key for querying
    let queryInput = document.getElementById("query");
    if (queryInput) {
        queryInput.addEventListener("keypress", function (e) {
            if (e.key === "Enter") {
                askQuestion();
            }
        });
    }
});

// -------------- Upload Files --------------
function uploadFile() {
    let formData = new FormData();
    let fileInput = document.getElementById("file-upload");
    let metadata = document.getElementById("metadata").value;
    let uploadStatus = document.getElementById("upload-status");
    let loadingMessage = document.getElementById("loading-message");

    // Clear previous messages
    uploadStatus.innerText = "";
    loadingMessage.style.display = "block";
    loadingMessage.innerText = "Uploading file, please wait...";

    for (let file of fileInput.files) {
        formData.append("files", file);
    }
    formData.append("metadata", metadata);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingMessage.style.display = "none";
        if (Array.isArray(data) && data.length > 0) {
            uploadStatus.innerText = "File uploaded successfully!";
            const firstResp = data[0];
            document.getElementById("extracted-text").innerText = firstResp.extracted_text || "N/A";
            showChunkData(firstResp.chunk_data || []);
        } else {
            uploadStatus.innerText = "Error: No data returned.";
        }
    })
    .catch(error => {
        console.error("Upload Error:", error);
        uploadStatus.innerText = "Upload failed.";
        loadingMessage.style.display = "none";
    });
}

// -------------- Upload Text --------------
function uploadText() {
    let textInput = document.getElementById("text-input").value;
    let metadata = document.getElementById("metadata").value;
    let textStatus = document.getElementById("text-status");
    let loadingMessage = document.getElementById("loading-message");

    // Clear status
    textStatus.innerText = "";
    loadingMessage.style.display = "block";
    loadingMessage.innerText = "Processing text, please wait...";

    fetch("/upload-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput, metadata: metadata })
    })
    .then(response => response.json())
    .then(data => {
        loadingMessage.style.display = "none";
        textStatus.innerText = "Text uploaded successfully!";
        document.getElementById("extracted-text").innerText = data.extracted_text || "N/A";
        showChunkData(data.chunk_data || []);
    })
    .catch(error => {
        console.error("Upload Error:", error);
        textStatus.innerText = "Text upload failed.";
        loadingMessage.style.display = "none";
    });
}

// -------------- Show Chunk Data --------------
function showChunkData(chunkData) {
    let chunkResultsDiv = document.getElementById("chunked-results");
    chunkResultsDiv.innerHTML = ""; 

    if (!chunkData.length) {
        chunkResultsDiv.innerHTML = "<p>No chunks generated.</p>";
        return;
    }

    let ul = document.createElement("ul");
    chunkData.forEach((obj, index) => {
        let chunkText = obj.chunk;
        let embedPreview = JSON.stringify(obj.embedding).slice(0, 60) + "...";
        let li = document.createElement("li");
        li.innerHTML = `<strong>• ${chunkText}</strong><br>[${embedPreview}]`;
        ul.appendChild(li);
    });
    chunkResultsDiv.appendChild(ul);
}

// -------------- Ask a Question --------------
function askQuestion() {
    let queryText = document.getElementById("query").value;
    let responseElement = document.getElementById("response");
    let kValue = document.getElementById("k-value").value;
    let responseMode = document.querySelector('input[name="response_mode"]:checked').value;
    let loadingMessage = document.getElementById("loading-message");

    if (!queryText.trim()) {
        responseElement.innerText = "Please enter a question.";
        return;
    }

    loadingMessage.style.display = "block";
    loadingMessage.innerText = "Fetching response...";

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText, response_mode: responseMode, k: kValue })
    })
    .then(response => response.json())
    .then(data => {
        loadingMessage.style.display = "none";
        const queryEmbedding = data.query_embedding || "None";
        const chunkArray = data.retrieved_chunk || [];
        const embeddingArray = data.retrieved_embedding || [];
        const llmResponse = data.response || "No response.";

        document.getElementById("query-embedding").innerText = JSON.stringify(queryEmbedding);

        if (chunkArray.length > 0) {
            document.getElementById("retrieved-text").innerText = chunkArray.join("\n---\n");
        } else {
            document.getElementById("retrieved-text").innerText = "No retrieved chunks.";
        }

        if (embeddingArray.length > 0) {
            let embedText = embeddingArray.map((emb, idx) => {
                return `Chunk ${idx+1} Embedding: ${JSON.stringify(emb).slice(0,50)}...`;
            }).join("\n");
            document.getElementById("retrieved-embedding").innerText = embedText;
        } else {
            document.getElementById("retrieved-embedding").innerText = "No retrieved embeddings.";
        }

        responseElement.innerText = llmResponse;
    })
    .catch(error => {
        console.error("Query Error:", error);
        loadingMessage.style.display = "none";
        responseElement.innerText = "Failed to fetch response.";
    });
}

// -------------- Reset the DB --------------
function resetDB() {
    fetch("/reset-db", {
        method: "POST"
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            alert(data.message || "Database reset successful!");
        }
    })
    .catch(error => {
        console.error("Error resetting DB:", error);
        alert("Error resetting DB.");
    });
}

// -------------- Show Stored Data --------------
function showDBContents() {
    fetch("/db-contents")
    .then(response => response.json())
    .then(data => {
        let dbOutput = document.getElementById("db-output");
        if (!dbOutput) return;

        dbOutput.innerHTML = ""; // Clear old content

        if (data.error) {
            dbOutput.innerText = "Error: " + data.error;
            return;
        }
        if (data.documents && data.documents.length > 0) {
            let html = "<ul>";
            data.documents.forEach(doc => {
                html += `<li><strong>ID:</strong> ${doc.id}<br>` +
                        `<strong>Raw Text:</strong> ${doc.metadata.raw_text}<br>` +
                        `<strong>Metadata:</strong> ${JSON.stringify(doc.metadata.metadata)}</li><br>`;
            });
            html += "</ul>";
            dbOutput.innerHTML = html;
        } else {
            dbOutput.innerText = "No documents found in the database.";
        }
    })
    .catch(error => {
        console.error("Error fetching DB contents:", error);
        let dbOutput = document.getElementById("db-output");
        if (dbOutput) dbOutput.innerText = "Error fetching DB contents.";
    });
}
