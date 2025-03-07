document.getElementById('mainForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const output = document.getElementById('responseOutput');
    const submitBtn = document.querySelector('#mainForm button[type="submit"]');
    
    output.innerHTML = "<div class='loading'>🔄 Processing...</div>";
    submitBtn.disabled = true;
    submitBtn.classList.add('processing');

    const fileInput = document.getElementById('fileInput');
    const textInput = document.getElementById('textInput').value;
    const formData = new FormData();

    try {
        if (fileInput.files.length > 0) {
            // File upload
            formData.append('file', fileInput.files[0]);
            formData.append('text', textInput);

            const uploadStart = Date.now();
            const res = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || `Upload failed (${res.status})`);
            }
            
            const result = await res.json();
            output.innerHTML = `<div class="success">${result.message}</div>`;
            fileInput.value = "";
        } else if (textInput.trim()) {
            // Query processing
            const queryStart = Date.now();
            const res = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ query: textInput })
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || `Query failed (${res.status})`);
            }
            
            const result = await res.json();
            const processTime = res.headers.get('X-Process-Time');
            
            let responseHTML = result.response.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            if (result.context_used > 0) {
                responseHTML += `<div class="context-used">🔍 Used ${result.context_used} knowledge sources</div>`;
            }
            responseHTML += `<div class="processing-time">⏱️ Processed in ${processTime}s</div>`;
            
            output.innerHTML = `<div class="response">${responseHTML}</div>`;
        }
        
        document.getElementById('textInput').value = "";
    } catch (err) {
        output.innerHTML = `<div class="error">❌ ${err.message}</div>`;
        console.error('Operation failed:', err);
    } finally {
        submitBtn.disabled = false;
        submitBtn.classList.remove('processing');
    }
});