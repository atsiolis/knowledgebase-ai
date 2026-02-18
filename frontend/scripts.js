const API_URL = "http://localhost:8000";

async function handleAsk() {
    const input = document.getElementById('userInput');
    const chatWindow = document.getElementById('chatWindow');
    const loading = document.getElementById('loading');
    const question = input.value.trim();

    if (!question) return;

    // Remove welcome message if present
    const welcomeMsg = chatWindow.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    // 1. Add User Message
    chatWindow.innerHTML += `<div class="message user">${question}</div>`;
    input.value = "";
    loading.classList.remove('hidden');
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        // 2. Call FastAPI
        const response = await fetch(`${API_URL}/ask?question=${encodeURIComponent(question)}`);
        const data = await response.json();

        // 3. Prepare Source citations
        const sourceHtml = data.sources && data.sources.length > 0 
            ? `<div class="sources"><strong>Sources:</strong> ${[...new Set(data.sources)].join(", ")}</div>` 
            : "";

        // 4. Add Bot Message
        chatWindow.innerHTML += `
            <div class="message bot">
                ${data.answer}
                ${sourceHtml}
            </div>
        `;
    } catch (error) {
        chatWindow.innerHTML += `<div class="message bot" style="color: red;">Failed to connect to backend.</div>`;
    } finally {
        loading.classList.add('hidden');
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

async function handleUpload() {
    const status = document.getElementById('uploadStatus');
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.innerHTML = "⏳ Uploading...";
    try {
        const resp = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
        const res = await resp.json();
        
        if (res.status === "processing" && res.upload_id) {
            // Start polling for progress
            pollProgress(res.upload_id);
        } else if (res.status === "error") {
            status.innerHTML = `❌ Error: ${res.detail}`;
        }
    } catch (e) {
        status.innerHTML = "❌ Upload Failed";
    }
}

async function pollProgress(uploadId) {
    const status = document.getElementById('uploadStatus');
    
    const pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`${API_URL}/upload/status/${uploadId}`);
            const data = await resp.json();
            
            if (data.status === "not_found") {
                clearInterval(pollInterval);
                status.innerHTML = "❌ Upload not found";
                return;
            }
            
            // Update progress bar
            const progress = data.progress || 0;
            status.innerHTML = `
                <div class="progress-container">
                    <div class="progress-bar" style="width: ${progress}%"></div>
                </div>
                <div class="progress-text">${data.message}</div>
            `;
            
            // Stop polling when complete or error
            if (data.status === "complete") {
                clearInterval(pollInterval);
                setTimeout(() => {
                    status.innerHTML = `<div class="progress-text" style="color: #059669;">${data.message}</div>`;
                    loadDocuments(); // Refresh document list
                }, 500);
            } else if (data.status === "error") {
                clearInterval(pollInterval);
                status.innerHTML = `<div class="progress-text" style="color: #dc2626;">${data.message}</div>`;
            }
        } catch (e) {
            clearInterval(pollInterval);
            status.innerHTML = "❌ Connection error";
        }
    }, 500); // Poll every 500ms for smooth updates
}

// Load and display documents
async function loadDocuments() {
    try {
        const resp = await fetch(`${API_URL}/documents`);
        const docs = await resp.json();
        
        const docsList = document.getElementById('documentsList');
        
        if (!docs || docs.length === 0) {
            docsList.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>No documents yet.<br>Upload a file to get started.</p>
                </div>
            `;
            return;
        }
        
        docsList.innerHTML = docs.map(doc => `
            <div class="document-item">
                <i class="fa-solid fa-file-pdf"></i>
                <span class="document-name">${doc.name}</span>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

// Add 'Enter' key support
document.getElementById("userInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleAsk();
});

// Trigger upload when file is selected
document.getElementById("fileInput").onchange = handleUpload;

// Load documents on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
});